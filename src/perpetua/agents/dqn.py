"""Continuing masked DQN/IQL with a bounded replay, explicit RNG and target sync.

Standard DQN under Perpetua's lifetime contract: no episodic resets or done
masking, constant exploration, a fixed-shape ring replay, explicit sampling
keys and periodic target-network synchronization. ``DQN`` shares one
Q-network across fixed agent slots with active-masked losses; ``IQL`` keeps
independent per-slot models, optimizers, targets and sampling keys, mirroring
how IPPO extends PPO.
"""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import functools
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
from gymnasium import spaces

from perpetua import compute, metrics, probes, runtime
from perpetua.agents._common import (
    Bridge,
    advance_cursor,
    as_float32,
    bridge_step,
    check_discrete_actions,
    check_environment_pixel_scale,
    check_torso_observations,
    checked_hyperparameters,
    compile_chunk,
    init_bridge,
    l2_regularized,
    periodic_reset,
    runner_metadata,
    set_floats,
    training_health,
    weighted_mean,
)
from perpetua.agents.encoders import (
    Activation,
    CNNTorso,
    Torso,
    check_pixel_scale,
    resolve_pixel_scale,
)
from perpetua.agents.plasticity import Params, adam, crelu, graft_top_layers
from perpetua.core import Clock, Environment, Info, JSONValue, as_int

__all__ = [
    "Transition",
    "DQNConfig",
    "QNetwork",
    "Replay",
    "DQNHyperparameters",
    "DQNState",
    "DQNStats",
    "DQNTrainingState",
    "Pending",
    "init_replay",
    "insert",
    "sample",
    "masked_greedy",
    "DQN",
    "reset_dqn_layers",
    "IQL",
    "init_dqn",
    "DQNRunner",
    "HostDQNRunner",
]

type Transition = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class DQNConfig:
    """DQN structure plus defaults for the traced hyperparameters.

    The continuous values that ``DQNHyperparameters`` also carries are only
    defaults; the copy in ``DQNState`` is used. The L2 penalty is gated
    statically by ``l2_coefficient`` here. There is no finite-lifetime
    annealing schedule.

    Sizes count ticks, not transitions: one replay row is one tick and holds
    every agent slot. Shared DQN averages its loss over ``batch_size * slots``
    transitions per gradient step (IQL gives each slot ``batch_size``), and
    ``buffer_capacity`` ticks hold ``buffer_capacity * slots`` transitions.

    - ``rollout_steps``: ticks collected per runner call, before its updates.
    - ``buffer_capacity``: ticks kept in the replay ring.
    - ``batch_size``: ticks sampled (with replacement) per gradient step.
    - ``gradient_steps``: gradient steps per runner call.
    - ``hidden_size``: width of the hidden layers (CReLU doubles it).
    - ``learning_rate``, ``gamma``: step size and discount of the TD update.
    - ``reward_centering``, ``centering_step_size``: subtract a running average
      reward rate from the reward, updated at this rate.
    - ``epsilon``: probability of a uniformly random valid action per slot.
    - ``target_update_interval``: gradient steps between target-network copies.
    - ``max_grad_norm``: global gradient-norm clip.
    - ``activation``, ``torso``: ``"tanh"`` or ``"crelu"``; ``"mlp"`` for
      ``(slots, features)`` observations or ``"cnn"`` for pixels.
    - ``l2_coefficient``: weight of an L2 penalty on the parameters (0 is off).
    - ``reset_interval``: gradient steps between resets of the last
      ``reset_last_layers`` layers (``None`` never resets).
    - ``pixel_scale``: what the CNN torso divides observations by. ``None``
      derives it once, when the learner is built, from the
      ``observation_space`` given to the learner (255 for bounds of 0 and 255,
      else 1; 1 without a space); a number sets it whatever the space says.
    """

    rollout_steps: int = 128
    buffer_capacity: int = 4096
    batch_size: int = 128
    gradient_steps: int = 4
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    reward_centering: bool = False
    centering_step_size: float = 0.01
    epsilon: float = 0.05
    target_update_interval: int = 64
    max_grad_norm: float = 10.0
    activation: Activation = "tanh"
    torso: Torso = "mlp"
    l2_coefficient: float = 0.0
    reset_interval: int | None = None
    reset_last_layers: int = 2
    pixel_scale: float | None = None

    def __post_init__(self) -> None:
        if self.torso not in {"mlp", "cnn"}:
            raise ValueError("torso must be 'mlp' or 'cnn'.")
        check_pixel_scale(self.pixel_scale)
        if self.pixel_scale is not None:
            object.__setattr__(self, "pixel_scale", float(self.pixel_scale))
        for name in (
            "rollout_steps",
            "buffer_capacity",
            "batch_size",
            "gradient_steps",
            "hidden_size",
            "target_update_interval",
        ):
            value = as_int(getattr(self, name), f"{name} must be a positive integer.", minimum=1)
            object.__setattr__(self, name, value)
        if self.batch_size > self.buffer_capacity:
            raise ValueError("batch_size must not exceed buffer_capacity.")
        if self.rollout_steps > self.buffer_capacity:
            raise ValueError(
                f"rollout_steps ({self.rollout_steps}) must not exceed buffer_capacity "
                f"({self.buffer_capacity}): both count ticks, and the oldest "
                f"{self.rollout_steps - self.buffer_capacity} ticks of every rollout would be "
                "overwritten before any update could sample them."
            )
        if self.activation not in {"tanh", "crelu"}:
            raise ValueError("activation must be 'tanh' or 'crelu'.")
        if self.reset_interval is not None:
            interval = as_int(
                self.reset_interval, "reset_interval must be None or a positive integer.", minimum=1
            )
            object.__setattr__(self, "reset_interval", interval)
        last_layers = as_int(
            self.reset_last_layers, "reset_last_layers must be a nonnegative integer.", minimum=0
        )
        object.__setattr__(self, "reset_last_layers", last_layers)
        set_floats(
            self,
            (
                "learning_rate",
                "gamma",
                "centering_step_size",
                "epsilon",
                "max_grad_norm",
                "l2_coefficient",
            ),
        )
        for name in ("learning_rate", "gamma", "epsilon", "max_grad_norm", "l2_coefficient"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if type(self.reward_centering) is not bool:
            raise ValueError("reward_centering must be a boolean.")
        if not math.isfinite(self.centering_step_size):
            raise ValueError("centering_step_size must be finite.")
        if not 0 <= self.centering_step_size <= 1:
            raise ValueError("centering_step_size must be in [0, 1].")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Learning rate and maximum gradient norm must be positive.")
        if self.l2_coefficient < 0:
            raise ValueError("l2_coefficient must be nonnegative.")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be in [0, 1].")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1].")


class QNetwork(nn.Module):
    """Two-layer Q-network mirroring the reference actor initialization.

    ``activation`` selects tanh or concatenated ReLU hidden activations;
    CReLU widens each hidden output to ``2 * hidden_size``. With
    ``torso="cnn"`` a ``CNNTorso`` encodes pixel observations first.
    """

    action_size: int
    hidden_size: int
    activation: Activation = "tanh"
    torso: Torso = "mlp"
    pixel_scale: float = 1.0

    def _activate(self, value: jax.Array) -> jax.Array:
        return crelu(value) if self.activation == "crelu" else nn.tanh(value)

    @nn.compact
    def __call__(
        self, obs: jax.Array, probe: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        if self.torso == "cnn":
            obs = CNNTorso(self.hidden_size, self.activation, self.pixel_scale)(obs)
        hidden = obs
        for _ in range(2):
            hidden = self._activate(
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.orthogonal(math.sqrt(2)),
                    bias_init=nn.initializers.zeros_init(),
                )(hidden)
            )
        q_values = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        if probe:
            return q_values, hidden
        return q_values


class Replay(NamedTuple):
    """Fixed-shape ring replay; ``cursor`` counts insertions, bounded once full.

    It equals the insertion count until ``2 * capacity`` and then stays in
    ``[capacity, 2 * capacity)``, so it never overflows.
    """

    obs: jax.Array
    mask: jax.Array
    active: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    next_mask: jax.Array
    bridge: jax.Array
    cursor: jax.Array

    @property
    def filled(self) -> jax.Array:
        """Number of valid entries, saturating at the fixed capacity."""
        return jnp.minimum(self.cursor, self.obs.shape[0])


class DQNHyperparameters(NamedTuple):
    """Continuous hyperparameters carried as traced state.

    Structural choices (rollout length, buffer capacity, batch size, target
    sync interval, hidden size) stay in ``DQNConfig``; these values live in
    ``DQNState`` so sweeps over them are data and can be vmapped.
    """

    learning_rate: jax.Array
    gamma: jax.Array
    centering_step_size: jax.Array
    epsilon: jax.Array
    max_grad_norm: jax.Array
    l2_coefficient: jax.Array

    @classmethod
    def from_config(cls, config: "DQNConfig") -> "DQNHyperparameters":
        """Read the configured defaults as float32 scalars."""
        return cls(*[jnp.asarray(getattr(config, name), dtype=jnp.float32) for name in cls._fields])

    def as_dict(self) -> dict[str, float | list[float]]:
        """Host-side values for reports; stacked per-slot values become lists.

        Entries must be concrete (no tracers): call it on a training state,
        not inside a compiled update.
        """
        report: dict[str, float | list[float]] = {}
        for name in self._fields:
            value = np.asarray(jnp.asarray(getattr(self, name)))
            if value.ndim == 0:
                report[name] = float(value)
            else:
                report[name] = [float(entry) for entry in value.reshape(-1)]
        return report

    def validate(self) -> None:
        """Check concrete overrides; skipped inside traced transformations."""
        values = checked_hyperparameters(self)
        if values is None:
            return
        if not 0 <= values["gamma"] <= 1:
            raise ValueError("Hyperparameter gamma must be in [0, 1].")
        if not 0 <= values["epsilon"] <= 1:
            raise ValueError("Hyperparameter epsilon must be in [0, 1].")
        if values["l2_coefficient"] < 0:
            raise ValueError("Hyperparameter l2_coefficient must be nonnegative.")


class DQNState(NamedTuple):
    """Online/target parameters, optimizer moments and explicit step counters.

    ``reset_age`` counts ``update`` calls since the last periodic top-layer
    reset, including calls skipped because the batch was all inactive. IQL
    stacks every leaf, including ``reset_age``, on an agent axis.
    ``hyperparameters`` carries the continuous knobs as traced scalars.
    """

    params: Params
    target: Params
    opt_state: optax.OptState
    gradient_steps: Clock
    sync_age: jax.Array
    reset_age: jax.Array
    hyperparameters: DQNHyperparameters
    reward_rate: jax.Array


class DQNStats(NamedTuple):
    """Bounded diagnostics: scalar loss, mean action value and gradient norm."""

    loss: jax.Array
    q_mean: jax.Array
    grad_norm: jax.Array


class DQNTrainingState[StateT](NamedTuple):
    """Full lifetime, learner/target/optimizer, replay buffer and diagnostics."""

    lifetime: runtime.Lifetime[StateT]
    agent: DQNState
    replay: Replay
    optimizer_key: jax.Array
    updates: Clock
    statistics: DQNStats
    healthy: jax.Array
    pending: "Pending"


class Pending(NamedTuple):
    """What each slot observed and did when it last acted, until its transition is stored."""

    obs: jax.Array
    mask: jax.Array
    action: jax.Array
    bridge: Bridge


def init_replay(
    *, num_agents: int, obs_dim: int | tuple[int, ...], action_size: int, capacity: int
) -> Replay:
    """Allocate one bounded fixed-shape replay buffer."""
    num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
    if isinstance(obs_dim, int):
        obs_shape = (obs_dim,)
        if obs_dim < 1:
            raise ValueError("obs_dim must be a positive integer.")
    else:
        obs_shape = tuple(int(dimension) for dimension in obs_dim)
        if not obs_shape or any(dimension < 1 for dimension in obs_shape):
            raise ValueError("obs_dim must be a positive integer or a nonempty shape.")
    action_size = as_int(action_size, "action_size must be a positive integer.", minimum=1)
    capacity = as_int(capacity, "capacity must be a positive integer.", minimum=1)
    return Replay(
        jnp.zeros((capacity, num_agents, *obs_shape), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents, action_size), dtype=jnp.bool_),
        jnp.zeros((capacity, num_agents), dtype=jnp.bool_),
        jnp.zeros((capacity, num_agents), dtype=jnp.int32),
        jnp.zeros((capacity, num_agents), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents, *obs_shape), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents, action_size), dtype=jnp.bool_),
        jnp.ones((capacity, num_agents), dtype=jnp.float32),
        jnp.int32(0),
    )


def insert(replay: Replay, transition: Transition, bridge: jax.Array | None = None) -> Replay:
    """Write one (time, agents) transition into the ring, wrapping in place.

    ``bridge`` is the extra discount of a transition that spans inactive
    ticks; it defaults to 1, an ordinary one-tick transition.
    """
    obs, mask, active, action, reward, next_obs, next_mask = transition
    if bridge is None:
        bridge = jnp.ones(replay.bridge.shape[1:], dtype=jnp.float32)
    expected = (
        (replay.obs.shape[1:], obs),
        (replay.mask.shape[1:], mask),
        (replay.active.shape[1:], active),
        (replay.action.shape[1:], action),
        (replay.reward.shape[1:], reward),
        (replay.next_obs.shape[1:], next_obs),
        (replay.next_mask.shape[1:], next_mask),
        (replay.bridge.shape[1:], bridge),
    )
    for shape, value in expected:
        if jnp.shape(value) != shape:
            raise ValueError("Replay transitions must match fixed agent/observation shapes.")
    index = replay.cursor % replay.obs.shape[0]

    def write(buffer: jax.Array, value: jax.Array) -> jax.Array:
        return buffer.at[index].set(value)

    return Replay(
        write(replay.obs, obs),
        write(replay.mask, mask),
        write(replay.active, active),
        write(replay.action, action),
        write(replay.reward, reward),
        write(replay.next_obs, next_obs),
        write(replay.next_mask, next_mask),
        write(replay.bridge, bridge),
        advance_cursor(replay.cursor, replay.obs.shape[0]),
    )


def sample(replay: Replay, key: jax.Array, batch_size: int) -> Replay:
    """Draw a with-replacement batch from the filled prefix using explicit RNG.

    Requires at least one inserted transition; the cursor is left unchanged.
    """
    batch_size = as_int(batch_size, "batch_size must be a positive integer.", minimum=1)
    indices = jax.random.randint(key, (batch_size,), 0, replay.filled)
    return replay._replace(
        obs=replay.obs[indices],
        mask=replay.mask[indices],
        active=replay.active[indices],
        action=replay.action[indices],
        reward=replay.reward[indices],
        next_obs=replay.next_obs[indices],
        next_mask=replay.next_mask[indices],
        bridge=replay.bridge[indices],
    )


def masked_greedy(q_values: jax.Array, mask: jax.Array) -> jax.Array:
    """Argmax over valid actions only; masked entries are never selected."""
    minimum = cast(float, jnp.finfo(jnp.float32).min)
    return jnp.argmax(jnp.where(mask, q_values, minimum), axis=-1).astype(jnp.int32)


class DQN:
    """Masked continuing DQN with bounded replay and periodic target sync.

    With ``torso="cnn"``, pass ``observation_space=env.observation_space`` so
    raw 8-bit pixels are divided by 255. The divisor is fixed here for the
    learner's life and kept as ``pixel_scale`` (see ``DQNConfig``).
    """

    def __init__(
        self,
        *,
        action_size: int,
        config: DQNConfig | None = None,
        observation_space: spaces.Space[Any] | None = None,
    ) -> None:
        action_size = as_int(action_size, "action_size must be a positive integer.", minimum=1)
        config = DQNConfig() if config is None else config
        self.config = config
        self.action_size = action_size
        self.pixel_scale = resolve_pixel_scale(config.pixel_scale, observation_space)
        self.network = QNetwork(
            action_size=action_size,
            hidden_size=config.hidden_size,
            activation=config.activation,
            torso=config.torso,
            pixel_scale=self.pixel_scale,
        )
        self.optimizer = adam(
            config.reset_last_layers if config.reset_interval is not None else None
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Host-side learner configuration recorded in every training checkpoint."""
        configuration: dict[str, JSONValue] = {
            "algorithm": "dqn",
            "action_size": self.action_size,
            "config": asdict(self.config),
        }
        if self.config.torso == "cnn":
            configuration["pixel_scale"] = self.pixel_scale
        return configuration

    def init(
        self,
        obs: jax.Array,
        key: jax.Array,
        hyperparameters: DQNHyperparameters | None = None,
    ) -> DQNState:
        """Initialize from observations of shape (agent slots, features).

        With ``torso="cnn"`` observations are (agent slots, H, W, C).
        ``hyperparameters`` overrides the configured defaults as traced
        scalars; the values are stored in the returned state so sweeps are
        data and can be vmapped.
        """
        if obs.ndim < 2 or obs.shape[0] < 1 or obs.size < 1:
            raise ValueError("DQN observations must have shape (agents, features).")
        check_torso_observations(obs, self.config.torso, "DQN")
        params = cast(Params, self.network.init(key, obs.astype(jnp.float32)))
        resolved = (
            DQNHyperparameters.from_config(self.config)
            if hyperparameters is None
            else cast(DQNHyperparameters, jax.tree.map(as_float32, hyperparameters))
        )
        resolved.validate()
        return DQNState(
            params,
            params,
            self.optimizer.init(params),
            Clock.from_int(0),
            jnp.int32(0),
            jnp.int32(0),
            resolved,
            jnp.zeros((), dtype=jnp.float32),
        )

    def q_values(self, state: DQNState, obs: jax.Array) -> jax.Array:
        """Return action values for observations (agents, features).

        With ``torso="cnn"`` observations are (agents, H, W, C).
        """
        return cast(jax.Array, self.network.apply(state.params, obs.astype(jnp.float32)))

    def probe(
        self,
        state: DQNState,
        obs: jax.Array,
        *,
        threshold: float = 1e-3,
        weights: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Health probes on the final Q-network hidden layer: dormant units and rank.

        On-demand diagnostics outside the compiled update path; they consume no
        RNG and never mutate the learner. The network is shared across slots, so
        one report covers every slot; IQL probes each slot separately.
        """
        _, hidden = cast(
            tuple[jax.Array, jax.Array],
            self.network.apply(state.params, obs.astype(jnp.float32), probe=True),
        )
        return {
            "dormant_q": probes.dormant_unit_ratio(hidden, threshold=threshold, weights=weights),
            "rank_q": probes.effective_rank(hidden, weights=weights),
        }

    def zero_statistics(self, num_agents: int) -> DQNStats:
        """All-zero diagnostics matching this learner's statistics shape."""
        del num_agents
        zero = jnp.zeros((), dtype=jnp.float32)
        return DQNStats(zero, zero, zero)

    def act(self, state: DQNState, obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
        """Epsilon-greedy masked actions; inactive slots select placeholder zero."""
        q_values = self.q_values(state, obs)
        greedy = masked_greedy(q_values, info["action_mask"])
        explore_key, uniform_key = jax.random.split(key)
        logits = jnp.where(info["action_mask"], 0.0, -jnp.inf)
        uniform = jax.random.categorical(uniform_key, logits, axis=-1).astype(jnp.int32)
        take_uniform = jax.random.bernoulli(
            explore_key, p=state.hyperparameters.epsilon, shape=greedy.shape
        )
        action = jnp.where(take_uniform, uniform, greedy)
        return jnp.where(info["active_mask"], action, 0).astype(jnp.int32)

    def _loss(
        self,
        hyperparameters: DQNHyperparameters,
        params: Params,
        target: Params,
        batch: Replay,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        minimum = cast(float, jnp.finfo(jnp.float32).min)
        q_values = cast(jax.Array, self.network.apply(params, batch.obs))
        action_q = jnp.take_along_axis(q_values, batch.action[..., None], axis=-1)[..., 0]
        next_q = cast(jax.Array, self.network.apply(target, batch.next_obs))
        masked_max = jnp.max(jnp.where(batch.next_mask, next_q, minimum), axis=-1)
        # A slot with no valid next action (it leaves the population) has nothing to
        # bootstrap from; without this the target would be the float32 minimum.
        masked_max = jnp.where(jnp.any(batch.next_mask, axis=-1), masked_max, 0.0)
        discount = hyperparameters.gamma * batch.bridge
        td_target = jax.lax.stop_gradient(batch.reward + discount * masked_max)
        losses = optax.huber_loss(action_q, td_target, delta=1.0)
        loss = weighted_mean(losses, batch.active)
        loss = l2_regularized(
            loss, params, hyperparameters.l2_coefficient, configured=self.config.l2_coefficient
        )
        q_mean = weighted_mean(action_q, batch.active)
        td_mean = weighted_mean(td_target - action_q, batch.active)
        return loss, (q_mean, td_mean)

    def update(self, state: DQNState, replay: Replay, key: jax.Array) -> tuple[DQNState, DQNStats]:
        """One replay-sampled gradient step plus scheduled target synchronization."""
        hyperparameters = state.hyperparameters
        reset_key = key
        if self.config.reset_interval is not None:
            key, reset_key = jax.random.split(key)
        batch = sample(replay, key, self.config.batch_size)
        if self.config.reward_centering:
            batch = batch._replace(reward=batch.reward - state.reward_rate)
        (loss, (q_mean, td_mean)), grads = jax.value_and_grad(self._loss, argnums=1, has_aux=True)(
            state.hyperparameters, state.params, state.target, batch
        )
        rate = (
            state.reward_rate + hyperparameters.centering_step_size * td_mean
            if self.config.reward_centering
            else state.reward_rate
        )

        def sync_leaf(new: jax.Array, old: jax.Array, *, sync: jax.Array) -> jax.Array:
            return jnp.where(sync, new, old)

        def apply_gradients(agent: DQNState) -> DQNState:
            norm = optax.tree.norm(grads)
            scale = jnp.minimum(1.0, agent.hyperparameters.max_grad_norm / (norm + 1e-6))

            def rescale(leaf: Any) -> Any:
                return leaf * scale

            def descend(leaf: Any, update: Any) -> Any:
                return leaf - agent.hyperparameters.learning_rate * update

            clipped = jax.tree.map(rescale, grads)
            updates, opt_state = self.optimizer.update(clipped, agent.opt_state, agent.params)
            params = cast(Params, jax.tree.map(descend, agent.params, updates))
            steps = agent.gradient_steps.increment()
            age = agent.sync_age + jnp.int32(1)
            sync = age >= self.config.target_update_interval
            target = cast(
                Params,
                jax.tree.map(functools.partial(sync_leaf, sync=sync), params, agent.target),
            )
            return DQNState(
                params,
                target,
                opt_state,
                steps,
                jnp.where(sync, jnp.int32(0), age),
                agent.reset_age,
                agent.hyperparameters,
                rate,
            )

        def unchanged(agent: DQNState) -> DQNState:
            return agent

        trained = cast(
            DQNState, jax.lax.cond(jnp.any(batch.active), apply_gradients, unchanged, state)
        )
        if self.config.reset_interval is not None:

            def perform_reset(agent: DQNState) -> DQNState:
                fresh_params = cast(
                    Params, self.network.init(reset_key, batch.obs[0].astype(jnp.float32))
                )
                fresh = DQNState(
                    fresh_params,
                    agent.target,
                    self.optimizer.init(fresh_params),
                    agent.gradient_steps,
                    agent.sync_age,
                    jnp.int32(0),
                    agent.hyperparameters,
                    agent.reward_rate,
                )
                return reset_dqn_layers(agent, fresh, self.config.reset_last_layers)

            trained = periodic_reset(trained, self.config.reset_interval, perform_reset)
        return trained, DQNStats(loss, q_mean, optax.tree.norm(grads))


def reset_dqn_layers(state: DQNState, fresh: DQNState, count: int) -> DQNState:
    """Replace the final ``count`` layers of online/target params and moments.

    The target network receives the same fresh top layers as the online
    network (their trunks keep their own trained values), so bootstrapping
    does not immediately fight the reset. Ages and counters are preserved
    except for ``reset_age``, which restarts at zero.
    """
    params, opt_state = graft_top_layers(
        state.params, state.opt_state, fresh.params, fresh.opt_state, count
    )
    target, _ = graft_top_layers(
        state.target, state.opt_state, fresh.params, fresh.opt_state, count
    )
    return DQNState(
        params,
        target,
        opt_state,
        state.gradient_steps,
        state.sync_age,
        jnp.int32(0),
        state.hyperparameters,
        state.reward_rate,
    )


class IQL:
    """Independent DQN models/optimizers per fixed agent slot, without sharing.

    Mirrors IPPO: every slot keeps its own parameters, target network,
    optimizer moments and sampling subkeys; a slot's rewards/gradients cannot
    update another slot, and wholly inactive slots skip updates. The shared
    bounded replay stores every slot's transitions; each agent's update
    samples its own column with independent keys.
    """

    def __init__(
        self,
        *,
        action_size: int,
        config: DQNConfig | None = None,
        observation_space: spaces.Space[Any] | None = None,
    ) -> None:
        self._dqn = DQN(action_size=action_size, config=config, observation_space=observation_space)
        self.config = self._dqn.config
        self.action_size = self._dqn.action_size
        self.pixel_scale = self._dqn.pixel_scale

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record independent, fixed-slot learners rather than one shared model."""
        return {**self._dqn.configuration, "algorithm": "iql", "parameter_sharing": False}

    def init(
        self,
        obs: jax.Array,
        key: jax.Array,
        hyperparameters: DQNHyperparameters | None = None,
    ) -> DQNState:
        """Initialize separate trees for observations (agents, features).

        With ``torso="cnn"`` observations are (agents, H, W, C).
        ``hyperparameters`` may be scalar (broadcast to every slot) or carry a
        leading agent axis; ``None`` uses the configured defaults.
        """
        if obs.ndim < 2 or obs.shape[0] < 1 or obs.size < 1:
            raise ValueError("IQL observations must have shape (agents, features).")
        check_torso_observations(obs, self.config.torso, "IQL")
        keys = jax.random.split(key, obs.shape[0])
        if hyperparameters is None:
            return jax.vmap(self._dqn.init)(obs[:, None, :], keys)

        def per_slot_leaf(leaf: Any) -> jax.Array:
            if jnp.ndim(leaf) == 1:
                return as_float32(leaf)
            return jnp.broadcast_to(as_float32(leaf), (obs.shape[0],))

        per_slot = cast(DQNHyperparameters, jax.tree.map(per_slot_leaf, hyperparameters))
        # Inside vmap the values are traced and DQN's own check is skipped, so
        # check each slot's concrete values here.
        for slot in range(obs.shape[0]):

            def pick(leaf: jax.Array, slot: int = slot) -> jax.Array:
                return leaf[slot]

            cast(DQNHyperparameters, jax.tree.map(pick, per_slot)).validate()
        return jax.vmap(self._dqn.init)(obs[:, None, :], keys, per_slot)

    def zero_statistics(self, num_agents: int) -> DQNStats:
        """All-zero per-slot diagnostics."""
        zero = jnp.zeros((num_agents,), dtype=jnp.float32)
        return DQNStats(zero, zero, zero)

    def act(self, state: DQNState, obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
        """Epsilon-greedy per-slot actions with independent keys and masks."""

        def draw(
            learner: DQNState,
            obs: jax.Array,
            mask: jax.Array,
            active: jax.Array,
            respawn: jax.Array,
            key: jax.Array,
        ) -> jax.Array:
            local: Info = {
                "action_mask": mask[None, :],
                "active_mask": active[None],
                "respawn": respawn[None],
                "t": info["t"],
            }
            return self._dqn.act(learner, obs[None, :], local, key)[0]

        return jax.vmap(draw)(
            state,
            obs,
            info["action_mask"],
            info["active_mask"],
            info["respawn"],
            jax.random.split(key, obs.shape[0]),
        )

    def probe(
        self,
        state: DQNState,
        obs: jax.Array,
        *,
        threshold: float = 1e-3,
        weights: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Probe every independent slot on the same observation batch.

        IQL keeps separate models per slot, so the report carries a leading
        agent axis (unlike shared DQN). On-demand diagnostics outside the
        compiled update path; they consume no RNG and never mutate the learner.
        """

        def one(learner: DQNState) -> dict[str, jax.Array]:
            return self._dqn.probe(learner, obs, threshold=threshold, weights=weights)

        return jax.vmap(one)(state)

    def update(self, state: DQNState, replay: Replay, key: jax.Array) -> tuple[DQNState, DQNStats]:
        """Optimize each (slot, replay column) independently."""
        if state.gradient_steps.lo.ndim != 1:
            raise ValueError("IQL requires per-slot stacked gradient clocks.")
        key, draw_key = jax.random.split(key)

        by_agent = replay._replace(
            obs=jnp.swapaxes(replay.obs, 0, 1),
            mask=jnp.swapaxes(replay.mask, 0, 1),
            active=jnp.swapaxes(replay.active, 0, 1),
            action=jnp.swapaxes(replay.action, 0, 1),
            reward=jnp.swapaxes(replay.reward, 0, 1),
            next_obs=jnp.swapaxes(replay.next_obs, 0, 1),
            next_mask=jnp.swapaxes(replay.next_mask, 0, 1),
            bridge=jnp.swapaxes(replay.bridge, 0, 1),
        )

        def single_agent(value: jax.Array) -> jax.Array:
            return value[:, None]

        def update_agent(
            learner: DQNState, data: Replay, agent_key: jax.Array
        ) -> tuple[DQNState, DQNStats]:
            data = data._replace(
                obs=single_agent(data.obs),
                mask=single_agent(data.mask),
                active=single_agent(data.active),
                action=single_agent(data.action),
                reward=single_agent(data.reward),
                next_obs=single_agent(data.next_obs),
                next_mask=single_agent(data.next_mask),
                bridge=single_agent(data.bridge),
            )
            return self._dqn.update(learner, data, agent_key)

        # The scalar cursor is shared across slots and must not be mapped.
        axes = cast(
            Replay,
            Replay(*(0,) * 8 + (None,)),  # type: ignore[arg-type]
        )
        trained, statistics = jax.vmap(update_agent, in_axes=(0, axes, 0))(
            state, by_agent, jax.random.split(draw_key, state.gradient_steps.lo.shape[0])
        )
        return trained, statistics


def init_dqn[State](
    env: Environment[State],
    agent: DQN | IQL,
    *,
    seed: int | jax.Array,
    window_size: int,
    hyperparameters: DQNHyperparameters | None = None,
) -> DQNTrainingState[State]:
    """Initialize a continuing lifetime, learner, replay buffer and counters."""
    lifetime = runtime.init_lifetime(env, seed=seed, window_size=window_size)
    agent_key, init_key, optimizer_key = jax.random.split(lifetime.agent_key, 3)
    lifetime = lifetime._replace(agent_key=agent_key)
    mask = lifetime.info["action_mask"]
    check_discrete_actions(env, mask, agent.action_size, type(agent).__name__)
    check_environment_pixel_scale(env, agent, type(agent).__name__)
    state = agent.init(lifetime.obs, init_key, hyperparameters)
    replay = init_replay(
        num_agents=mask.shape[0],
        obs_dim=lifetime.obs.shape[1:],
        action_size=mask.shape[-1],
        capacity=agent.config.buffer_capacity,
    )
    return DQNTrainingState(
        lifetime,
        state,
        replay,
        optimizer_key,
        Clock.from_int(0),
        agent.zero_statistics(mask.shape[0]),
        jnp.asarray(True),
        Pending(
            jnp.zeros_like(lifetime.obs),
            jnp.zeros_like(mask),
            jnp.zeros(mask.shape[0], dtype=jnp.int32),
            init_bridge(mask.shape[0]),
        ),
    )


def _collect[State](
    env: Environment[State],
    agent: DQN | IQL,
    carry: DQNTrainingState[State],
    *,
    act: Callable[[DQNState, jax.Array, Info, jax.Array], jax.Array] | None = None,
    reference: runtime.Policy | None = None,
    update: metrics.Updater = metrics.update,
    store: Callable[[Replay, Transition, jax.Array], Replay] = insert,
) -> DQNTrainingState[State]:
    """One shared-interaction tick plus the replay insert (native and host loops)."""
    policy = agent.act if act is None else act

    def choose(obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
        return policy(carry.agent, obs, info, key)

    lifetime, action, reward = runtime.interact(
        env, carry.lifetime, choose, reference=reference, update=update
    )
    # A slot that is inactive on the next tick has no real next observation
    # yet, so its transition waits and is stored when the slot acts again.
    active = carry.lifetime.info["active_mask"]
    bridge, complete = bridge_step(
        carry.pending.bridge,
        active,
        lifetime.info["active_mask"],
        reward,
        carry.agent.hyperparameters.gamma,
    )
    pending = Pending(
        jnp.where(_per_slot(active, carry.lifetime.obs), carry.lifetime.obs, carry.pending.obs),
        jnp.where(active[:, None], carry.lifetime.info["action_mask"], carry.pending.mask),
        jnp.where(active, action, carry.pending.action),
        bridge,
    )
    transition = (
        pending.obs,
        pending.mask,
        complete,
        pending.action,
        bridge.reward,
        lifetime.obs,
        lifetime.info["action_mask"],
    )
    replay = store(carry.replay, transition, bridge.discount)
    return carry._replace(lifetime=lifetime, replay=replay, pending=pending)


def _per_slot(flag: jax.Array, like: jax.Array) -> jax.Array:
    return flag.reshape(flag.shape + (1,) * (like.ndim - 1))


class DQNRunner[StateT]:
    """Collect/learn continuing chunks and checkpoint only between updates.

    The compiled chunk uses per-function deterministic GPU compilation so a
    fresh process resumes exactly; this may cost throughput. Process-global
    XLA flags are never changed. With ``reference`` each tick also scores a
    stateless reference policy from the same state (one extra environment
    step) and the metrics carry accumulates same-state regret.
    ``donate=True`` donates the input carry's buffers to the compiled call, so
    large states (replay buffers, windows) are updated in place instead of
    copied. The carry passed in is then invalid: keep only the returned one.
    Results are unchanged.

    The runner is already compiled and must be the outermost compiled call:
    use ``runner(carry)`` or a bare ``jax.vmap(runner)``, never inside
    ``jax.jit``, which is refused.
    """

    def __init__(
        self,
        env: Environment[StateT],
        agent: DQN | IQL,
        *,
        reference: runtime.Policy | None = None,
        donate: bool = False,
    ) -> None:
        self._donate = donate
        if env.metadata.get("execution") == "host":
            raise ValueError("Use HostDQNRunner for mutable CPU simulators.")
        check_environment_pixel_scale(env, agent, type(agent).__name__)
        runtime.check_reference(env, reference)
        self.env = env
        self.agent = agent
        self.reference = reference
        self.metadata = runner_metadata(env, agent, reference)

        def run(carry: DQNTrainingState[StateT]) -> DQNTrainingState[StateT]:
            def collect(
                state: DQNTrainingState[StateT], _: None
            ) -> tuple[DQNTrainingState[StateT], None]:
                return _collect(self.env, self.agent, state, reference=self.reference), None

            carry, _ = jax.lax.scan(collect, carry, None, length=self.agent.config.rollout_steps)

            def learn(
                state: DQNTrainingState[StateT], _: None
            ) -> tuple[DQNTrainingState[StateT], None]:
                key, step_key = jax.random.split(state.optimizer_key)
                agent, statistics = self.agent.update(state.agent, state.replay, step_key)
                healthy = training_health(
                    state.healthy, state.lifetime.metrics.healthy, agent, statistics
                )
                return (
                    state._replace(
                        optimizer_key=key, agent=agent, statistics=statistics, healthy=healthy
                    ),
                    None,
                )

            carry, _ = jax.lax.scan(learn, carry, None, length=self.agent.config.gradient_steps)
            return carry._replace(updates=carry.updates.increment())

        self._run = compile_chunk(run, donate=donate)

    def init(
        self,
        *,
        seed: int | jax.Array = 0,
        window_size: int = 128,
        hyperparameters: DQNHyperparameters | None = None,
    ) -> DQNTrainingState[StateT]:
        """Initialize without changing the native environment's seeded RNG stream."""
        return init_dqn(
            self.env,
            self.agent,
            seed=seed,
            window_size=window_size,
            hyperparameters=hyperparameters,
        )

    @property
    def donates(self) -> bool:
        """Whether a call consumes the carry passed in (``donate=True``)."""
        return self._donate

    def __call__(self, state: DQNTrainingState[StateT]) -> DQNTrainingState[StateT]:
        """Collect one rollout, learn from replay, and advance the update counter."""
        return self._run(runtime.unalias(state) if self.donates else state)


class HostDQNRunner[StateT]:
    """Reuse the same DQN learner with host-controlled, batched CPU simulation.

    Policy evaluation and optimization remain compiled JAX operations; the
    simulator loop is explicit host work. Checkpoint support is determined by
    the backend. Host kernels use per-function deterministic GPU compilation.
    ``reference`` adds same-state regret exactly as in :class:`DQNRunner`.
    A call copies the replay buffer once, so the carry passed in stays
    readable, and every later tick of the call writes into that copy in place.
    """

    def __init__(
        self,
        env: Environment[StateT],
        agent: DQN | IQL,
        *,
        reference: runtime.Policy | None = None,
    ) -> None:
        check_environment_pixel_scale(env, agent, type(agent).__name__)
        runtime.check_reference(env, reference)
        self.env = env
        self.agent = agent
        compiler_options = compute.measured_compiler_options()
        self.reference = runtime.compile_reference(reference, compiler_options)
        self.metadata = runner_metadata(
            env, agent, reference, host_compiler_options=compiler_options
        )
        self._act = cast(
            Callable[[DQNState, jax.Array, Info, jax.Array], jax.Array],
            jax.jit(agent.act, compiler_options=compiler_options),
        )
        self._update = cast(
            Callable[[DQNState, Replay, jax.Array], tuple[DQNState, DQNStats]],
            jax.jit(agent.update, compiler_options=compiler_options),
        )
        self._metrics = cast(
            metrics.Updater,
            jax.jit(metrics.update, compiler_options=compiler_options),
        )
        self._insert = cast(
            Callable[[Replay, Transition, jax.Array], Replay],
            jax.jit(insert, compiler_options=compiler_options),
        )
        self._insert_in_place = cast(
            Callable[[Replay, Transition, jax.Array], Replay],
            jax.jit(insert, compiler_options=compiler_options, donate_argnums=0),
        )

    def init(
        self,
        *,
        seed: int = 0,
        window_size: int = 128,
        hyperparameters: DQNHyperparameters | None = None,
    ) -> DQNTrainingState[StateT]:
        """Initialize external simulation and the same full learner/metric carry."""
        return init_dqn(
            self.env,
            self.agent,
            seed=seed,
            window_size=window_size,
            hyperparameters=hyperparameters,
        )

    @property
    def donates(self) -> bool:
        """Always False: a call leaves the carry passed in readable."""
        return False

    def __call__(self, carry: DQNTrainingState[StateT]) -> DQNTrainingState[StateT]:
        """Collect a bounded CPU rollout, then learn from the replay buffer."""
        # The first insert copies the caller's replay; the rest write into that copy.
        store = self._insert
        for _ in range(self.agent.config.rollout_steps):
            carry = _collect(
                self.env,
                self.agent,
                carry,
                act=self._act,
                reference=self.reference,
                update=self._metrics,
                store=store,
            )
            store = self._insert_in_place
        for _ in range(self.agent.config.gradient_steps):
            key, step_key = jax.random.split(carry.optimizer_key)
            agent, statistics = self._update(carry.agent, carry.replay, step_key)
            healthy = training_health(
                carry.healthy, carry.lifetime.metrics.healthy, agent, statistics
            )
            carry = carry._replace(
                optimizer_key=key, agent=agent, statistics=statistics, healthy=healthy
            )
        return carry._replace(updates=carry.updates.increment())
