"""Continuing soft actor-critic with a bounded replay and explicit RNG.

Standard SAC under Perpetua's lifetime contract: no episodic resets or done
masking, a fixed-shape ring replay with explicit sampling keys, twin critics
with soft-updated targets and a tanh-squashed Gaussian policy. The entropy
temperature is the fixed ``alpha`` by default; ``autotune_alpha`` learns
``log_alpha`` toward ``target_entropy`` instead. Continuous actions are mapped
onto the environment's bounds.
"""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax.typing import ArrayLike

from perpetua import compute, metrics, probes, runtime
from perpetua.agents._common import (
    Bridge,
    advance_cursor,
    as_float32,
    bridge_step,
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
from perpetua.agents.plasticity import Params, adam, crelu, graft_top_layers
from perpetua.core import Clock, Environment, Info, JSONValue, as_int

__all__ = [
    "SACTransition",
    "SACActivation",
    "SACKernelInit",
    "SACConfig",
    "GaussianPolicy",
    "ContinuousQNetwork",
    "SACReplay",
    "SACParams",
    "SACOptimizerState",
    "SACHyperparameters",
    "SACState",
    "SACAction",
    "SACStats",
    "SACTrainingState",
    "SACPending",
    "init_sac_replay",
    "insert",
    "sample",
    "SAC",
    "init_sac",
    "SACRunner",
    "HostSACRunner",
]

type SACTransition = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0
# The Gaussian policy's mean and log-std heads read the same hidden layer.
_POLICY_HEADS = 2
_LOG_TWO_PI = math.log(2.0 * math.pi)

#: Hidden-layer activations of the SAC networks.
type SACActivation = Literal["tanh", "crelu", "leaky_relu"]
type SACKernelInit = Literal["orthogonal", "glorot_uniform"]


@dataclass(frozen=True)
class SACConfig:
    """SAC structure plus defaults for the traced hyperparameters.

    The continuous values that ``SACHyperparameters`` also carries are only
    defaults; the copy in ``SACState`` is used. The L2 penalty is gated
    statically by ``l2_coefficient`` here. There is no finite-lifetime
    annealing schedule. ``kernel_init`` selects orthogonal weights (the
    default) or Glorot uniform for every Dense layer of the actor and critics.

    ``target_entropy`` is in the units of the environment's actions, because
    the policy's log-probability is. Unset, it defaults to ``-action_dim +
    sum(log((high - low) / 2))``: the usual ``-action_dim`` heuristic in
    actions normalised to [-1, 1], which equals ``-action_dim`` for bounds of
    +-1. A fixed target of ``-action_dim`` is unreachable once an action range
    (``high - low``) is narrower than about 0.37 (a box of +-0.18 or less), and
    the temperature then grows without limit.
    """

    rollout_steps: int = 128
    buffer_capacity: int = 65536
    batch_size: int = 256
    gradient_steps: int = 128
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    reward_centering: bool = False
    centering_step_size: float = 0.01
    tau: float = 0.005
    alpha: float = 0.2
    max_grad_norm: float = 10.0
    autotune_alpha: bool = False
    target_entropy: float | None = None
    activation: SACActivation = "tanh"
    l2_coefficient: float = 0.0
    reset_interval: int | None = None
    reset_last_layers: int = 2
    hidden_layers: int = 2
    layer_norm: bool = False
    task_boundary_reset: bool = False
    random_steps: int = 0
    update_after: int = 0
    mask_respawns: bool = False
    kernel_init: SACKernelInit = "orthogonal"

    def __post_init__(self) -> None:
        for name in (
            "rollout_steps",
            "buffer_capacity",
            "batch_size",
            "gradient_steps",
            "hidden_size",
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
        if type(self.autotune_alpha) is not bool:
            raise ValueError("autotune_alpha must be a boolean.")
        if self.target_entropy is not None:
            set_floats(self, ("target_entropy",))
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite when provided.")
        if self.activation not in {"tanh", "crelu", "leaky_relu"}:
            raise ValueError("activation must be 'tanh', 'crelu' or 'leaky_relu'.")
        if self.kernel_init not in {"orthogonal", "glorot_uniform"}:
            raise ValueError("kernel_init must be 'orthogonal' or 'glorot_uniform'.")
        layers = as_int(self.hidden_layers, "hidden_layers must be a positive integer.", minimum=1)
        object.__setattr__(self, "hidden_layers", layers)
        for name in ("layer_norm", "task_boundary_reset", "mask_respawns"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        for name in ("random_steps", "update_after"):
            value = as_int(getattr(self, name), f"{name} must be a nonnegative integer.", minimum=0)
            object.__setattr__(self, name, value)
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
                "tau",
                "alpha",
                "max_grad_norm",
                "l2_coefficient",
            ),
        )
        for name in ("learning_rate", "gamma", "tau", "alpha", "max_grad_norm", "l2_coefficient"):
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
        if not 0 < self.tau <= 1:
            raise ValueError("Tau must be in (0, 1].")
        if self.alpha < 0:
            raise ValueError("Alpha must be nonnegative.")


def _activation(name: SACActivation, value: jax.Array) -> jax.Array:
    if name == "crelu":
        return crelu(value)
    if name == "leaky_relu":
        return nn.leaky_relu(value, negative_slope=0.2)
    return nn.tanh(value)


def _kernel(name: SACKernelInit, scale: float) -> nn.initializers.Initializer:
    """Orthogonal with gain ``scale``, or Glorot uniform (which ignores ``scale``)."""
    if name == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    return nn.initializers.orthogonal(scale)


def _hidden_stack(
    inputs: jax.Array,
    width: int,
    layers: int,
    layer_norm: bool,
    activate: Callable[[jax.Array], jax.Array],
    kernel_init: SACKernelInit = "orthogonal",
) -> jax.Array:
    """``layers`` Dense layers, each followed by ``activate``.

    With ``layer_norm`` the first layer is Dense, LayerNorm, tanh (Continual
    World's network) and later layers use the configured activation.
    """
    hidden = inputs
    for index in range(layers):
        hidden = nn.Dense(
            width,
            kernel_init=_kernel(kernel_init, math.sqrt(2)),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        if layer_norm and index == 0:
            hidden = nn.tanh(nn.LayerNorm()(hidden))
        else:
            hidden = activate(hidden)
    return hidden


class GaussianPolicy(nn.Module):
    """Policy MLP returning mean and clamped log standard deviation.

    ``hidden_layers`` sets the depth and ``layer_norm`` the first layer (see
    ``_hidden_stack``). ``activation`` selects tanh, concatenated ReLU or
    leaky ReLU (slope 0.2) hidden activations; CReLU widens each hidden output
    to ``2 * hidden_size``.
    """

    action_dim: int
    hidden_size: int
    activation: SACActivation = "tanh"
    hidden_layers: int = 2
    layer_norm: bool = False
    kernel_init: SACKernelInit = "orthogonal"

    def _activate(self, value: jax.Array) -> jax.Array:
        return _activation(self.activation, value)

    @nn.compact
    def __call__(
        self, obs: jax.Array, probe: bool = False
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
        hidden = _hidden_stack(
            obs,
            self.hidden_size,
            self.hidden_layers,
            self.layer_norm,
            self._activate,
            self.kernel_init,
        )
        mean = nn.Dense(
            self.action_dim,
            kernel_init=_kernel(self.kernel_init, 0.01),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        log_std = jnp.clip(
            nn.Dense(
                self.action_dim,
                kernel_init=_kernel(self.kernel_init, 0.01),
                bias_init=nn.initializers.zeros_init(),
            )(hidden),
            _LOG_STD_MIN,
            _LOG_STD_MAX,
        )
        if probe:
            return mean, log_std, hidden
        return mean, log_std


class ContinuousQNetwork(nn.Module):
    """Critic MLP mapping (obs, action) pairs to scalar values.

    ``hidden_layers``, ``layer_norm`` and ``activation`` (tanh, concatenated
    ReLU or leaky ReLU) configure it as in ``GaussianPolicy``.
    """

    hidden_size: int
    activation: SACActivation = "tanh"
    hidden_layers: int = 2
    layer_norm: bool = False
    kernel_init: SACKernelInit = "orthogonal"

    def _activate(self, value: jax.Array) -> jax.Array:
        return _activation(self.activation, value)

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array, probe: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        hidden = _hidden_stack(
            jnp.concatenate((obs, action), axis=-1),
            self.hidden_size,
            self.hidden_layers,
            self.layer_norm,
            self._activate,
            self.kernel_init,
        )
        value = nn.Dense(
            1,
            kernel_init=_kernel(self.kernel_init, 1.0),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        value = jnp.squeeze(value, axis=-1)
        if probe:
            return value, hidden
        return value


class SACReplay(NamedTuple):
    """Fixed-shape ring replay with continuous actions.

    ``cursor`` equals the insertion count until ``2 * capacity`` and then stays
    in ``[capacity, 2 * capacity)``, so it never overflows.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    active: jax.Array
    bridge: jax.Array
    cursor: jax.Array

    @property
    def filled(self) -> jax.Array:
        """Number of valid entries, saturating at the fixed capacity."""
        return jnp.minimum(self.cursor, self.obs.shape[0])


class SACParams(NamedTuple):
    """Actor, twin critics and their soft-updated target copies."""

    actor: Params
    q1: Params
    q2: Params
    q1_target: Params
    q2_target: Params


class SACOptimizerState(NamedTuple):
    """Separate optimizer moments for the actor, each critic and log-alpha."""

    actor: optax.OptState
    q1: optax.OptState
    q2: optax.OptState
    alpha: optax.OptState


class SACHyperparameters(NamedTuple):
    """Continuous hyperparameters carried as traced state.

    Structural choices (rollout length, buffer capacity, batch size, hidden
    size, alpha autotuning) stay in ``SACConfig``; these values live in
    ``SACState`` so sweeps over them are data and can be vmapped.
    """

    learning_rate: jax.Array
    gamma: jax.Array
    centering_step_size: jax.Array
    tau: jax.Array
    max_grad_norm: jax.Array
    l2_coefficient: jax.Array
    alpha: jax.Array
    target_entropy: jax.Array

    @classmethod
    def from_config(
        cls, config: "SACConfig", *, default_target_entropy: float | None = None
    ) -> "SACHyperparameters":
        """Read the configured defaults as float32 scalars.

        An unset ``target_entropy`` takes ``default_target_entropy``. Without
        one it becomes ``inf`` here, which ``SAC.init`` resolves to the
        agent's default because only the agent knows the action bounds.
        """
        values = []
        for name in cls._fields:
            raw = getattr(config, name)
            if raw is None:
                raw = float("inf") if default_target_entropy is None else default_target_entropy
            values.append(jnp.asarray(raw, dtype=jnp.float32))
        return cls(*values)

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
        if not 0 < values["tau"] <= 1:
            raise ValueError("Hyperparameter tau must be in (0, 1].")
        if values["l2_coefficient"] < 0 or values["alpha"] < 0:
            raise ValueError("Hyperparameter l2_coefficient and alpha must be nonnegative.")


class SACState(NamedTuple):
    """Parameters, optimizer moments, log-alpha and the gradient-step clock.

    ``reset_age`` counts ``update`` calls since the last periodic top-layer
    reset of the actor, both online critics and their target copies, including
    calls skipped because the batch was all inactive.

    With ``reward_centering``, ``reward_rate`` is the average of the entropy
    augmented reward: it settles at the reward plus ``gamma * alpha`` times the
    policy entropy, so it is not comparable with ``metrics.mean`` unless
    ``alpha`` is zero.
    """

    params: SACParams
    opt_state: SACOptimizerState
    log_alpha: jax.Array
    gradient_steps: Clock
    reset_age: jax.Array
    hyperparameters: SACHyperparameters
    reward_rate: jax.Array


class SACAction(NamedTuple):
    """Sampled continuous action per slot and its squashed log probability."""

    action: jax.Array
    log_prob: jax.Array


class SACStats(NamedTuple):
    """Bounded diagnostics: actor loss, critic loss, entropy and temperature."""

    actor_loss: jax.Array
    critic_loss: jax.Array
    entropy: jax.Array
    alpha: jax.Array


class SACTrainingState[StateT](NamedTuple):
    """Full lifetime, learner/targets/optimizers, replay buffer and diagnostics.

    ``since_boundary`` counts ticks since the lifetime started or, with
    ``task_boundary_reset``, since the last task switch; it saturates at the
    largest of ``random_steps`` and ``update_after`` (at least 1), so it never
    overflows.
    """

    lifetime: runtime.Lifetime[StateT]
    agent: SACState
    replay: SACReplay
    optimizer_key: jax.Array
    updates: Clock
    statistics: SACStats
    healthy: jax.Array
    since_boundary: jax.Array
    pending: "SACPending"


class SACPending(NamedTuple):
    """What each slot observed and did when it last acted, until its transition is stored.

    ``respawned`` (agents,) is true for a waiting transition whose slot has
    been flagged ``respawn`` since it acted; ``mask_respawns`` reads it.
    """

    obs: jax.Array
    action: jax.Array
    bridge: Bridge
    respawned: jax.Array


def init_sac_replay(*, num_agents: int, obs_dim: int, action_dim: int, capacity: int) -> SACReplay:
    """Allocate one bounded fixed-shape continuous-action replay buffer."""
    num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
    obs_dim = as_int(obs_dim, "obs_dim must be a positive integer.", minimum=1)
    action_dim = as_int(action_dim, "action_dim must be a positive integer.", minimum=1)
    capacity = as_int(capacity, "capacity must be a positive integer.", minimum=1)
    return SACReplay(
        jnp.zeros((capacity, num_agents, obs_dim), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents, action_dim), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents, obs_dim), dtype=jnp.float32),
        jnp.zeros((capacity, num_agents), dtype=jnp.bool_),
        jnp.ones((capacity, num_agents), dtype=jnp.float32),
        jnp.int32(0),
    )


def insert(
    replay: SACReplay, transition: SACTransition, bridge: jax.Array | None = None
) -> SACReplay:
    """Write one (time, agents) transition into the ring, wrapping in place.

    ``bridge`` is the extra discount of a transition that spans inactive
    ticks; it defaults to 1, an ordinary one-tick transition.
    """
    obs, action, reward, next_obs, active = transition
    if bridge is None:
        bridge = jnp.ones(replay.bridge.shape[1:], dtype=jnp.float32)
    expected = (
        (replay.obs.shape[1:], obs),
        (replay.action.shape[1:], action),
        (replay.reward.shape[1:], reward),
        (replay.next_obs.shape[1:], next_obs),
        (replay.active.shape[1:], active),
        (replay.bridge.shape[1:], bridge),
    )
    for shape, value in expected:
        if jnp.shape(value) != shape:
            raise ValueError("Replay transitions must match fixed agent/observation shapes.")
    index = replay.cursor % replay.obs.shape[0]
    return SACReplay(
        replay.obs.at[index].set(obs),
        replay.action.at[index].set(action),
        replay.reward.at[index].set(reward),
        replay.next_obs.at[index].set(next_obs),
        replay.active.at[index].set(active),
        replay.bridge.at[index].set(bridge),
        advance_cursor(replay.cursor, replay.obs.shape[0]),
    )


def sample(replay: SACReplay, key: jax.Array, batch_size: int) -> SACReplay:
    """Draw a with-replacement batch from the filled prefix using explicit RNG."""
    batch_size = as_int(batch_size, "batch_size must be a positive integer.", minimum=1)
    indices = jax.random.randint(key, (batch_size,), 0, replay.filled)
    return SACReplay(
        replay.obs[indices],
        replay.action[indices],
        replay.reward[indices],
        replay.next_obs[indices],
        replay.active[indices],
        replay.bridge[indices],
        replay.cursor,
    )


class SAC:
    """Masked continuing SAC with bounded replay and soft target updates.

    ``action_low`` and ``action_high`` are the bounds of **one slot's** action
    vector, shape ``(action_dim,)``: pass ``env.action_space.low[0]`` and
    ``env.action_space.high[0]``. The actor is shared by every slot, so the
    environment's slots must all use these bounds.
    """

    def __init__(
        self,
        *,
        action_low: ArrayLike | Sequence[float],
        action_high: ArrayLike | Sequence[float],
        config: SACConfig | None = None,
    ) -> None:
        for name, bound in (("action_low", action_low), ("action_high", action_high)):
            shape = np.shape(bound)
            if len(shape) > 1 and shape[0] != 1:
                raise ValueError(
                    f"SAC {name} must be one slot's bounds, shape (action_dim,); got shape "
                    f"{tuple(shape)}, which would flatten {shape[0]} slots into one action "
                    "vector. Pass env.action_space.low[0] and env.action_space.high[0]."
                )
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if low.shape != high.shape or low.size < 1:
            raise ValueError("SAC action bounds must be equally shaped nonempty vectors.")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low >= high):
            raise ValueError("SAC action bounds must be finite and strictly ordered.")
        config = SACConfig() if config is None else config
        self.config = config
        self.action_low = low
        self.action_high = high
        self.action_dim = int(low.size)
        self._scale = jnp.asarray((high - low) / 2.0)
        self._bias = jnp.asarray((high + low) / 2.0)
        self.policy = GaussianPolicy(
            action_dim=self.action_dim,
            hidden_size=config.hidden_size,
            activation=config.activation,
            hidden_layers=config.hidden_layers,
            layer_norm=config.layer_norm,
            kernel_init=config.kernel_init,
        )
        self.critic = ContinuousQNetwork(
            hidden_size=config.hidden_size,
            activation=config.activation,
            hidden_layers=config.hidden_layers,
            layer_norm=config.layer_norm,
            kernel_init=config.kernel_init,
        )
        reset_layers = config.reset_last_layers if config.reset_interval is not None else None
        self.actor_optimizer = adam(reset_layers, heads=_POLICY_HEADS)
        self.critic_optimizer = adam(reset_layers)
        self.alpha_optimizer = optax.scale_by_adam(eps=1e-5)
        # The policy's log-probability is in the environment's action units, so
        # the default target is -action_dim in normalised units shifted by the
        # log of the half-width of each bound (zero for bounds of +-1).
        self.target_entropy = (
            float(-self.action_dim) + sum(math.log(float(width)) for width in (high - low) / 2.0)
            if config.target_entropy is None
            else config.target_entropy
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Host-side learner configuration recorded in every training checkpoint."""
        return {
            "algorithm": "sac",
            "action_dim": self.action_dim,
            "action_low": cast(JSONValue, self.action_low.tolist()),
            "action_high": cast(JSONValue, self.action_high.tolist()),
            "config": asdict(self.config),
        }

    def init(
        self,
        obs: jax.Array,
        key: jax.Array,
        hyperparameters: SACHyperparameters | None = None,
    ) -> SACState:
        """Initialize actor, twin critics, target copies and optimizer moments.

        ``hyperparameters`` overrides the configured defaults as traced
        scalars; the values are stored in the returned state so sweeps are
        data and can be vmapped. ``log_alpha`` starts from the configured
        ``alpha``, not the traced one; with ``autotune_alpha`` the traced
        ``alpha`` is unused.
        """
        if obs.ndim != 2 or obs.shape[0] < 1 or obs.shape[1] < 1:
            raise ValueError("SAC observations must have shape (agents, features).")
        actor_key, q1_key, q2_key = jax.random.split(key, 3)
        actor = cast(Params, self.policy.init(actor_key, obs.astype(jnp.float32)))
        zeros = jnp.zeros((obs.shape[0], self.action_dim), dtype=jnp.float32)
        # Twin critics need different initial weights: identical twins trained on
        # the same batches stay identical, and min(q1, q2) then does nothing.
        q1 = cast(Params, self.critic.init(q1_key, obs.astype(jnp.float32), zeros))
        q2 = cast(Params, self.critic.init(q2_key, obs.astype(jnp.float32), zeros))
        params = SACParams(actor, q1, q2, q1, q2)
        log_alpha = jnp.log(jnp.asarray(max(self.config.alpha, 1e-8), dtype=jnp.float32))
        resolved = (
            SACHyperparameters.from_config(self.config, default_target_entropy=self.target_entropy)
            if hyperparameters is None
            else cast(SACHyperparameters, jax.tree.map(as_float32, hyperparameters))
        )
        resolved = resolved._replace(
            target_entropy=jnp.where(
                jnp.isposinf(resolved.target_entropy),
                jnp.asarray(self.target_entropy, dtype=jnp.float32),
                resolved.target_entropy,
            )
        )
        resolved.validate()
        return SACState(
            params,
            SACOptimizerState(
                self.actor_optimizer.init(actor),
                self.critic_optimizer.init(q1),
                self.critic_optimizer.init(q2),
                self.alpha_optimizer.init(log_alpha),
            ),
            log_alpha,
            Clock.from_int(0),
            jnp.int32(0),
            resolved,
            jnp.zeros((), dtype=jnp.float32),
        )

    def begin_task(self, state: SACState) -> SACState:
        """Start a new task: fresh optimizer moments and the initial temperature.

        Parameters and target networks are kept (fine-tuning). Used by
        ``task_boundary_reset``; the replay buffer is cleared by the runner.
        """
        log_alpha = jnp.log(jnp.asarray(max(self.config.alpha, 1e-8), dtype=jnp.float32))
        params = state.params
        return state._replace(
            opt_state=SACOptimizerState(
                self.actor_optimizer.init(params.actor),
                self.critic_optimizer.init(params.q1),
                self.critic_optimizer.init(params.q2),
                self.alpha_optimizer.init(log_alpha),
            ),
            log_alpha=log_alpha,
        )

    def _squashed(self, actor: Params, obs: jax.Array, key: jax.Array) -> SACAction:
        mean, log_std = cast(
            tuple[jax.Array, jax.Array], self.policy.apply(actor, obs.astype(jnp.float32))
        )
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        pre_tanh = mean + jnp.exp(log_std) * noise
        squashed = jnp.tanh(pre_tanh)
        leading = (1,) * (mean.ndim - 1)
        scale = self._scale.reshape((*leading, -1))
        action = squashed * scale + self._bias.reshape((*leading, -1))
        # From the noise itself: (pre_tanh - mean) / std cancels in float32
        # once std is near the spacing of floats around the mean.
        normal = -0.5 * jnp.square(noise) - log_std - 0.5 * _LOG_TWO_PI
        # log(1 - tanh(u)**2) in its exact, numerically stable form. An
        # epsilon-floored log1p(-tanh(u)**2 + eps) has zero gradient once tanh
        # saturates, so a policy whose mean ran away could never recover: its
        # entropy stayed pinned and an automatic temperature grew without bound.
        correction = 2.0 * (math.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
        log_scale = jnp.log(scale)
        return SACAction(action, jnp.sum(normal - correction - log_scale, axis=-1))

    def act(self, state: SACState, obs: jax.Array, info: Info, key: jax.Array) -> SACAction:
        """Sample a bounded continuous action per active slot from the policy."""
        sampled = self._squashed(state.params.actor, obs, key)
        action = jnp.where(info["active_mask"][:, None], sampled.action, 0.0)
        return SACAction(action, sampled.log_prob)

    def probe(
        self,
        state: SACState,
        obs: jax.Array,
        *,
        threshold: float = 1e-3,
        weights: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Health probes on the actor and first-critic hidden layers.

        On-demand diagnostics outside the compiled update path; they consume no
        RNG and never mutate the learner. The critic is probed on the policy's
        mean action, squashed and scaled to the action bounds as in training, so
        the report needs no action input; the twin critics share an architecture
        and the first is reported.
        """
        mean, _, actor_hidden = cast(
            tuple[jax.Array, jax.Array, jax.Array],
            self.policy.apply(state.params.actor, obs.astype(jnp.float32), probe=True),
        )
        action = jnp.tanh(mean) * self._scale + self._bias
        _, critic_hidden = cast(
            tuple[jax.Array, jax.Array],
            self.critic.apply(state.params.q1, obs.astype(jnp.float32), action, probe=True),
        )
        return {
            "dormant_actor": probes.dormant_unit_ratio(
                actor_hidden, threshold=threshold, weights=weights
            ),
            "dormant_critic": probes.dormant_unit_ratio(
                critic_hidden, threshold=threshold, weights=weights
            ),
            "rank_actor": probes.effective_rank(actor_hidden, weights=weights),
            "rank_critic": probes.effective_rank(critic_hidden, weights=weights),
        }

    def _critic_loss(
        self,
        hyperparameters: SACHyperparameters,
        q_params: Params,
        target: jax.Array,
        batch: SACReplay,
    ) -> jax.Array:
        values = cast(jax.Array, self.critic.apply(q_params, batch.obs, batch.action))
        squared = jnp.square(values - target)
        return l2_regularized(
            0.5 * weighted_mean(squared, batch.active),
            q_params,
            hyperparameters.l2_coefficient,
            configured=self.config.l2_coefficient,
        )

    def update(
        self, state: SACState, replay: SACReplay, key: jax.Array
    ) -> tuple[SACState, SACStats]:
        """One replay-sampled actor/twin-critic step plus soft target updates.

        With ``autotune_alpha`` the temperature is learned toward
        ``target_entropy`` (default ``-action_dim`` plus the log half-width of
        each bound, see ``SACConfig``) with a detached
        ``-(log_alpha * (log_prob + target_entropy))`` objective; otherwise
        the fixed ``alpha`` hyperparameter is used (the configured value unless
        a hyperparameter override replaced it).
        """
        hyperparameters = state.hyperparameters
        if self.config.reset_interval is not None:
            sample_key, next_key, actor_key, reset_key = jax.random.split(key, 4)
        else:
            sample_key, next_key, actor_key = jax.random.split(key, 3)
            reset_key = key
        batch = sample(replay, sample_key, self.config.batch_size)
        if self.config.reward_centering:
            batch = batch._replace(reward=batch.reward - state.reward_rate)
        alpha = jnp.exp(state.log_alpha) if self.config.autotune_alpha else hyperparameters.alpha
        next_action, next_log_prob = self._squashed(state.params.actor, batch.next_obs, next_key)
        q1_next = cast(
            jax.Array, self.critic.apply(state.params.q1_target, batch.next_obs, next_action)
        )
        q2_next = cast(
            jax.Array, self.critic.apply(state.params.q2_target, batch.next_obs, next_action)
        )
        target = jax.lax.stop_gradient(
            batch.reward
            + (hyperparameters.gamma * batch.bridge)
            * (jnp.minimum(q1_next, q2_next) - alpha * next_log_prob)
        )
        if self.config.reward_centering:
            q1_values = cast(jax.Array, self.critic.apply(state.params.q1, batch.obs, batch.action))
            rate = state.reward_rate + hyperparameters.centering_step_size * weighted_mean(
                target - q1_values, batch.active
            )
        else:
            rate = state.reward_rate
        q1_loss, q1_grads = jax.value_and_grad(self._critic_loss, argnums=1)(
            hyperparameters, state.params.q1, target, batch
        )
        q2_loss, q2_grads = jax.value_and_grad(self._critic_loss, argnums=1)(
            hyperparameters, state.params.q2, target, batch
        )

        def actor_loss(actor: Params) -> jax.Array:
            action, log_prob = self._squashed(actor, batch.obs, actor_key)
            q1 = cast(
                jax.Array,
                self.critic.apply(jax.lax.stop_gradient(state.params.q1), batch.obs, action),
            )
            q2 = cast(
                jax.Array,
                self.critic.apply(jax.lax.stop_gradient(state.params.q2), batch.obs, action),
            )
            return l2_regularized(
                weighted_mean(alpha * log_prob - jnp.minimum(q1, q2), batch.active),
                actor,
                hyperparameters.l2_coefficient,
                configured=self.config.l2_coefficient,
            )

        actor_objective, actor_grads = jax.value_and_grad(actor_loss)(state.params.actor)

        def alpha_loss(log_alpha: jax.Array) -> jax.Array:
            _, log_prob = self._squashed(state.params.actor, batch.obs, actor_key)
            detached = jax.lax.stop_gradient(log_prob + hyperparameters.target_entropy)
            return -weighted_mean(detached * log_alpha, batch.active)

        alpha_objective, alpha_grads = jax.value_and_grad(alpha_loss)(state.log_alpha)
        del alpha_objective

        def clipped_step(
            optimizer: optax.GradientTransformation,
            grads: Params,
            opt_state: optax.OptState,
            params: Params,
            learning_rate: jax.Array,
        ) -> tuple[Params, optax.OptState]:
            norm = optax.tree.norm(grads)
            scale = jnp.minimum(1.0, hyperparameters.max_grad_norm / (norm + 1e-6))

            def rescale(leaf: Any) -> Any:
                return leaf * scale

            def descend(leaf: Any, update: Any) -> Any:
                return leaf - learning_rate * update

            clipped = jax.tree.map(rescale, grads)
            updates, opt_state = optimizer.update(clipped, opt_state, params)
            return cast(Params, jax.tree.map(descend, params, updates)), opt_state

        def apply_gradients(agent: SACState) -> SACState:
            actor, actor_opt = clipped_step(
                self.actor_optimizer,
                actor_grads,
                agent.opt_state.actor,
                agent.params.actor,
                hyperparameters.learning_rate,
            )
            q1, q1_opt = clipped_step(
                self.critic_optimizer,
                q1_grads,
                agent.opt_state.q1,
                agent.params.q1,
                hyperparameters.learning_rate,
            )
            q2, q2_opt = clipped_step(
                self.critic_optimizer,
                q2_grads,
                agent.opt_state.q2,
                agent.params.q2,
                hyperparameters.learning_rate,
            )
            log_alpha = agent.log_alpha
            alpha_opt = agent.opt_state.alpha
            if self.config.autotune_alpha:
                alpha_updates, alpha_opt = self.alpha_optimizer.update(
                    alpha_grads, agent.opt_state.alpha, agent.log_alpha
                )
                log_alpha = jnp.asarray(
                    agent.log_alpha
                    - hyperparameters.learning_rate * cast(jax.Array, alpha_updates),
                    dtype=jnp.float32,
                )

            def soft(target: Params, online: Params) -> Params:
                def soft_leaf(new: jax.Array, old: jax.Array) -> jax.Array:
                    return hyperparameters.tau * new + (1.0 - hyperparameters.tau) * old

                return cast(Params, jax.tree.map(soft_leaf, online, target))

            params = SACParams(
                actor, q1, q2, soft(agent.params.q1_target, q1), soft(agent.params.q2_target, q2)
            )
            return SACState(
                params,
                SACOptimizerState(actor_opt, q1_opt, q2_opt, alpha_opt),
                log_alpha,
                agent.gradient_steps.increment(),
                agent.reset_age,
                agent.hyperparameters,
                rate,
            )

        def unchanged(agent: SACState) -> SACState:
            return agent

        trained = cast(
            SACState, jax.lax.cond(jnp.any(batch.active), apply_gradients, unchanged, state)
        )
        if self.config.reset_interval is not None:

            def perform_reset(agent: SACState) -> SACState:
                fresh = self.init(batch.obs[0].astype(jnp.float32), reset_key)
                count = self.config.reset_last_layers
                actor, actor_opt = graft_top_layers(
                    agent.params.actor,
                    agent.opt_state.actor,
                    fresh.params.actor,
                    fresh.opt_state.actor,
                    count,
                    heads=_POLICY_HEADS,
                )
                q1, q1_opt = graft_top_layers(
                    agent.params.q1, agent.opt_state.q1, fresh.params.q1, fresh.opt_state.q1, count
                )
                q2, q2_opt = graft_top_layers(
                    agent.params.q2, agent.opt_state.q2, fresh.params.q2, fresh.opt_state.q2, count
                )
                q1_target, _ = graft_top_layers(
                    agent.params.q1_target,
                    agent.opt_state.q1,
                    fresh.params.q1_target,
                    fresh.opt_state.q1,
                    count,
                )
                q2_target, _ = graft_top_layers(
                    agent.params.q2_target,
                    agent.opt_state.q2,
                    fresh.params.q2_target,
                    fresh.opt_state.q2,
                    count,
                )
                return SACState(
                    SACParams(actor, q1, q2, q1_target, q2_target),
                    SACOptimizerState(actor_opt, q1_opt, q2_opt, agent.opt_state.alpha),
                    agent.log_alpha,
                    agent.gradient_steps,
                    jnp.int32(0),
                    agent.hyperparameters,
                    agent.reward_rate,
                )

            trained = periodic_reset(trained, self.config.reset_interval, perform_reset)
        entropy = -weighted_mean(next_log_prob, batch.active)
        statistics = SACStats(
            actor_objective,
            0.5 * (q1_loss + q2_loss),
            entropy,
            jnp.exp(trained.log_alpha) if self.config.autotune_alpha else alpha,
        )
        return trained, statistics


def init_sac[State](
    env: Environment[State],
    agent: SAC,
    *,
    seed: int | jax.Array,
    window_size: int,
    hyperparameters: SACHyperparameters | None = None,
) -> SACTrainingState[State]:
    """Initialize a continuing lifetime, learner, replay buffer and counters."""
    import gymnasium as gym

    lifetime = runtime.init_lifetime(env, seed=seed, window_size=window_size)
    agent_key, init_key, optimizer_key = jax.random.split(lifetime.agent_key, 3)
    lifetime = lifetime._replace(agent_key=agent_key)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("SAC requires a Box action space.")
    expected = (lifetime.obs.shape[0], agent.action_dim)
    if env.action_space.shape != expected:
        raise ValueError(
            f"SAC has action_dim={agent.action_dim}, but the environment's action_space has shape "
            f"{env.action_space.shape} (slots, action dims per slot), so {expected} was expected. "
            "Build the agent from one slot's bounds: "
            "SAC(action_low=env.action_space.low[0], action_high=env.action_space.high[0])."
        )
    space_low = np.asarray(cast(Any, env.action_space.low), dtype=np.float32)
    space_high = np.asarray(cast(Any, env.action_space.high), dtype=np.float32)
    if space_low.ndim == 2 and space_low.shape[1] == agent.action_dim:
        # A shared multi-agent actor has one set of per-slot bounds. Require
        # every slot to use those bounds rather than flattening them together.
        if not np.allclose(space_low, space_low[0]) or not np.allclose(space_high, space_high[0]):
            raise ValueError("Shared multi-agent SAC requires identical per-slot action bounds.")
        low, high = space_low[0], space_high[0]
    else:
        low, high = space_low.reshape(-1), space_high.reshape(-1)
    if not np.allclose(low, agent.action_low, atol=1e-6) or not np.allclose(
        high, agent.action_high, atol=1e-6
    ):
        raise ValueError(
            "SAC action bounds do not match the environment action space: the agent has "
            f"low={agent.action_low.tolist()} high={agent.action_high.tolist()}, the "
            f"environment's per-slot bounds are low={low.tolist()} high={high.tolist()}."
        )
    if agent.config.task_boundary_reset and "boundary" not in env.metadata["oracle_exposure"]:
        raise ValueError(
            "task_boundary_reset reads the task switch, which is privileged: wrap the "
            'environment in pp.Oracle(env, expose=["boundary", ...]) so the exposure is recorded.'
        )
    state = agent.init(lifetime.obs, init_key, hyperparameters)
    replay = init_sac_replay(
        num_agents=lifetime.obs.shape[0],
        obs_dim=lifetime.obs.shape[-1],
        action_dim=agent.action_dim,
        capacity=agent.config.buffer_capacity,
    )
    zero = jnp.zeros((), dtype=jnp.float32)
    return SACTrainingState(
        lifetime,
        state,
        replay,
        optimizer_key,
        Clock.from_int(0),
        SACStats(zero, zero, zero, zero),
        jnp.asarray(True),
        jnp.int32(0),
        SACPending(
            jnp.zeros_like(lifetime.obs),
            jnp.zeros((lifetime.obs.shape[0], agent.action_dim), dtype=jnp.float32),
            init_bridge(lifetime.obs.shape[0]),
            jnp.zeros(lifetime.obs.shape[0], dtype=jnp.bool_),
        ),
    )


def _env_action(sampled: SACAction) -> jax.Array:
    return sampled.action


def _collect[State](
    env: Environment[State],
    carry: SACTrainingState[State],
    act: Callable[[SACState, jax.Array, Info, jax.Array], SACAction],
    agent: SAC,
    *,
    reference: runtime.Policy | None = None,
    update: metrics.Updater = metrics.update,
    store: Callable[[SACReplay, SACTransition, jax.Array], SACReplay] = insert,
) -> SACTrainingState[State]:
    """One shared-interaction tick plus the replay insert (native and host loops).

    With ``random_steps`` the first ticks after the start (and after each task
    switch with ``task_boundary_reset``) take uniform actions within the
    bounds. ``mask_respawns`` marks transitions that cross a respawn inactive,
    so they never enter a TD target (their next observation is a fresh start).
    A transition that waits while its slot is inactive crosses a respawn when
    the slot is flagged on any tick from the one it acted on to the one that
    completes it.
    ``task_boundary_reset`` clears the replay buffer and restarts the
    optimizer and temperature when the Oracle's ``boundary`` fires.
    """
    config = agent.config

    def choose(obs: jax.Array, info: Info, key: jax.Array) -> SACAction:
        sampled = act(carry.agent, obs, info, key)
        if config.random_steps == 0:
            return sampled
        uniform = jax.random.uniform(
            jax.random.fold_in(key, 1),
            sampled.action.shape,
            dtype=sampled.action.dtype,
            minval=jnp.asarray(agent.action_low),
            maxval=jnp.asarray(agent.action_high),
        )
        uniform = jnp.where(info["active_mask"][:, None], uniform, 0.0)
        explore = carry.since_boundary < config.random_steps
        return SACAction(jnp.where(explore, uniform, sampled.action), sampled.log_prob)

    lifetime, sampled, reward = runtime.interact(
        env, carry.lifetime, choose, action=_env_action, reference=reference, update=update
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
    respawned = carry.pending.respawned
    if config.mask_respawns:
        crossed = (respawned & ~active) | lifetime.info["respawn"]
        complete = complete & ~crossed
        respawned = crossed & bridge.open
    pending = SACPending(
        jnp.where(active[:, None], carry.lifetime.obs, carry.pending.obs),
        jnp.where(active[:, None], sampled.action, carry.pending.action),
        bridge,
        respawned,
    )
    transition = (pending.obs, pending.action, bridge.reward, lifetime.obs, complete)
    replay = store(carry.replay, transition, bridge.discount)
    cap = max(config.random_steps, config.update_after, 1)
    since = jnp.minimum(carry.since_boundary + 1, cap).astype(jnp.int32)
    learner = carry.agent
    if config.task_boundary_reset:
        oracle = cast(dict[str, jax.Array], cast(Any, lifetime.info)["oracle"])
        boundary = jnp.any(oracle["boundary"])
        replay = replay._replace(cursor=jnp.where(boundary, jnp.int32(0), replay.cursor))
        learner = cast(SACState, jax.lax.cond(boundary, agent.begin_task, _same_agent, learner))
        since = jnp.where(boundary, jnp.int32(0), since)
        pending = pending._replace(bridge=bridge._replace(open=bridge.open & ~boundary))
    return carry._replace(
        lifetime=lifetime, replay=replay, agent=learner, since_boundary=since, pending=pending
    )


def _same_agent(state: SACState) -> SACState:
    return state


def _same_training(state: SACTrainingState[Any]) -> SACTrainingState[Any]:
    return state


def _ready(carry: SACTrainingState[Any], agent: SAC) -> jax.Array:
    """Whether there is something to learn from, ``update_after`` ticks after a switch.

    A task boundary on a rollout's last tick empties the replay; sampling from
    it would then return copies of one stale transition of the previous task.
    """
    return (carry.since_boundary >= agent.config.update_after) & (carry.replay.filled > 0)


class SACRunner[StateT]:
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
        agent: SAC,
        *,
        reference: runtime.Policy | None = None,
        donate: bool = False,
    ) -> None:
        self._donate = donate
        if env.metadata.get("execution") == "host":
            raise ValueError("Use HostSACRunner for mutable CPU simulators.")
        runtime.check_reference(env, reference)
        self.env = env
        self.agent = agent
        self.reference = reference
        self.metadata = runner_metadata(env, agent, reference)

        def run(carry: SACTrainingState[StateT]) -> SACTrainingState[StateT]:
            def collect(
                state: SACTrainingState[StateT], _: None
            ) -> tuple[SACTrainingState[StateT], None]:
                return (
                    _collect(self.env, state, self.agent.act, self.agent, reference=self.reference),
                    None,
                )

            carry, _ = jax.lax.scan(collect, carry, None, length=self.agent.config.rollout_steps)

            def learn(
                state: SACTrainingState[StateT], _: None
            ) -> tuple[SACTrainingState[StateT], None]:
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

            def train(state: SACTrainingState[StateT]) -> SACTrainingState[StateT]:
                trained, _ = jax.lax.scan(
                    learn, state, None, length=self.agent.config.gradient_steps
                )
                return trained

            if self.agent.config.update_after > 0 or self.agent.config.task_boundary_reset:
                carry = cast(
                    SACTrainingState[StateT],
                    jax.lax.cond(_ready(carry, self.agent), train, _same_training, carry),
                )
            else:
                carry = train(carry)
            return carry._replace(updates=carry.updates.increment())

        self._run = compile_chunk(run, donate=donate)

    def init(
        self,
        *,
        seed: int | jax.Array = 0,
        window_size: int = 128,
        hyperparameters: SACHyperparameters | None = None,
    ) -> SACTrainingState[StateT]:
        """Initialize without changing the native environment's seeded RNG stream."""
        return init_sac(
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

    def __call__(self, state: SACTrainingState[StateT]) -> SACTrainingState[StateT]:
        """Collect one rollout, learn from replay, and advance the update counter."""
        return self._run(runtime.unalias(state) if self.donates else state)


class HostSACRunner[StateT]:
    """Reuse the same SAC learner with host-controlled, batched CPU simulation.

    Policy sampling and optimization remain compiled JAX operations; the
    simulator loop is explicit host work. Checkpoint support is determined by
    the backend. Host kernels use per-function deterministic GPU compilation.
    ``reference`` adds same-state regret exactly as in :class:`SACRunner`.
    A call copies the replay buffer once, so the carry passed in stays
    readable, and every later tick of the call writes into that copy in place.
    """

    def __init__(
        self,
        env: Environment[StateT],
        agent: SAC,
        *,
        reference: runtime.Policy | None = None,
    ) -> None:
        runtime.check_reference(env, reference)
        self.env = env
        self.agent = agent
        compiler_options = compute.measured_compiler_options()
        self.reference = runtime.compile_reference(reference, compiler_options)
        self.metadata = runner_metadata(
            env, agent, reference, host_compiler_options=compiler_options
        )
        self._act = cast(
            Callable[[SACState, jax.Array, Info, jax.Array], SACAction],
            jax.jit(agent.act, compiler_options=compiler_options),
        )
        self._update = cast(
            Callable[[SACState, SACReplay, jax.Array], tuple[SACState, SACStats]],
            jax.jit(agent.update, compiler_options=compiler_options),
        )
        self._metrics = cast(
            metrics.Updater,
            jax.jit(metrics.update, compiler_options=compiler_options),
        )
        self._insert = cast(
            Callable[[SACReplay, SACTransition, jax.Array], SACReplay],
            jax.jit(insert, compiler_options=compiler_options),
        )
        self._insert_in_place = cast(
            Callable[[SACReplay, SACTransition, jax.Array], SACReplay],
            jax.jit(insert, compiler_options=compiler_options, donate_argnums=0),
        )

    def init(
        self,
        *,
        seed: int = 0,
        window_size: int = 128,
        hyperparameters: SACHyperparameters | None = None,
    ) -> SACTrainingState[StateT]:
        """Initialize external simulation and the same full learner/metric carry."""
        return init_sac(
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

    def __call__(self, carry: SACTrainingState[StateT]) -> SACTrainingState[StateT]:
        """Collect a bounded CPU rollout, then learn from the replay buffer."""
        # The first insert copies the caller's replay; the rest write into that copy.
        store = self._insert
        for _ in range(self.agent.config.rollout_steps):
            carry = _collect(
                self.env,
                carry,
                self._act,
                self.agent,
                reference=self.reference,
                update=self._metrics,
                store=store,
            )
            store = self._insert_in_place
        steps = self.agent.config.gradient_steps if bool(_ready(carry, self.agent)) else 0
        for _ in range(steps):
            key, step_key = jax.random.split(carry.optimizer_key)
            agent, statistics = self._update(carry.agent, carry.replay, step_key)
            healthy = training_health(
                carry.healthy, carry.lifetime.metrics.healthy, agent, statistics
            )
            carry = carry._replace(
                optimizer_key=key, agent=agent, statistics=statistics, healthy=healthy
            )
        return carry._replace(updates=carry.updates.increment())
