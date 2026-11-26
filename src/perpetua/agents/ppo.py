"""Continuing PPO adapted from PureJaxRL's JAX PPO reference implementation.

Copyright 2023 Chris Lu. Licensed under Apache-2.0; see LICENSE-PureJaxRL.
Modified for Perpetua: no episodic resets/done masks, explicit action masks,
constant learning rate, full lifetime carry, health checks and configuration.
Upstream revision and changes are recorded in NOTICE.
"""

# JAX/Flax/Optax expose some partially annotated transformation helpers.
# pyright: reportUnknownMemberType=false

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
from jax.typing import ArrayLike

from perpetua import compute, metrics, probes, runtime
from perpetua.agents._common import (
    as_float32,
    check_discrete_actions,
    check_environment_pixel_scale,
    check_torso_observations,
    checked_hyperparameters,
    compile_chunk,
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
    "PPOConfig",
    "PPOHyperparameters",
    "ActorCritic",
    "PPOState",
    "reset_state_layers",
    "Action",
    "discounted_tick_count",
    "Rollout",
    "PPOStats",
    "compute_gae",
    "PPO",
    "IPPO",
    "TrainingState",
    "PPORunner",
    "HostPPORunner",
]


@dataclass(frozen=True)
class PPOConfig:
    """PPO structure plus defaults for the traced hyperparameters.

    The continuous values that ``PPOHyperparameters`` also carries are only
    defaults; the copy in ``PPOState`` is used. The L2 penalty is gated
    statically by ``l2_coefficient`` here. There is no finite-lifetime
    annealing schedule.

    ``pixel_scale`` is what the CNN torso divides observations by. ``None``
    derives it once, when the learner is built, from the ``observation_space``
    given to the learner (255 for bounds of 0 and 255, else 1; 1 without a
    space); a number sets it whatever the space says.
    """

    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    reward_centering: bool = False
    centering_step_size: float = 0.01
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
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
        for name in ("rollout_steps", "epochs", "minibatches", "hidden_size"):
            value = as_int(getattr(self, name), f"{name} must be a positive integer.", minimum=1)
            object.__setattr__(self, name, value)
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
                "gae_lambda",
                "clip_epsilon",
                "value_coefficient",
                "entropy_coefficient",
                "max_grad_norm",
                "l2_coefficient",
            ),
        )
        for name in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "value_coefficient",
            "entropy_coefficient",
            "max_grad_norm",
            "l2_coefficient",
        ):
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
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Gamma and GAE lambda must be in [0, 1].")
        if not 0 < self.clip_epsilon <= 1:
            raise ValueError("clip_epsilon must be in (0, 1].")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("Loss coefficients must be nonnegative.")
        if self.l2_coefficient < 0:
            raise ValueError("l2_coefficient must be nonnegative.")


class ActorCritic(nn.Module):
    """Separate two-layer actor and critic, as in the reference baseline.

    ``activation`` selects tanh or concatenated ReLU hidden activations;
    CReLU widens each hidden output to ``2 * hidden_size``. With
    ``torso="cnn"`` both read features from one shared ``CNNTorso``. Dense
    layers are created alternately: ``Dense_0``/``Dense_2`` are the actor's
    hidden layers, ``Dense_1``/``Dense_3`` the critic's, ``Dense_4`` the
    policy head and ``Dense_5`` the value head.
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
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if self.torso == "cnn":
            obs = CNNTorso(self.hidden_size, self.activation, self.pixel_scale)(obs)
        actor = obs
        critic = obs
        for _ in range(2):
            actor = self._activate(
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.orthogonal(math.sqrt(2)),
                    bias_init=nn.initializers.zeros_init(),
                )(actor)
            )
            critic = self._activate(
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.orthogonal(math.sqrt(2)),
                    bias_init=nn.initializers.zeros_init(),
                )(critic)
            )
        logits = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros_init(),
        )(actor)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros_init(),
        )(critic)
        if probe:
            return logits, jnp.squeeze(value, axis=-1), actor, critic
        return logits, jnp.squeeze(value, axis=-1)


class PPOHyperparameters(NamedTuple):
    """Continuous hyperparameters carried as traced state.

    Structural choices (epochs, minibatches, hidden size, reset schedule)
    stay in ``PPOConfig``; these values live in ``PPOState`` so a sweep over
    them is data, not new compiled programs, and can be vmapped.
    """

    learning_rate: jax.Array
    gamma: jax.Array
    centering_step_size: jax.Array
    gae_lambda: jax.Array
    clip_epsilon: jax.Array
    value_coefficient: jax.Array
    entropy_coefficient: jax.Array
    max_grad_norm: jax.Array
    l2_coefficient: jax.Array

    @classmethod
    def from_config(cls, config: "PPOConfig") -> "PPOHyperparameters":
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
        if not 0 <= values["gamma"] <= 1 or not 0 <= values["gae_lambda"] <= 1:
            raise ValueError("Hyperparameter gamma and gae_lambda must be in [0, 1].")
        if not 0 < values["clip_epsilon"] <= 1:
            raise ValueError("Hyperparameter clip_epsilon must be in (0, 1].")
        if values["value_coefficient"] < 0 or values["entropy_coefficient"] < 0:
            raise ValueError("Hyperparameter loss coefficients must be nonnegative.")
        if values["l2_coefficient"] < 0:
            raise ValueError("Hyperparameter l2_coefficient must be nonnegative.")


class PPOState(NamedTuple):
    """Parameters, optimizer moments, gradient count, reset age and hyperparameters.

    PPO uses one parameter tree and scalar clock limbs. IPPO stacks every leaf
    on an agent axis, including uint32 gradient-clock limbs (agents,).
    ``reset_age`` counts ``update`` calls (one per rollout) since the last
    periodic top-layer reset.
    ``hyperparameters`` carries the continuous knobs as traced scalars.
    """

    params: Params
    opt_state: optax.OptState
    gradient_steps: Clock
    reset_age: jax.Array
    hyperparameters: PPOHyperparameters
    reward_rate: jax.Array


def reset_state_layers(state: PPOState, fresh: PPOState, count: int) -> PPOState:
    """Replace the final ``count`` layers of parameters and optimizer moments.

    Fresh values come from a re-initialized state with identical structure;
    non-reset layers, the gradient count and the hyperparameters are preserved
    exactly.
    """
    params, opt_state = graft_top_layers(
        state.params, state.opt_state, fresh.params, fresh.opt_state, count
    )
    return PPOState(
        params,
        opt_state,
        state.gradient_steps,
        jnp.int32(0),
        state.hyperparameters,
        state.reward_rate,
    )


class Action(NamedTuple):
    """Action, sampled log probability and pre-transition value per agent slot."""

    action: jax.Array
    log_prob: jax.Array
    value: jax.Array


def discounted_tick_count(gamma: jax.Array, elapsed_ticks: jax.Array) -> jax.Array:
    """``sum_{i<k} gamma**i`` for each decision's tick count ``k``.

    ``(1 - gamma**k) / (1 - gamma)``, and ``k`` when ``gamma`` is 1. A decision's
    reward is the discounted sum of its ticks' rewards, so this is how many
    ticks' worth of average reward it contains.
    """
    k = elapsed_ticks.astype(jnp.float32)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    safe = jnp.where(gamma < 1.0, 1.0 - gamma, 1.0)
    return jnp.where(gamma < 1.0, (1.0 - jnp.power(gamma, k)) / safe, k)


class Rollout(NamedTuple):
    """Bounded on-policy data: leading axes are (time, agent slots)."""

    obs: jax.Array
    mask: jax.Array
    active: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    elapsed_ticks: jax.Array | None = None


class PPOStats(NamedTuple):
    """Minibatch-averaged diagnostics: scalars for PPO, (agents,) arrays for IPPO."""

    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    grad_norm: jax.Array


def _bridged_value(
    next_value: jax.Array, reward: jax.Array, value: jax.Array, discount: jax.Array, on: jax.Array
) -> jax.Array:
    """A tick's value, or its one-step bootstrap where the slot did not act."""
    return jnp.where(on, value, reward + discount * next_value)


def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    last_value: jax.Array,
    gamma: ArrayLike,
    gae_lambda: ArrayLike,
    elapsed_ticks: jax.Array | None = None,
    active: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute continuing GAE, bootstrapping from the last observation's value.

    Rewards/values have shape (time, agents), last_value has shape (agents,).
    No respawn, drift or rollout boundary zeros the continuation discount.
    Optional uint32 elapsed_ticks (time, agents) describes semi-Markov decisions:
    rewards are already discounted over each interval, with gamma**ticks for
    bootstrap and (gamma * gae_lambda)**ticks for the continuing trace.

    Optional bool ``active`` (time, agents) marks the ticks on which a slot
    acted. An inactive tick's value was computed from a placeholder
    observation the critic is never trained on, so it is replaced by its own
    one-step bootstrap: the tick before an inactive stretch then bootstraps
    from the next active tick, discounted across the stretch and including its
    rewards. Advantages and targets on inactive ticks are not meaningful. The
    final ``last_value`` is used as given.
    """
    if active is not None and (active.shape != values.shape or active.dtype != jnp.bool_):
        raise ValueError("active must be boolean with the reward/value shape.")
    if rewards.shape != values.shape or values.ndim != 2 or last_value.shape != values.shape[1:]:
        raise ValueError("GAE requires matching (time, agents) arrays and final agent values.")

    if elapsed_ticks is not None:
        if elapsed_ticks.shape != values.shape or elapsed_ticks.dtype != jnp.uint32:
            raise ValueError("elapsed_ticks must be uint32 with the reward/value shape.")
        discounts = jnp.power(gamma, elapsed_ticks.astype(jnp.float32))
        traces = jnp.power(gamma * gae_lambda, elapsed_ticks.astype(jnp.float32))
    else:
        discounts = jnp.full_like(values, gamma)
        traces = jnp.full_like(values, gamma * gae_lambda)

    acted = jnp.ones_like(values, dtype=jnp.bool_) if active is None else active

    def backward(
        carry: tuple[jax.Array, jax.Array],
        transition: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        advantage, next_value = carry
        reward, value, discount, trace, on = transition
        value = _bridged_value(next_value, reward, value, discount, on)
        delta = reward + discount * next_value - value
        advantage = delta + trace * advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        backward,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, discounts, traces, acted),
        reverse=True,
    )
    return advantages, advantages + values


def _log_probs(logits: jax.Array, mask: jax.Array) -> jax.Array:
    # A finite minimum prevents 0 * -inf entropy/gradient NaNs for masked actions.
    minimum = cast(float, jnp.finfo(jnp.float32).min)
    return jax.nn.log_softmax(jnp.where(mask, logits, minimum), axis=-1)


class PPO:
    """Masked, discrete PPO with a shared policy across fixed agent slots.

    With ``torso="cnn"``, pass ``observation_space=env.observation_space`` so
    raw 8-bit pixels are divided by 255. The divisor is fixed here for the
    learner's life and kept as ``pixel_scale`` (see ``PPOConfig``).
    """

    def __init__(
        self,
        *,
        action_size: int,
        config: PPOConfig | None = None,
        observation_space: spaces.Space[Any] | None = None,
    ) -> None:
        action_size = as_int(action_size, "action_size must be a positive integer.", minimum=1)
        config = PPOConfig() if config is None else config
        self.config = config
        self.action_size = action_size
        self.pixel_scale = resolve_pixel_scale(config.pixel_scale, observation_space)
        self.network = ActorCritic(
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
            "algorithm": "ppo",
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
        hyperparameters: PPOHyperparameters | None = None,
    ) -> PPOState:
        """Initialize from observations of shape (agent slots, features).

        With ``torso="cnn"`` observations are (agent slots, H, W, C).
        ``hyperparameters`` overrides the configured defaults as traced
        scalars; the values are stored in the returned state so sweeps are
        data and can be vmapped.
        """
        if obs.ndim < 2 or obs.shape[0] < 1 or obs.size < 1:
            raise ValueError("PPO observations must have shape (agents, features).")
        check_torso_observations(obs, self.config.torso, "PPO")
        params = cast(Params, self.network.init(key, obs.astype(jnp.float32)))
        resolved = (
            PPOHyperparameters.from_config(self.config)
            if hyperparameters is None
            else cast(PPOHyperparameters, jax.tree.map(as_float32, hyperparameters))
        )
        resolved.validate()
        return PPOState(
            params,
            self.optimizer.init(params),
            Clock.from_int(0),
            jnp.int32(0),
            resolved,
            jnp.zeros((), dtype=jnp.float32),
        )

    def evaluate(self, state: PPOState, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return logits/value without sampling or consuming RNG."""
        return cast(tuple[jax.Array, jax.Array], self.network.apply(state.params, obs))

    def probe(
        self,
        state: PPOState,
        obs: jax.Array,
        *,
        threshold: float = 1e-3,
        weights: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Health probes on the final hidden layers: dormant units and rank.

        On-demand diagnostics outside the compiled update path; they consume no
        RNG and never mutate the learner. IPPO's stacked per-agent parameters
        are not probed here.
        """
        logits, value, actor_hidden, critic_hidden = cast(
            tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            self.network.apply(state.params, obs, probe=True),
        )
        del logits, value
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

    def act(self, state: PPOState, obs: jax.Array, info: Info, key: jax.Array) -> Action:
        """Sample only valid actions from public observations and masks."""
        logits, value = self.evaluate(state, obs)
        log_probs = _log_probs(logits, info["action_mask"])
        action = jax.random.categorical(key, log_probs, axis=-1).astype(jnp.int32)
        selected = jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]
        return Action(action, selected, value)

    def _loss(
        self,
        hyperparameters: PPOHyperparameters,
        params: Params,
        batch: Rollout,
        advantages: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        logits, value = cast(tuple[jax.Array, jax.Array], self.network.apply(params, batch.obs))
        log_probs = _log_probs(logits, batch.mask)
        log_prob = jnp.take_along_axis(log_probs, batch.action[..., None], axis=-1)[..., 0]
        log_ratio = log_prob - batch.log_prob
        ratio = jnp.exp(log_ratio)
        mean = weighted_mean(advantages, batch.active)
        variance = weighted_mean(jnp.square(advantages - mean), batch.active)
        normalized = (advantages - mean) / jnp.sqrt(variance + 1e-8)
        clipped = jnp.clip(
            ratio, 1 - hyperparameters.clip_epsilon, 1 + hyperparameters.clip_epsilon
        )
        policy_loss = -weighted_mean(
            jnp.minimum(ratio * normalized, clipped * normalized), batch.active
        )
        clipped_value = batch.value + jnp.clip(
            value - batch.value, -hyperparameters.clip_epsilon, hyperparameters.clip_epsilon
        )
        value_loss = 0.5 * weighted_mean(
            jnp.maximum(jnp.square(value - targets), jnp.square(clipped_value - targets)),
            batch.active,
        )
        entropy = weighted_mean(-jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1), batch.active)
        loss = (
            policy_loss
            + hyperparameters.value_coefficient * value_loss
            - hyperparameters.entropy_coefficient * entropy
        )
        loss = l2_regularized(
            loss, params, hyperparameters.l2_coefficient, configured=self.config.l2_coefficient
        )
        kl = weighted_mean((ratio - 1) - log_ratio, batch.active)
        clip_fraction = weighted_mean(
            (jnp.abs(ratio - 1) > hyperparameters.clip_epsilon).astype(jnp.float32), batch.active
        )
        return loss, (policy_loss, value_loss, entropy, kl, clip_fraction)

    def update(
        self, state: PPOState, rollout: Rollout, last_value: jax.Array, key: jax.Array
    ) -> tuple[PPOState, jax.Array, PPOStats]:
        """Optimize a consumed rollout; no history is retained between updates.

        With ``reward_centering`` the rate moves by the mean per-tick TD error
        of the active ticks. Write ``s_j = sum_{i<k_j} gamma**i`` for the
        discounted tick count of rollout step ``j`` (``k_j`` world ticks; 1
        without ``elapsed_ticks``) and ``d_j = gamma**k_j``. An active step
        ``t`` followed by inactive steps ``t+1..t+m`` has the bridged error

            td_t = sum_{j=t}^{t+m} D_j (r_j - rate * s_j) + D_{t+m+1} V' - V_t,
            D_j = d_t * ... * d_{j-1},

        so ``d td_t / d rate = -W_t`` with ``W_t = sum_{j=t}^{t+m} D_j s_j``,
        the discounted number of world ticks the error spans (it equals
        ``sum_{i<K} gamma**i`` for the ``K`` ticks of the whole span). The
        update is ``rate += step * mean_t(td_t / W_t)``. ``W_t`` obeys
        ``W_j = s_j + d_j * W_{j+1}`` along the inactive steps, starting from 0
        at the next active step or the rollout end, and is ``s_t`` when the
        next step is active.
        """
        hyperparameters = state.hyperparameters
        # A decision that spans k world ticks (compute-aware time) carries the
        # discounted sum of k tick rewards, so centering removes the average
        # reward once per tick: rate * (1 - gamma**k) / (1 - gamma), or rate * k
        # when gamma is 1. With one tick per decision the scale is exactly 1.
        scale = (
            discounted_tick_count(hyperparameters.gamma, rollout.elapsed_ticks)
            if self.config.reward_centering and rollout.elapsed_ticks is not None
            else jnp.ones_like(rollout.reward)
        )
        centered = (
            rollout.reward - state.reward_rate * scale
            if self.config.reward_centering
            else rollout.reward
        )
        advantages, targets = compute_gae(
            rewards=centered,
            values=rollout.value,
            last_value=last_value,
            gamma=hyperparameters.gamma,
            gae_lambda=hyperparameters.gae_lambda,
            elapsed_ticks=rollout.elapsed_ticks,
            active=rollout.active,
        )
        if self.config.reward_centering:
            # Bootstrap from the same bridged values the advantages use, so a
            # slot that sits out ticks adds nothing from its placeholder values.
            if rollout.elapsed_ticks is not None:
                discounts = jnp.power(
                    hyperparameters.gamma, rollout.elapsed_ticks.astype(jnp.float32)
                )
            else:
                discounts = jnp.full_like(rollout.value, hyperparameters.gamma)

            def bridge(
                carry: tuple[jax.Array, jax.Array],
                tick: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                next_value, waiting = carry
                reward, value, discount, on, count = tick
                value = _bridged_value(next_value, reward, value, discount, on)
                span = count + discount * waiting
                return (value, jnp.where(on, 0.0, span)), (value, span)

            _, (bridged, spans) = jax.lax.scan(
                bridge,
                (last_value, jnp.zeros_like(last_value)),
                (centered, rollout.value, discounts, rollout.active, scale),
                reverse=True,
            )
            next_values = jnp.concatenate([bridged[1:], last_value[None]], axis=0)
            td = centered + discounts * next_values - rollout.value
            # Per-tick TD error, so the rate stays an average reward per tick.
            rate = state.reward_rate + hyperparameters.centering_step_size * weighted_mean(
                td / spans, rollout.active
            )
        else:
            rate = state.reward_rate
        sample_count = rollout.reward.shape[0] * rollout.reward.shape[1]
        if sample_count % self.config.minibatches:
            raise ValueError(
                f"Rollout samples ({sample_count}) must divide evenly into "
                f"{self.config.minibatches} minibatches."
            )
        if sample_count // self.config.minibatches < 2:
            raise ValueError(
                f"Rollout samples ({sample_count}) split into {self.config.minibatches} "
                f"minibatches leave {sample_count // self.config.minibatches} sample per "
                "minibatch; advantage normalisation needs at least 2. Use a longer rollout or "
                "lower PPOConfig.minibatches."
            )

        def flatten(value: jax.Array) -> jax.Array:
            return value.reshape((sample_count,) + value.shape[2:])

        flat = jax.tree.map(
            flatten,
            (rollout, jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(targets)),
        )

        def minibatch(
            state: PPOState, data: tuple[Rollout, jax.Array, jax.Array]
        ) -> tuple[PPOState, PPOStats]:
            batch, advantage, target = data
            (loss, details), grads = jax.value_and_grad(self._loss, argnums=1, has_aux=True)(
                state.hyperparameters, state.params, batch, advantage, target
            )

            def apply_gradients(state: PPOState) -> PPOState:
                norm = optax.tree.norm(grads)
                scale = jnp.minimum(1.0, state.hyperparameters.max_grad_norm / (norm + 1e-6))

                def rescale(leaf: Any) -> Any:
                    return leaf * scale

                def descend(leaf: Any, update: Any) -> Any:
                    return leaf - state.hyperparameters.learning_rate * update

                clipped = jax.tree.map(rescale, grads)
                updates, opt_state = self.optimizer.update(clipped, state.opt_state, state.params)
                params = cast(Params, jax.tree.map(descend, state.params, updates))
                return PPOState(
                    params,
                    opt_state,
                    state.gradient_steps.increment(),
                    state.reset_age,
                    state.hyperparameters,
                    rate,
                )

            def unchanged(state: PPOState) -> PPOState:
                return state

            state = cast(
                PPOState, jax.lax.cond(jnp.any(batch.active), apply_gradients, unchanged, state)
            )
            policy_loss, value_loss, entropy, kl, clip_fraction = cast(
                tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], details
            )
            return state, PPOStats(
                loss, policy_loss, value_loss, entropy, kl, clip_fraction, optax.tree.norm(grads)
            )

        def epoch(
            carry: tuple[PPOState, jax.Array], _: None
        ) -> tuple[tuple[PPOState, jax.Array], PPOStats]:
            state, key = carry
            key, shuffle_key = jax.random.split(key)
            permutation = jax.random.permutation(shuffle_key, sample_count)

            def shuffle(value: jax.Array) -> jax.Array:
                return value[permutation].reshape(
                    (self.config.minibatches, sample_count // self.config.minibatches)
                    + value.shape[1:]
                )

            batches = jax.tree.map(shuffle, flat)
            state, statistics = jax.lax.scan(minibatch, state, batches)
            return (state, key), statistics

        (state, key), statistics = jax.lax.scan(
            epoch, (state, key), None, length=self.config.epochs
        )

        def average(value: jax.Array) -> jax.Array:
            return jnp.mean(value)

        statistics = jax.tree.map(average, statistics)
        if self.config.reset_interval is not None:
            key, reset_key = jax.random.split(key)

            def perform_reset(state: PPOState) -> PPOState:
                fresh_params = cast(
                    Params, self.network.init(reset_key, rollout.obs[0].astype(jnp.float32))
                )
                fresh = PPOState(
                    fresh_params,
                    self.optimizer.init(fresh_params),
                    state.gradient_steps,
                    jnp.int32(0),
                    state.hyperparameters,
                    state.reward_rate,
                )
                return reset_state_layers(state, fresh, self.config.reset_last_layers)

            state = periodic_reset(state, self.config.reset_interval, perform_reset)
        return state, key, statistics


class IPPO:
    """Independent PPO models/optimizers per fixed agent slot, without sharing.

    Observations, actions and continuing GAE retain the native agent axis.
    Each model consumes only its own public obs and action mask. Initialization,
    sampling and shuffle subkeys are split per slot from saved root keys; a slot's
    rewards/gradients cannot update another slot's parameters or optimizer.
    """

    def __init__(
        self,
        *,
        action_size: int,
        config: PPOConfig | None = None,
        observation_space: spaces.Space[Any] | None = None,
    ) -> None:
        self._ppo = PPO(action_size=action_size, config=config, observation_space=observation_space)
        self.config = self._ppo.config
        self.action_size = self._ppo.action_size
        self.pixel_scale = self._ppo.pixel_scale

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record independent, fixed-slot learners rather than task-indexed heads."""
        return {**self._ppo.configuration, "algorithm": "ippo", "parameter_sharing": False}

    def init(
        self,
        obs: jax.Array,
        key: jax.Array,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> PPOState:
        """Initialize separate trees for observations (agents, features).

        With ``torso="cnn"`` observations are (agents, H, W, C).
        ``hyperparameters`` may be scalar (broadcast to every slot) or carry a
        leading agent axis; ``None`` uses the configured defaults.
        """
        if obs.ndim < 2 or obs.shape[0] < 1 or obs.size < 1:
            raise ValueError("IPPO observations must have shape (agents, features).")
        check_torso_observations(obs, self.config.torso, "IPPO")
        keys = jax.random.split(key, obs.shape[0])
        if hyperparameters is None:
            return jax.vmap(self._ppo.init)(obs[:, None, :], keys)

        def per_slot_leaf(leaf: Any) -> jax.Array:
            if jnp.ndim(leaf) == 1:
                return as_float32(leaf)
            return jnp.broadcast_to(as_float32(leaf), (obs.shape[0],))

        per_slot = cast(PPOHyperparameters, jax.tree.map(per_slot_leaf, hyperparameters))
        # Inside vmap the values are traced and PPO's own check is skipped, so
        # check each slot's concrete values here.
        for slot in range(obs.shape[0]):

            def pick(leaf: jax.Array, slot: int = slot) -> jax.Array:
                return leaf[slot]

            cast(PPOHyperparameters, jax.tree.map(pick, per_slot)).validate()
        return jax.vmap(self._ppo.init)(obs[:, None, :], keys, per_slot)

    def evaluate(self, state: PPOState, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return logits (agents, actions) and values (agents,) from local models."""
        logits, value = jax.vmap(self._ppo.evaluate)(state, obs[:, None, :])
        return logits[:, 0, :], value[:, 0]

    def act(self, state: PPOState, obs: jax.Array, info: Info, key: jax.Array) -> Action:
        """Sample per-agent actions with independent keys and public masks."""

        def draw(
            learner: PPOState,
            obs: jax.Array,
            mask: jax.Array,
            active: jax.Array,
            respawn: jax.Array,
            key: jax.Array,
        ) -> Action:
            local: Info = {
                "action_mask": mask[None, :],
                "active_mask": active[None],
                "respawn": respawn[None],
                "t": info["t"],
            }
            return self._ppo.act(learner, obs[None, :], local, key)

        sampled = jax.vmap(draw)(
            state,
            obs,
            info["action_mask"],
            info["active_mask"],
            info["respawn"],
            jax.random.split(key, obs.shape[0]),
        )
        return Action(sampled.action[:, 0], sampled.log_prob[:, 0], sampled.value[:, 0])

    def update(
        self, state: PPOState, rollout: Rollout, last_value: jax.Array, key: jax.Array
    ) -> tuple[PPOState, jax.Array, PPOStats]:
        """Optimize each (time, singleton-agent) rollout independently."""
        if rollout.reward.ndim != 2 or last_value.shape != rollout.reward.shape[1:]:
            raise ValueError("IPPO requires (time, agents) rewards and final agent values.")
        key, draw_key = jax.random.split(key)

        def agent_axis(value: jax.Array) -> jax.Array:
            return jnp.swapaxes(value, 0, 1)

        def single_agent(value: jax.Array) -> jax.Array:
            return value[:, None]

        by_agent = jax.tree.map(agent_axis, rollout)

        def update_agent(
            learner: PPOState, data: Rollout, last: jax.Array, key: jax.Array
        ) -> tuple[PPOState, PPOStats]:
            data = jax.tree.map(single_agent, data)
            trained, _, statistics = self._ppo.update(learner, data, last[None], key)
            return trained, statistics

        trained, statistics = jax.vmap(update_agent)(
            state, by_agent, last_value, jax.random.split(draw_key, last_value.shape[0])
        )
        return trained, key, statistics


class TrainingState[StateT](NamedTuple):
    """Full lifetime, learner/optimizer, shuffle RNG, update count and diagnostics."""

    lifetime: runtime.Lifetime[StateT]
    agent: PPOState
    optimizer_key: jax.Array
    updates: Clock
    statistics: PPOStats
    healthy: jax.Array


def init_training[State](
    env: Environment[State],
    agent: PPO | IPPO,
    *,
    seed: int | jax.Array,
    window_size: int,
    hyperparameters: PPOHyperparameters | None = None,
) -> TrainingState[State]:
    """Initialize a continuing lifetime, learner, optimizer key and counters.

    ``hyperparameters`` overrides the configured defaults (validated when
    concrete).
    """
    lifetime = runtime.init_lifetime(env, seed=seed, window_size=window_size)
    agent_key, init_key, optimizer_key = jax.random.split(lifetime.agent_key, 3)
    lifetime = lifetime._replace(agent_key=agent_key)
    check_discrete_actions(
        env, lifetime.info["action_mask"], agent.action_size, type(agent).__name__
    )
    check_environment_pixel_scale(env, agent, type(agent).__name__)
    state = agent.init(lifetime.obs, init_key, hyperparameters)
    zero = jnp.zeros(state.gradient_steps.lo.shape, dtype=jnp.float32)
    return TrainingState(
        lifetime,
        state,
        optimizer_key,
        Clock.from_int(0),
        PPOStats(zero, zero, zero, zero, zero, zero, zero),
        jnp.asarray(True),
    )


def _env_action(sampled: Action) -> jax.Array:
    return sampled.action


class PPORunner[StateT]:
    """Collect/optimize continuing rollouts and checkpoint only between updates.

    The compiled chunk uses per-function deterministic GPU compilation so a
    fresh process resumes exactly; this may cost throughput. Process-global
    XLA flags are never changed. With ``reference`` each tick also scores a
    stateless reference policy from the same state (one extra environment
    step) and the metrics carry accumulates same-state regret.
    ``donate=True`` donates the input carry's buffers to the compiled call, so
    large states (metric windows, parameters) are updated in place instead of
    copied. The carry passed in is then invalid: keep only the returned one.
    Results are unchanged.

    The runner is already compiled and must be the outermost compiled call:
    use ``runner(carry)`` or a bare ``jax.vmap(runner)``, never inside
    ``jax.jit``, which is refused.
    """

    def __init__(
        self,
        env: Environment[StateT],
        agent: PPO | IPPO,
        *,
        reference: runtime.Policy | None = None,
        donate: bool = False,
    ) -> None:
        self._donate = donate
        if env.metadata.get("execution") == "host":
            raise ValueError("Use HostPPORunner for mutable CPU simulators.")
        validate_minibatches(agent, env.num_agents)
        check_environment_pixel_scale(env, agent, type(agent).__name__)
        runtime.check_reference(env, reference)
        self.env = env
        self.agent = agent
        self.reference = reference
        self.metadata = runner_metadata(env, agent, reference)

        def run(carry: TrainingState[StateT]) -> TrainingState[StateT]:
            def collect(
                lifetime: runtime.Lifetime[StateT], _: None
            ) -> tuple[runtime.Lifetime[StateT], Rollout]:
                def act(obs: jax.Array, info: Info, key: jax.Array) -> Action:
                    return agent.act(carry.agent, obs, info, key)

                advanced, sampled, reward = runtime.interact(
                    env, lifetime, act, action=_env_action, reference=reference
                )
                transition = Rollout(
                    lifetime.obs,
                    lifetime.info["action_mask"],
                    lifetime.info["active_mask"],
                    sampled.action,
                    sampled.value,
                    reward,
                    sampled.log_prob,
                )
                return advanced, transition

            lifetime, rollout = jax.lax.scan(
                collect, carry.lifetime, None, length=agent.config.rollout_steps
            )
            _, last_value = agent.evaluate(carry.agent, lifetime.obs)
            trained, optimizer_key, statistics = agent.update(
                carry.agent, rollout, last_value, carry.optimizer_key
            )
            healthy = training_health(carry.healthy, lifetime.metrics.healthy, trained, statistics)
            return TrainingState(
                lifetime, trained, optimizer_key, carry.updates.increment(), statistics, healthy
            )

        self._run = compile_chunk(run, donate=donate)

    def init(
        self,
        *,
        seed: int | jax.Array = 0,
        window_size: int = 128,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> TrainingState[StateT]:
        """Initialize without changing the native environment's seeded RNG stream.

        ``hyperparameters`` overrides the configured defaults for this
        lifetime; the values are stored as traced state.
        """
        return init_training(
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

    def __call__(self, state: TrainingState[StateT]) -> TrainingState[StateT]:
        """Advance one rollout and optimize it without resets or offline evaluation."""
        return self._run(runtime.unalias(state) if self.donates else state)


class HostPPORunner[StateT]:
    """Reuse the same PPO learner with host-controlled, batched CPU simulation.

    Policy evaluation, GAE and optimization remain compiled JAX operations.
    Simulator stepping and host/device transfers are explicit, not hidden in a
    supposedly pure scan. Host kernels use per-function deterministic GPU
    compilation for exact fresh-process continuation; this may cost throughput.
    Checkpoint support is determined by the backend. ``reference`` adds
    same-state regret exactly as in :class:`PPORunner`.
    """

    def __init__(
        self,
        env: Environment[StateT],
        agent: PPO | IPPO,
        *,
        reference: runtime.Policy | None = None,
    ) -> None:
        validate_minibatches(agent, env.num_agents)
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
            Callable[[PPOState, jax.Array, Info, jax.Array], Action],
            jax.jit(agent.act, compiler_options=compiler_options),
        )
        self._evaluate = cast(
            Callable[[PPOState, jax.Array], tuple[jax.Array, jax.Array]],
            jax.jit(agent.evaluate, compiler_options=compiler_options),
        )
        self._update = cast(
            Callable[
                [PPOState, Rollout, jax.Array, jax.Array], tuple[PPOState, jax.Array, PPOStats]
            ],
            jax.jit(agent.update, compiler_options=compiler_options),
        )
        self._metrics = cast(
            metrics.Updater,
            jax.jit(metrics.update, compiler_options=compiler_options),
        )

    def init(
        self,
        *,
        seed: int = 0,
        window_size: int = 128,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> TrainingState[StateT]:
        """Initialize external simulation and the same full learner/metric carry."""
        return init_training(
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

    def __call__(self, carry: TrainingState[StateT]) -> TrainingState[StateT]:
        """Collect a bounded CPU rollout, then call the shared PPO optimizer."""
        lifetime = carry.lifetime
        transitions: list[Rollout] = []
        for _ in range(self.agent.config.rollout_steps):

            def act(obs: jax.Array, info: Info, key: jax.Array) -> Action:
                return self._act(carry.agent, obs, info, key)

            advanced, sampled, reward = runtime.interact(
                self.env,
                lifetime,
                act,
                action=_env_action,
                reference=self.reference,
                update=self._metrics,
            )
            transitions.append(
                Rollout(
                    lifetime.obs,
                    lifetime.info["action_mask"],
                    lifetime.info["active_mask"],
                    sampled.action,
                    sampled.value,
                    reward,
                    sampled.log_prob,
                )
            )
            lifetime = advanced

        def stack(*values: jax.Array) -> jax.Array:
            return jnp.stack(values)

        rollout = jax.tree.map(stack, *transitions)
        _, last_value = self._evaluate(carry.agent, lifetime.obs)
        trained, optimizer_key, statistics = self._update(
            carry.agent, rollout, last_value, carry.optimizer_key
        )
        healthy = training_health(carry.healthy, lifetime.metrics.healthy, trained, statistics)
        return TrainingState(
            lifetime, trained, optimizer_key, carry.updates.increment(), statistics, healthy
        )


def validate_minibatches(agent: PPO | IPPO, num_agents: int) -> None:
    """Refuse a rollout that does not split into minibatches of at least two samples.

    PPO pools ``rollout_steps * num_agents`` samples; IPPO splits each agent's
    ``rollout_steps`` samples on their own. A one-sample minibatch normalises
    its advantage to exactly zero, so the policy loss would be zero too.
    """
    steps, minibatches = agent.config.rollout_steps, agent.config.minibatches
    if isinstance(agent, IPPO):
        samples, counted = steps, f"rollout_steps={steps} per agent"
    else:
        samples = steps * num_agents
        counted = f"rollout_steps={steps} x {num_agents} agents, pooled"
    if samples % minibatches:
        raise ValueError(
            f"Rollout samples ({samples}: {counted}) must divide evenly into "
            f"{minibatches} minibatches."
        )
    if samples // minibatches < 2:
        raise ValueError(
            f"Rollout samples ({samples}: {counted}) split into {minibatches} minibatches "
            f"leave {samples // minibatches} sample per minibatch; advantage normalisation "
            "needs at least 2. Raise rollout_steps or lower PPOConfig.minibatches."
        )
