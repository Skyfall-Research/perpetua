"""Parallel revisit evaluation: score stacked past policies in one vmap."""

# JAX device members are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from typing import cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from perpetua import metrics, runtime
from perpetua.core import Environment, Info, as_int

__all__ = [
    "Act",
    "revisit",
    "recovery_ticks",
    "forgetting",
    "recent_rewards",
]

# Candidate windows scored per batch in recovery_ticks (bounds its memory).
_WINDOW_BATCH = 1024

type Act[AgentT] = Callable[[AgentT, jax.Array, Info, jax.Array], jax.Array]


def revisit[StateT, AgentT](
    env: Environment[StateT],
    act: Act[AgentT],
    states: AgentT,
    *,
    seeds: ArrayLike,
    ticks: int,
) -> jax.Array:
    """Evaluate every stacked past policy from identical seeded lifetimes.

    ``states`` is a policy stack with a leading axis of length K; ``act`` maps
    ``(agent_state, obs, info, key) -> action`` for one unstacked policy (for
    example ``lambda state, obs, info, key: ppo.act(state, obs, info, key).action``).
    Each of the S seeds starts a fresh lifetime, and all randomness
    (environment and action keys) is derived from the seed alone, so every
    policy sees identical lifetimes per seed (common random numbers). Returns
    mean lifetime rewards of shape (K, S), evaluated in one vmapped pass
    rather than a sequential loop. ``revisit`` takes no key and needs a JAX
    environment: a host environment fails under ``vmap``.

    The score averages the per-tick mean reward over the learner slots: the
    slots whose ``info["active_mask"]`` was true at the start or after any
    tick of the evaluated lifetime. Scripted partner and opponent slots never
    are, so their reward (the negated learner reward on the zero-sum
    ``OpponentGrid``) is left out. A learner slot's inactive ticks still count
    as zero-reward ticks, and when every slot acted, or none did, the score
    is the mean over all slots.
    """
    seeds = jnp.asarray(seeds)
    if seeds.ndim != 1 or seeds.size < 1 or seeds.dtype.kind not in {"i", "u"}:
        raise ValueError("Revisit seeds must be a nonempty one-dimensional integer array.")
    ticks = as_int(ticks, "ticks must be a positive integer.", minimum=1)

    def rollout(state: AgentT, seed: jax.Array) -> jax.Array:
        # Only the lifetime mean is read, so the reward ring stays one entry.
        lifetime = runtime.init_lifetime(env, seed=seed, window_size=1)

        def policy(obs: jax.Array, info: Info, act_key: jax.Array) -> jax.Array:
            return act(state, obs, info, act_key)

        def step(
            carry: tuple[runtime.Lifetime[StateT], jax.Array], _: None
        ) -> tuple[tuple[runtime.Lifetime[StateT], jax.Array], None]:
            current, learner = carry
            following = runtime.tick(env, policy, current)
            return (following, learner | following.info["active_mask"]), None

        (final, learner), _ = jax.lax.scan(
            step, (lifetime, lifetime.info["active_mask"]), None, length=ticks
        )
        means = final.metrics.mean
        every = jnp.all(learner) | ~jnp.any(learner)
        weights = learner.astype(means.dtype)
        masked = jnp.sum(means * weights) / jnp.maximum(jnp.sum(weights), 1.0)
        return jnp.where(every, jnp.mean(means), masked)

    return cast(
        jax.Array,
        jax.jit(jax.vmap(jax.vmap(rollout, in_axes=(None, 0)), in_axes=(0, None)))(states, seeds),
    )


def recovery_ticks(
    rewards: ArrayLike,
    switches: ArrayLike,
    *,
    window: int = 32,
    fraction: float = 0.9,
) -> jax.Array:
    """Ticks after each switch until the windowed mean recovers.

    ``rewards`` is a one-dimensional per-tick reward trace; ``switches`` are
    trace indices of the switches. The ``boundary`` Oracle field is raised by
    the step that returns the last reward earned under the old parameters, so
    its trace index is one below the first reward under the new ones (for
    ``Cyclic(values, period)``: ``period - 1`` against ``period``); pass the
    index you mean and state it. For each switch at ``c`` the reference is
    the mean of the ``window`` ticks before it; when ``c < window`` the
    reference window starts at tick 0 and so includes ticks from ``c``
    onward. Recovery is the first
    ``t >= c`` whose following ``window`` mean is at least
    ``reference - (1 - fraction) * abs(reference)`` (a signed-agnostic
    tolerance that also works for negative rewards). Values are capped at the
    trace end; callers extend traces or treat the cap as not-recovered.
    """
    values = jnp.asarray(rewards, dtype=jnp.float32)
    positions = np.asarray(switches)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("Recovery traces must be a nonempty one-dimensional array.")
    if positions.ndim != 1 or positions.dtype.kind not in {"i", "u"}:
        raise ValueError("Switch indices must be a one-dimensional integer array.")
    window = as_int(window, "window must be a positive integer.", minimum=1)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    length = values.shape[0]

    def window_mean(start: jax.Array) -> jax.Array:
        positions = start + jnp.arange(window)
        gathered = values[jnp.minimum(positions, length - 1)]
        return jnp.mean(gathered)

    def one(change: jax.Array) -> jax.Array:
        reference = window_mean(jnp.maximum(change - window, 0))
        threshold = reference - (1.0 - fraction) * jnp.abs(reference)
        offsets = jnp.arange(length)
        candidates = change + offsets
        valid = candidates + window <= length
        # In batches: all candidates at once would hold length x window values.
        means = cast(
            jax.Array,
            jax.lax.map(
                window_mean, jnp.minimum(candidates, length - window), batch_size=_WINDOW_BATCH
            ),
        )
        recovered = means >= threshold
        first = jnp.argmax((valid & recovered).astype(jnp.int32))
        found = jnp.any(valid & recovered)
        return jnp.where(found, first, length - change).astype(jnp.int32)

    if positions.size == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    # Compared as int64 on the host: a narrow or unsigned dtype would wrap in
    # the range check and in ``change - window``.
    wide = positions.astype(np.int64)
    if positions.dtype == np.uint64:
        wide = np.where(positions > np.iinfo(np.int64).max, -1, wide)
    if np.any(wide < 0) or np.any(wide >= length):
        raise ValueError("Switch indices must lie inside the reward trace.")
    return jax.vmap(one)(jnp.asarray(wide, dtype=jnp.int32))


def forgetting(scores: ArrayLike) -> jax.Array:
    """Mean forgetting of earlier phases by the final policy.

    ``scores[k, j]`` is the score of the policy saved after phase ``k``
    evaluated under phase ``j`` conditions (for example from ``revisit`` with
    phase-conditioned lifetimes), shape ``(phases, phases)`` with at least two
    phases. For each earlier phase ``j`` forgetting is the best score any
    policy saved from phase ``j`` up to the second-to-last phase achieved on
    ``j``, minus the final policy's score on ``j`` (Chaudhry et al., 2018);
    the result averages over phases ``0 .. phases - 2``. Scores from policies
    saved before phase ``j`` was learned are ignored, and negative values mean
    the final policy improved on an earlier phase (backward transfer).
    """
    values = jnp.asarray(scores, dtype=jnp.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("Forgetting requires a square (phases, phases) matrix with phases >= 2.")
    phases = values.shape[0]
    earlier = values[: phases - 1, : phases - 1]
    learned = jnp.arange(phases - 1)[:, None] >= jnp.arange(phases - 1)[None, :]
    best = jnp.max(jnp.where(learned, earlier, -jnp.inf), axis=0)
    return jnp.mean(best - values[phases - 1, : phases - 1])


def recent_rewards(metrics_state: metrics.Metrics, *, steps: int) -> jax.Array:
    """Return the most recent per-tick rewards in chronological order.

    Reads the bounded metrics window ring as a ``(steps, num_agents)``
    float32 array, so long training runs can feed ``recovery_ticks`` without
    retaining their own reward trace. ``steps`` must not exceed the populated
    window entries (``metrics_state.filled``); outside traced code that is
    validated here.
    """
    steps = as_int(steps, "steps must be a positive integer.", minimum=1)
    window = metrics_state.window
    if steps > window.shape[0]:
        raise ValueError("steps must not exceed the metrics window size.")
    filled = metrics_state.filled
    if not isinstance(filled, jax.core.Tracer) and int(filled) < steps:
        raise ValueError("steps must not exceed the populated metrics entries.")
    indices = (metrics_state.cursor - steps + jnp.arange(steps)) % window.shape[0]
    return window[indices]
