"""Bounded-memory, float32 online reward metrics per fixed agent slot."""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import math
import warnings
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from perpetua.core import Clock, as_int

__all__ = [
    "Metrics",
    "Updater",
    "init",
    "update",
]


def _renamed(old: str, new: str) -> None:
    warnings.warn(
        f"Metrics.{old} is deprecated; use Metrics.{new}, which holds the same value.",
        DeprecationWarning,
        stacklevel=3,
    )


class Metrics(NamedTuple):
    """Streaming carry; storage is O(window_size * agents), not lifetime length.

    Lifetime sums use compensated float32 accumulation. The integer count is
    exact; means have float32 precision. Inactive-slot reward zeros still count
    as lifetime ticks. Non-finite rewards set ``healthy`` false and propagate.

    Recovery tracking stays inert until a caller reports switches through
    ``update``: ``switch_baseline`` holds the pre-switch window mean,
    ``has_switch`` gates the feature, ``since_switch`` counts ticks since the
    last switch and ``recovered_at`` holds the number of ticks since the
    switch at which the trailing window first held only post-switch rewards
    inside the tolerance (``-1`` while unrecovered). ``change_baseline``,
    ``has_change`` and ``since_change`` are deprecated aliases. The tracker
    pools every agent slot, scripted and opponent slots included, so it is
    not meaningful where those rewards cancel (zero-sum ``OpponentGrid``) or
    belong to someone else; use ``evaluation.recovery_ticks`` on the learner
    slots' reward trace there.

    Regret tracking is inert until a caller reports a reference reward:
    ``regret_total`` accumulates the per-tick shortfall of the realized
    transition against the reference policy's same-state reward,
    ``reference - reward`` (compensated float32, like ``total``), so it is
    positive when the learner earns less than the reference and negative when
    it does better; ``regret_window`` keeps the bounded ring of per-tick
    regret. It is a same-state comparison, not best-in-hindsight
    re-planning, and it is exactly zero for every pair of policies where a
    tick's reward does not depend on that tick's action (CartPole, for
    example); :func:`perpetua.testing.reward_action_sensitivity` measures
    that for an environment.

    ``welford_mean`` and ``welford_m2`` are the running mean and the sum of
    squared deviations of the per-tick reward, so ``reward_variance`` and
    ``reward_std`` cost O(1) memory. Each tick adds ``(reward - mean)**2 *
    (n - 1) / n`` to ``welford_m2`` with compensated float32 accumulation
    (``welford_compensation``), where the mean before the tick comes from the
    compensated ``total``. The values keep float32 relative precision at any
    tick count; the float32 tick count they divide by (``Clock.as_float``) is
    the remaining approximation.
    """

    ticks: Clock
    total: jax.Array
    compensation: jax.Array
    ema: jax.Array
    decay: jax.Array
    window: jax.Array
    window_total: jax.Array
    cursor: jax.Array
    filled: jax.Array
    healthy: jax.Array
    switch_baseline: jax.Array
    has_switch: jax.Array
    since_switch: jax.Array
    recovered_at: jax.Array
    regret_total: jax.Array
    regret_compensation: jax.Array
    regret_window: jax.Array
    regret_window_total: jax.Array
    welford_mean: jax.Array
    welford_m2: jax.Array
    welford_compensation: jax.Array

    @property
    def change_baseline(self) -> jax.Array:
        """Deprecated alias of ``switch_baseline``."""
        _renamed("change_baseline", "switch_baseline")
        return self.switch_baseline

    @property
    def has_change(self) -> jax.Array:
        """Deprecated alias of ``has_switch``."""
        _renamed("has_change", "has_switch")
        return self.has_switch

    @property
    def since_change(self) -> jax.Array:
        """Deprecated alias of ``since_switch``."""
        _renamed("since_change", "since_switch")
        return self.since_switch

    @property
    def mean(self) -> jax.Array:
        """Mean reward per lifetime tick and agent; zero before the first tick."""
        return self.total / jnp.maximum(self.ticks.as_float(), 1.0)[..., None]

    @property
    def window_mean(self) -> jax.Array:
        """Mean over the most recent populated window entries."""
        return self.window_total / jnp.maximum(self.filled, 1)[..., None]

    @property
    def reward_variance(self) -> jax.Array:
        """Sample variance of the per-tick reward (Welford); zero before two ticks."""
        count = self.ticks.as_float()[..., None]
        return jnp.where(count > 1, self.welford_m2 / jnp.maximum(count - 1, 1.0), 0.0)

    @property
    def reward_std(self) -> jax.Array:
        """Sample standard deviation of the per-tick reward."""
        return jnp.sqrt(self.reward_variance)

    @property
    def regret_mean(self) -> jax.Array:
        """Mean per-tick regret (``reference - reward``); zero before the first tick."""
        return self.regret_total / jnp.maximum(self.ticks.as_float(), 1.0)[..., None]

    @property
    def regret_window_mean(self) -> jax.Array:
        """Mean regret over the most recent populated window entries."""
        return self.regret_window_total / jnp.maximum(self.filled, 1)[..., None]


class Updater(Protocol):
    """Callable shape of :func:`update`, including compiled ``jax.jit`` versions."""

    def __call__(
        self,
        state: Metrics,
        reward: ArrayLike,
        *,
        switch: ArrayLike | None = None,
        fraction: float = 0.9,
        reference_reward: ArrayLike | None = None,
    ) -> Metrics: ...


def init(*, num_agents: int, window_size: int = 128, ema_decay: float = 0.99) -> Metrics:
    """Create metric state for float32 reward vectors of shape (num_agents,)."""
    num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
    window_size = as_int(window_size, "window_size must be a positive integer.", minimum=1)
    if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be finite and in [0, 1).")
    zero = jnp.zeros(num_agents, dtype=jnp.float32)
    return Metrics(
        Clock.from_int(0),
        zero,
        zero,
        zero,
        jnp.asarray(ema_decay, dtype=jnp.float32),
        jnp.zeros((window_size, num_agents), dtype=jnp.float32),
        zero,
        jnp.int32(0),
        jnp.int32(0),
        jnp.asarray(True),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(False),
        jnp.int32(0),
        jnp.int32(-1),
        zero,
        zero,
        jnp.zeros((window_size, num_agents), dtype=jnp.float32),
        zero,
        zero,
        zero,
        zero,
    )


def update(
    state: Metrics,
    reward: ArrayLike,
    *,
    switch: ArrayLike | None = None,
    fraction: float = 0.9,
    reference_reward: ArrayLike | None = None,
    change: ArrayLike | None = None,
) -> Metrics:
    """Accumulate one reward vector without retaining transition history.

    ``switch`` is an optional boolean scalar marking a declared switch on this
    tick (typically the backend's hidden flag, see ``runtime.drift_switch``).
    With it, the window mean before the switch becomes ``switch_baseline`` and
    ``recovered_at`` records the ticks since the switch at which the trailing
    window first lies within ``1 - fraction`` of that baseline (``-1`` while
    unrecovered). Recovery is only judged once the window holds no pre-switch
    rewards and the switch-tick reward has left it, so the earliest window
    starts one tick after the switch and the smallest reported value is
    ``window_size``. When the post-hoc ``evaluation.recovery_ticks`` finds
    recovery at ``r >= 1`` inside the trace, the reported value is
    ``r + window_size - 1`` (the tracker's window is trailing, the post-hoc
    one leading); when ``r`` is 0 the tracker reports ``s + window_size - 1``
    for the first recovering window start ``s >= 1``. A switch reported
    before any reward (on the first ``update``) is not tracked: there is no
    window to take a baseline from, so the four fields stay as they were.
    ``since_switch`` is int32 and stops at ``2**31 - 1`` instead of wrapping,
    so a recovery later than that after a switch reports that value. Without
    ``switch`` the four recovery fields pass through untouched. ``change`` is
    a deprecated alias of ``switch``.

    ``reference_reward`` is the optional reward a reference policy would earn
    from the same state (see ``runtime.shadow_reward``). With it, the per-tick
    regret ``reference_reward - reward`` is added to ``regret_total`` with
    compensated float32 accumulation and recorded in the ``regret_window``
    ring; without it the regret fields pass through untouched.
    """
    if change is not None:
        if switch is not None:
            raise TypeError("Pass switch= only; change= is its deprecated alias.")
        warnings.warn(
            "metrics.update(change=...) is deprecated; pass switch=, which means the same.",
            DeprecationWarning,
            stacklevel=2,
        )
        switch = change
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    reward = jnp.asarray(reward, dtype=jnp.float32)
    if reward.shape != state.total.shape:
        raise ValueError("Reward shape must match the configured agent slots.")
    adjusted = reward - state.compensation
    total = state.total + adjusted
    compensation = (total - state.total) - adjusted
    first = (state.ticks.hi == 0) & (state.ticks.lo == 0)
    ema = jnp.where(first, reward, state.decay * state.ema + (1 - state.decay) * reward)
    count = state.ticks.increment().as_float()
    previous_mean = (state.total - state.compensation) / jnp.maximum(count - 1.0, 1.0)
    welford_mean = (total - compensation) / count
    delta = reward - previous_mean
    adjusted_m2 = delta * (delta * ((count - 1.0) / count)) - state.welford_compensation
    welford_m2 = state.welford_m2 + adjusted_m2
    welford_compensation = (welford_m2 - state.welford_m2) - adjusted_m2
    window = state.window.at[state.cursor].set(reward)
    # Re-summed each tick: a running add/subtract would keep the float32
    # rounding error of a large reward long after it left the window.
    window_total = jnp.sum(window, axis=0)
    if switch is None:
        switch_baseline = state.switch_baseline
        has_switch = state.has_switch
        since_switch = state.since_switch
        recovered_at = state.recovered_at
    else:
        changed = jnp.asarray(switch)
        if changed.shape != () or changed.dtype.kind not in {"b", "i", "u"}:
            raise ValueError(
                "switch must be a scalar boolean (or integer) flag; got shape "
                f"{tuple(changed.shape)} and dtype {changed.dtype}."
            )
        # A switch before any reward has no window to take a baseline from.
        changed = changed.astype(jnp.bool_) & (state.filled > 0)
        has_switch = state.has_switch | changed
        reference = jnp.mean(state.window_total / jnp.maximum(state.filled, 1))
        switch_baseline = jnp.where(changed, reference, state.switch_baseline)
        since_switch = jnp.where(
            changed,
            jnp.int32(0),
            jnp.where(
                has_switch,
                jnp.minimum(state.since_switch, jnp.int32(2**31 - 2)) + jnp.int32(1),
                jnp.int32(0),
            ),
        )
        threshold = switch_baseline - (1.0 - fraction) * jnp.abs(switch_baseline)
        recovered = (since_switch >= window.shape[0]) & (
            jnp.mean(window_total / window.shape[0]) >= threshold
        )
        recovered_at = jnp.where(
            changed | ~has_switch,
            jnp.int32(-1),
            jnp.where(
                state.recovered_at >= 0,
                state.recovered_at,
                jnp.where(recovered, since_switch, jnp.int32(-1)),
            ),
        )
    if reference_reward is None:
        regret_total = state.regret_total
        regret_compensation = state.regret_compensation
        regret_window = state.regret_window
        regret_window_total = state.regret_window_total
    else:
        reference = jnp.asarray(reference_reward, dtype=jnp.float32)
        if reference.shape != state.total.shape:
            raise ValueError("Reference reward shape must match the configured agent slots.")
        step_regret = reference - reward
        adjusted_regret = step_regret - state.regret_compensation
        regret_total = state.regret_total + adjusted_regret
        regret_compensation = (regret_total - state.regret_total) - adjusted_regret
        regret_window = state.regret_window.at[state.cursor].set(step_regret)
        regret_window_total = jnp.sum(regret_window, axis=0)
    return Metrics(
        state.ticks.increment(),
        total,
        compensation,
        ema,
        state.decay,
        window,
        window_total,
        (state.cursor + 1) % state.window.shape[0],
        jnp.minimum(state.filled + 1, state.window.shape[0]),
        state.healthy
        & jnp.all(jnp.isfinite(reward))
        & jnp.all(jnp.isfinite(total))
        & jnp.all(jnp.isfinite(ema))
        & jnp.all(jnp.isfinite(window_total))
        & jnp.all(jnp.isfinite(regret_total))
        & jnp.all(jnp.isfinite(welford_m2)),
        switch_baseline,
        has_switch,
        since_switch,
        recovered_at,
        regret_total,
        regret_compensation,
        regret_window,
        regret_window_total,
        welford_mean,
        welford_m2,
        welford_compensation,
    )
