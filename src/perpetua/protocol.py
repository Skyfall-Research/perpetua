"""The MORPHEUS paper's six-metric protocol for behaviour around configuration changes.

Ported, with permission, from an earlier Skyfall evaluation harness, whose arithmetic was
differentially verified against the paper's reference evaluation code; a frozen slice of those
recorded answers is replayed by ``tests/test_protocol.py``. It reads a recorded trace of
``(step, regime_id, reward)`` rows -- no environment, model or JAX -- so any Perpetua run whose
configuration labels are known (for example from the Oracle's ``*_phase`` fields) can be scored.

    1. Per-configuration reward ``J_k``     mean reward over configuration *k*
    2. Adaptation speed ``tau_adapt``       steps after a shift to reach the adapted level
    3. Forgetting ``F_k``                   mean(first encounter) - mean(second encounter)
    4. Recovery time ``tau_rec``            steps for the running mean to settle
    5. Stability ``sigma^2``                within-configuration reward variance
    6. Performance gap ``Delta_k``          upper bound minus the first occurrence's tail mean

Conventions of the source metrics, which differ from :mod:`perpetua.evaluation`:

* **Within-trace forgetting** (:func:`within_trace_forgetting`, the suite's "forgetting") is
  positive when the second encounter is worse and compares encounters of the same configuration
  *within one training trace*. ``perpetua.evaluation.forgetting`` instead re-scores saved policies
  on fresh lifetimes. The names differ because they answer different questions.
* **Unavailable metrics are** ``None`` -- a configuration seen once has no forgetting, a
  single-step interval has no variance, and no adaptation speed exists when no steps of the
  target configuration follow the shift.
* Segmentation follows the configuration label, never chunk or rollout boundaries. A revisit
  recorded under a new label (``regime_origin``) still counts as the same configuration for
  forgetting.
* Adaptation and recovery default to the **segment anchor** (relative to the interval's own
  peak/asymptote); the **ceiling anchor** measures against a theoretical upper bound and is the
  formulation the recorded reference answers use.
* The segment-anchored adaptation speed is **not shift-invariant**: an interval whose best running
  mean is zero or negative returns the interval length, the same value as "never adapted", so it
  needs rewards with a positive peak. For negative-reward tasks (costs, Pendulum, MPE) use
  ``Anchor.CEILING`` with an ``upper_bound`` on the same scale as the rewards. The ceiling
  defaults assume a positive bound: with a negative ``upper_bound`` also pass ``alpha >= 1``
  to :func:`adaptation_speed` and a positive ``epsilon`` to :func:`recovery_time`, or both
  return ``None`` for every trace. A bound of zero needs the ``epsilon`` as well.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from statistics import mean, variance
from typing import Any

from perpetua.core import as_int

__all__ = [
    "AdaptationWindow",
    "Anchor",
    "ProtocolStep",
    "ProtocolTrace",
    "adaptation_speed",
    "cumulative_reward",
    "default_alpha",
    "per_configuration_reward",
    "performance_gap",
    "recovery_time",
    "report",
    "shift_boundaries",
    "stability",
    "tail_mean_reward",
    "undiscounted_reward",
    "within_trace_forgetting",
    "zero_shot_reward",
]

DEFAULT_ALPHA_SEGMENT = 0.9
DEFAULT_ALPHA_CEILING = 0.5
DEFAULT_EPSILON_FRACTION = 0.30
DEFAULT_EPSILON_ABS_FLOOR = 0.005
DEFAULT_TAIL_FRACTION = 0.2
DEFAULT_ZERO_SHOT_WINDOW = 10


class AdaptationWindow(enum.Enum):
    """Which steps count after a shift: the configuration's own interval, or all later steps."""

    SEGMENT = "segment"
    POST_SHIFT = "post_shift"


class Anchor(enum.Enum):
    """Measure adaptation/recovery against the segment's own peak, or a theoretical ceiling."""

    SEGMENT = "segment"
    CEILING = "ceiling"


def default_alpha(anchor: Anchor) -> float:
    """The adapted fraction that means something for this anchor."""
    return DEFAULT_ALPHA_SEGMENT if anchor is Anchor.SEGMENT else DEFAULT_ALPHA_CEILING


@dataclass(frozen=True, slots=True)
class ProtocolStep:
    """One recorded tick: its index, configuration label and reward."""

    step: int
    regime_id: str | None
    reward: float
    regime_origin: str | None = None


@dataclass(frozen=True, slots=True)
class ProtocolTrace:
    """One policy's recorded run; boundaries come from the labels themselves."""

    steps: tuple[ProtocolStep, ...]
    gamma: float = 0.99
    meta: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_arrays(
        cls,
        rewards: Sequence[float],
        labels: Sequence[str],
        *,
        origins: Sequence[str | None] | None = None,
        start: int = 0,
        gamma: float = 0.99,
    ) -> ProtocolTrace:
        """Build a trace from per-tick rewards and configuration labels."""
        if len(rewards) != len(labels) or (origins is not None and len(origins) != len(labels)):
            raise ValueError("rewards, labels and origins must have one entry per tick.")
        return cls(
            tuple(
                ProtocolStep(
                    start + index,
                    str(label),
                    float(reward),
                    None if origins is None else origins[index],
                )
                for index, (reward, label) in enumerate(zip(rewards, labels, strict=True))
            ),
            gamma,
        )

    @property
    def regime_ids(self) -> list[str]:
        """Every configuration this trace ran under, in the order first seen."""
        seen: list[str] = []
        for step in self.steps:
            if step.regime_id is not None and step.regime_id not in seen:
                seen.append(step.regime_id)
        return seen


def cumulative_reward(trace: ProtocolTrace, gamma: float | None = None) -> float:
    """Discounted return, discounted by position in the trace."""
    rate = trace.gamma if gamma is None else gamma
    return sum((rate**t) * step.reward for t, step in enumerate(trace.steps))


def undiscounted_reward(trace: ProtocolTrace) -> float:
    """Plain sum of every step's reward."""
    return sum(step.reward for step in trace.steps)


def per_configuration_reward(trace: ProtocolTrace, regime_id: str) -> float | None:
    """Mean reward over every step under one configuration (all of its occurrences)."""
    rewards = [s.reward for s in trace.steps if s.regime_id == regime_id]
    return mean(rewards) if rewards else None


def adaptation_speed(
    trace: ProtocolTrace,
    shift_step: int,
    target_regime: str,
    upper_bound: float | None = None,
    alpha: float | None = None,
    window: AdaptationWindow = AdaptationWindow.SEGMENT,
    anchor: Anchor = Anchor.SEGMENT,
) -> int | None:
    """Steps after a shift for the running mean to reach the adapted level.

    Segment anchor (the default): ``alpha`` of the interval's own peak running mean; never
    crossing returns the interval length. If that peak is zero or negative the anchor is
    meaningless and the interval length is returned too, so a negative-reward interval reads as
    "never adapted" whatever the learning curve (adding a constant to every reward changes the
    answer); pass ``anchor=Anchor.CEILING`` and an ``upper_bound`` for such tasks. Ceiling
    anchor: ``alpha * upper_bound``; never crossing returns ``None``. With a negative
    ``upper_bound`` the default ``alpha`` of 0.5 puts that level above the bound itself, where
    no running mean can reach it, so pass ``alpha >= 1`` (1 is the bound; a larger value is a
    level below it). With no steps to measure, both anchors return ``None``. ``alpha`` defaults
    to :func:`default_alpha` for the anchor.
    """
    if alpha is None:
        alpha = default_alpha(anchor)
    rewards = _window_rewards(trace, shift_step, target_regime, window)
    if not rewards:
        return None
    running = _running_means(rewards)
    if anchor is Anchor.SEGMENT:
        peak = max(running)
        if peak <= 0:
            return len(rewards)
        threshold = alpha * peak
        for tau, mean_so_far in enumerate(running, start=1):
            if mean_so_far >= threshold:
                return tau
        return len(rewards)
    if upper_bound is None:
        raise ValueError("Anchor.CEILING needs an upper_bound; pass one or use Anchor.SEGMENT.")
    threshold = alpha * upper_bound
    for tau, mean_so_far in enumerate(running, start=1):
        if mean_so_far >= threshold:
            return tau
    return None


def within_trace_forgetting(trace: ProtocolTrace, regime_id: str) -> float | None:
    """``mean(first occurrence) - mean(second)``: positive means worse on the revisit.

    The MORPHEUS suite calls this "forgetting". It is measured within one
    training trace, unlike :func:`perpetua.evaluation.forgetting`, which
    re-scores saved policies on fresh lifetimes.

    ``None`` unless the configuration occurs at least twice; a revisit under a new label counts
    through ``regime_origin``.
    """
    segments = _contiguous_segments(trace, regime_id, by_identity=True)
    if len(segments) < 2:
        return None
    first = _mean([s.reward for s in segments[0]])
    second = _mean([s.reward for s in segments[1]])
    return first - second


def recovery_time(
    trace: ProtocolTrace,
    shift_step: int,
    target_regime: str,
    upper_bound: float | None = None,
    epsilon: float | None = None,
    window: AdaptationWindow = AdaptationWindow.SEGMENT,
    anchor: Anchor = Anchor.SEGMENT,
    epsilon_fraction: float = DEFAULT_EPSILON_FRACTION,
    epsilon_abs_floor: float = DEFAULT_EPSILON_ABS_FLOOR,
) -> int | None:
    """Steps after a shift for the running mean to settle.

    Segment anchor: within ``max(epsilon_fraction * |tail mean|, epsilon_abs_floor)`` of the
    interval's tail mean. Ceiling anchor: within ``epsilon`` (default half the bound) of it.
    Half of a zero or negative ``upper_bound`` is a band no running mean can fall in (a zero
    bound is met only by a mean exactly equal to it), so pass a positive ``epsilon`` on the
    reward scale for such bounds.
    """
    rewards = _window_rewards(trace, shift_step, target_regime, window)
    if not rewards:
        return None
    if anchor is Anchor.SEGMENT:
        tail = max(1, math.ceil(len(rewards) * DEFAULT_TAIL_FRACTION))
        asymptote = _mean(rewards[-tail:])
        band = max(epsilon_fraction * abs(asymptote), epsilon_abs_floor)
        for tau, mean_so_far in enumerate(_running_means(rewards), start=1):
            if abs(mean_so_far - asymptote) <= band:
                return tau
        return None
    if upper_bound is None:
        raise ValueError("Anchor.CEILING needs an upper_bound; pass one or use Anchor.SEGMENT.")
    if epsilon is None:
        epsilon = 0.5 * upper_bound
    for tau, mean_so_far in enumerate(_running_means(rewards), start=1):
        if abs(mean_so_far - upper_bound) <= epsilon:
            return tau
    return None


def stability(trace: ProtocolTrace, regime_id: str) -> dict[str, Any]:
    """Within-configuration reward variance: ``{"per_segment": [...], "pooled": ...}``."""
    per_segment: list[float | None] = []
    pooled: list[float] = []
    for segment in _contiguous_segments(trace, regime_id):
        rewards = [s.reward for s in segment]
        pooled.extend(rewards)
        per_segment.append(variance(rewards) if len(rewards) >= 2 else None)
    return {"per_segment": per_segment, "pooled": variance(pooled) if len(pooled) >= 2 else None}


def tail_mean_reward(
    trace: ProtocolTrace, regime_id: str, tail_fraction: float = DEFAULT_TAIL_FRACTION
) -> float | None:
    """The settled reward: mean of each occurrence's final ``tail_fraction``, averaged.

    Each tail holds ``max(1, ceil(len(occurrence) * tail_fraction))`` steps; ``None`` if the
    configuration never occurs.
    """
    means: list[float] = []
    for segment in _contiguous_segments(trace, regime_id):
        tail = max(1, math.ceil(len(segment) * tail_fraction))
        means.append(mean([s.reward for s in segment[-tail:]]))
    return mean(means) if means else None


def performance_gap(
    trace: ProtocolTrace,
    regime_id: str,
    upper_bound: float,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> float | None:
    """``upper_bound`` minus the mean of the first occurrence's final ``tail_fraction``.

    Only the first occurrence counts, and its tail holds
    ``max(1, round(len(occurrence) * tail_fraction))`` steps, so this is not
    ``upper_bound - tail_mean_reward(...)``, which averages every occurrence and rounds the tail
    up. ``None`` if the configuration never occurs.
    """
    segments = _contiguous_segments(trace, regime_id)
    if not segments:
        return None
    segment = segments[0]
    tail = max(1, round(len(segment) * tail_fraction))
    return upper_bound - mean(s.reward for s in segment[-tail:])


def zero_shot_reward(trace: ProtocolTrace, shift_step: int, delta: int) -> float | None:
    """Mean reward over the first ``delta`` steps from a shift; ``None`` past the end."""
    delta = as_int(delta, "delta must be a positive integer.", minimum=1)
    post_shift = [s.reward for s in trace.steps if s.step >= shift_step]
    return _mean(post_shift[:delta]) if post_shift else None


def shift_boundaries(trace: ProtocolTrace) -> list[tuple[int, str, str]]:
    """Every configuration change as ``(step, from, to)``, derived from the labels."""
    boundaries: list[tuple[int, str, str]] = []
    if not trace.steps:
        return boundaries
    previous = trace.steps[0].regime_id
    for step in trace.steps[1:]:
        if step.regime_id != previous:
            if previous is not None and step.regime_id is not None:
                boundaries.append((step.step, previous, step.regime_id))
            previous = step.regime_id
    return boundaries


def report(
    trace: ProtocolTrace,
    *,
    upper_bound: float | None = None,
    anchor: Anchor = Anchor.SEGMENT,
    window: AdaptationWindow = AdaptationWindow.SEGMENT,
    zero_shot_window: int = DEFAULT_ZERO_SHOT_WINDOW,
) -> dict[str, Any]:
    """All six metrics for one trace: per configuration and per shift."""
    per_regime: dict[str, Any] = {}
    for regime_id in trace.regime_ids:
        per_regime[regime_id] = {
            "per_configuration_reward": per_configuration_reward(trace, regime_id),
            "within_trace_forgetting": within_trace_forgetting(trace, regime_id),
            "stability": stability(trace, regime_id),
            "tail_mean_reward": tail_mean_reward(trace, regime_id),
            "performance_gap": (
                None if upper_bound is None else performance_gap(trace, regime_id, upper_bound)
            ),
        }
    shifts = [
        {
            "shift_step": step,
            "from": source,
            "to": target,
            "adaptation_speed": adaptation_speed(
                trace, step, target, upper_bound, window=window, anchor=anchor
            ),
            "recovery_time": recovery_time(
                trace, step, target, upper_bound, window=window, anchor=anchor
            ),
            "zero_shot_reward": zero_shot_reward(trace, step, zero_shot_window),
        }
        for step, source, target in shift_boundaries(trace)
    ]
    return {
        "cumulative_reward": cumulative_reward(trace),
        "undiscounted_reward": undiscounted_reward(trace),
        "per_regime": per_regime,
        "shifts": shifts,
        "anchor": anchor.value,
        "window": window.value,
    }


def _identity(step: ProtocolStep) -> str | None:
    return step.regime_origin or step.regime_id


def _contiguous_segments(
    trace: ProtocolTrace, regime_id: str, *, by_identity: bool = False
) -> list[list[ProtocolStep]]:
    """Each unbroken run of steps under one configuration, in order."""
    wanted: set[str | None] = {regime_id}
    if by_identity:
        wanted |= {
            _identity(s)
            for s in trace.steps
            if s.regime_id == regime_id or s.regime_origin == regime_id
        }
        wanted.discard(None)
    key: Callable[[ProtocolStep], str | None] = (
        _identity if by_identity else (lambda s: s.regime_id)
    )
    segments: list[list[ProtocolStep]] = []
    current: list[ProtocolStep] = []
    for step in trace.steps:
        if key(step) in wanted:
            current.append(step)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _window_rewards(
    trace: ProtocolTrace, shift_step: int, regime_id: str, window: AdaptationWindow
) -> list[float]:
    if window is AdaptationWindow.SEGMENT:
        for segment in _contiguous_segments(trace, regime_id):
            if segment and segment[0].step >= shift_step:
                return [s.reward for s in segment]
        return []
    return [s.reward for s in trace.steps if s.step >= shift_step]


def _running_means(values: Sequence[float]) -> list[float]:
    """Mean of each prefix, in one pass and rounded exactly as ``statistics.mean`` rounds."""
    if not all(math.isfinite(value) for value in values):
        return [_mean(values[: count + 1]) for count in range(len(values))]
    total = Fraction(0)
    means: list[float] = []
    for count, value in enumerate(values, start=1):
        total += Fraction(value)
        means.append(float(total / count))
    return means


def _mean(values: Sequence[float]) -> float:
    """Mean, or ``0.0`` for nothing (used only where an empty sum really is zero)."""
    return mean(values) if values else 0.0
