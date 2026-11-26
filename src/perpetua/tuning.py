"""A declared tuning protocol: k-percent tuning for unbounded lifetimes.

A lifetime has no fixed horizon to tune for, so the protocol is part of the
benchmark specification (budget fraction, seeds, selection rule, reporting).
:func:`tune` implements k-percent tuning (Mesbahi et al. 2024): every candidate
runs on the first ``fraction`` of the lifetime for a few tuning seeds, the best
by the declared rule is kept, and only that candidate is evaluated on the whole
lifetime with fresh, disjoint seeds. The result records the protocol, every
tuning score and the environment ticks spent, so the cost of tuning is
reported next to the score it bought.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import numpy as np
from jax.typing import ArrayLike

from perpetua import reporting
from perpetua.core import JSONValue, as_int

__all__ = [
    "Score",
    "TuningProtocol",
    "declared",
    "TuningResult",
    "tune",
]

type Score[T] = Callable[[T, Sequence[int], int], ArrayLike | Sequence[float]]


@dataclass(frozen=True)
class TuningProtocol:
    """Budget fraction, seeds and selection rule, recorded with every result.

    ``fraction`` is the share of the lifetime each candidate is tuned on;
    ``tuning_seeds`` and ``evaluation_seeds`` are disjoint (evaluation seeds
    start at ``seed_offset``); ``selection`` ranks candidates by the mean or by
    the interquartile mean of their tuning scores: the mean of the middle half
    of the sorted scores, with fractional weight at the cuts, so it differs
    from the mean even with three seeds. (``reporting.interquartile_mean``,
    which trims ``n // 4`` scores per end, would not.) The class default is
    ``"mean"``; the presets' ``DEFAULT_TUNING`` selects by ``"iqm"``.
    """

    fraction: float
    tuning_seeds: int = 3
    evaluation_seeds: int = 10
    selection: Literal["mean", "iqm"] = "mean"
    seed_offset: int = 1_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        for name in ("tuning_seeds", "evaluation_seeds"):
            value = as_int(getattr(self, name), f"{name} must be a positive integer.", minimum=1)
            object.__setattr__(self, name, value)
        offset = as_int(
            self.seed_offset,
            "seed_offset must keep evaluation seeds disjoint from tuning seeds.",
            minimum=self.tuning_seeds,
        )
        object.__setattr__(self, "seed_offset", offset)
        if self.selection not in ("mean", "iqm"):
            raise ValueError("selection must be 'mean' or 'iqm'.")

    def tuning_ticks(self, lifetime_ticks: int) -> int:
        """Ticks each candidate gets: ``ceil(fraction * lifetime_ticks)``, at least one."""
        lifetime_ticks = as_int(
            lifetime_ticks, "lifetime_ticks must be a positive integer.", minimum=1
        )
        return max(1, math.ceil(round(self.fraction * lifetime_ticks, 6)))

    @property
    def tuning_seed_list(self) -> list[int]:
        """Tuning seeds ``0 .. tuning_seeds - 1``."""
        return list(range(self.tuning_seeds))

    @property
    def evaluation_seed_list(self) -> list[int]:
        """Evaluation seeds ``seed_offset .. seed_offset + evaluation_seeds - 1``."""
        return list(range(self.seed_offset, self.seed_offset + self.evaluation_seeds))

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """JSON-ready protocol description for result metadata."""
        return {
            "protocol": "k-percent",
            "fraction": self.fraction,
            "tuning_seeds": self.tuning_seeds,
            "evaluation_seeds": self.evaluation_seeds,
            "selection": self.selection,
            "seed_offset": self.seed_offset,
        }


def declared(env: Any) -> TuningProtocol:
    """The tuning protocol a preset declares in ``metadata["configuration"]["preset"]``."""
    configuration: dict[str, Any] = dict(env.metadata.get("configuration", {}))
    preset: Any = configuration.get("preset")
    tuning: Any = cast(dict[str, Any], preset).get("tuning") if isinstance(preset, dict) else None
    if not isinstance(tuning, dict) or cast(dict[str, Any], tuning).get("protocol") != "k-percent":
        raise ValueError("This environment declares no tuning protocol; build a TuningProtocol.")
    fields = cast(dict[str, Any], tuning)
    return TuningProtocol(
        float(fields["fraction"]),
        tuning_seeds=int(fields["tuning_seeds"]),
        evaluation_seeds=int(fields["evaluation_seeds"]),
        selection=fields["selection"],
        seed_offset=int(fields["seed_offset"]),
    )


@dataclass(frozen=True)
class TuningResult[T]:
    """Everything needed to report a tuned result honestly."""

    protocol: dict[str, JSONValue]
    candidates: list[T]
    tuning_scores: np.ndarray[Any, np.dtype[np.float64]]  # (candidates, tuning seeds)
    selected: int
    evaluation_scores: np.ndarray[Any, np.dtype[np.float64]]  # (evaluation seeds,)
    intervals: dict[str, reporting.BootstrapInterval]
    tuning_ticks: int
    lifetime_ticks: int

    @property
    def best(self) -> T:
        """The selected candidate, ``candidates[selected]``."""
        return self.candidates[self.selected]

    @property
    def ticks_spent(self) -> dict[str, int]:
        """Environment ticks per phase, summed over candidates and seeds."""
        tuning = self.tuning_ticks * self.tuning_scores.size
        evaluation = self.lifetime_ticks * self.evaluation_scores.size
        return {"tuning": tuning, "evaluation": evaluation, "total": tuning + evaluation}


def _middle_half_mean(row: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Mean of the middle half of the sorted scores, with fractional weight at the cuts.

    Equal to ``reporting.interquartile_mean`` whenever the number of scores is
    a multiple of four; for the few tuning seeds a protocol usually has (three,
    say) that statistic trims nothing and would be the plain mean, so the
    weights here cut a quarter of the mass from each end instead.
    """
    values = np.sort(row)
    n = values.size
    edges = np.arange(n + 1.0)
    weights = np.clip(np.minimum(edges[1:], 0.75 * n) - np.maximum(edges[:-1], 0.25 * n), 0, None)
    return float(np.sum(weights * values) / np.sum(weights))


def _rank(scores: np.ndarray[Any, np.dtype[np.float64]], selection: str) -> np.ndarray[Any, Any]:
    if selection == "mean":
        return scores.mean(axis=1)
    return np.asarray([_middle_half_mean(row) for row in scores])


def tune[T](
    protocol: TuningProtocol,
    candidates: Sequence[T],
    score: Score[T],
    *,
    lifetime_ticks: int,
    key: jax.Array,
) -> TuningResult[T]:
    """Tune ``candidates`` by k-percent tuning and evaluate the winner.

    ``score(candidate, seeds, ticks)`` runs one lifetime per seed for ``ticks``
    ticks and returns one score per seed (online reward over that run is the
    usual choice). It may batch the seeds, for example by vmapping traced
    hyperparameters. Higher scores are better: the candidate with the highest
    mean or IQM tuning score is selected, and ties keep the earlier one.
    ``key`` only seeds the bootstrap confidence intervals of the evaluation
    scores; it does not choose seeds.
    """
    if not candidates:
        raise ValueError("candidates must not be empty.")
    tuning_ticks = protocol.tuning_ticks(lifetime_ticks)

    def run(candidate: T, seeds: list[int], ticks: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        values = np.asarray(score(candidate, seeds, ticks), dtype=np.float64).reshape(-1)
        if values.shape != (len(seeds),) or not np.all(np.isfinite(values)):
            raise ValueError("score must return one finite value per seed.")
        return values

    tuning = np.stack([run(c, protocol.tuning_seed_list, tuning_ticks) for c in candidates])
    selected = int(np.argmax(_rank(tuning, protocol.selection)))
    evaluation = run(candidates[selected], protocol.evaluation_seed_list, lifetime_ticks)
    intervals = reporting.report(np.asarray(evaluation, dtype=np.float32)[:, None], key=key)
    return TuningResult(
        protocol.configuration,
        list(candidates),
        tuning,
        selected,
        evaluation,
        intervals,
        tuning_ticks,
        lifetime_ticks,
    )
