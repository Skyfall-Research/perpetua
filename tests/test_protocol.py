"""The six-metric protocol replays answers recorded from the reference implementation.

``fixtures/morpheus_metrics_oracle.json`` comes from an earlier Skyfall evaluation harness, where
it froze a slice of a 175,404-comparison differential check against the MORPHEUS paper's
evaluation code.
The cases are generated, so they include single-step intervals, zero and negative bounds,
undiscounted runs and configurations that recur. They were recorded under the ceiling anchor and
the post-shift window, so those are pinned explicitly here.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from perpetua import protocol
from perpetua.protocol import AdaptationWindow, Anchor, ProtocolStep, ProtocolTrace

CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "morpheus_metrics_oracle.json").read_text()
)
TOL = 1e-9


def trace_of(case: dict[str, Any]) -> ProtocolTrace:
    return ProtocolTrace(
        tuple(ProtocolStep(s, label, reward) for s, label, reward, _ in case["steps"]),
        case["gamma"],
    )


def same(ours: float | None, theirs: float | None, what: str) -> None:
    if ours is None or theirs is None:
        assert ours is theirs, f"{what}: ours={ours!r}, reference={theirs!r}"
        return
    assert ours == pytest.approx(theirs, rel=TOL, abs=TOL), what


def test_fixture_covers_the_awkward_regions() -> None:
    assert len(CASES) == 140
    bounds = {case["upper_bound"] for case in CASES}
    assert 0.0 in bounds
    assert any(bound < 0 for bound in bounds)
    assert any(case["gamma"] == 1.0 for case in CASES)
    assert any(len(case["steps"]) == 1 for case in CASES)
    assert any(
        entry["forgetting"] is not None
        for case in CASES
        for entry in case["expected"]["per_regime"].values()
    )


@pytest.mark.parametrize("index", range(len(CASES)))
def test_recorded_case_gives_the_recorded_answer(index: int) -> None:
    case = CASES[index]
    trace, want = trace_of(case), case["expected"]
    bound, alpha = case["upper_bound"], case["alpha"]
    epsilon, tail = case["epsilon"], case["tail_fraction"]
    same(protocol.cumulative_reward(trace), want["cumulative_reward"], "return")
    same(protocol.undiscounted_reward(trace), want["undiscounted_reward"], "undiscounted")
    assert [list(b) for b in protocol.shift_boundaries(trace)] == want["shift_boundaries"]
    for label, expected in want["per_regime"].items():
        same(
            protocol.per_configuration_reward(trace, label),
            expected["per_configuration_reward"],
            f"J[{label}]",
        )
        # The recorded answers use the suite's name, "forgetting".
        same(protocol.within_trace_forgetting(trace, label), expected["forgetting"], f"F[{label}]")
        same(
            protocol.performance_gap(trace, label, bound, tail),
            expected["performance_gap"],
            f"gap[{label}]",
        )
        ours = protocol.stability(trace, label)
        same(ours["pooled"], expected["stability"]["pooled"], f"stability[{label}]")
        assert len(ours["per_segment"]) == len(expected["stability"]["per_segment"])
        for a, b in zip(ours["per_segment"], expected["stability"]["per_segment"], strict=True):
            same(a, b, f"stability segment[{label}]")
    for shift in want["shifts"]:
        step, label = shift["shift_step"], shift["to"]
        same(
            protocol.adaptation_speed(
                trace, step, label, bound, alpha, AdaptationWindow.POST_SHIFT, Anchor.CEILING
            ),
            shift["adaptation_speed"],
            f"adaptation@{step}",
        )
        same(
            protocol.recovery_time(
                trace, step, label, bound, epsilon, AdaptationWindow.POST_SHIFT, Anchor.CEILING
            ),
            shift["recovery_time"],
            f"recovery@{step}",
        )
        same(
            protocol.zero_shot_reward(trace, step, case["zero_shot_window"]),
            shift["zero_shot_reward"],
            f"zero-shot@{step}",
        )


def test_segment_anchor_and_revisit_identity() -> None:
    # A: 4 ticks at 1.0; B: ramps 0, 0.5, 1, 1; A again under a new label at 0.5.
    rewards = [1, 1, 1, 1, 0, 0.5, 1, 1, 0.5, 0.5]
    labels = ["A"] * 4 + ["B"] * 4 + ["A2"] * 2
    origins = [None] * 8 + ["A", "A"]
    trace = ProtocolTrace.from_arrays(rewards, labels, origins=origins)
    assert protocol.within_trace_forgetting(trace, "A") == pytest.approx(0.5)
    assert protocol.within_trace_forgetting(trace, "A2") == pytest.approx(
        0.5
    )  # same conditions, same number
    assert protocol.adaptation_speed(trace, 4, "B") == 4  # running mean 0.625 >= 0.9 * 0.625
    assert protocol.stability(trace, "B")["per_segment"][0] == pytest.approx(0.22916666666666666)
    result = protocol.report(trace)
    assert [shift["to"] for shift in result["shifts"]] == ["B", "A2"]
    assert result["per_regime"]["A"]["performance_gap"] is None  # no bound given


def test_segment_adaptation_speed_is_not_shift_invariant_for_nonpositive_rewards() -> None:
    # The same learning curve: -2, -1, 0 reads as "never adapted" (the interval length)
    # while 8, 9, 10 adapts at tick 2. The ceiling anchor needs a bound on the reward scale.
    labels = ["A", "A", "B", "B", "B"]
    negative = ProtocolTrace.from_arrays([0.0, 0.0, -2.0, -1.0, 0.0], labels)
    shifted = ProtocolTrace.from_arrays([10.0, 10.0, 8.0, 9.0, 10.0], labels)
    assert protocol.adaptation_speed(negative, 2, "B") == 3
    assert protocol.adaptation_speed(shifted, 2, "B") == 2

    # Running means are -2, -1.5, -1: a bound of 0 is never reached, a bound of -1 is reached
    # on tick 3 with alpha=1, and the default alpha of 0.5 asks for -0.5, above that bound.
    def ceiling(upper_bound: float, alpha: float | None) -> int | None:
        return protocol.adaptation_speed(
            negative, 2, "B", upper_bound=upper_bound, alpha=alpha, anchor=Anchor.CEILING
        )

    assert ceiling(0.0, 1.0) is None
    assert ceiling(-1.0, 1.0) == 3
    assert ceiling(-1.0, None) is None
    assert protocol.recovery_time(negative, 2, "B", upper_bound=-1.0, anchor=Anchor.CEILING) is None
    assert (
        protocol.recovery_time(
            negative, 2, "B", upper_bound=-1.0, epsilon=0.5, anchor=Anchor.CEILING
        )
        == 2
    )


@pytest.mark.parametrize("anchor", [Anchor.SEGMENT, Anchor.CEILING])
def test_adaptation_speed_is_none_when_nothing_follows_the_shift(anchor: Anchor) -> None:
    trace = ProtocolTrace.from_arrays([1.0, 1.0, 0.0, 0.0], ["A", "A", "B", "B"])
    assert protocol.adaptation_speed(trace, 4, "B", upper_bound=1.0, anchor=anchor) is None


def test_running_means_match_statistics_mean_on_every_prefix() -> None:
    import random
    from statistics import mean

    rng = random.Random(0)
    for size in (0, 1, 2, 17, 64):
        values = [rng.gauss(0.0, 1.0) * 10.0 ** rng.randint(-30, 30) for _ in range(size)]
        expected = [mean(values[: count + 1]) for count in range(size)]
        assert protocol._running_means(values) == expected  # pyright: ignore[reportPrivateUsage]
    assert protocol._running_means([1.0, float("inf")]) == [1.0, float("inf")]  # pyright: ignore[reportPrivateUsage]


def test_zero_shot_reward_needs_a_positive_window() -> None:
    trace = ProtocolTrace.from_arrays([1.0, 1.0, 0.0, 2.0], ["A", "A", "B", "B"])
    assert protocol.zero_shot_reward(trace, 2, 1) == 0.0
    assert protocol.zero_shot_reward(trace, 2, 2) == 1.0
    for delta in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match=r"delta must be a positive integer"):
            protocol.zero_shot_reward(trace, 2, delta)  # pyright: ignore[reportArgumentType]
