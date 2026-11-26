"""Property-based checks for drift arithmetic, metrics and protocol invariants.

These complement the example-based suites: Hypothesis generates the awkward
combinations (repeated phases, unusual periods, rewards that cancel, recurrent
configurations) rather than a short hand-written list owning them.
"""

# JAX transformation/device annotations are partial.
# pyright: reportUnknownMemberType=false

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import perpetua as pp


def _normal_in_float32(value: float) -> bool:
    """Reject values that only exist as float32 subnormals (XLA may flush them)."""
    return value == 0.0 or abs(value) >= float(np.finfo(np.float32).tiny)


FLOAT = st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False).filter(
    _normal_in_float32
)
SMALL = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False).filter(
    _normal_in_float32
)
BOUNDED = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False).filter(
    _normal_in_float32
)
SETTINGS = settings(max_examples=20, deadline=None)


@SETTINGS
@given(
    phases=st.lists(
        st.lists(SMALL, min_size=2, max_size=2),
        min_size=1,
        max_size=5,
    ),
    period=st.integers(min_value=1, max_value=8),
    ticks=st.integers(min_value=0, max_value=40),
)
def test_cyclic_phase_value_and_switch_timing_are_exact(
    phases: list[list[float]], period: int, ticks: int
) -> None:
    process = pp.drift.Cyclic(phases, period)
    values = jnp.asarray(phases, dtype=jnp.float32)
    state = process.init(jnp.zeros(2, dtype=jnp.float32), jax.random.key(0))
    for tick in range(1, ticks + 1):
        before = state
        state = process.step(state, jax.random.key(tick))
        phase = (tick // period) % len(phases)
        assert int(state.phase) == phase
        assert int(state.elapsed) == tick % period
        np.testing.assert_array_equal(np.asarray(state.value), np.asarray(values[phase]))
        assert bool(pp.drift.switched(before, state)) == (len(phases) > 1 and tick % period == 0)


@SETTINGS
@given(
    low=BOUNDED,
    width=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False).filter(
        _normal_in_float32
    ),
    scale=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False).filter(
        _normal_in_float32
    ),
    start=SMALL,
)
def test_random_walk_stays_inside_its_bounds_and_never_claims_a_switch(
    low: float, width: float, scale: float, start: float
) -> None:
    high = low + width
    process = pp.drift.RandomWalk(low=low, high=high, scale=scale)
    state = process.init(jnp.asarray([start]), jax.random.key(0))
    for tick in range(20):
        before = state
        state = process.step(state, jax.random.key(tick))
        values = np.asarray(state.value)
        assert np.all(values >= np.float32(low))
        assert np.all(values <= np.float32(high))
        assert not bool(pp.drift.switched(before, state))


@SETTINGS
@given(
    low=BOUNDED,
    width=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False).filter(
        _normal_in_float32
    ),
    period=st.integers(min_value=1, max_value=6),
    ticks=st.integers(min_value=0, max_value=30),
)
def test_procedural_redraws_only_on_the_declared_period(
    low: float, width: float, period: int, ticks: int
) -> None:
    high = low + width
    process = pp.drift.Procedural(low=low, high=high, period=period)
    state = process.init(jnp.asarray([low]), jax.random.key(0))
    switches = 0
    for tick in range(1, ticks + 1):
        before = state
        state = process.step(state, jax.random.key(tick))
        switches += bool(pp.drift.switched(before, state))
        values = np.asarray(state.value)
        assert np.all(values >= np.float32(low))
        assert np.all(values <= np.float32(high))
    assert switches == ticks // period
    assert int(state.phase) == ticks // period
    assert int(state.elapsed) == ticks % period


@SETTINGS
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=5), min_size=2, max_size=4),
    ticks=st.integers(min_value=0, max_value=25),
)
def test_sequence_boundaries_are_the_cumulative_tick_budgets(
    lengths: list[int], ticks: int
) -> None:
    processes = [pp.drift.Cyclic([[float(index)]], 10_000) for index in range(len(lengths))]
    process = pp.drift.Sequence(list(zip(processes, lengths, strict=True)))
    state = process.init(jnp.zeros(1, dtype=jnp.float32), jax.random.key(0))
    boundaries = np.cumsum(lengths[:-1]).tolist()
    for _ in range(ticks):
        state = process.step(state, jax.random.key(0))
    expected_segment = sum(1 for boundary in boundaries if ticks >= boundary)
    np.testing.assert_array_equal(
        np.asarray(state.value), np.asarray([expected_segment], dtype=np.float32)
    )
    assert int(state.phase) == len([b for b in boundaries if ticks >= b])


@SETTINGS
@given(values=st.lists(FLOAT, min_size=1, max_size=200))
def test_streaming_metrics_match_a_float64_reference(values: list[float]) -> None:
    metrics = pp.metrics.init(num_agents=1, window_size=7)
    for value in values:
        metrics = pp.metrics.update(metrics, jnp.asarray([value]))
    expected = float(np.sum(np.asarray(values, dtype=np.float64)))
    assert metrics.ticks.to_int() == len(values)
    assert float(metrics.total[0]) == pytest.approx(expected, rel=2e-6, abs=0.5)
    tail = np.asarray(values[-7:], dtype=np.float64)
    assert float(metrics.window_mean[0]) == pytest.approx(float(tail.mean()), rel=1e-5, abs=0.5)
    # Welford's variance against a float64 reference on the float32 inputs it saw.
    seen = np.asarray(values, dtype=np.float32).astype(np.float64)
    variance = float(np.var(seen, ddof=1)) if seen.size > 1 else 0.0
    scale = float(np.max(np.abs(seen))) ** 2
    assert float(metrics.reward_variance[0]) == pytest.approx(
        variance, rel=1e-3, abs=1e-5 * scale + 1e-6
    )
    assert bool(metrics.healthy)


@SETTINGS
@given(values=st.lists(FLOAT, min_size=1, max_size=60))
def test_interquartile_mean_matches_its_definition(values: list[float]) -> None:
    data = np.sort(np.asarray(values, dtype=np.float32))
    k = data.size // 4
    expected = float(np.mean(data[k : data.size - k].astype(np.float64)))
    actual = float(pp.reporting.interquartile_mean(np.asarray(values)))
    # JAX sums in float32; cancellation can leave an absolute error that is
    # large next to a near-zero mean even though the result is correct for the
    # declared dtype. Relative accuracy still has to hold for large means.
    assert actual == pytest.approx(expected, rel=1e-6, abs=0.5)


@SETTINGS
@given(
    runs=st.integers(min_value=1, max_value=8),
    tasks=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_bootstrap_intervals_are_reproducible_and_ordered(runs: int, tasks: int, seed: int) -> None:
    scores = np.random.default_rng(seed).normal(size=(runs, tasks)).astype(np.float32)
    first = pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(9), num_bootstrap=32)
    second = pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(9), num_bootstrap=32)
    for field in ("point", "low", "high"):
        np.testing.assert_array_equal(
            np.asarray(getattr(first, field)), np.asarray(getattr(second, field))
        )
    assert float(first.low) <= float(first.high)


@SETTINGS
@given(
    rewards=st.lists(SMALL, min_size=1, max_size=40),
    labels=st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=40),
)
def test_protocol_results_follow_labels_and_stay_in_range(
    rewards: list[float], labels: list[str]
) -> None:
    count = min(len(rewards), len(labels))
    rewards, labels = rewards[:count], labels[:count]
    trace = pp.protocol.ProtocolTrace.from_arrays(rewards, labels)
    result = pp.protocol.report(trace)
    expected_boundaries = [
        [index, labels[index - 1], labels[index]]
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]
    assert [list(boundary) for boundary in pp.protocol.shift_boundaries(trace)] == (
        expected_boundaries
    )
    for label in set(labels):
        positions = [index for index, value in enumerate(labels) if value == label]
        expected = float(np.mean(np.asarray([rewards[i] for i in positions], dtype=np.float64)))
        assert pp.protocol.per_configuration_reward(trace, label) == pytest.approx(
            expected, abs=1e-6
        )
        stability = pp.protocol.stability(trace, label)["pooled"]
        assert stability is None or stability >= 0.0
    for shift in result["shifts"]:
        step, target = shift["shift_step"], shift["to"]
        end = next(
            (index for index in range(step, len(labels)) if labels[index] != target),
            len(labels),
        )
        speed = shift["adaptation_speed"]
        assert (speed is None) if end == step else (0 <= speed <= end - step)
    assert all(entry["performance_gap"] is None for entry in result["per_regime"].values())


@SETTINGS
@given(
    first=st.lists(SMALL, min_size=1, max_size=8),
    second=st.lists(SMALL, min_size=1, max_size=8),
)
def test_protocol_forgetting_changes_sign_when_encounters_swap(
    first: list[float], second: list[float]
) -> None:
    labels = ["a"] * len(first) + ["b"] + ["a2"] * len(second)
    origins = [None] * len(first) + [None] + ["a"] * len(second)
    forward = pp.protocol.ProtocolTrace.from_arrays(first + [0.0] + second, labels, origins=origins)
    backward = pp.protocol.ProtocolTrace.from_arrays(
        second + [0.0] + first,
        ["a"] * len(second) + ["b"] + ["a2"] * len(first),
        origins=[None] * len(second) + [None] + ["a"] * len(first),
    )
    expected = float(np.mean(np.asarray(first, dtype=np.float64))) - float(
        np.mean(np.asarray(second, dtype=np.float64))
    )
    assert pp.protocol.within_trace_forgetting(forward, "a") == pytest.approx(expected, abs=1e-6)
    assert pp.protocol.within_trace_forgetting(backward, "a") == pytest.approx(-expected, abs=1e-6)


def _same_tree(left: object, right: object) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(
        cast(list[jax.Array], jax.tree.leaves(left)),
        cast(list[jax.Array], jax.tree.leaves(right)),
        strict=True,
    ):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@settings(max_examples=4, deadline=None)
@given(
    first=st.integers(min_value=1, max_value=4),
    second=st.integers(min_value=1, max_value=4),
)
def test_chunked_runs_equal_one_uninterrupted_run(first: int, second: int) -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 3))
    initial = pp.runtime.init_lifetime(env, seed=7, window_size=4)
    split = pp.runtime.Runner(env, pp.policies.random, ticks=first)(initial)
    split = pp.runtime.Runner(env, pp.policies.random, ticks=second)(split)
    complete = pp.runtime.Runner(env, pp.policies.random, ticks=first + second)(initial)
    _same_tree(split, complete)
