"""Drift combinators: Sequence and Sum, checked against hand-computed schedules."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp


def trace(process: pp.drift.Drift, initial: float, ticks: int) -> tuple[list[float], list[int]]:
    """Values and switch counters after each of ``ticks`` steps."""
    state = process.init(jnp.asarray([initial], dtype=jnp.float32), jax.random.key(0))
    values: list[float] = []
    phases: list[int] = []
    for tick in range(ticks):
        state = process.step(state, jax.random.key(tick + 1))
        values.append(float(state.value[0]))
        phases.append(int(state.phase))
    return values, phases


def test_sequence_runs_segments_in_order_and_then_holds_the_last() -> None:
    # Stationary at 5.0 for 3 ticks, then a period-2 cycle between 1 and 2.
    process = pp.drift.Sequence(
        [(pp.drift.Stationary(), 3), (pp.drift.Cyclic([[1.0], [2.0]], 2), 4)]
    )
    values, phases = trace(process, 5.0, 9)
    # Ticks 1-2 stay at 5.0; tick 3 spends the stationary budget and enters the
    # cycle, which starts at its first value 1.0. The cycle then switches every
    # 2 ticks (ticks 5, 7, 9) and, being the last segment, keeps running past
    # its 4-tick budget.
    assert values == [5.0, 5.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0]
    # One switch on entering the cycle, then one per cycle switch.
    assert phases == [0, 0, 1, 1, 2, 2, 3, 3, 4]


def test_sequence_carries_the_value_into_the_next_process() -> None:
    walk = pp.drift.RandomWalk(low=0.0, high=10.0, scale=0.0)
    process = pp.drift.Sequence([(pp.drift.Stationary(), 2), (walk, 3)])
    values, _ = trace(process, 7.0, 4)
    # A zero-scale walk entered at 7.0 holds 7.0: the parameter is continuous
    # across the segment boundary instead of jumping to the walk's own start.
    assert values == [7.0, 7.0, 7.0, 7.0]


def test_sequence_continuity_depends_on_the_process() -> None:
    # Cyclic starts at its own first phase, not at the value it is handed, so
    # the parameter jumps at the boundary (2.0 -> 7.0 here).
    process = pp.drift.Sequence(
        [(pp.drift.Stationary(), 2), (pp.drift.Cyclic([[7.0], [8.0]], 5), 3)]
    )
    values, phases = trace(process, 2.0, 3)
    assert values == [2.0, 7.0, 7.0]
    assert phases == [0, 1, 1]


def test_sequence_accepts_segments_with_different_state_layouts() -> None:
    # A Sum segment (with sub-process parts) after a Cyclic one: "clean cycles,
    # then noisy cycles". Stacking every segment's state used to reject this.
    clean = pp.drift.Cyclic([[1.0], [2.0]], 2)
    noisy = pp.drift.Sum(
        [pp.drift.Cyclic([[1.0], [2.0]], 2), pp.drift.RandomWalk(low=-0.5, high=0.5, scale=0.1)]
    )
    process = pp.drift.Sequence([(clean, 4), (noisy, 4)])
    values, phases = trace(process, 0.0, 8)
    assert values[:3] == [1.0, 2.0, 2.0]
    assert values[3] == 1.0  # tick 4 enters the noisy segment at its base value
    assert any(abs(value - round(value)) > 1e-6 for value in values[4:])  # noise on top
    assert phases[3] == phases[2] + 1
    # It also runs compiled inside an environment and checkpoints its layout.
    env = pp.make(
        "ContinualGrid-v0",
        size=5,
        drift=pp.drift.Sequence(
            [
                (pp.drift.Cyclic([[1.0, 1.0], [3.0, 3.0]], 2), 4),
                (
                    pp.drift.Sum(
                        [pp.drift.Cyclic([[0.0, 4.0]], 1), pp.drift.RandomWalk(low=-1, high=1)]
                    ),
                    4,
                ),
            ]
        ),
    )
    carry = pp.runtime.Runner(env, pp.policies.random, ticks=12)(
        pp.runtime.init_lifetime(env, seed=0, window_size=4)
    )
    assert carry.state.t.to_int() == 12


def test_sequence_stacks_same_layout_segments() -> None:
    # Twenty Cyclic segments share one state layout, so they are stored as one
    # stacked state plus the segment counter rather than twenty states.
    tasks = [(pp.drift.Cyclic([[float(i)], [float(i) + 0.5]], 3), 10) for i in range(20)]
    process = pp.drift.Sequence(tasks)
    state = process.init(jnp.asarray([0.0]), jax.random.key(0))
    assert len(state.parts) == 2
    assert isinstance(state.parts[0], pp.drift.DriftState)
    assert state.parts[0].value.shape == (20, 1)
    assert isinstance(state.parts[1], pp.drift.SegmentCounter)
    values, _ = trace(process, 0.0, 25)
    assert values[9] == 1.0
    assert values[19] == 2.0


def test_sum_adds_offsets_to_a_base_schedule() -> None:
    process = pp.drift.Sum(
        [pp.drift.Cyclic([[1.0], [3.0]], 2), pp.drift.Cyclic([[0.0], [10.0]], 3)]
    )
    values, phases = trace(process, 0.0, 6)
    # Base switches at ticks 2, 4, 6; offset at ticks 3, 6.
    # Base:   1 3 3 1 1 3      offset: 0 0 10 10 10 0
    assert values == [1.0, 3.0, 13.0, 11.0, 11.0, 3.0]
    # Tick 6 has both switching at once and counts once.
    assert phases == [0, 1, 2, 3, 3, 4]


def test_combinators_validate_and_record_their_configuration() -> None:
    with pytest.raises(ValueError, match=r"Sequence needs at least one \(process, ticks\)"):
        pp.drift.Sequence([])
    with pytest.raises(ValueError, match=r"Sequence segment ticks must be integers in"):
        pp.drift.Sequence([(pp.drift.Stationary(), 0)])
    with pytest.raises(ValueError, match=r"Sum needs a base process and at least one"):
        pp.drift.Sum([pp.drift.Stationary()])
    sequence = pp.drift.Sequence([(pp.drift.Stationary(), 3), (pp.drift.Cyclic([[1.0]], 2), 4)])
    assert sequence.configuration["type"] == "sequence"
    assert (
        pp.drift.Sum([pp.drift.Stationary(), pp.drift.Stationary()]).configuration["type"] == "sum"
    )


def test_combinators_drive_an_environment_and_declare_switches() -> None:
    process = pp.drift.Sequence(
        [(pp.drift.Stationary(), 4), (pp.drift.Cyclic([(0.0, 0.0), (2.0, 2.0)], 3), 9)]
    )
    env = pp.make("ContinualGrid-v0", size=3, drift=process)
    carry = pp.runtime.Runner(env, pp.policies.greedy, ticks=16)(
        pp.runtime.init_lifetime(env, seed=0, window_size=4)
    )
    # Switches: entering the cycle at tick 4, then cycle switches at 7, 10, 13, 16.
    assert int(carry.state.drift["goal"].phase) == 5
    assert bool(carry.metrics.has_switch)
    assert bool(carry.metrics.healthy)
    np.testing.assert_array_equal(
        np.asarray(env.privileged(carry.state)["goal_parameters"]), [0.0, 0.0]
    )


@pytest.mark.parametrize("combinator", ["sequence", "sum"])
def test_combinators_refuse_agent_driven_processes_up_front(combinator: str) -> None:
    accumulate = pp.drift.Accumulate(gain=0.1, decay=0.99, low=0.5, high=5.0)
    if combinator == "sequence":
        with pytest.raises(ValueError, match="Accumulate"):
            pp.drift.Sequence([(accumulate, 100), (pp.drift.Stationary(), 1)])
    else:
        with pytest.raises(ValueError, match="Accumulate"):
            pp.drift.Sum([accumulate, pp.drift.Stationary()])


@pytest.mark.parametrize("period", [2**31, 2**32 - 1])
def test_cyclic_and_procedural_step_with_periods_above_int32(period: int) -> None:
    key = jax.random.key(0)
    for process in (
        pp.drift.Cyclic([0.0, 1.0], period),
        pp.drift.Procedural(low=0.0, high=1.0, period=period),
    ):
        state = process.init(jnp.zeros(()), key)
        state = process.step(state, key)
        assert int(state.elapsed) == 1
        assert int(state.phase) == 0


def test_schedule_start_refuses_values_of_the_wrong_parameter_shape() -> None:
    process = pp.drift.Schedule([(0, 0), (1, 1)], 2)
    process.start(pp.drift.schedule([(2, 2), (0, 1)], 3))
    with pytest.raises(ValueError, match=r"must have shape \(2,\) per phase, got \(\)"):
        process.start(pp.drift.schedule([0.0, 3.0], 2))
    with pytest.raises(ValueError, match=r"must have shape \(2,\) per phase, got \(3,\)"):
        process.start(pp.drift.schedule([(0, 0, 0), (1, 1, 1)], 2))
