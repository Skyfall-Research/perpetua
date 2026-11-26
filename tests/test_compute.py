"""Compute latency must advance real dynamics, clocks and metrics without backdating actions."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import ast
import re
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.compute import (
    ComputeRunner,
    MeasuredTimingTrace,
    TimedLifetime,
    TimingTrace,
    advance,
    record_trace,
    tick_count,
)
from perpetua.core import Info
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def change_command(obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
    del key
    return jnp.full(obs.shape[0], jnp.where(info["t"].lo == 0, 4, 2), dtype=jnp.int32)


def test_integer_latency_rounding_and_clock_addition_are_exact() -> None:
    assert [tick_count(ns, 100) for ns in [0, 1, 100, 101, 200, 201]] == [1, 1, 1, 2, 2, 3]
    with pytest.raises(ValueError, match=r"compute_ns must be an integer in"):
        tick_count(-1, 100)
    with pytest.raises(ValueError, match=r"quantum_ns must be a positive"):
        tick_count(1, 0)
    with pytest.raises(OverflowError):
        tick_count(2**32, 1)
    add = cast(Callable[[pp.Clock, pp.Clock], pp.Clock], jax.jit(pp.Clock.add))
    clock = add(pp.Clock.from_int(2**32 - 1), pp.Clock.from_int(10_000_000_000))
    assert clock.to_int() == 2**32 - 1 + 10_000_000_000


def test_previous_command_is_held_and_new_command_arrives_on_final_tick() -> None:
    env = pp.make("ContinualGrid-v0", size=5)
    runner = ComputeRunner(env, change_command, decisions=1, quantum_ns=100)
    state = runner.init(seed=0)
    world = state.lifetime.state._replace(
        env=state.lifetime.state.env._replace(positions=jnp.asarray([[0, 0]], dtype=jnp.int32))
    )
    obs, info = env.observe(world)
    state = state._replace(lifetime=state.lifetime._replace(state=world, obs=obs, info=info))
    state = runner(state, TimingTrace.from_ticks([3], quantum_ns=100))
    np.testing.assert_array_equal(state.lifetime.state.env.positions, [[0, 1]])
    assert state.lifetime.state.t.to_int() == state.lifetime.metrics.ticks.to_int() == 3
    state = runner(state, TimingTrace.from_ticks([3], quantum_ns=100))
    np.testing.assert_array_equal(state.lifetime.state.env.positions, [[1, 3]])
    assert state.timing.decisions.to_int() == 2
    assert state.timing.compute_ns.to_int() == 600


def test_catchup_uses_every_world_key_drift_tick_and_reward() -> None:
    env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], 1))
    initial = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    actual = advance(env, initial, jnp.asarray([0]), jnp.asarray([4]), jnp.uint32(3), gamma=1.0)
    expected = initial
    for action in [0, 0, 4]:
        env_key, step_key = jax.random.split(expected.env_key)
        world, obs, reward, info = env.step(expected.state, jnp.asarray([action]), step_key)
        expected = expected._replace(
            state=world,
            obs=obs,
            info=info,
            env_key=env_key,
            metrics=pp.metrics.update(
                expected.metrics, reward, switch=pp.runtime.drift_switch(world)
            ),
        )
    assert_same(actual.lifetime, expected)
    assert actual.lifetime.state.t.to_int() == 3
    assert int(actual.lifetime.state.drift["goal"].phase) == 1


def test_discounted_interval_reward_does_not_replace_world_tick_metrics() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    initial = pp.runtime.init_lifetime(env, seed=0)
    world = initial.state._replace(
        env=initial.state.env._replace(positions=jnp.asarray([[3, 2]], dtype=jnp.int32))
    )
    obs, info = env.observe(world)
    initial = initial._replace(state=world, obs=obs, info=info)
    result = advance(env, initial, jnp.asarray([0]), jnp.asarray([4]), jnp.uint32(3), gamma=0.5)
    np.testing.assert_array_equal(result.reward, [0.25])
    np.testing.assert_array_equal(result.lifetime.metrics.total, [1.0])
    assert result.lifetime.metrics.ticks.to_int() == 3


def test_stale_commands_are_repaired_against_each_new_observation() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    life = pp.runtime.init_lifetime(env)
    world = life.state._replace(
        env=life.state.env._replace(positions=jnp.asarray([[1, 1]], dtype=jnp.int32))
    )
    obs, info = env.observe(world)
    life = life._replace(state=world, obs=obs, info=info)
    result = advance(env, life, jnp.asarray([1]), jnp.asarray([1]), jnp.uint32(3))
    np.testing.assert_array_equal(result.lifetime.state.env.positions, [[0, 1]])
    assert result.lifetime.state.t.to_int() == 3
    assert bool(result.healthy)


def test_invalid_masks_do_not_advance_physics_and_bad_discount_is_refused() -> None:
    env = pp.make("ContinualGrid-v0")
    life = pp.runtime.init_lifetime(env)
    info = life.info.copy()
    info["action_mask"] = jnp.zeros_like(info["action_mask"])
    invalid = life._replace(info=info)
    result = advance(env, invalid, jnp.asarray([0]), jnp.asarray([0]), jnp.uint32(2))
    assert not bool(result.healthy)
    assert result.lifetime.state.t.to_int() == 0
    for gamma in [-1.0, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="gamma"):
            advance(env, life, jnp.asarray([0]), jnp.asarray([0]), jnp.uint32(1), gamma=gamma)


def test_advance_accepts_python_and_numpy_tick_counts() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    life = pp.runtime.init_lifetime(env)
    held, action = jnp.asarray([0]), jnp.asarray([4])
    expected = advance(env, life, held, action, jnp.uint32(3))
    counts: list[Any] = [3, np.int64(3), tick_count(2_500_000, 1_000_000)]
    for ticks in counts:
        result = advance(env, life, held, action, ticks)
        assert result.lifetime.state.t.to_int() == int(ticks)
    np.testing.assert_array_equal(
        advance(env, life, held, action, 3).lifetime.state.env.positions,
        expected.lifetime.state.env.positions,
    )
    with pytest.raises(ValueError, match=r"scalar uint32 count.*Got float 3\.0"):
        advance(env, life, held, action, 3.0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"got shape \(2,\) and dtype uint32"):
        advance(env, life, held, action, jnp.asarray([1, 2], dtype=jnp.uint32))


def test_a_trace_of_the_wrong_length_names_both_lengths() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    runner = ComputeRunner(env, pp.policies.random, decisions=3, quantum_ns=100)
    state = runner.init(seed=0)
    short = TimingTrace.from_ticks([1, 2], quantum_ns=100)
    with pytest.raises(ValueError, match=r"field ticks .*shape \(3,\).*got shape \(2,\)"):
        runner(state, short)


def test_a_trace_built_for_another_quantum_is_refused() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    runner = ComputeRunner(env, pp.policies.random, decisions=3, quantum_ns=1000)
    state = runner.init(seed=0)
    other = TimingTrace.from_ticks([3, 1, 2], quantum_ns=5)
    match = r"decision 0 records 3 ticks for 15 ns, but this runner's quantum_ns=1000 gives 1"
    with pytest.raises(ValueError, match=match):
        runner(state, other)
    with pytest.raises(ValueError, match=match):
        runner(state, MeasuredTimingTrace(other.ticks, other.compute_ns))
    done = runner(state, TimingTrace.from_ticks([3, 1, 2], quantum_ns=1000))
    assert done.timing.compute_ns.to_int() == 6000
    # A measured cost rounds up to whole ticks, and no cost is still one tick.
    measured = record_trace([0, 1, 2500], quantum_ns=1000)
    np.testing.assert_array_equal(np.asarray(measured.ticks), [1, 1, 3])
    assert runner(state, measured).lifetime.state.t.to_int() == 5

    from perpetua.agents import PPO, ComputePPORunner

    trainer = ComputePPORunner(env, PPO(action_size=5), decisions=3, quantum_ns=1000)
    with pytest.raises(ValueError, match=match):
        trainer(trainer.init(seed=0), other)


def test_compute_runners_refuse_continuous_actions_when_built() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    with pytest.raises(ValueError, match=r"discrete \(MultiDiscrete\) action space.*Box"):
        ComputeRunner(env, pp.policies.random, decisions=2)
    from perpetua.agents import PPO, ComputePPORunner

    with pytest.raises(ValueError, match=r"ComputePPORunner needs a discrete"):
        ComputePPORunner(env, PPO(action_size=1))


def test_replay_is_chunk_continuous_vmappable_and_fully_checkpointable(tmp_path: Path) -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    runner = ComputeRunner(env, pp.policies.random, decisions=2, quantum_ns=100)
    whole = ComputeRunner(env, pp.policies.random, decisions=4, quantum_ns=100)
    initial = runner.init(seed=0, window_size=4)
    first = TimingTrace.from_ticks([1, 3], quantum_ns=100)
    second = TimingTrace.from_ticks([2, 4], quantum_ns=100)
    expected = runner(runner(initial, first), second)
    actual = whole(initial, TimingTrace.from_ticks([1, 3, 2, 4], quantum_ns=100))
    assert_same(actual, expected)
    path = tmp_path / "compute"
    saved = runner(initial, first)
    pp.checkpoint.save(path, saved, metadata=runner.metadata)
    restored = pp.checkpoint.restore(
        path, runner.init(seed=999, window_size=4), expected_metadata=runner.metadata
    )
    assert_same(runner(restored, second), expected)
    assert expected.lifetime.state.t.to_int() == 10
    assert expected.timing.decisions.to_int() == 4
    assert "oracle" not in expected.lifetime.info

    def initialize(seed: jax.Array) -> TimedLifetime[DriftEnvState[GridState]]:
        return runner.init(seed=seed, window_size=4)

    def run(
        state: TimedLifetime[DriftEnvState[GridState]],
    ) -> TimedLifetime[DriftEnvState[GridState]]:
        return runner(state, first)

    batch = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(run)(batch)
    np.testing.assert_array_equal(result.lifetime.state.t.lo, [4, 4])


def test_measured_trace_replays_without_timers_or_lost_rng() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    runner = ComputeRunner(env, pp.policies.random, decisions=3, quantum_ns=1_000_000_000)
    initial = runner.init(seed=0)
    measured, trace = runner.measure(initial)
    assert_same(measured, runner(initial, trace))
    assert measured.lifetime.state.t.to_int() == int(np.asarray(trace.ticks).sum())
    assert measured.timing.compute_ns.to_int() > 0
    assert trace.compute_ns.hi.shape == trace.compute_ns.lo.shape == (3,)


def test_invalid_schedule_is_refused_or_marks_compiled_carry_unhealthy() -> None:
    for values in [[0], [-1], [1.5]]:
        with pytest.raises((TypeError, ValueError)):
            TimingTrace.from_ticks(values, quantum_ns=100)
    env = pp.make("ContinualGrid-v0")
    runner = ComputeRunner(env, pp.policies.random, decisions=1, quantum_ns=100)
    initial = runner.init()
    trace = TimingTrace.from_ticks([1], quantum_ns=100)._replace(
        ticks=jnp.asarray([0], dtype=jnp.uint32)
    )
    result = runner(initial, trace)
    assert not bool(result.timing.healthy)
    assert result.lifetime.state.t.to_int() == result.timing.decisions.to_int() == 0


def test_cpu_snapshot_backend_can_replay_timed_steps_into_a_fresh_instance() -> None:
    import gymnasium as gym

    first = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    second = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    try:
        a = ComputeRunner(first, pp.policies.random, decisions=2, quantum_ns=100)
        b = ComputeRunner(second, pp.policies.random, decisions=2, quantum_ns=100)
        state = a.init(seed=0)
        b.init(seed=999)
        trace = TimingTrace.from_ticks([2, 4], quantum_ns=100)
        assert_same(a(state, trace), b(state, trace))
    finally:
        first.close()
        second.close()


def test_ci_embedded_python_has_valid_syntax() -> None:
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    blocks = re.findall(r"python - <<'PY'\n(.*?)\n\s*PY", workflow, flags=re.DOTALL)
    assert blocks
    for code in blocks:
        ast.parse(textwrap.dedent(code))


class Benched:
    """Wrap a two-slot world so slot 1 is always inactive with no valid action."""

    def __init__(self, env: Any) -> None:
        self.env = env
        for name in ("num_agents", "observation_space", "action_space", "oracle_fields"):
            setattr(self, name, getattr(env, name))
        self.metadata = env.metadata

    def _bench(self, info: Any) -> Any:
        on = jnp.asarray([True, False])
        return {
            **info,
            "active_mask": info["active_mask"] & on,
            "action_mask": info["action_mask"] & on[:, None],
        }

    def init(self, key: jax.Array) -> tuple[Any, jax.Array, Any]:
        state, obs, info = self.env.init(key)
        return state, obs, self._bench(info)

    def step(
        self, state: Any, action: Any, key: jax.Array
    ) -> tuple[Any, jax.Array, jax.Array, Any]:
        state, obs, reward, info = self.env.step(state, action, key)
        return state, obs, reward, self._bench(info)

    def observe(self, state: Any) -> tuple[jax.Array, Any]:
        obs, info = self.env.observe(state)
        return obs, self._bench(info)

    def privileged(self, state: Any) -> dict[str, jax.Array]:
        return self.env.privileged(state)


def test_an_inactive_slot_without_a_valid_action_does_not_fail_the_world_tick() -> None:
    def build() -> Any:
        return Benched(pp.make("CooperativeGrid-v0"))

    pp.testing.check_env(build)
    env = build()
    life = pp.runtime.init_lifetime(env)
    assert not bool(jnp.any(life.info["action_mask"][1]))
    command = jnp.zeros(2, dtype=jnp.int32)
    result = advance(env, life, command, command, 3)
    assert bool(result.healthy)
    assert result.lifetime.state.t.to_int() == 3
    info = life.info.copy()
    info["action_mask"] = jnp.zeros_like(info["action_mask"])
    stuck = advance(env, life._replace(info=info), command, command, 3)
    assert not bool(stuck.healthy)


def _costs(trace: TimingTrace) -> list[int]:
    hi, lo = np.asarray(trace.compute_ns.hi), np.asarray(trace.compute_ns.lo)
    return [(int(high) << 32) | int(low) for high, low in zip(hi, lo, strict=True)]


def test_numpy_scalars_do_not_wrap_the_recorded_compute_cost() -> None:
    trace = TimingTrace.from_ticks([5], quantum_ns=cast(int, np.uint32(10**9)))
    assert _costs(trace) == [5 * 10**9]
    costs = cast(list[int], [np.uint32(3 * 10**9), np.int32(7)])
    measured = record_trace(costs, quantum_ns=cast(int, np.uint32(10**9)))
    np.testing.assert_array_equal(np.asarray(measured.ticks), [3, 1])
    assert _costs(measured) == [3 * 10**9, 7]
    with pytest.raises(ValueError, match=r"quantum_ns must be a positive"):
        TimingTrace.from_ticks([5], quantum_ns=0)
