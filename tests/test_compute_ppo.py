"""Timed PPO/IPPO must discount real elapsed ticks and checkpoint pending rollouts."""

# JAX transformations are partially annotated.
# pyright: reportUnknownMemberType=false

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.agents import IPPO, PPO, ComputePPORunner, PPOConfig
from perpetua.agents.compute_ppo import TimedTrainingState
from perpetua.agents.ppo import compute_gae
from perpetua.compute import TimingTrace
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def test_gae_uses_world_tick_discounts_for_variable_decision_intervals() -> None:
    advantages, targets = compute_gae(
        rewards=jnp.asarray([[1.0], [2.0]]),
        values=jnp.asarray([[0.5], [0.75]]),
        last_value=jnp.asarray([1.0]),
        gamma=0.9,
        gae_lambda=0.8,
        elapsed_ticks=jnp.asarray([[2], [3]], dtype=jnp.uint32),
    )
    np.testing.assert_allclose(advantages, [[2.1334136], [1.979]], rtol=1e-6)
    np.testing.assert_allclose(targets, [[2.6334136], [2.729]], rtol=1e-6)


@pytest.mark.parametrize("independent", [False, True])
def test_optimizer_is_charged_at_next_decision_and_partial_buffers_resume(
    tmp_path: Path,
    independent: bool,
) -> None:
    env = pp.make("CooperativeGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 3))
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    agent = IPPO(action_size=5, config=config) if independent else PPO(action_size=5, config=config)
    runner = ComputePPORunner(env, agent, decisions=2, quantum_ns=100)
    first = TimingTrace.from_ticks([1, 3], quantum_ns=100)
    second = TimingTrace.from_ticks([2, 2], quantum_ns=100)
    initial = runner.init(seed=0, window_size=4)
    partial = runner(initial, first)
    assert int(partial.filled) == 2
    assert partial.training.updates.to_int() == 0
    path = tmp_path / "timed-training"
    pp.checkpoint.save(path, partial, metadata=runner.metadata)
    restored = pp.checkpoint.restore(
        path, runner.init(seed=999, window_size=4), expected_metadata=runner.metadata
    )
    full = runner(partial, second)
    assert int(full.filled) == 4
    assert full.training.updates.to_int() == 0
    expected = runner(full, first)
    actual = runner(runner(restored, second), first)
    assert_same(expected, actual)
    assert expected.training.updates.to_int() == 1
    assert expected.training.lifetime.state.t.to_int() == 12
    assert expected.timing.decisions.to_int() == 6
    assert int(expected.filled) == 2
    assert np.all(np.asarray(expected.training.agent.gradient_steps.lo) == 1)
    assert bool(expected.training.healthy)


def test_measured_training_includes_optimization_and_replays_all_state() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    agent = IPPO(
        action_size=5, config=PPOConfig(rollout_steps=2, epochs=1, minibatches=1, hidden_size=8)
    )
    runner = ComputePPORunner(env, agent, decisions=3, quantum_ns=1_000_000_000)
    initial = runner.init(seed=0)
    state, trace = runner.measure(initial)
    assert state.training.updates.to_int() == 1
    assert np.all(np.asarray(state.training.agent.gradient_steps.lo) == 1)
    assert_same(state, runner(initial, trace))
    assert state.training.lifetime.metrics.ticks.to_int() == int(np.asarray(trace.ticks).sum())
    assert state.timing.compute_ns.to_int() > 0


def test_timed_ippo_worlds_vmap_and_cross_large_clocks() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    agent = IPPO(
        action_size=5, config=PPOConfig(rollout_steps=2, epochs=1, minibatches=1, hidden_size=8)
    )
    runner = ComputePPORunner(env, agent, decisions=2, quantum_ns=100)
    trace = TimingTrace.from_ticks([2, 3], quantum_ns=100)

    def initialize(seed: jax.Array) -> TimedTrainingState[DriftEnvState[GridState]]:
        return runner.init(seed=seed, window_size=4)

    def replay(
        state: TimedTrainingState[DriftEnvState[GridState]],
    ) -> TimedTrainingState[DriftEnvState[GridState]]:
        return runner(state, trace)

    states = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(replay)(states)
    np.testing.assert_array_equal(result.training.lifetime.state.t.lo, [5, 5])
    state = runner.init(seed=0, window_size=4)
    clock = pp.Clock.from_int(10_000_000_000)
    world = state.training.lifetime.state._replace(t=clock)
    obs, info = env.observe(world)
    life = state.training.lifetime._replace(
        state=world,
        obs=obs,
        info=info,
        metrics=state.training.lifetime.metrics._replace(ticks=clock),
    )
    state = state._replace(
        training=state.training._replace(lifetime=life),
        timing=state.timing._replace(decisions=clock, compute_ns=clock),
    )
    result = runner(state, trace)
    assert result.training.lifetime.state.t.to_int() == 10_000_000_005
    assert result.timing.decisions.to_int() == 10_000_000_002
    assert result.timing.compute_ns.to_int() == 10_000_000_500


def test_timed_host_ppo_uses_numeric_cpu_snapshots_and_public_work_only(tmp_path: Path) -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    first = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    second = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    config = PPOConfig(rollout_steps=2, epochs=1, minibatches=1, hidden_size=8)
    try:
        a = ComputePPORunner(first, PPO(action_size=2, config=config), decisions=2, quantum_ns=100)
        b = ComputePPORunner(second, PPO(action_size=2, config=config), decisions=2, quantum_ns=100)
        trace = TimingTrace.from_ticks([2, 2], quantum_ns=100)
        state = a(a.init(seed=0), trace)
        path = tmp_path / "host-compute"
        pp.checkpoint.save(path, state, metadata=a.metadata)
        restored = pp.checkpoint.restore(path, b.init(seed=999), expected_metadata=b.metadata)
        assert_same(a(state, trace), b(restored, trace))
    finally:
        first.close()
        second.close()


def test_measured_replay_preserves_optimizer_kernel_partition_after_full_buffer() -> None:
    env = pp.make("CooperativeGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 7))
    agent = IPPO(
        action_size=5, config=PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=16)
    )
    runner = ComputePPORunner(env, agent, decisions=4, quantum_ns=1_000_000_000)
    before = runner.init(seed=0, window_size=16)
    before, _ = runner.measure(before)
    before, _ = runner.measure(before)
    assert int(before.filled) == 8
    measured, trace = runner.measure(before)
    assert measured.training.updates.to_int() == 1
    assert_same(measured, runner(before, trace))
