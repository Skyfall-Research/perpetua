"""Schedules as data: each vmapped lifetime follows its own drift schedule."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownVariableType=false

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner
from perpetua.agents.ppo import PPOHyperparameters

GOALS = [(float(i % 4), float(i // 4 % 4)) for i in range(20)]


def grid(process: Any) -> Any:
    return pp.make("ContinualGrid-v0", size=4, drift=process)


def assigned(env: Any, seed: jax.Array, data: pp.drift.ScheduleData) -> Any:
    return pp.runtime.assign_schedule(env, pp.runtime.init_lifetime(env, seed=seed), data)


def bank() -> pp.drift.ScheduleData:
    return pp.drift.stack_schedules(
        [
            pp.drift.schedule(GOALS, 5, offset=15),  # starts in phase 3
            pp.drift.schedule(GOALS, 5, offset=85),  # starts in phase 17
            pp.drift.schedule(GOALS[:3], 2, phases=20),  # three phases, padded
            pp.drift.schedule(GOALS, 7),
        ]
    )


def test_each_lifetime_in_a_batch_follows_its_own_schedule() -> None:
    env = grid(pp.drift.Schedule(GOALS, 5))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=10)
    batch = jax.jit(jax.vmap(lambda s, d: assigned(env, s, d)))(jnp.arange(4), bank())
    np.testing.assert_array_equal(np.asarray(batch.state.drift["goal"].phase), [3, 17, 0, 0])
    batch = jax.jit(jax.vmap(runner))(batch)
    np.testing.assert_array_equal(np.asarray(batch.state.drift["goal"].phase), [5, 19, 2, 1])
    # Each batched lifetime equals the same schedule run on its own.
    for index in range(4):
        data = cast(pp.drift.ScheduleData, jax.tree.map(lambda leaf, i=index: leaf[i], bank()))
        alone = runner(assigned(env, jnp.asarray(index), data))
        np.testing.assert_array_equal(np.asarray(alone.obs), np.asarray(batch.obs[index]))
        np.testing.assert_array_equal(
            np.asarray(alone.metrics.total), np.asarray(batch.metrics.total[index])
        )


def test_an_unassigned_schedule_matches_cyclic_exactly() -> None:
    def run(process: Any) -> Any:
        env = grid(process)
        return pp.runtime.Runner(env, pp.policies.random, ticks=40)(
            pp.runtime.init_lifetime(env, seed=2)
        )

    schedule, cyclic = run(pp.drift.Schedule(GOALS[:4], 6)), run(pp.drift.Cyclic(GOALS[:4], 6))
    np.testing.assert_array_equal(np.asarray(schedule.obs), np.asarray(cyclic.obs))
    np.testing.assert_array_equal(
        np.asarray(schedule.metrics.total), np.asarray(cyclic.metrics.total)
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.state.drift["goal"].value), np.asarray(cyclic.state.drift["goal"].value)
    )


def test_new_schedules_reuse_the_compiled_programs() -> None:
    env = grid(pp.drift.Schedule(GOALS, 5))
    init = jax.jit(jax.vmap(lambda s, d: assigned(env, s, d)))
    run = jax.jit(jax.vmap(pp.runtime.Runner(env, pp.policies.random, ticks=8)))
    data, seeds = bank(), jnp.arange(4)
    jax.block_until_ready(run(init(seeds, data)))
    other = data._replace(offset=data.offset + jnp.uint32(3), period=data.period + jnp.uint32(1))
    with pp.measurement.count_compilations() as compiles:
        jax.block_until_ready(run(init(seeds, other)))
    assert compiles.count == 0


def test_phase_changes_are_declared_switches() -> None:
    env = grid(pp.drift.Schedule(GOALS[:3], 3))
    lifetime = assigned(env, jnp.asarray(0), pp.drift.schedule(GOALS[:3], 3, offset=1))
    switches = []
    for _ in range(7):
        lifetime = pp.runtime.tick(env, pp.policies.random, lifetime)
        switches.append(bool(lifetime.state.drift_changed))
    # Offset 1 into a period of 3: the first switch is two ticks in, then every three.
    assert switches == [False, True, False, False, True, False, False]


def test_assigned_schedules_survive_a_checkpoint(tmp_path: Path) -> None:
    env = grid(pp.drift.Schedule(GOALS, 5))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=6)
    lifetime = runner(assigned(env, jnp.asarray(1), pp.drift.schedule(GOALS, 2, offset=31)))
    pp.checkpoint.save(tmp_path / "schedule", lifetime, metadata=runner.metadata)
    restored = pp.checkpoint.restore(
        tmp_path / "schedule",
        pp.runtime.init_lifetime(env, seed=9),
        expected_metadata=runner.metadata,
    )
    np.testing.assert_array_equal(
        np.asarray(runner(restored).obs), np.asarray(runner(lifetime).obs)
    )


def test_seeds_schedules_and_hyperparameters_vmap_in_one_call() -> None:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=4, drift=pp.drift.Schedule(GOALS[:4], 8))
    config = PPOConfig(rollout_steps=16)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    base = PPOHyperparameters.from_config(config)
    rates = jnp.asarray([1e-4, 3e-4, 1e-3, 3e-3], dtype=jnp.float32)
    schedules = pp.drift.stack_schedules(
        [pp.drift.schedule(GOALS[:4], 8, offset=offset) for offset in (0, 8, 16, 24)]
    )

    def init(seed: jax.Array, rate: jax.Array, data: pp.drift.ScheduleData) -> Any:
        state = trainer.init(seed=seed, hyperparameters=base._replace(learning_rate=rate))
        return state._replace(lifetime=pp.runtime.assign_schedule(env, state.lifetime, data))

    states = jax.jit(jax.vmap(init))(jnp.arange(4), rates, schedules)
    np.testing.assert_array_equal(
        np.asarray(states.lifetime.state.drift["goal"].phase), [0, 1, 2, 3]
    )
    states = jax.vmap(trainer)(states)
    assert bool(jnp.all(states.healthy))
    np.testing.assert_allclose(np.asarray(states.agent.hyperparameters.learning_rate), rates)


def test_invalid_schedules_are_refused() -> None:
    with pytest.raises(ValueError, match=r"Schedule values must contain finite, equally"):
        pp.drift.schedule([], 3)
    with pytest.raises(ValueError, match=r"Schedule period must be an integer in"):
        pp.drift.schedule(GOALS, 0)
    with pytest.raises(ValueError, match=r"phases must be an integer at least the number"):
        pp.drift.schedule(GOALS, 3, phases=2)
    env = grid(pp.drift.Schedule(GOALS, 5))
    with pytest.raises(ValueError, match="phase rows"):
        assigned(env, jnp.asarray(0), pp.drift.schedule(GOALS[:3], 5))
    cyclic = grid(pp.drift.Cyclic(GOALS, 5))
    with pytest.raises(ValueError, match="Schedule"):
        assigned(cyclic, jnp.asarray(0), pp.drift.schedule(GOALS, 5))
