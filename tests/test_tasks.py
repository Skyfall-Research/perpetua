"""Task switching: one lifetime over several environments, drifted by task index."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters.gymnax_params import GymnaxParameters
from perpetua.envs.continual_grid import ContinualGrid


def two_tasks(period: int = 4, **kwargs: Any) -> Any:
    """A small grid and Acrobot, alternating every ``period`` ticks."""
    return pp.tasks.switch(
        [ContinualGrid(size=3), GymnaxParameters("Acrobot-v1")],
        pp.drift.Cyclic([0.0, 1.0], period),
        **kwargs,
    )


def test_task_switch_pads_observations_and_masks_per_task() -> None:
    env = two_tasks(period=2)
    state, obs, info = env.init(jax.random.key(0))
    # Common observation shape is (1, 6); the grid's four features are padded.
    assert obs.shape == (1, 6)
    np.testing.assert_array_equal(np.asarray(obs[:, 4:]), np.zeros((1, 2)))
    assert info["action_mask"].shape == (1, 5)
    state, obs, _, info = env.step(state, jnp.asarray([0]), jax.random.key(0))
    state, obs, _, info = env.step(state, jnp.asarray([0]), jax.random.key(1))
    # Acrobot has three actions, so the two extra mask entries are disabled.
    assert int(state.drift["task"].value) == 1
    np.testing.assert_array_equal(
        np.asarray(info["action_mask"]), [[True, True, True, False, False]]
    )
    assert obs.shape == (1, 6)


def test_task_switch_never_recompiles_across_switches() -> None:
    env = two_tasks(period=2)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    carry = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    with pp.measurement.count_compilations() as cold:
        carry = runner(carry)
    assert cold.count >= 1
    with pp.measurement.count_compilations() as warm:
        for _ in range(4):
            carry = runner(carry)
    assert warm.count == 0
    assert carry.state.t.to_int() == 40
    assert bool(carry.metrics.has_switch)


def test_inactive_tasks_resume_unchanged() -> None:
    env = two_tasks(period=2)
    state, _, _ = env.init(jax.random.key(0))
    # Two ticks on task 0, then four ticks on task 1.
    for tick in range(2):
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(tick))
    before = jax.tree.leaves(state.env.states[0])
    for tick in range(2, 6):
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(tick))
    after = jax.tree.leaves(state.env.states[0])
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
    # The other task did advance.
    assert int(state.drift["task"].value) == 1


def test_restart_reinitialises_the_entered_task_and_reports_respawn() -> None:
    env = two_tasks(period=2, on_switch="restart")
    state, _, _ = env.init(jax.random.key(0))
    state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(0))
    assert not bool(info["respawn"][0])
    # The switch happens on the second tick, and the entered task restarts on
    # that same tick (it used to restart one tick late).
    state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(1))
    assert bool(env.privileged(state)["drift_changed"])
    assert bool(info["respawn"][0])


def test_task_switches_are_declared_switches_for_recovery_tracking() -> None:
    env = two_tasks(period=4)
    carry = pp.runtime.Runner(env, pp.policies.greedy, ticks=16)(
        pp.runtime.init_lifetime(env, seed=0, window_size=2)
    )
    assert bool(carry.metrics.has_switch)
    assert int(carry.metrics.since_switch) < 4


def test_task_switch_refuses_incompatible_environments() -> None:
    with pytest.raises(ValueError, match="agent count"):
        pp.tasks.switch([ContinualGrid(size=3, num_agents=2), ContinualGrid(size=3)])
    with pytest.raises(ValueError, match="discrete"):
        pp.tasks.switch([ContinualGrid(size=3), GymnaxParameters("Pendulum-v1")])
    with pytest.raises(ValueError, match="at least two"):
        pp.tasks.switch([ContinualGrid(size=3)])
    with pytest.raises(ValueError, match="on_switch"):
        pp.tasks.switch([ContinualGrid(size=3), ContinualGrid(size=3)], on_switch="bogus")  # pyright: ignore[reportArgumentType]
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    host = GymnasiumLifetime(gym.make("CartPole-v1"))
    try:
        with pytest.raises(ValueError, match="functional JAX"):
            pp.tasks.switch([ContinualGrid(size=3), cast(Any, host)])
    finally:
        host.close()


def test_per_task_parameters_are_addressable_by_name() -> None:
    env = pp.tasks.switch(
        [ContinualGrid(size=3), ContinualGrid(size=4)],
        {
            "task": pp.drift.Cyclic([0.0, 1.0], 4),
            "tasks[1].goal": pp.drift.Cyclic([(0.0, 0.0), (3.0, 3.0)], 2),
        },
    )
    assert set(env.drifts) == {"task", "tasks[1].goal"}
    state, _, _ = env.init(jax.random.key(0))
    assert "tasks[1].goal_parameters" in env.privileged(state)


class CountingHost:
    """Delegate a host lifetime while counting observations."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observations = 0

    def observe(self, state: Any) -> Any:
        self.observations += 1
        return self.env.observe(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def test_host_observe_only_calls_the_active_task() -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    envs = [CountingHost(GymnasiumLifetime(gym.make("CartPole-v1"))) for _ in range(2)]
    env = pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic([0.0, 1.0], 2))
    try:
        state, _, _ = env.init(jax.random.key(0))
        assert [env.observations for env in envs] == [1, 0]
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(0))
        assert [env.observations for env in envs] == [2, 0]
        # On a switch the framework's traced mask comparison may inspect all
        # host masks; ordinary ticks still observe only the active task.
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(1))
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(2))
        assert envs[0].observations == 3
        assert envs[1].observations == 3
    finally:
        for item in envs:
            item.close()


def test_host_switch_declares_checkpoint_capability(tmp_path: Any) -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    envs = [GymnasiumLifetime(gym.make("Pendulum-v1")) for _ in range(2)]
    env = pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic([0.0, 1.0], 2))
    try:
        assert env.metadata.get("checkpointable") is False
        runner = pp.runtime.HostRunner(
            env, lambda obs, info, key: jnp.zeros((1, 1), dtype=jnp.float32), ticks=2
        )
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=2))
        with pytest.raises(NotImplementedError, match="checkpointable"):
            pp.checkpoint.save(tmp_path / "unsupported", carry, metadata=runner.metadata)
    finally:
        for item in envs:
            item.close()


def test_snapshot_capable_host_switch_checkpoints_and_resumes(tmp_path: Any) -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    def make() -> Any:
        envs = [GymnasiumLifetime(gym.make("CartPole-v1")) for _ in range(2)]
        return pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic([0.0, 1.0], 3))

    env, target = make(), make()
    try:
        assert env.metadata.get("checkpointable") is True
        runner = pp.runtime.HostRunner(env, pp.policies.random, ticks=4)
        target_runner = pp.runtime.HostRunner(target, pp.policies.random, ticks=4)
        carry = runner(pp.runtime.init_lifetime(env, seed=1, window_size=4))
        pp.checkpoint.save(tmp_path / "switch", carry, metadata=runner.metadata)
        restored = pp.checkpoint.restore(
            tmp_path / "switch",
            pp.runtime.init_lifetime(target, seed=9, window_size=4),
            expected_metadata=target_runner.metadata,
        )
        # Crossing the period-3 switch after restore must continue identically.
        expected, actual = runner(carry), target_runner(restored)
        assert expected.state.t.to_int() == actual.state.t.to_int() == 8
        np.testing.assert_array_equal(np.asarray(expected.obs), np.asarray(actual.obs))
        np.testing.assert_array_equal(
            np.asarray(expected.metrics.total), np.asarray(actual.metrics.total)
        )
        assert int(expected.state.drift["task"].phase) == int(actual.state.drift["task"].phase)
    finally:
        env.close()
        target.close()


def test_oracle_task_id_and_boundary_follow_the_active_task() -> None:
    # Phase order (1, 0, 2) differs from task order, so task_id must be the
    # active task index rather than the schedule phase.
    env = pp.Oracle(
        pp.tasks.switch(
            [ContinualGrid(size=3), GymnaxParameters("Acrobot-v1"), ContinualGrid(size=4)],
            pp.drift.Cyclic([1.0, 0.0, 2.0], 2),
        ),
        expose=["task_id", "boundary"],
    )
    state, _, info = env.init(jax.random.key(0))

    def read(info: pp.Info) -> tuple[int, bool]:
        oracle = cast("dict[str, jax.Array]", info.get("oracle"))
        return int(oracle["task_id"]), bool(oracle["boundary"])

    seen = [read(info)]
    for tick in range(6):
        state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(tick))
        seen.append(read(info))
    assert seen == [
        (1, False),
        (1, False),
        (0, True),
        (0, False),
        (2, True),
        (2, False),
        (1, True),
    ]
    assert env.metadata["oracle_exposure"] == ("task_id", "boundary")


def test_restart_reinitialises_on_the_switch_tick() -> None:
    """The incoming task restarts on the tick it becomes active, and no action is lost."""
    env = pp.tasks.switch(
        [pp.make("ContinualGrid-v0", size=3), pp.make("ContinualGrid-v0", size=4)],
        pp.drift.Cyclic([0.0, 1.0], 2),
        on_switch="restart",
    )
    state, _, _ = env.init(jax.random.key(0))
    switches, respawns = [], []
    for tick in range(1, 7):
        state, _, _, info = env.step(state, jnp.zeros(1, dtype=jnp.int32), jax.random.key(tick))
        switches.append(bool(env.privileged(state)["drift_changed"]))
        respawns.append(bool(info["respawn"][0]))
    assert switches == [False, True, False, True, False, True]
    # Every switch tick reports the restart respawn; ContinualGrid-v0 with action 0
    # (stay) never reaches a goal on its own, so no other tick respawns.
    assert respawns == switches


def test_switch_refuses_a_task_with_its_own_fixed_value() -> None:
    """switch() keeps each task's defaults, so a fixed value would be silently dropped."""
    held = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Stationary(jnp.asarray([2.0, 2.0])))
    with pytest.raises(ValueError, match="must not drift"):
        pp.tasks.switch([held, pp.make("ContinualGrid-v0", size=4)])
    plain = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Stationary())
    pp.tasks.switch([plain, pp.make("ContinualGrid-v0", size=4)])


def test_a_resumed_task_does_not_repeat_its_last_respawn() -> None:
    env = pp.tasks.switch(
        [pp.make("ContinualGrid-v0", size=3), pp.make("ContinualGrid-v0", size=4)],
        pp.drift.Cyclic([0.0, 1.0], 4),
    )
    world = cast(Any, env).env
    state, _, _ = env.init(jax.random.key(0))
    inner = state.env._replace(respawn=state.env.respawn.at[1].set(True))  # task 1's last tick
    params = cast(Any, env)._params_of(state.drift)
    entering = params._replace(task=jnp.asarray(1.0))
    resumed = world.after_drift(inner, params, entering, jax.random.key(1))
    assert not bool(np.asarray(resumed.respawn[1]).any())
    assert int(resumed.active) == 1


def test_a_tasks_own_mask_parameter_reports_space_change_inside_a_switch() -> None:
    doors = pp.drift.Cyclic([(1, 1, 1, 1), (0, 0, 0, 0)], 2)

    def flags(env: Any) -> list[bool]:
        state, _, _ = env.init(jax.random.key(0))
        seen: list[bool] = []
        for tick in range(6):
            state, _, _, info = env.step(state, jnp.zeros((1,), jnp.int32), jax.random.key(tick))
            seen.append(bool(info["space_change"][0]))
        return seen

    alone = flags(pp.make("DoorGrid-v0", size=3, drift=doors))
    switched = flags(
        pp.tasks.switch(
            [pp.make("DoorGrid-v0", size=3), pp.make("ContinualGrid-v0", size=3)],
            {"tasks[0].doors": doors},
        )
    )
    assert alone == [False, True, False, True, False, True]
    assert switched == alone


def test_task_init_keys_never_equal_the_drift_stream_key() -> None:
    from perpetua.tasks import TaskSwitch

    seen: list[jax.Array] = []

    class Recording(ContinualGrid):
        def init(self, key: jax.Array, params: Any) -> Any:
            seen.append(jax.random.key_data(key))
            return super().init(key, params)

    for count in (2, 3, 4):
        seen.clear()
        key = jax.random.key(7)
        world = TaskSwitch([Recording(size=3) for _ in range(count)])
        pp.params.DriftEnv(world, pp.drift.Procedural(low=0, high=count - 1, period=1)).init(key)
        plain_splits = [np.asarray(jax.random.key_data(k)) for k in jax.random.split(key, 3)]
        for task_key in seen:
            assert not any(np.array_equal(np.asarray(task_key), r) for r in plain_splits)


def test_switch_accepts_a_drift_env_whose_world_has_no_env_id() -> None:
    class Anonymous(ContinualGrid):
        def __init__(self) -> None:
            super().__init__(size=3)
            metadata = {k: v for k, v in self.metadata.items() if k != "env_id"}
            cast(Any, self).metadata = metadata

    wrapped = pp.params.DriftEnv(Anonymous(), env_id="anonymous-v0")
    env = pp.tasks.switch(
        [wrapped, pp.make("ContinualGrid-v0", size=3)], pp.drift.Cyclic([0.0, 1.0], 2)
    )
    assert "Anonymous" in json.dumps(cast(Any, env.metadata)["configuration"])


def test_continuous_tasks_keep_their_own_action_mask() -> None:
    tasks = [GymnaxParameters("Swimmer-misc"), GymnaxParameters("Swimmer-misc")]
    env = pp.tasks.switch(tasks, pp.drift.Cyclic([0.0, 1.0], 2))
    state, _, info = env.init(jax.random.key(0))
    assert env.action_space.shape == (1, 2)
    assert bool(np.asarray(info["action_mask"]).all())
    action = jnp.zeros((1, 2), dtype=jnp.float32)
    for tick in range(3):
        state, _, _, info = env.step(state, action, jax.random.key(tick))
        # Both tasks use both action dimensions: no padded False may appear.
        assert bool(np.asarray(info["action_mask"]).all())


class Unsavable(ContinualGrid):
    """A grid that declares it cannot be checkpointed, like a compiled simulator bridge."""

    def __init__(self) -> None:
        super().__init__(size=3)
        self.metadata = {**self.metadata, "checkpointable": False, "forkable": False}


def test_a_switch_over_a_non_checkpointable_task_refuses_checkpoints(tmp_path: Any) -> None:
    env = pp.tasks.switch([ContinualGrid(size=3), Unsavable()], pp.drift.Cyclic([0.0, 1.0], 4))
    metadata = cast(dict[str, Any], env.metadata)
    assert metadata["execution"] == "jax"
    assert metadata["checkpointable"] is False
    assert metadata["forkable"] is False
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
    with pytest.raises(NotImplementedError, match="checkpointable"):
        pp.checkpoint.save(tmp_path / "refused", carry, metadata=runner.metadata)
    with pytest.raises(ValueError, match="same state"):
        pp.runtime.Runner(env, pp.policies.random, ticks=4, reference=pp.policies.random)
    pp.testing.check_env(lambda: pp.tasks.switch([ContinualGrid(size=3), Unsavable()]), ticks=4)
    plain = pp.tasks.switch([ContinualGrid(size=3), ContinualGrid(size=4)])
    assert "checkpointable" not in plain.metadata


def test_a_switch_records_each_tasks_configuration_in_order(tmp_path: Any) -> None:
    def make(sizes: tuple[int, int]) -> Any:
        return pp.tasks.switch(
            [ContinualGrid(size=sizes[0]), ContinualGrid(size=sizes[1])],
            pp.drift.Cyclic([0.0, 1.0], 4),
        )

    env, swapped, resized = make((3, 4)), make((4, 3)), make((3, 5))
    tasks = env.metadata["configuration"]["tasks"]
    assert [task["env_id"] for task in tasks] == ["ContinualGrid-v0", "ContinualGrid-v0"]
    assert tasks != swapped.metadata["configuration"]["tasks"]
    assert tasks != resized.metadata["configuration"]["tasks"]
    json.dumps(env.metadata["configuration"], allow_nan=False)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
    pp.checkpoint.save(tmp_path / "saved", carry, metadata=runner.metadata)
    for other in (swapped, resized):
        other_runner = pp.runtime.Runner(other, pp.policies.random, ticks=4)
        with pytest.raises(ValueError, match="tasks"):
            pp.checkpoint.restore(
                tmp_path / "saved",
                pp.runtime.init_lifetime(other, seed=1, window_size=4),
                expected_metadata=other_runner.metadata,
            )


def test_a_task_schedule_outside_the_task_range_is_refused() -> None:
    tasks = [ContinualGrid(size=3), ContinualGrid(size=4)]
    with pytest.raises(ValueError, match=r"holds \[2\.0\].*2 tasks.*0\.\.1"):
        pp.tasks.switch(tasks, pp.drift.Cyclic([1.0, 2.0], 2))
    with pytest.raises(ValueError, match=r"\[0\.5\]"):
        pp.tasks.switch(tasks, pp.drift.Cyclic([0.5, 1.0], 2))
    with pytest.raises(ValueError, match=r"\[-1\.0\]"):
        pp.tasks.switch(tasks, {"task": pp.drift.Stationary(-1.0)})
    with pytest.raises(ValueError, match=r"\[3\.0\]"):
        pp.tasks.switch(tasks, pp.drift.Sequence([(pp.drift.Cyclic([0.0, 3.0], 2), 6)]))
    with pytest.raises(ValueError, match=r"\[2\.0\]"):
        pp.tasks.switch(tasks, pp.drift.Schedule([0.0, 2.0], 2))
    pp.tasks.switch(tasks, pp.drift.Cyclic([1.0, 0.0], 2))
    pp.tasks.switch(tasks, {"tasks[0].goal": pp.drift.Stationary()})
    # A gradual process is held to the values that round to a task.
    pp.tasks.switch(tasks, pp.drift.RandomWalk(low=-0.5, high=1.5, scale=0.1))
    pp.tasks.switch(tasks, pp.drift.Procedural(low=-0.5, high=1.5, period=3))
    walk = pp.drift.RandomWalk(low=-0.4, high=0.4, scale=0.1)
    pp.tasks.switch(tasks, pp.drift.Sum([pp.drift.Cyclic([0.0, 1.0], 2), walk]))
    with pytest.raises(ValueError, match=r"can reach values in \[-3, 3\].*within \[-0\.5, 1\.5\]"):
        pp.tasks.switch(tasks, pp.drift.RandomWalk(low=-3.0, high=3.0, scale=0.1))
    with pytest.raises(ValueError, match=r"can reach values in \[0, 2\]"):
        pp.tasks.switch(tasks, pp.drift.Procedural(low=0.0, high=2.0, period=3))


def test_a_sum_that_can_leave_the_task_range_is_refused() -> None:
    tasks = [ContinualGrid(size=3), ContinualGrid(size=4)]
    D = pp.drift
    with pytest.raises(ValueError, match=r"can reach values in \[0, 7\].*2 tasks"):
        pp.tasks.switch(tasks, D.Sum([D.Cyclic([0.0, 7.0], 3), D.Stationary()]))
    with pytest.raises(ValueError, match=r"can reach values in \[0, 2\]"):
        pp.tasks.switch(tasks, D.Sum([D.Cyclic([0.0, 1.0], 3), D.Cyclic([0.0, 1.0], 5)]))
    with pytest.raises(ValueError, match=r"can reach values in \[-1, 2\]"):
        pp.tasks.switch(tasks, D.Sum([D.Stationary(), D.RandomWalk(low=-1.0, high=2.0, scale=0.1)]))
    with pytest.raises(ValueError, match=r"can reach values in \[0, 3\]"):
        pp.tasks.switch(
            tasks,
            D.Sequence(
                [(D.Cyclic([0.0, 1.0], 2), 4), (D.Sum([D.Stationary(2.0), D.Stationary(1.0)]), 4)]
            ),
        )
    env = pp.tasks.switch(tasks, D.Cyclic([0.0, 1.0], 3))
    with pytest.raises(ValueError, match=r"can reach values in \[0, 7\]"):
        env.with_drift(D.Sum([D.Cyclic([0.0, 7.0], 3), D.Stationary()]))


def test_an_assigned_task_schedule_outside_the_task_range_is_refused() -> None:
    tasks = [ContinualGrid(size=3), ContinualGrid(size=4)]
    env = pp.tasks.switch(tasks, pp.drift.Schedule([0.0, 1.0], 5, phases=3))
    life = pp.runtime.init_lifetime(env, seed=0)
    with pytest.raises(ValueError, match=r"holds \[3\.0, 7\.0\].*2 tasks.*0\.\.1"):
        pp.runtime.assign_schedule(env, life, pp.drift.schedule([3.0, 7.0], 5, phases=3))
    with pytest.raises(ValueError, match=r"holds \[0\.5\]"):
        pp.runtime.assign_schedule(env, life, pp.drift.schedule([0.5, 1.0], 5, phases=3))
    given = pp.runtime.assign_schedule(env, life, pp.drift.schedule([1.0, 0.0, 1.0], 5))
    assert int(env.privileged(given.state)["task_id"]) == 1
    good = pp.drift.stack_schedules(
        [pp.drift.schedule([1.0, 0.0], 5, phases=3), pp.drift.schedule([0.0, 1.0, 0.0], 2)]
    )
    env.check_schedule("task", good)
    bad = pp.drift.stack_schedules([good, good])._replace(
        values=jnp.stack([good.values, good.values.at[1, 2].set(4.0)])
    )
    with pytest.raises(ValueError, match=r"holds \[4\.0\]"):
        env.check_schedule("task", bad)
    # Rows past a schedule's length are padding and are not read.
    env.check_schedule("task", good._replace(values=good.values.at[0, 2].set(9.0)))
    # A schedule on another field is not a task table.
    grid = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Schedule([(0.0, 0.0), (3.0, 3.0)], 5))
    pp.runtime.assign_schedule(
        grid, pp.runtime.init_lifetime(grid, seed=0), pp.drift.schedule([(9.0, 9.0), (3.0, 3.0)], 5)
    )


def oracle_trace(schedule: Any, ticks: int, **kwargs: Any) -> list[tuple[int, bool, bool, int]]:
    """Per tick: task id, boundary, task_changed and the task process's phase."""
    env = pp.Oracle(
        pp.tasks.switch(
            [ContinualGrid(size=3), GymnaxParameters("Acrobot-v1")], schedule, **kwargs
        ),
        expose=["task_id", "boundary", "task_changed", "task_phase"],
    )
    state, _, info = env.init(jax.random.key(0))
    step: Any = jax.jit(env.step)
    rows: list[tuple[int, bool, bool, int]] = []
    for tick in range(ticks + 1):
        oracle = cast("dict[str, jax.Array]", info.get("oracle"))
        rows.append(
            (
                int(oracle["task_id"]),
                bool(oracle["boundary"]),
                bool(oracle["task_changed"]),
                int(oracle["task_phase"]),
            )
        )
        state, _, _, info = step(state, jnp.asarray([0]), jax.random.key(tick))
    return rows


@pytest.mark.parametrize(
    "schedule",
    [
        pp.drift.RandomWalk(low=0.0, high=1.0, scale=0.4),
        pp.drift.Sum([pp.drift.Stationary(), pp.drift.RandomWalk(low=0.0, high=1.0, scale=0.4)]),
    ],
)
@pytest.mark.parametrize("on_switch", ["resume", "restart"])
def test_a_gradual_task_schedule_declares_every_task_change(schedule: Any, on_switch: Any) -> None:
    rows = oracle_trace(schedule, 200, on_switch=on_switch)
    changed = [now[0] != before[0] for before, now in zip(rows, rows[1:], strict=False)]
    assert sum(changed) > 20
    assert [row[1] for row in rows[1:]] == changed
    assert [row[2] for row in rows[1:]] == changed
    env = pp.tasks.switch(
        [ContinualGrid(size=3), GymnaxParameters("Acrobot-v1")], schedule, on_switch=on_switch
    )
    carry = pp.runtime.Runner(env, pp.policies.random, ticks=200)(
        pp.runtime.init_lifetime(env, seed=0, window_size=4)
    )
    assert bool(carry.metrics.has_switch)


class ProcessSwitchesOnly(pp.tasks.TaskSwitch):
    """The switch as it was before it declared task changes itself."""

    declared_switches = cast(Any, None)


TABLES: dict[str, Any] = {
    "cyclic": pp.drift.Cyclic([0.0, 1.0], 3),
    "cyclic-repeated": pp.drift.Cyclic([0.0, 0.0, 1.0, 1.0, 0.0], 2),
    "schedule": pp.drift.Schedule([1.0, 0.0, 1.0], 4, offset=2),
    "sequence": pp.drift.Sequence(
        [
            (pp.drift.Stationary(1.0), 5),
            (pp.drift.Cyclic([0.0, 1.0], 3), 9),
            (pp.drift.Stationary(0.0), 4),
        ]
    ),
    "mapping": {
        "task": pp.drift.Cyclic([1.0, 0.0], 5),
        "tasks[0].goal": pp.drift.Cyclic([(0.0, 0.0), (2.0, 2.0)], 3),
    },
}


@pytest.mark.parametrize("name", sorted(TABLES))
@pytest.mark.parametrize("on_switch", ["resume", "restart"])
def test_table_schedules_are_unchanged_by_the_task_change_rule(name: str, on_switch: Any) -> None:
    def lifetime(world: type[pp.tasks.TaskSwitch]) -> Any:
        tasks = [ContinualGrid(size=3), GymnaxParameters("Acrobot-v1")]
        env = pp.params.DriftEnv(
            world(tasks, on_switch=on_switch), TABLES[name], env_id="task-switch-v0"
        )
        runner = pp.runtime.Runner(env, pp.policies.random, ticks=64)
        return runner(pp.runtime.init_lifetime(env, seed=3, window_size=4))

    now, before = lifetime(pp.tasks.TaskSwitch), lifetime(ProcessSwitchesOnly)
    assert bool(now.metrics.has_switch)
    for a, b in zip(jax.tree.leaves(now), jax.tree.leaves(before), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_repeated_table_value_is_still_a_declared_switch() -> None:
    rows = oracle_trace(pp.drift.Cyclic([0.0, 0.0, 1.0], 2), 12)
    assert [row[0] for row in rows] == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    # The switch between the two phases that hold task 0 (ticks 2 and 8) counts.
    assert [tick for tick, row in enumerate(rows) if row[1]] == [2, 4, 6, 8, 10, 12]
    assert [row[1] for row in rows] == [row[2] for row in rows]


class Guarded(ContinualGrid):
    """A grid that refuses a goal table of more than two phases."""

    def check_drift(self, drifts: Any) -> None:
        process = drifts.get("goal")
        if process is not None and len(process.configuration.get("values", [])) > 2:
            raise ValueError("Guarded takes at most two goals.")


def test_per_task_drift_passes_the_tasks_own_guards() -> None:
    D = pp.drift
    task = D.Cyclic([0.0, 1.0], 50)
    acrobot = GymnaxParameters("Acrobot-v1", drift_fields=("link_mass_1",))
    with pytest.raises(
        ValueError,
        match=r"Task 0 \(GymnaxParameters\) accepts drift only on \['link_mass_1'\], not \['dt'\]",
    ):
        pp.tasks.switch(
            [acrobot, ContinualGrid(size=3)], {"task": task, "tasks[0].dt": D.Cyclic([0.2, 5.0], 3)}
        )
    env = pp.tasks.switch(
        [acrobot, ContinualGrid(size=3)],
        {"task": task, "tasks[0].link_mass_1": D.Cyclic([1.0, 2.0], 3)},
    )
    with pytest.raises(ValueError, match=r"Task 0 \(GymnaxParameters\) accepts drift only on"):
        env.with_drift({"tasks[0].dt": D.Cyclic([0.2, 5.0], 3)})
    three = D.Cyclic([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], 3)
    with pytest.raises(ValueError, match=r"Task 1 \(Guarded\): Guarded takes at most two goals\."):
        pp.tasks.switch([ContinualGrid(size=3), Guarded(size=3)], {"tasks[1].goal": three})
    pp.tasks.switch([ContinualGrid(size=3), Guarded(size=3)], {"tasks[0].goal": three})


def test_an_endogenous_process_is_refused_when_bound_not_at_the_first_step() -> None:
    from perpetua.envs.forager_grid import ForagerGrid

    accumulate = pp.drift.Accumulate(gain=4.0, decay=0.9, low=1.0, high=8.0)
    with pytest.raises(ValueError, match=r"Accumulate on 'tasks\[0\]\.mass'.*task switch"):
        pp.tasks.switch(
            [ForagerGrid(size=3), ContinualGrid(size=3)],
            {"task": pp.drift.Cyclic([0.0, 1.0], 4), "tasks[0].mass": accumulate},
        )
    with pytest.raises(ValueError, match=r"Accumulate on 'goal'.*ContinualGrid defines no signal"):
        pp.make("ContinualGrid-v0", drift=pp.drift.Accumulate(low=0.0, high=4.0))
    grid = pp.make("ContinualGrid-v0")
    with pytest.raises(ValueError, match=r"ContinualGrid defines no signal"):
        pp.compose(grid, pp.drift.Accumulate(low=0.0, high=4.0))
    pp.make("ForagerGrid-v0", drift=accumulate)


@pytest.mark.parametrize("wrapper", [pp.Oracle, pp.OneHotObservation])
def test_switch_refuses_wrapped_tasks_and_says_where_the_wrapper_goes(wrapper: Any) -> None:
    inner = pp.make("ContinualGrid-v0", size=3)
    wrapped = (
        pp.Oracle(inner, expose=["task_id"])
        if wrapper is pp.Oracle
        else wrapper(pp.Oracle(inner, expose=["task_id"]), field="task_id", size=2)
    )
    with pytest.raises(TypeError, match=rf"unwrapped.*{wrapper.__name__}.*wrap the switch"):
        pp.tasks.switch([wrapped, pp.make("ContinualGrid-v0", size=4)])


@pytest.mark.parametrize("env_id", ["PartnerGrid-v0", "OpponentGrid-v0"])
def test_composite_grids_report_respawn_inside_a_switch(env_id: str) -> None:
    alone = pp.make(env_id, size=3)
    switched = pp.tasks.switch(
        [pp.make(env_id, size=3), pp.make(env_id, size=4)], pp.drift.Cyclic([0.0, 1.0], 150)
    )

    def count(env: Any) -> int:
        def tick(state: Any, key: jax.Array) -> tuple[Any, jax.Array]:
            action_key, step_key = jax.random.split(key)
            action = jax.random.randint(action_key, (env.num_agents,), 0, 5)
            state, _, _, info = env.step(state, action, step_key)
            return state, jnp.sum(info["respawn"])

        state, _, _ = env.init(jax.random.key(0))
        _, counts = jax.lax.scan(tick, state, jax.random.split(jax.random.key(1), 300))
        return int(jnp.sum(counts))

    assert count(alone) > 0
    assert count(switched) > 0


def test_a_respawn_on_the_switch_tick_is_reported() -> None:
    env = pp.tasks.switch(
        [GymnaxParameters("BernoulliBandit-misc"), GymnaxParameters("BernoulliBandit-misc")],
        pp.drift.Cyclic([0.0, 1.0], 100),
    )

    def tick(state: Any, key: jax.Array) -> tuple[Any, jax.Array]:
        state, _, _, info = env.step(state, jnp.zeros((1,), jnp.int32), key)
        return state, info["respawn"][0]

    state, _, _ = env.init(jax.random.key(0))
    _, flags = jax.lax.scan(tick, state, jax.random.split(jax.random.key(1), 400))
    # Each bandit ends its 100-step episode on the tick that hands over to the other.
    assert np.flatnonzero(np.asarray(flags)).tolist() == [99, 199, 299, 399]


class AlwaysRespawning:
    """Delegate a host lifetime whose own observation always claims a respawn."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def observe(self, state: Any) -> Any:
        obs, info = self.env.observe(state)
        return obs, {**info, "respawn": jnp.ones_like(info["respawn"])}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def test_a_host_switch_reports_its_own_respawn_flags_not_the_hosts_stale_ones() -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    envs = [AlwaysRespawning(GymnasiumLifetime(gym.make("CartPole-v1"))) for _ in range(2)]
    env = pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic([0.0, 1.0], 2))
    try:
        state, _, info = env.init(jax.random.key(0))
        assert not bool(info["respawn"][0])
        for tick in range(4):
            state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(tick))
            assert bool(info["respawn"][0]) == bool(np.asarray(state.env.respawn).any())
    finally:
        for item in envs:
            item.close()


def test_a_bad_task_table_is_refused_on_every_way_of_binding_drift() -> None:
    from types import MappingProxyType

    tasks = [ContinualGrid(size=3), ContinualGrid(size=4)]
    bad = pp.drift.Cyclic([0.0, 5.0], 3)
    refused = pytest.raises(ValueError, match=r"holds \[5\.0\].*2 tasks.*0\.\.1")
    with refused:
        pp.tasks.switch(tasks, bad)
    with refused:
        pp.tasks.switch(tasks, cast(Any, MappingProxyType({"task": bad})))
    env = pp.tasks.switch(tasks, pp.drift.Cyclic([0.0, 1.0], 3))
    with refused:
        env.with_drift(bad)
    with refused:
        env.with_drift(MappingProxyType({"task": bad}))
    with refused:
        pp.compose(env, bad)
    rebound = cast(Any, pp.compose(env, pp.drift.Cyclic([1.0, 0.0], 3)))
    assert rebound.drifts["task"].configuration["values"] == [1.0, 0.0]


def test_a_task_with_different_action_counts_per_slot_is_refused_when_built() -> None:
    from gymnasium import spaces

    uneven = ContinualGrid(size=3, num_agents=2)
    uneven.action_space = spaces.MultiDiscrete([5, 3])
    with pytest.raises(ValueError, match=r"task 1 \(ContinualGrid\).*nvec=\[5, 3\]"):
        pp.tasks.switch([ContinualGrid(size=3, num_agents=2), uneven])
