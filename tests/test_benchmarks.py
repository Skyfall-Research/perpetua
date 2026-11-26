"""MEAL and Meta-World integrations, including their continual presets."""

# Third-party environments have partially annotated interfaces.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

import perpetua as pp


def zero_action(obs: Any, info: Any, key: Any) -> jax.Array:
    return jnp.zeros((1, 4), dtype=jnp.float32)


def test_meal_runs_compiled_and_trains_with_the_cnn() -> None:
    pytest.importorskip("meal", reason="optional meal group")
    from perpetua.agents import PPO, PPOConfig, PPORunner

    env = pp.make("meal/overcooked-v0")
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
    assert carry.state.t.to_int() == 4
    config = PPOConfig(torso="cnn", rollout_steps=2, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=6, config=config))
    state = trainer(trainer.init(seed=0))
    assert bool(state.healthy)
    assert state.agent.gradient_steps.to_int() == 1


def test_meal_sequence_switches_layouts() -> None:
    pytest.importorskip("meal", reason="optional meal group")
    env = pp.presets.meal_sequence(sequence_length=2, period=2)
    state, obs, info = env.init(jax.random.key(0))
    assert obs.shape == (2, 9, 9, 26)
    for tick in range(4):
        state, obs, reward, info = env.step(
            state, jnp.zeros(2, dtype=jnp.int32), jax.random.key(tick)
        )
        assert reward.shape == (2,)
    assert int(state.drift["task"].value) == 0
    assert info["t"].to_int() == 4


def test_metaworld_runs_and_trains_through_host_sac() -> None:
    from perpetua.agents import SAC, HostSACRunner, SACConfig

    env = pp.make("metaworld/MT1-v0", env_name="hammer-v3")
    try:
        space = cast(Any, env.action_space)
        config = SACConfig(
            rollout_steps=2, buffer_capacity=16, batch_size=4, gradient_steps=1, hidden_size=8
        )
        agent = SAC(action_low=space.low[0], action_high=space.high[0], config=config)
        trainer = HostSACRunner(env, agent)
        state = trainer(trainer.init(seed=0))
        assert bool(state.healthy)
        assert state.agent.gradient_steps.to_int() == 1
    finally:
        cast(Any, env).close()


def test_continual_world_preset_switches_continuous_host_tasks() -> None:
    env = pp.presets.continual_world(("hammer-v3", "push-v3"), period=2)
    try:
        runner = pp.runtime.HostRunner(env, zero_action, ticks=4)
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
        assert carry.state.t.to_int() == 4
        assert int(carry.state.drift["task"].value) == 0
        assert carry.obs.shape == (1, 39)
    finally:
        env.close()


def test_cw20_preset_is_cw10_twice_on_one_simulator_per_task() -> None:
    env = pp.presets.cw20(period=2)
    try:
        preset = cast(dict[str, Any], env.metadata.get("configuration"))["preset"]
        assert preset == {
            "name": "cw20",
            "tasks": list(pp.presets.CW10),
            "repetitions": 2,
            "seed": 0,
            "max_episode_steps": 200,
            "compiled": False,
            "randomize_goals": False,
            "horizon_ticks": 40,
            "tuning": pp.presets.DEFAULT_TUNING.configuration,
        }
        runner = pp.runtime.HostRunner(env, zero_action, ticks=22)
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
        # Twenty-two ticks at two per task: the cycle is back at task 1, the
        # second pass over the same ten simulators.
        assert int(carry.state.drift["task"].value) == 1
    finally:
        env.close()


def test_compiled_continual_world_trains_in_one_program_with_the_paper_protocol() -> None:
    import math

    from perpetua.agents import SAC, SACConfig, SACRunner

    tasks = pp.presets.continual_world(("hammer-v3", "push-wall-v3"), period=40, compiled=True)
    env = pp.OneHotObservation(
        pp.Oracle(tasks, expose=["task_id", "boundary"]), field="task_id", size=2
    )
    try:
        assert env.metadata.get("execution") != "host"
        box = cast(Any, env.action_space)
        config = SACConfig(
            rollout_steps=20,
            gradient_steps=2,
            buffer_capacity=256,
            batch_size=8,
            hidden_size=16,
            hidden_layers=4,
            layer_norm=True,
            activation="leaky_relu",
            alpha=math.e,
            autotune_alpha=True,
            task_boundary_reset=True,
            random_steps=10,
            update_after=5,
            mask_respawns=True,
        )
        runner = SACRunner(env, SAC(action_low=box.low[0], action_high=box.high[0], config=config))
        state = runner.init(seed=0)
        cursors = []
        for _ in range(4):
            state = runner(state)
            cursors.append(int(state.replay.cursor))
        assert state.lifetime.state.t.to_int() == 80
        assert bool(state.healthy)
        assert cursors == [20, 0, 20, 0]  # the switches at ticks 40 and 80 empty the replay
        assert env.observation_space.shape == (1, 41)
    finally:
        env.close()


def test_switch_refuses_tasks_that_drift_on_their_own() -> None:
    drifting = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (12.0,)], 8))
    other = pp.make("gymnax/Pendulum-v1")
    with pytest.raises(ValueError, match="must not drift"):
        pp.tasks.switch([drifting, other], pp.drift.Cyclic([0.0, 1.0], 16))
    # Stationary DriftEnv tasks are composed through their worlds.
    env = pp.tasks.switch([other, pp.make("gymnax/Pendulum-v1")], pp.drift.Cyclic([0.0, 1.0], 4))
    state, obs, _ = env.init(jax.random.key(0))
    for tick in range(6):
        state, obs, _, _ = env.step(state, jnp.zeros((1, 1)), jax.random.key(tick))
    assert obs.shape == (1, 3)


def test_metaworld_can_draw_a_new_goal_every_episode_reproducibly() -> None:
    import numpy as np

    from perpetua.adapters.host import MetaWorldLifetime

    def goals(randomize: bool) -> list[tuple[float, ...]]:
        env = MetaWorldLifetime(env_name="reach-v3", max_episode_steps=4, randomize_goals=randomize)
        try:
            state, obs, _ = env.init(jax.random.key(0))
            seen = [tuple(np.asarray(obs)[0, -3:].round(4).tolist())]
            for tick in range(12):
                state, obs, _, info = env.step(state, jnp.zeros((1, 4)), jax.random.key(tick))
                if bool(info["respawn"][0]):
                    seen.append(tuple(np.asarray(obs)[0, -3:].round(4).tolist()))
            return seen
        finally:
            env.close()

    assert len(set(goals(False))) == 1  # the pinned goal
    drawn = goals(True)
    # A fresh draw from MT1's 50 configurations each episode (repeats can occur).
    assert len(drawn) == 4
    assert len(set(drawn)) >= 3
    assert goals(True) == drawn  # and the same ones for the same seed


def test_metaworld_observation_space_contains_what_it_returns_without_warnings() -> None:
    import warnings

    import numpy as np

    from perpetua.adapters.host import MetaWorldLifetime

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = MetaWorldLifetime(env_name="hammer-v3", max_episode_steps=4)
        try:
            space = env.observation_space
            # Upstream declares the goal entries as the point 0; observed values lie off it.
            assert np.isinf(space.low[0, 36:39]).all()
            assert np.isinf(space.high[0, 36:39]).all()
            state, obs, _ = env.init(jax.random.key(0))
            assert space.contains(np.asarray(obs))
            for tick in range(6):
                state, obs, _, _ = env.step(state, jnp.zeros((1, 4)), jax.random.key(tick))
                assert space.contains(np.asarray(obs))
        finally:
            env.close()


def test_metaworld_refuses_an_episode_length_above_its_own_horizon() -> None:
    from perpetua.adapters.host import MetaWorldLifetime

    with pytest.raises(ValueError, match=r"max_episode_steps=501 exceeds.*500 steps"):
        MetaWorldLifetime(env_name="hammer-v3", max_episode_steps=501)
    env = MetaWorldLifetime(env_name="hammer-v3", max_episode_steps=500)
    try:
        assert env.metadata["configuration"]["backend"]["horizon"] == 500  # type: ignore[index]
    finally:
        env.close()


def test_a_compiled_continual_world_switch_cannot_be_checkpointed(tmp_path: Any) -> None:
    def make() -> Any:
        return pp.presets.continual_world(
            ("reach-v3", "push-v3"), period=3, compiled=True, max_episode_steps=50, seed=4
        )

    env = make()
    try:
        assert env.metadata["checkpointable"] is False
        assert env.metadata["forkable"] is False
        preset = env.metadata["configuration"]["preset"]
        assert (preset["seed"], preset["max_episode_steps"], preset["compiled"]) == (4, 50, True)
        assert env.metadata["configuration"]["tasks"][0]["configuration"]["backend"]["seed"] == 4
        runner = pp.runtime.Runner(env, zero_action, ticks=2)
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=2))
        with pytest.raises(NotImplementedError, match="checkpointable"):
            pp.checkpoint.save(tmp_path / "saved", carry, metadata=runner.metadata)
    finally:
        env.close()
    pp.testing.check_env(make, ticks=4)
