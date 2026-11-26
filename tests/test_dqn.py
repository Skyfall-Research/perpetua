"""Continuing DQN must respect masks, bound memory and checkpoint exactly."""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents.dqn import (
    DQN,
    DQNConfig,
    DQNRunner,
    DQNTrainingState,
    HostDQNRunner,
    init_dqn,
    init_replay,
    insert,
    masked_greedy,
    sample,
)
from perpetua.envs.continual_grid import GridParams, GridState
from perpetua.params import DriftEnv, DriftEnvState


def test_masked_greedy_and_epsilon_never_select_invalid_actions() -> None:
    q = jnp.asarray([[3.0, 2.0, 1.0], [0.0, 5.0, 4.0]])
    mask = jnp.asarray([[True, False, False], [True, True, False]])
    np.testing.assert_array_equal(masked_greedy(q, mask), [0, 1])

    env = pp.make("ContinualGrid-v0", size=3)
    grid, _, _ = env.init(jax.random.key(0))
    grid = grid._replace(env=grid.env._replace(positions=jnp.asarray([[0, 0]])))
    obs, info = env.observe(grid)
    info["action_mask"] = info["action_mask"].at[0, 1:].set(False)
    config = DQNConfig(rollout_steps=2, buffer_capacity=8, batch_size=2, gradient_steps=1)
    greedy = DQN(action_size=5, config=config)
    state = greedy.init(obs, jax.random.key(1))
    for seed in range(8):
        assert int(greedy.act(state, obs, info, jax.random.key(seed))[0]) == 0
    exploring = DQN(
        action_size=5,
        config=DQNConfig(
            rollout_steps=2, buffer_capacity=8, batch_size=2, gradient_steps=1, epsilon=1.0
        ),
    )
    state = exploring.init(obs, jax.random.key(1))
    chosen = {int(exploring.act(state, obs, info, jax.random.key(seed))[0]) for seed in range(64)}
    assert chosen == {0}
    info["action_mask"] = info["action_mask"].at[0, 0].set(False)
    info["action_mask"] = info["action_mask"].at[0, 1].set(True)
    assert int(exploring.act(state, obs, info, jax.random.key(0))[0]) == 1
    info["active_mask"] = jnp.asarray([False])
    assert int(exploring.act(state, obs, info, jax.random.key(0))[0]) == 0


def test_replay_buffer_wraps_bounded_and_samples_only_filled() -> None:
    replay = init_replay(num_agents=1, obs_dim=3, action_size=5, capacity=4)
    for step in range(6):
        transition = (
            jnp.full((1, 3), float(step)),
            jnp.ones((1, 5), dtype=jnp.bool_),
            jnp.asarray([True]),
            jnp.asarray([step % 5], dtype=jnp.int32),
            jnp.asarray([float(step)]),
            jnp.full((1, 3), float(step) + 0.5),
            jnp.ones((1, 5), dtype=jnp.bool_),
        )
        replay = insert(replay, transition)
    assert int(replay.cursor) == 6
    assert int(replay.filled) == 4
    # Ring wrap: the two oldest entries were overwritten by steps 4 and 5.
    np.testing.assert_array_equal(np.asarray(replay.obs)[:, 0, 0], [4.0, 5.0, 2.0, 3.0])
    batch = sample(replay, jax.random.key(0), 3)
    assert batch.obs.shape == (3, 1, 3)
    assert bool(jnp.all(batch.reward <= 5.0))
    again = sample(replay, jax.random.key(0), 3)
    np.testing.assert_array_equal(np.asarray(batch.obs), np.asarray(again.obs))
    different = sample(replay, jax.random.key(1), 3)
    assert not np.array_equal(np.asarray(batch.action), np.asarray(different.action))


def test_target_network_syncs_on_schedule() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(
        rollout_steps=2,
        buffer_capacity=8,
        batch_size=2,
        gradient_steps=3,
        target_update_interval=3,
    )
    agent = DQN(action_size=5, config=config)
    carry = seed_replay(env, agent, seed=0, ticks=4)
    for step_key in (jax.random.key(2), jax.random.key(2)):
        agent_state, statistics = agent.update(carry.agent, carry.replay, step_key)
        carry = carry._replace(agent=agent_state, statistics=statistics)
    assert not tree_equal(carry.agent.params, carry.agent.target)
    assert int(carry.agent.sync_age) == 2
    agent_state, statistics = agent.update(carry.agent, carry.replay, jax.random.key(3))
    carry = carry._replace(agent=agent_state, statistics=statistics)
    assert tree_equal(carry.agent.params, carry.agent.target)
    assert int(carry.agent.sync_age) == 0


def test_dqn_checkpoint_restores_every_leaf_including_replay(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(rollout_steps=4, buffer_capacity=8, batch_size=2, gradient_steps=1)
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    path = tmp_path / "dqn"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert int(restored.replay.cursor) == 4
    assert_same(trainer(state), trainer(restored))


def test_dqn_checkpoint_resumes_across_a_drift_switch(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 3))
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1)
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    # One update spans 4 ticks, so the goal switch inside it is already history.
    assert int(state.lifetime.state.drift["goal"].phase) == 1
    path = tmp_path / "switch"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert int(restored.lifetime.state.drift["goal"].phase) == 1
    assert_same(trainer(state), trainer(restored))


def test_host_dqn_runner_resumes_a_cpu_cartpole_lifetime(tmp_path: Path) -> None:
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1)
    first = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    second = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    try:
        trainer = HostDQNRunner(first, DQN(action_size=2, config=config))
        state = trainer(trainer.init(seed=0, window_size=4))
        assert bool(state.healthy)
        assert state.lifetime.state.t.to_int() == 4
        path = tmp_path / "host"
        pp.checkpoint.save(path, state, metadata=trainer.metadata)
        resume = HostDQNRunner(second, DQN(action_size=2, config=config))
        template = resume.init(seed=999, window_size=4)
        restored = pp.checkpoint.restore(path, template, expected_metadata=resume.metadata)
        assert_same(state, restored)
        assert_same(trainer(state), resume(restored))
    finally:
        first.close()
        second.close()


def test_invalid_dqn_configuration() -> None:
    with pytest.raises(ValueError, match=r"rollout_steps must be a positive integer"):
        DQNConfig(rollout_steps=0)
    with pytest.raises(ValueError, match=r"buffer_capacity must be a positive integer"):
        DQNConfig(buffer_capacity=0)
    with pytest.raises(ValueError, match=r"batch_size must be a positive integer"):
        DQNConfig(batch_size=0)
    with pytest.raises(ValueError, match=r"batch_size must not exceed buffer_capacity"):
        DQNConfig(batch_size=5, buffer_capacity=4)
    with pytest.raises(ValueError, match=r"gradient_steps must be a positive integer"):
        DQNConfig(gradient_steps=0)
    with pytest.raises(ValueError, match=r"target_update_interval must be a positive integer"):
        DQNConfig(target_update_interval=0)
    with pytest.raises(ValueError, match=r"Epsilon must be in \[0, 1\]"):
        DQNConfig(epsilon=1.5)
    with pytest.raises(ValueError, match=r"Gamma must be in \[0, 1\]"):
        DQNConfig(gamma=1.5)
    with pytest.raises(ValueError, match=r"Learning rate and maximum gradient norm must be"):
        DQNConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match=r"Learning rate and maximum gradient norm must be"):
        DQNConfig(max_grad_norm=0.0)


def test_dqn_learns_above_random_on_a_continuing_grid() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    config = DQNConfig(
        rollout_steps=32,
        buffer_capacity=4096,
        batch_size=64,
        gradient_steps=8,
        hidden_size=64,
        epsilon=0.1,
        target_update_interval=64,
    )
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer.init(seed=0, window_size=128)
    for _ in range(200):
        state = trainer(state)
    assert bool(state.healthy)
    learned = float(np.asarray(state.lifetime.metrics.mean)[0])
    baseline_env = pp.make("ContinualGrid-v0", size=4)
    baseline = pp.runtime.Runner(baseline_env, pp.policies.random, ticks=6400)(
        pp.runtime.init_lifetime(baseline_env, seed=0, window_size=128)
    )
    baseline_mean = float(np.asarray(baseline.metrics.mean)[0])
    assert learned > 2 * baseline_mean
    print(f"DQN mean {learned:.4f} vs random {baseline_mean:.4f}")


def seed_replay(
    env: DriftEnv[GridState, GridParams], agent: DQN, *, seed: int, ticks: int
) -> DQNTrainingState[DriftEnvState[GridState]]:
    """Insert real transitions without running the compiled runner."""
    carry = init_dqn(env, agent, seed=seed, window_size=4)
    for _ in range(ticks):
        agent_key, act_key = jax.random.split(carry.lifetime.agent_key)
        env_key, step_key = jax.random.split(carry.lifetime.env_key)
        action = agent.act(carry.agent, carry.lifetime.obs, carry.lifetime.info, act_key)
        state, obs, reward, info = env.step(carry.lifetime.state, action, step_key)
        lifetime = carry.lifetime._replace(
            state=state, obs=obs, info=info, env_key=env_key, agent_key=agent_key
        )
        transition = (
            carry.lifetime.obs,
            carry.lifetime.info["action_mask"],
            carry.lifetime.info["active_mask"],
            action,
            reward,
            obs,
            info["action_mask"],
        )
        carry = carry._replace(lifetime=lifetime, replay=insert(carry.replay, transition))
    return carry


def tree_equal(left: object, right: object) -> bool:
    return all(
        bool(np.array_equal(np.asarray(a), np.asarray(b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_dqn_trains_through_cyclic_goal_drift() -> None:
    env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], 256))
    config = DQNConfig(
        rollout_steps=32,
        buffer_capacity=4096,
        batch_size=64,
        gradient_steps=8,
        hidden_size=64,
        epsilon=0.1,
        target_update_interval=64,
    )
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer.init(seed=0, window_size=128)
    trace: list[np.ndarray] = []
    for _ in range(200):
        state = trainer(state)
        trace.append(np.asarray(pp.evaluation.recent_rewards(state.lifetime.metrics, steps=32)))
    assert bool(state.healthy)
    # 6400 ticks with period 256: 25 switches, cyclic table index wraps to 1.
    assert int(state.lifetime.state.drift["goal"].elapsed) == 0
    assert int(state.lifetime.state.drift["goal"].phase) == 1
    rewards = np.concatenate(trace)[:, 0]
    learned = float(np.asarray(state.lifetime.metrics.mean)[0])
    baseline_env = pp.make("ContinualGrid-v0", size=4)
    baseline = pp.runtime.Runner(baseline_env, pp.policies.random, ticks=6400)(
        pp.runtime.init_lifetime(baseline_env, seed=0, window_size=128)
    )
    random_mean = float(np.asarray(baseline.metrics.mean)[0])
    assert learned > 2 * random_mean
    # Late-lifetime performance does not collapse after the switches.
    assert rewards[-1600:].mean() > random_mean
    changes = np.asarray([255 + 256 * k for k in range(25)])
    ticks = np.asarray(pp.evaluation.recovery_ticks(rewards, changes, window=64, fraction=0.5))
    capped = int(np.sum(ticks >= len(rewards) - changes))
    assert float(ticks.mean()) < 160.0
    assert capped <= 3
    print(
        f"DQN through drift: mean {learned:.4f} vs random {random_mean:.4f}, "
        f"recovery mean {ticks.mean():.1f}, capped {capped}"
    )


def test_replay_cursor_never_overflows_and_keeps_its_ring_position() -> None:
    from perpetua.agents._common import advance_cursor  # pyright: ignore[reportPrivateUsage]

    capacity = 4
    for advance in (advance_cursor,):
        cursor, count = jnp.int32(0), 0
        for _ in range(50):
            cursor, count = advance(cursor, capacity), count + 1
            assert int(cursor) % capacity == count % capacity  # same write index
            assert min(int(cursor), capacity) == min(count, capacity)  # same fill level
            assert int(cursor) < 2 * capacity
        # Near the int32 limit a plain increment would wrap negative.
        assert int(advance(jnp.int32(2 * capacity - 1), capacity)) == capacity


def test_a_transition_with_no_valid_next_action_bootstraps_from_zero() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(rollout_steps=2, buffer_capacity=8, batch_size=4, gradient_steps=1)
    agent = DQN(action_size=5, config=config)
    carry = seed_replay(env, agent, seed=0, ticks=8)
    replay = carry.replay._replace(next_mask=jnp.zeros_like(carry.replay.next_mask))
    _, statistics = agent.update(carry.agent, replay, jax.random.key(1))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(statistics))
    assert float(jnp.max(jnp.abs(statistics.loss))) < 10.0


def test_dqn_action_size_may_be_a_numpy_integer_and_mismatches_name_both_numbers() -> None:
    from perpetua.agents import DQN, IQL, DQNConfig, DQNRunner

    env = pp.make("ContinualGrid-v0", size=3, num_agents=2)
    size: Any = env.action_space.nvec[0]
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, hidden_size=8)
    DQNRunner(env, DQN(action_size=size, config=config)).init(seed=0)
    for agent in (DQN(action_size=3, config=config), IQL(action_size=3, config=config)):
        name = type(agent).__name__
        with pytest.raises(
            ValueError, match=rf"{name} was built with action_size=3.*has 5 actions"
        ):
            DQNRunner(env, agent).init(seed=0)
    with pytest.raises(ValueError, match="needs a discrete action space"):
        DQNRunner(pp.make("gymnax/Pendulum-v1"), DQN(action_size=1, config=config)).init(seed=0)


def test_dqn_config_refuses_a_rollout_longer_than_the_buffer() -> None:
    with pytest.raises(ValueError, match=r"rollout_steps \(32\).*buffer_capacity \(16\).*16 ticks"):
        DQNConfig(rollout_steps=32, buffer_capacity=16, batch_size=2)
    DQNConfig(rollout_steps=16, buffer_capacity=16, batch_size=2)
    doc = DQNConfig.__doc__ or ""
    for field in ("rollout_steps", "buffer_capacity", "batch_size", "gradient_steps"):
        assert f"``{field}``" in doc


def test_the_mlp_torso_refuses_pixel_observations_for_dqn_and_iql() -> None:
    from perpetua.agents import DQN, IQL, DQNConfig, DQNRunner

    env = pp.make("gymnax/Breakout-MinAtar")
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, hidden_size=8)
    for agent in (DQN(action_size=3, config=config), IQL(action_size=3, config=config)):
        with pytest.raises(ValueError, match=r"MLP torso needs \(agents, features\)"):
            DQNRunner(env, agent).init(seed=0)


def test_reward_centering_settings_are_validated_at_construction() -> None:
    config: Any = DQNConfig
    for value in ("no", 1, None):
        with pytest.raises(ValueError, match=r"reward_centering must be a boolean"):
            config(reward_centering=value)
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"centering_step_size must be finite"):
            config(centering_step_size=value)
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"centering_step_size must be in \[0, 1\]"):
            config(centering_step_size=value)
    assert config(reward_centering=True, centering_step_size=1.0).reward_centering is True
