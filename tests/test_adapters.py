"""Adapters must preserve lifetime semantics and declare legacy boundaries."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp


def test_gymnasium_view_uses_familiar_spaces_and_return_values() -> None:
    from perpetua.adapters.gymnasium import GymnasiumAdapter

    env = pp.make("ContinualGrid-v0", size=3)
    view = GymnasiumAdapter(env, horizon=2)
    obs, info = view.reset(seed=0)
    assert view.observation_space.contains(obs)
    assert isinstance(view.action_space, gym.spaces.Discrete)
    assert view.action_space.n == 5
    assert info["t"].to_int() == 0
    obs, reward, terminated, truncated, info = view.step(0)
    assert view.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    _, _, terminated, truncated, info = view.step(0)
    assert not terminated
    assert truncated
    assert info["t"].to_int() == 2
    _, continued = view.reset()
    assert continued["t"].to_int() == 2
    _, _, _, truncated, info = view.step(0)
    assert not truncated
    assert info["t"].to_int() == 3


def test_gymnasium_view_cannot_silently_restart_a_lifetime() -> None:
    from perpetua.adapters.gymnasium import GymnasiumAdapter

    view = GymnasiumAdapter(pp.make("ContinualGrid-v0"))
    view.reset(seed=1)
    view.step(0)
    with pytest.raises(ValueError, match="seed"):
        view.reset(seed=2)
    with pytest.raises(ValueError, match=r"Action is outside the Gymnasium action space\."):
        view.step(99)
    with pytest.raises(ValueError, match=r"The Gymnasium view supports one agent; do not"):
        GymnasiumAdapter(pp.make("ContinualGrid-v0", num_agents=2))


def test_gymnax_cartpole_has_continuing_public_contract() -> None:
    env = pp.make("gymnax/CartPole-v1")
    state, obs, info = env.init(jax.random.key(0))
    assert obs.shape == (1, 4)
    assert env.observation_space.contains(np.asarray(obs))
    state, obs, reward, info = env.step(state, jnp.asarray([1]), jax.random.key(1))
    assert obs.shape == (1, 4)
    assert reward.shape == (1,)
    assert info["t"].to_int() == 1
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert not hasattr(env, "reset")


def test_gymnax_respawn_is_not_an_interface_boundary() -> None:
    env = pp.make("gymnax/CartPole-v1")
    state, _, _ = env.init(jax.random.key(0))
    physics = state.env.physics.replace(theta=jnp.asarray(0.5))
    state = state._replace(env=state.env._replace(physics=physics))
    state, obs, reward, info = env.step(state, jnp.asarray([0]), jax.random.key(1))
    assert bool(info["respawn"][0])
    assert info["t"].to_int() == 1
    assert abs(float(obs[0, 2])) < 0.05
    assert float(reward[0]) == 0.0
    assert int(state.env.physics.time) == 0


def test_gymnax_time_limit_is_not_a_lifetime_boundary() -> None:
    env = pp.make("gymnax/CartPole-v1")
    state, _, _ = env.init(jax.random.key(0))
    physics = state.env.physics.replace(time=jnp.int32(2**31 - 1))
    state = state._replace(env=state.env._replace(physics=physics))
    state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(1))
    assert not bool(info["respawn"][0])
    assert info["t"].to_int() == 1
    assert int(state.env.physics.time) == 1


def test_gymnax_drift_checkpoint_and_ppo_reuse(tmp_path: Path) -> None:
    from perpetua.agents import PPO, PPOConfig, PPORunner

    env = pp.make("gymnax/CartPole-v1", drift=pp.drift.Cyclic([(9.8,), (5.0,)], period=3))
    config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=2, config=config))
    state = trainer(trainer.init(seed=0))
    assert state.lifetime.state.t.to_int() == 8
    assert state.agent.gradient_steps.to_int() == 1
    assert bool(state.healthy)
    checkpoint = tmp_path / "cartpole"
    pp.checkpoint.save(checkpoint, state, metadata=trainer.metadata)
    restored = pp.checkpoint.restore(
        checkpoint, trainer.init(seed=999), expected_metadata=trainer.metadata
    )
    actual, expected = trainer(restored), trainer(state)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
        else:
            np.testing.assert_array_equal(left, right)


def test_gymnasium_cpu_env_is_lifted_without_public_done_flags(tmp_path: Path) -> None:
    from perpetua.adapters import GymnasiumLifetime

    env = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=2))
    try:
        runner = pp.runtime.HostRunner(env, pp.policies.random, ticks=2)
        initial = pp.runtime.init_lifetime(env, seed=0)
        state = runner(initial)
        assert state.state.t.to_int() == 2
        assert bool(state.info["respawn"][0])
        assert state.obs.shape == (1, 4)
        assert "terminated" not in state.info
        assert "truncated" not in state.info
        pp.checkpoint.save(tmp_path / "cartpole", state, metadata=env.metadata)
        replay = runner(initial)
        np.testing.assert_array_equal(replay.obs, state.obs)
    finally:
        env.close()


def test_host_ppo_runs_against_a_gymnasium_cpu_env() -> None:
    from perpetua.adapters import GymnasiumLifetime
    from perpetua.agents import PPO, HostPPORunner, PPOConfig

    env = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=10))
    try:
        agent = PPO(
            action_size=2, config=PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
        )
        trainer = HostPPORunner(env, agent)
        state = trainer(trainer.init(seed=0))
        assert state.lifetime.state.t.to_int() == 8
        assert state.agent.gradient_steps.to_int() == 1
        assert bool(state.healthy)
    finally:
        env.close()


def test_failed_mutable_backend_step_invalidates_old_state() -> None:
    from perpetua.adapters import GymnasiumLifetime

    class FailingEnv(gym.Env[Any, Any]):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)
            self.calls = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
            super().reset(seed=seed)
            return np.asarray([0.0], dtype=np.float32), {}

        def step(self, action: Any) -> Any:
            self.calls += 1
            raise RuntimeError("Backend failed after mutation")

    backend = FailingEnv()
    env = GymnasiumLifetime(backend)
    try:
        state = pp.runtime.init_lifetime(env)
        with pytest.raises(RuntimeError, match="after mutation"):
            pp.runtime.tick(env, pp.policies.random, state)
        assert backend.calls == 1
        with pytest.raises(ValueError, match="stale"):
            pp.runtime.tick(env, pp.policies.random, state)
    finally:
        env.close()


def test_an_unseeded_first_reset_is_seed_zero_and_a_seed_changes_it() -> None:
    from perpetua.adapters.gymnasium import GymnasiumAdapter

    def first(seed: int | None) -> np.ndarray:
        view = GymnasiumAdapter(pp.make("ContinualGrid-v0", size=5))
        return view.reset(seed=seed)[0]

    np.testing.assert_array_equal(first(None), first(None))
    np.testing.assert_array_equal(first(None), first(0))
    assert not np.array_equal(first(None), first(7))
    with pytest.raises(ValueError, match="options"):
        GymnasiumAdapter(pp.make("ContinualGrid-v0")).reset(options={"x": 1})


def test_a_host_lift_keeps_the_environments_observation_shape() -> None:
    from perpetua.adapters import GymnasiumLifetime

    class Frames(gym.Env[np.ndarray, np.int64]):
        observation_space = gym.spaces.Box(0, 255, (3, 4, 2), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            return np.full((3, 4, 2), 200, dtype=np.uint8), {}

        def step(self, action: np.int64) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            return np.full((3, 4, 2), 7, dtype=np.uint8), 0.0, False, False, {}

    env = GymnasiumLifetime(Frames())
    try:
        assert env.observation_space.shape == (1, 3, 4, 2)
        assert env.observation_space.dtype == np.float32
        state, obs, _ = env.init(jax.random.key(0))
        assert obs.shape == (1, 3, 4, 2)
        assert obs.dtype == jnp.float32
        assert env.observation_space.contains(np.asarray(obs))
        _, obs, _, _ = env.step(state, jnp.zeros((1,), jnp.int32), jax.random.key(1))
        assert obs.shape == (1, 3, 4, 2)
        assert float(obs.max()) == 7.0
    finally:
        env.close()


def test_vmapping_a_host_lift_says_host_simulators_cannot_be_vmapped() -> None:
    from perpetua.adapters import GymnasiumLifetime

    for compiled in (False, True):
        env = GymnasiumLifetime(gym.make("CartPole-v1"), compiled=compiled)
        try:
            with pytest.raises(TypeError, match=r"Host simulators hold one mutable process"):
                jax.vmap(env.init)(jax.random.split(jax.random.key(0), 2))
        finally:
            env.close()


@pytest.mark.parametrize("seed", [2**32, -1])
def test_a_reset_seed_that_jax_would_alias_is_refused(seed: int) -> None:
    from perpetua.adapters.gymnasium import GymnasiumAdapter

    view = GymnasiumAdapter(pp.make("ContinualGrid-v0", size=3))
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 2\*\*32\)"):
        view.reset(seed=seed)
    view.reset(seed=2**32 - 1)
