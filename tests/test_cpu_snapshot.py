"""Snapshot-backed Gymnasium lifetimes must restore full CPU and learner state."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.adapters.snapshots import CartPoleCodec, CartPoleSnapshot
from perpetua.agents import PPO, HostPPORunner, PPOConfig


def test_the_locked_gymnasium_is_the_version_the_snapshot_code_accepts() -> None:
    # The extras allow a range; a relock outside the accepted version must fail
    # here by name rather than turn CartPole checkpointing off.
    env = gym.make("CartPole-v1")
    try:
        assert CartPoleCodec.supports(env), f"gymnasium {version('gymnasium')} is not accepted"
    finally:
        env.close()


def test_cartpole_snapshot_preserves_float64_rng_and_wrapper_state() -> None:
    first_raw = gym.make("CartPole-v1", max_episode_steps=3)
    second_raw = gym.make("CartPole-v1", max_episode_steps=3)
    first, second = GymnasiumLifetime(first_raw), GymnasiumLifetime(second_raw)
    first_base, second_base = cast(Any, first_raw.unwrapped), cast(Any, second_raw.unwrapped)
    try:
        pp.runtime.init_lifetime(first, seed=0)
        pp.runtime.init_lifetime(second, seed=999)
        # Include the PCG64 cached uint32 value, not only its 128-bit state.
        first_raw.unwrapped.np_random.integers(0, 2**32, dtype=np.uint32)
        physical = np.asarray([np.nextafter(0.1, 1.0), 0.2, 0.03, -0.2], dtype=np.float64)
        first_base.state = physical.copy()
        first_raw.step(0)
        snapshot = first.backend.snapshot()
        assert snapshot.physics.dtype == np.float64
        second.backend.restore(snapshot)
        np.testing.assert_array_equal(first_base.state, second_base.state)
        np.testing.assert_array_equal(
            first_raw.unwrapped.np_random.integers(0, 2**32, size=8, dtype=np.uint32),
            second_raw.unwrapped.np_random.integers(0, 2**32, size=8, dtype=np.uint32),
        )
        for action in [1, 0, 1, 1]:
            assert_same(first_raw.step(action), second_raw.step(action))
    finally:
        first.close()
        second.close()


def test_cpu_checkpoint_restores_into_fresh_backend_across_respawn_and_drift(
    tmp_path: Path,
) -> None:
    drift = pp.drift.Cyclic([(1.0,), (0.5,)], period=2)
    first = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3), drift=drift)
    second = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3), drift=drift)
    try:
        runner = pp.runtime.HostRunner(first, pp.policies.random, ticks=5)
        state = runner(pp.runtime.init_lifetime(first, seed=0, window_size=4))
        assert first.metadata.get("checkpointable") is True
        assert state.state.env.snapshot is not None
        path = tmp_path / "cpu"
        pp.checkpoint.save(path, state, metadata=runner.metadata)
        template = pp.runtime.init_lifetime(second, seed=999, window_size=4)
        restored = pp.checkpoint.restore(path, template, expected_metadata=second.metadata)
        assert_same(state, restored)
        for _ in range(8):
            state = pp.runtime.tick(first, pp.policies.random, state)
            restored = pp.runtime.tick(second, pp.policies.random, restored)
            assert_same(state, restored)
        assert state.state.t.to_int() == 13
    finally:
        first.close()
        second.close()


def test_host_ppo_checkpoint_restores_all_learning_state(tmp_path: Path) -> None:
    first = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    second = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    try:
        a = HostPPORunner(first, PPO(action_size=2, config=config))
        b = HostPPORunner(second, PPO(action_size=2, config=config))
        state = a(a.init(seed=0))
        path = tmp_path / "ppo-cpu"
        pp.checkpoint.save(path, state, metadata=a.metadata)
        restored = pp.checkpoint.restore(path, b.init(seed=999), expected_metadata=b.metadata)
        assert_same(a(state), b(restored))
    finally:
        first.close()
        second.close()


def test_snapshot_state_can_be_replayed_without_mutable_backend_drift() -> None:
    env = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    try:
        state = pp.runtime.init_lifetime(env, seed=0)
        left = pp.runtime.tick(env, pp.policies.random, state)
        right = pp.runtime.tick(env, pp.policies.random, state)
        assert_same(left, right)
    finally:
        env.close()


def test_unsupported_wrapper_is_not_marked_checkpointable(tmp_path: Path) -> None:
    raw = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    env = GymnasiumLifetime(raw)
    try:
        state = pp.runtime.init_lifetime(env)
        assert env.metadata.get("checkpointable") is False
        with pytest.raises(NotImplementedError, match="checkpoint"):
            pp.checkpoint.save(tmp_path / "unsupported", state, metadata=env.metadata)
    finally:
        env.close()


def test_snapshot_version_and_configuration_changes_are_rejected(tmp_path: Path) -> None:
    first = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
    second = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=4))
    try:
        state = pp.runtime.init_lifetime(first)
        snapshot = first.backend.snapshot()
        with pytest.raises(ValueError, match="version"):
            first.backend.restore(snapshot._replace(version=np.asarray(999, dtype=np.uint32)))
        with pytest.raises(ValueError, match="configuration"):
            second.backend.restore(snapshot)
        path = tmp_path / "cpu"
        pp.checkpoint.save(path, state, metadata=first.metadata)
        template = pp.runtime.init_lifetime(second)
        with pytest.raises(ValueError, match="configuration"):
            pp.checkpoint.restore(path, template, expected_metadata=second.metadata)
    finally:
        first.close()
        second.close()


def test_malformed_snapshot_is_rejected_without_mutating_backend() -> None:
    env = GymnasiumLifetime(gym.make("CartPole-v1"))
    try:
        pp.runtime.init_lifetime(env)
        snapshot = env.backend.snapshot()
        assert isinstance(snapshot, CartPoleSnapshot)
        invalid = snapshot._replace(rng=snapshot.rng.copy())
        invalid.rng[8] = 2
        with pytest.raises(ValueError, match="flags"):
            env.backend.restore(invalid)
        assert_same(env.backend.snapshot(), snapshot)
        invalid = snapshot._replace(wrappers=snapshot.wrappers.copy())
        invalid.wrappers[1, 0] = 2
        with pytest.raises(ValueError, match="flags"):
            env.backend.restore(invalid)
        assert_same(env.backend.snapshot(), snapshot)
    finally:
        env.close()


def test_direct_snapshot_restore_rejects_reordered_wrappers() -> None:
    first_raw = gym.make("CartPole-v1")
    second_raw = gym.wrappers.PassiveEnvChecker(
        gym.wrappers.OrderEnforcing(
            gym.wrappers.TimeLimit(gym.make("CartPole-v1").unwrapped, max_episode_steps=500)
        )
    )
    first, second = GymnasiumLifetime(first_raw), GymnasiumLifetime(second_raw)
    try:
        pp.runtime.init_lifetime(first)
        pp.runtime.init_lifetime(second)
        with pytest.raises(ValueError, match="configuration"):
            second.backend.restore(first.backend.snapshot())
    finally:
        first.close()
        second.close()


def test_unsupported_rng_is_not_silently_serialized() -> None:
    raw = gym.make("CartPole-v1")
    env = GymnasiumLifetime(raw)
    try:
        pp.runtime.init_lifetime(env)
        cast(Any, raw.unwrapped)._np_random = np.random.Generator(np.random.MT19937(0))
        with pytest.raises(NotImplementedError, match="PCG64"):
            env.backend.snapshot()
    finally:
        env.close()


def test_checkpoint_continues_through_physical_failure(tmp_path: Path) -> None:
    raw = gym.make("CartPole-v1")
    first = GymnasiumLifetime(raw)
    second = GymnasiumLifetime(gym.make("CartPole-v1"))
    try:
        state = pp.runtime.init_lifetime(first, seed=0)
        physical = np.asarray([2.5, 0.0, 0.0, 0.0], dtype=np.float64)
        cast(Any, raw.unwrapped).state = physical.copy()
        obs = jnp.asarray(physical, dtype=jnp.float32)[None, :]
        world = state.state.env._replace(snapshot=first.backend.snapshot(), obs=obs)
        state = state._replace(state=state.state._replace(env=world), obs=obs)
        path = tmp_path / "physical-failure"
        pp.checkpoint.save(path, state, metadata=first.metadata)
        template = pp.runtime.init_lifetime(second, seed=999)
        restored = pp.checkpoint.restore(path, template, expected_metadata=second.metadata)
        actual = pp.runtime.tick(first, pp.policies.random, state)
        expected = pp.runtime.tick(second, pp.policies.random, restored)
        assert bool(actual.info["respawn"][0])
        assert actual.state.t.to_int() == 1
        assert_same(actual, expected)
        assert_same(
            pp.runtime.tick(first, pp.policies.random, actual),
            pp.runtime.tick(second, pp.policies.random, expected),
        )
    finally:
        first.close()
        second.close()


def test_snapshot_recovers_after_a_real_transition_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    env = GymnasiumLifetime(gym.make("CartPole-v1"))
    try:
        state = pp.runtime.init_lifetime(env, seed=0)
        expected = pp.runtime.tick(env, pp.policies.random, state)
        original_step = env.backend.step

        def fail_after_step(action: np.ndarray[Any, Any], reset_seed: int) -> None:
            original_step(action, reset_seed)
            raise RuntimeError("Failure after a real CartPole transition")

        with monkeypatch.context() as patch:
            patch.setattr(env.backend, "step", fail_after_step)
            with pytest.raises(RuntimeError, match="real CartPole"):
                pp.runtime.tick(env, pp.policies.random, state)
        assert_same(pp.runtime.tick(env, pp.policies.random, state), expected)
    finally:
        env.close()
