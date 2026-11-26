"""Real EnvPool dynamics and complete snapshot continuation, including pending resets."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters.snapshots import EnvPoolSnapshot
from perpetua.agents import PPO, HostPPORunner, PPOConfig

pytestmark = pytest.mark.skipif(
    find_spec("perpetua_envpool_snapshot") is None or sys.platform != "linux",
    reason="Requires the optional Linux native/envpool_snapshot binding",
)


@pytest.mark.parametrize("num_threads", [1, 2, 4])
def test_native_binding_matches_real_envpool_dynamics(num_threads: int) -> None:
    envpool = cast(Any, import_module("envpool"))
    native = cast(Any, import_module("perpetua_envpool_snapshot"))

    first: Any = envpool.make_gymnasium(
        "CartPole-v1",
        num_envs=4,
        batch_size=4,
        num_threads=num_threads,
        max_episode_steps=3,
        seed=0,
    )
    second: Any = native.CartPolePool(4, num_threads, 3, 0)
    try:
        a, b = first.reset(), second.reset()
        np.testing.assert_array_equal(a[0], b[0])
        for i in range(64):
            action = np.asarray([(i + j) % 2 for j in range(4)], dtype=np.int32)
            a, b = first.step(action), second.step(action)
            for x, y in zip(a[:4], b[:4], strict=True):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(a[4]["env_id"], b[4]["env_id"])
    finally:
        first.close()
        second.close()


def test_snapshot_preserves_pending_resets_drift_metrics_and_replay(tmp_path: Path) -> None:
    drift = pp.drift.Cyclic([(1.0,), (0.5,)], period=2)
    first = pp.make(
        "envpool/CartPole-v1",
        num_envs=4,
        num_threads=2,
        episode_steps=3,
        snapshot=True,
        drift=drift,
    )
    second = pp.make(
        "envpool/CartPole-v1",
        num_envs=4,
        num_threads=2,
        episode_steps=3,
        snapshot=True,
        drift=drift,
    )
    try:
        runner = pp.runtime.HostRunner(first, pp.policies.random, ticks=3)
        state = runner(pp.runtime.init_lifetime(first, seed=0, window_size=4))
        assert first.metadata.get("checkpointable") is True
        assert not bool(jnp.any(state.info["active_mask"]))
        path = tmp_path / "envpool"
        pp.checkpoint.save(path, state, metadata=first.metadata)
        restored = pp.checkpoint.restore(
            path,
            pp.runtime.init_lifetime(second, seed=999, window_size=4),
            expected_metadata=second.metadata,
        )
        for _ in range(16):
            a = pp.runtime.tick(first, pp.policies.random, state)
            assert_same(a, pp.runtime.tick(first, pp.policies.random, state))
            b = pp.runtime.tick(second, pp.policies.random, restored)
            assert_same(a, b)
            state, restored = a, b
        assert state.state.t.to_int() == 19
    finally:
        first.close()
        second.close()


def test_snapshot_preserves_all_ppo_state(tmp_path: Path) -> None:
    first = pp.make(
        "envpool/CartPole-v1", num_envs=4, num_threads=2, episode_steps=3, snapshot=True
    )
    second = pp.make(
        "envpool/CartPole-v1", num_envs=4, num_threads=2, episode_steps=3, snapshot=True
    )
    config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    try:
        a, b = (
            HostPPORunner(first, PPO(action_size=2, config=config)),
            HostPPORunner(second, PPO(action_size=2, config=config)),
        )
        state = a(a.init(seed=0))
        path = tmp_path / "ppo-envpool"
        pp.checkpoint.save(path, state, metadata=a.metadata)
        restored = pp.checkpoint.restore(path, b.init(seed=999), expected_metadata=b.metadata)
        assert_same(a(state), b(restored))
    finally:
        first.close()
        second.close()


def test_native_restore_validates_every_slot_before_mutation() -> None:
    native = cast(Any, import_module("perpetua_envpool_snapshot"))

    pool: Any = native.CartPolePool(4, 2, 3, 0)
    try:
        with pytest.raises(RuntimeError, match="Initialize"):
            pool.snapshot()
        pool.reset()
        state = pool.snapshot()
        invalid = [value.copy() for value in state]
        invalid[2][-1, -1] = 999
        with pytest.raises(ValueError, match="MT19937"):
            pool.restore(*invalid)
        assert_same(state, pool.snapshot())
        invalid = [value.copy() for value in state]
        invalid[0][0] = 999
        with pytest.raises(ValueError, match="version"):
            pool.restore(*invalid)
        assert_same(state, pool.snapshot())
        invalid = [value.copy() for value in state]
        invalid[1] = invalid[1].astype(np.float32)
        with pytest.raises(ValueError, match="dtype"):
            pool.restore(*invalid)
        assert_same(state, pool.snapshot())
        invalid = [value.copy() for value in state]
        physical = invalid[1]
        unaligned = np.ndarray(
            physical.shape, dtype=np.float64, buffer=bytearray(physical.nbytes + 1), offset=1
        )
        unaligned[:] = physical
        invalid[1] = unaligned
        with pytest.raises(ValueError, match="layout"):
            pool.restore(*invalid)
        assert_same(state, pool.snapshot())
    finally:
        pool.close()


def test_checkpoint_preserves_physical_failure_and_reset_only_credit(tmp_path: Path) -> None:
    first = pp.make("envpool/CartPole-v1", num_envs=4, snapshot=True)
    second = pp.make("envpool/CartPole-v1", num_envs=4, snapshot=True)
    try:
        state, _, _ = first.init(jax.random.key(0))
        assert isinstance(state.env.snapshot, EnvPoolSnapshot)
        physics = state.env.snapshot.physics.copy()
        physics[0] = [2.5, 0.0, 0.0, 0.0]
        world = state.env._replace(snapshot=state.env.snapshot._replace(physics=physics))
        state = state._replace(env=world)
        path = tmp_path / "failure"
        pp.checkpoint.save(path, state, metadata=first.metadata)
        template, _, _ = second.init(jax.random.key(999))
        restored = pp.checkpoint.restore(path, template, expected_metadata=second.metadata)
        action = np.zeros(4, dtype=np.int32)
        a = first.step(state, action, jax.random.key(1))
        b = second.step(restored, action, jax.random.key(1))
        assert_same(a, b)
        assert not bool(a[0].env.active[0])
        np.testing.assert_array_equal(a[2], np.ones(4))
        a = first.step(a[0], action, jax.random.key(2))
        b = second.step(b[0], action, jax.random.key(2))
        assert_same(a, b)
        assert bool(a[0].env.active[0])
        assert bool(a[0].env.respawn[0])
        assert float(a[2][0]) == 0.0
    finally:
        first.close()
        second.close()


def test_concurrent_snapshot_refuses_inflight_work_without_deadlock() -> None:
    native = cast(Any, import_module("perpetua_envpool_snapshot"))
    pool = native.CartPolePool(128, 2, 3, 0)
    barrier = threading.Barrier(2)
    try:
        pool.reset()
        action = np.zeros(128, dtype=np.int32)

        def advance() -> None:
            barrier.wait(timeout=10)
            for _ in range(512):
                pool.step(action)

        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(advance)
            barrier.wait(timeout=10)
            refused = False
            for _ in range(512):
                try:
                    pool.snapshot()
                except RuntimeError as error:
                    assert "in flight" in str(error)  # noqa: PT017 - polled until refused
                    refused = True
                    break
            job.result(timeout=30)
        assert refused
        snapshot = pool.snapshot()
        pool.restore(*snapshot)
        assert_same(snapshot, pool.snapshot())
    finally:
        pool.close()
