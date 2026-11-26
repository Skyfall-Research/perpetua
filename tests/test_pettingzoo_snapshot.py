"""Supported real MPE2 lifetimes must snapshot physics, RNG and all AEC bookkeeping."""

# Third-party/JAX transformation interfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest
from helpers import assert_same
from pettingzoo import EnvSpec, ParallelEnv, make  # pyright: ignore[reportMissingTypeStubs]
from pettingzoo.utils.wrappers import BaseParallelWrapper  # pyright: ignore[reportMissingTypeStubs]

import perpetua as pp
from perpetua.adapters import PettingZooLifetime
from perpetua.adapters.mpe2_snapshot import MPE2Codec
from perpetua.adapters.snapshots import MPE2Snapshot
from perpetua.agents import IPPO, ComputePPORunner, HostPPORunner, PPOConfig
from perpetua.compute import TimingTrace


def spread(*, cycles: int = 3, **kwargs: Any) -> ParallelEnv[str, Any, Any]:
    spec = EnvSpec(
        id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
    )
    return cast(
        ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=cycles, **kwargs)
    )


def test_the_locked_mpe2_and_pettingzoo_are_the_versions_the_snapshot_code_accepts() -> None:
    # The extras allow ranges; a relock outside the accepted versions must fail
    # here by name rather than turn MPE2 checkpointing off.
    env = spread()
    try:
        installed = f"mpe2 {version('mpe2')} with pettingzoo {version('pettingzoo')}"
        assert MPE2Codec.supports(env), f"{installed} is not accepted"
    finally:
        env.close()


def test_mpe_snapshot_preserves_float64_cached_rng_and_partial_aec_actions() -> None:
    first_raw, second_raw = spread(), spread()
    first = PettingZooLifetime(first_raw, snapshot=True)
    second = PettingZooLifetime(second_raw, snapshot=True)
    try:
        pp.runtime.init_lifetime(first, seed=0)
        pp.runtime.init_lifetime(second, seed=999)
        base = cast(Any, first_raw.unwrapped)
        base.world.agents[0].state.p_pos[0] = np.nextafter(0.1, 1.0)
        base.np_random.integers(0, 2**32, dtype=np.uint32)
        # Capture between AEC actions, then finish the same physical cycle.
        cast(Any, first_raw).aec_env.step(2)
        snapshot = first.backend.snapshot()
        assert snapshot.physics.dtype == np.float64
        second.backend.restore(snapshot)
        other = cast(Any, second_raw.unwrapped)
        np.testing.assert_array_equal(
            base.world.agents[0].state.p_pos, other.world.agents[0].state.p_pos
        )
        np.testing.assert_array_equal(
            base.np_random.integers(0, 2**32, size=8, dtype=np.uint32),
            other.np_random.integers(0, 2**32, size=8, dtype=np.uint32),
        )
        for action in [3, 4]:
            cast(Any, first_raw).aec_env.step(action)
            cast(Any, second_raw).aec_env.step(action)
        for name in first_raw.possible_agents:
            np.testing.assert_array_equal(base.observe(name), other.observe(name))
        assert_same(first.backend.snapshot(), second.backend.snapshot())
    finally:
        first.close()
        second.close()


def test_full_lifetime_checkpoint_rebinds_and_continues_across_respawn_drift(
    tmp_path: Path,
) -> None:
    drift = pp.drift.Cyclic([(1.0,), (0.5,)], period=2)
    first = PettingZooLifetime(spread(), snapshot=True, drift=drift)
    second = PettingZooLifetime(spread(), snapshot=True, drift=drift)
    try:
        runner = pp.runtime.HostRunner(first, pp.policies.random, ticks=5)
        state = runner(pp.runtime.init_lifetime(first, seed=0, window_size=4))
        assert first.metadata.get("checkpointable") is True
        assert state.state.env.owner is None
        path = tmp_path / "mpe"
        pp.checkpoint.save(path, state, metadata=first.metadata)
        template = pp.runtime.init_lifetime(second, seed=999, window_size=4)
        restored = pp.checkpoint.restore(path, template, expected_metadata=second.metadata)
        for _ in range(10):
            a = pp.runtime.tick(first, pp.policies.random, state)
            assert_same(a, pp.runtime.tick(first, pp.policies.random, state))
            b = pp.runtime.tick(second, pp.policies.random, restored)
            assert_same(a, b)
            state, restored = a, b
        assert state.state.t.to_int() == 15
        assert set(state.info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("timed", [False, True])
def test_ippo_snapshot_preserves_optimizer_keys_and_pending_timed_buffers(
    tmp_path: Path,
    timed: bool,
) -> None:
    first = PettingZooLifetime(spread(), snapshot=True)
    second = PettingZooLifetime(spread(), snapshot=True)
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    try:
        trace = TimingTrace.from_ticks([1, 2, 1], quantum_ns=1_000_000)
        path = tmp_path / "training"
        if timed:
            a = ComputePPORunner(first, IPPO(action_size=5, config=config), decisions=3)
            b = ComputePPORunner(second, IPPO(action_size=5, config=config), decisions=3)
            state = a(a.init(seed=0), trace)
            pp.checkpoint.save(path, state, metadata=a.metadata)
            restored = pp.checkpoint.restore(path, b.init(seed=999), expected_metadata=b.metadata)
            assert_same(a(state, trace), b(restored, trace))
        else:
            c = HostPPORunner(first, IPPO(action_size=5, config=config))
            d = HostPPORunner(second, IPPO(action_size=5, config=config))
            trained = c(c.init(seed=0))
            pp.checkpoint.save(path, trained, metadata=c.metadata)
            loaded = pp.checkpoint.restore(path, d.init(seed=999), expected_metadata=d.metadata)
            assert_same(c(trained), d(loaded))
    finally:
        first.close()
        second.close()


def test_snapshot_validation_is_atomic_and_wrong_configuration_is_refused(tmp_path: Path) -> None:
    first = PettingZooLifetime(spread(), snapshot=True)
    other = PettingZooLifetime(spread(cycles=4), snapshot=True)
    try:
        state = pp.runtime.init_lifetime(first)
        pp.runtime.init_lifetime(other)
        snapshot = first.backend.snapshot()
        assert isinstance(snapshot, MPE2Snapshot)
        invalid = snapshot._replace(physics=snapshot.physics.copy())
        invalid.physics[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            first.backend.restore(invalid)
        assert_same(snapshot, first.backend.snapshot())
        with pytest.raises(ValueError, match="configuration"):
            other.backend.restore(snapshot)
        with pytest.raises(ValueError, match="version"):
            first.backend.restore(snapshot._replace(version=np.asarray(99, dtype=np.uint32)))
        invalid = snapshot._replace(rng=snapshot.rng.copy())
        invalid.rng[8] = 2
        with pytest.raises(ValueError, match="cache"):
            first.backend.restore(invalid)
        assert_same(snapshot, first.backend.snapshot())
        invalid = snapshot._replace(present=snapshot.present.copy())
        invalid.present[0, 0] = False
        with pytest.raises(ValueError, match="membership"):
            first.backend.restore(invalid)
        assert_same(snapshot, first.backend.snapshot())
        invalid = snapshot._replace(counters=snapshot.counters.copy())
        invalid.counters[5] = 1
        with pytest.raises(ValueError, match="counters"):
            first.backend.restore(invalid)
        assert_same(snapshot, first.backend.snapshot())
        pp.checkpoint.save(tmp_path / "checkpoint", state, metadata=first.metadata)
    finally:
        first.close()
        other.close()


def test_static_physics_changes_are_not_silently_mislabeled_as_same_configuration() -> None:
    raw = spread()
    env = PettingZooLifetime(raw, snapshot=True)
    try:
        pp.runtime.init_lifetime(env)
        snapshot = env.backend.snapshot()
        cast(Any, raw.unwrapped).world.dt = 0.2
        with pytest.raises(ValueError, match="configuration"):
            env.backend.snapshot()
        with pytest.raises(ValueError, match="configuration"):
            env.backend.restore(snapshot)
    finally:
        env.close()


def test_unknown_wrappers_and_global_rng_noise_are_not_checkpointable() -> None:
    wrapped = BaseParallelWrapper(spread())
    with pytest.raises(ValueError, match="snapshot"):
        PettingZooLifetime(wrapped, snapshot=True)
    wrapped.close()
    noisy = spread()
    cast(Any, noisy.unwrapped).world.agents[0].u_noise = 0.1
    with pytest.raises(ValueError, match="snapshot"):
        PettingZooLifetime(noisy, snapshot=True)
    noisy.close()


def test_success_termination_and_curriculum_state_restore_inside_lifetime(tmp_path: Path) -> None:
    first_raw = spread(curriculum=True, terminate_on_success=True)
    second_raw = spread(curriculum=True, terminate_on_success=True)
    first = PettingZooLifetime(first_raw, snapshot=True)
    second = PettingZooLifetime(second_raw, snapshot=True)
    try:
        state = pp.runtime.init_lifetime(first, seed=0)
        base = cast(Any, first_raw.unwrapped)
        base.set_curriculum_stage(1)
        actual_stage = base.curriculum_stage
        for agent, landmark in zip(base.world.agents, base.world.landmarks, strict=True):
            agent.state.p_pos = landmark.state.p_pos.copy()
            agent.state.p_vel = np.zeros(2, dtype=np.float64)
        state = state._replace(
            state=state.state._replace(
                env=state.state.env._replace(snapshot=first.backend.snapshot())
            )
        )
        path = tmp_path / "success"
        pp.checkpoint.save(path, state, metadata=first.metadata)
        restored = pp.checkpoint.restore(
            path, pp.runtime.init_lifetime(second, seed=999), expected_metadata=second.metadata
        )
        a = first.step(state.state, np.zeros(3, dtype=np.int32), jax.random.key(1))
        b = second.step(restored.state, np.zeros(3, dtype=np.int32), jax.random.key(1))
        assert_same(a, b)
        assert np.all(np.asarray(a[3]["respawn"]))
        assert cast(Any, second_raw.unwrapped).curriculum_stage == actual_stage
    finally:
        first.close()
        second.close()
