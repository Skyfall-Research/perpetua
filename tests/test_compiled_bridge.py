"""The compiled CPU bridge: host simulators inside compiled lifetime programs."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false

import sys
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import PPO, HostPPORunner, PPORunner


def cartpole(compiled: bool, **kwargs: Any) -> GymnasiumLifetime:
    return GymnasiumLifetime(gym.make("CartPole-v1"), compiled=compiled, **kwargs)


def test_compiled_runner_reproduces_the_host_runner_exactly() -> None:
    results: list[Any] = []
    for compiled in (False, True):
        env = cartpole(compiled, drift=pp.drift.Cyclic([(1.0,), (2.0,)], 50))
        try:
            runner = (pp.runtime.Runner if compiled else pp.runtime.HostRunner)(
                env, pp.policies.random, ticks=200
            )
            carry = runner(pp.runtime.init_lifetime(env, seed=3, window_size=16))
            results.append(carry)
        finally:
            env.close()
    host, compiled_carry = results
    np.testing.assert_array_equal(np.asarray(host.obs), np.asarray(compiled_carry.obs))
    np.testing.assert_array_equal(
        np.asarray(host.metrics.total), np.asarray(compiled_carry.metrics.total)
    )
    assert compiled_carry.state.t.to_int() == 200


def test_compiled_ppo_matches_host_ppo_bit_for_bit() -> None:
    leaves: list[list[np.ndarray[Any, Any]]] = []
    for compiled in (False, True):
        env = cartpole(compiled)
        try:
            runner = (PPORunner if compiled else HostPPORunner)(env, PPO(action_size=2))
            state = runner.init(seed=0)
            for _ in range(3):
                state = runner(state)
            leaves.append([np.asarray(leaf) for leaf in jax.tree.leaves(state.agent.params)])
        finally:
            env.close()
    for host, compiled_leaf in zip(*leaves, strict=True):
        np.testing.assert_array_equal(host, compiled_leaf)


def test_states_are_single_use_and_not_checkpointable(tmp_path: Path) -> None:
    env = cartpole(True)
    try:
        assert env.metadata.get("execution") == "jax"
        assert env.metadata.get("checkpointable") is False
        state, _, _ = env.init(jax.random.key(0))
        stepped, *_ = env.step(state, jnp.asarray([0]), jax.random.key(1))
        with pytest.raises(RuntimeError, match="stale or forked"):
            jax.block_until_ready(env.step(state, jnp.asarray([1]), jax.random.key(2)))
        del stepped
    finally:
        env.close()
    other = cartpole(True)
    try:
        carry = pp.runtime.init_lifetime(other, seed=0)
        with pytest.raises(NotImplementedError):
            pp.checkpoint.save(tmp_path / "bridge", carry, metadata=other.metadata)
    finally:
        other.close()


@pytest.mark.skipif(sys.platform != "linux", reason="EnvPool is Linux-only.")
def test_envpool_batches_run_compiled() -> None:
    env: Any = pp.make("envpool/CartPole-v1", num_envs=8, num_threads=2, compiled=True)
    try:
        carry = pp.runtime.Runner(env, pp.policies.random, ticks=64)(
            pp.runtime.init_lifetime(env, seed=0)
        )
        assert carry.obs.shape == (8, 4)
        assert bool(carry.metrics.healthy)
        with pytest.raises(ValueError, match="single-use"):
            pp.make("envpool/CartPole-v1", snapshot=True, compiled=True)
    finally:
        env.close()


def compare_compiled(make: Any, ticks: int, action: Any) -> None:
    """The same lifetime through the eager host loop and the compiled bridge."""
    carries: list[Any] = []
    for compiled in (False, True):
        env = make(compiled)
        try:
            runner = (pp.runtime.Runner if compiled else pp.runtime.HostRunner)(
                env, action, ticks=ticks
            )
            carries.append(runner(pp.runtime.init_lifetime(env, seed=1, window_size=8)))
        finally:
            env.close()
    eager, compiled_carry = carries
    np.testing.assert_array_equal(np.asarray(eager.obs), np.asarray(compiled_carry.obs))
    np.testing.assert_array_equal(
        np.asarray(eager.metrics.total), np.asarray(compiled_carry.metrics.total)
    )


def test_metaworld_runs_compiled_and_matches_eager() -> None:
    pytest.importorskip("metaworld")
    from perpetua.adapters.host import MetaWorldLifetime

    def zero(obs: jax.Array, info: Any, key: jax.Array) -> jax.Array:
        del obs, info, key
        return jnp.full((1, 4), 0.1, dtype=jnp.float32)

    compare_compiled(
        lambda compiled: MetaWorldLifetime(env_name="reach-v3", compiled=compiled), 30, zero
    )


def test_agarcl_runs_compiled_and_matches_eager() -> None:
    from perpetua.adapters import agarcl

    if not agarcl.available():
        pytest.skip("AgarCL runtime not installed (python -m perpetua.adapters.agarcl_setup).")

    def move(obs: jax.Array, info: Any, key: jax.Array) -> jax.Array:
        del obs, info
        return jax.random.uniform(key, (1, 3), minval=0.0, maxval=1.0)

    compare_compiled(
        lambda compiled: pp.make("agarcl/agario-grid-v0", grid_size=16, compiled=compiled),
        20,
        move,
    )
    with pytest.raises(ValueError, match="single-use"):
        pp.make("agarcl/agario-grid-v0", replay_capacity=8, compiled=True)


def test_coom_runs_compiled_and_matches_eager() -> None:
    from perpetua.adapters import coom

    if not coom.available():
        pytest.skip("COOM runtime not installed (python -m perpetua.adapters.coom_setup).")
    compare_compiled(
        lambda compiled: pp.make("coom/COOM-v0", compiled=compiled), 20, pp.policies.random
    )


class _Counter:
    """A fake simulator whose observation is the number of transitions it applied."""

    num_slots = 1
    uses_reset_seed = False
    checkpointable = False
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
    configuration: dict[str, Any] = {"backend": "fake"}

    def __init__(self, fail_at: int | None = None) -> None:
        self.applied = 0
        self.fail_at = fail_at

    def init(self, seed: int) -> Any:
        del seed
        return np.zeros((1, 1), np.float32)

    def step(self, action: Any, reset_seed: int) -> Any:
        from perpetua.adapters.host import BackendResult

        del action, reset_seed
        self.applied += 1  # the simulator mutates, then may fail
        if self.applied == self.fail_at:
            raise RuntimeError("native step failed after mutating")
        observation = np.full((1, 1), self.applied, np.float32)
        return BackendResult(
            observation, np.ones(1, np.float32), np.ones(1, bool), np.zeros(1, bool)
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("compiled", [False, True])
def test_a_state_is_not_accepted_again_after_its_step_failed(compiled: bool) -> None:
    from perpetua.adapters.host import CPULifetime

    env = CPULifetime(cast(Any, _Counter(fail_at=2)), env_id="fake", compiled=compiled)
    action = jnp.asarray([0], jnp.int32)
    state, _, _ = env.init(jax.random.key(0))
    state, *_ = env.step(state, action, jax.random.key(1))
    with pytest.raises(Exception, match="native step failed"):
        jax.block_until_ready(env.step(state, action, jax.random.key(2)))
    with pytest.raises((ValueError, RuntimeError), match="stale"):
        jax.block_until_ready(env.step(state, action, jax.random.key(2)))


def test_a_regret_reference_on_a_compiled_bridge_is_refused_before_any_step() -> None:
    from perpetua.adapters.host import CPULifetime

    counter = _Counter()
    env: Any = CPULifetime(cast(Any, counter), env_id="fake", compiled=True)
    assert env.metadata["forkable"] is False
    with pytest.raises(ValueError, match=r"Regret needs a shadow step.*compiled=True"):
        pp.runtime.Runner(env, pp.policies.random, ticks=2, reference=pp.policies.random)
    assert counter.applied == 0
    pp.runtime.Runner(env, pp.policies.random, ticks=2)


def test_compiled_generation_counter_survives_the_uint32_wrap() -> None:
    from perpetua.adapters.host import CPULifetime

    env: Any = CPULifetime(cast(Any, _Counter()), env_id="fake", compiled=True)
    action = jnp.asarray([0], jnp.int32)
    state, _, _ = env.init(jax.random.key(0))
    env.env._steps = 2**32 - 1
    state = state._replace(env=state.env._replace(generation=jnp.uint32(2**32 - 1)))
    state, *_ = env.step(state, action, jax.random.key(1))
    assert int(state.env.generation) == 0
    state, obs, _, _ = env.step(state, action, jax.random.key(2))  # refused before the fix
    assert float(obs[0, 0]) == 2.0


@pytest.mark.parametrize("unroll", [None, 4])
def test_open_loop_compiled_steps_run_in_order(unroll: int | None) -> None:
    from perpetua.adapters.host import CPULifetime

    action = jnp.asarray([0], jnp.int32)
    keys = jax.random.split(jax.random.key(1), 8)

    for _ in range(5):
        env: Any = CPULifetime(cast(Any, _Counter()), env_id="fake", compiled=True)
        state, _, _ = env.init(jax.random.key(0))

        def tick(state: Any, key: jax.Array, env: Any = env) -> tuple[Any, jax.Array]:
            state, obs, _, _ = env.step(state, action, key)
            return state, obs[0, 0]

        def lifetime(state: Any, tick: Any = tick) -> tuple[Any, jax.Array]:
            if unroll is not None:
                return jax.lax.scan(tick, state, keys, unroll=unroll)
            seen: list[jax.Array] = []
            for index in range(8):
                state, obs = tick(state, keys[index])
                seen.append(obs)
            return state, jnp.stack(seen)

        state, seen = cast(Any, jax.jit(lifetime))(state)
        np.testing.assert_array_equal(np.asarray(seen), np.arange(1, 9, dtype=np.float32))
        assert int(state.env.generation) == 8


class _ScriptedPool:
    """An EnvPool stand-in: slot 0's game ends at step 2 and resets at step 3."""

    def __init__(self) -> None:
        self.steps = 0
        self.actions: list[Any] = []

    def step(self, action: Any) -> Any:
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        frame = 0.0 if self.steps == 3 else float(self.steps)  # 0.0 is the reset frame
        values = np.asarray([frame, float(self.steps)], dtype=np.float32)
        obs = np.broadcast_to(values[:, None, None, None], (2, 4, 84, 84)).copy()
        terminated = np.asarray([self.steps == 2, False])
        info = {"env_id": np.arange(2)}
        return obs, np.ones(2, np.float32), terminated, np.zeros(2, bool), info


def test_envpool_atari_reports_the_respawn_with_the_reset_frame() -> None:
    from perpetua.adapters.host import _EnvPoolAtariBackend  # pyright: ignore[reportPrivateUsage]

    backend: Any = object.__new__(_EnvPoolAtariBackend)
    backend.num_slots = 2
    backend.pool = _ScriptedPool()
    backend.pending = np.zeros(2, dtype=np.bool_)
    action = np.zeros(2, dtype=np.int32)
    playing, ended, reset = (backend.step(action, 0) for _ in range(3))
    assert playing.active.tolist() == [True, True]
    assert playing.respawn.tolist() == [False, False]
    # The game ends: EnvPool ignores this slot's next action, so the slot is
    # inactive now, and the respawn is reported when the reset frame arrives.
    assert ended.active.tolist() == [False, True]
    assert ended.respawn.tolist() == [False, False]
    assert reset.active.tolist() == [True, True]
    assert reset.respawn.tolist() == [True, False]
    assert float(reset.obs[0, 0, 0, 0]) == 0.0


class _FramePool:
    """An EnvPool stand-in that returns slots out of order, frames first: (slots, 4, 84, 84)."""

    def __init__(self) -> None:
        counts = np.arange(2 * 4 * 84 * 84).reshape(2, 4, 84, 84)
        self.frames = (counts % 256).astype(np.uint8)

    def reset(self) -> Any:
        return self.frames[::-1], {"env_id": np.asarray([1, 0])}

    def step(self, action: Any) -> Any:
        del action
        info = {"env_id": np.asarray([1, 0])}
        return self.frames[::-1], np.asarray([1.0, 2.0]), np.zeros(2, bool), np.zeros(2, bool), info


def test_envpool_atari_frames_are_channel_last_in_slot_order() -> None:
    from perpetua.adapters.host import _EnvPoolAtariBackend  # pyright: ignore[reportPrivateUsage]

    # EnvPool itself cannot run off Linux: this feeds frames of the shape and
    # dtype EnvPool's Atari pool returns, (slots, 4, 84, 84) uint8, to the adapter.
    pool = _FramePool()
    backend: Any = object.__new__(_EnvPoolAtariBackend)
    backend.num_slots = 2
    backend.pool = pool
    backend.module = type(
        "FakeEnvPool", (), {"make_gymnasium": staticmethod(lambda *args, **kwargs: pool)}
    )
    backend.task = "Breakout-v5"
    backend.num_threads = 1
    backend.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8)
    expected = np.transpose(pool.frames, (0, 2, 3, 1)).astype(np.float32)
    first = backend.init(0)
    assert first.shape == (2, 84, 84, 4)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, expected)
    result = backend.step(np.zeros(2, dtype=np.int32), 0)
    np.testing.assert_array_equal(result.obs, expected)
    np.testing.assert_array_equal(result.reward, [2.0, 1.0])
    assert tuple(backend.observation_space.shape) == first.shape[1:]


class _HalfStarted(_Counter):
    """Two slots, the second of which has no observation yet (as EnvPool overlap=True)."""

    num_slots = 2
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)

    def init(self, seed: int) -> Any:
        del seed
        self.initial_active = np.asarray([True, False])
        return np.zeros((2, 1), np.float32)


def test_compiled_bridge_honours_the_backends_initial_active_slots() -> None:
    from perpetua.adapters.host import CPULifetime

    masks: list[Any] = []
    for compiled in (False, True):
        env = CPULifetime(cast(Any, _HalfStarted()), env_id="fake", compiled=compiled)
        _, _, info = env.init(jax.random.key(0))
        masks.append(
            (np.asarray(info["active_mask"]).tolist(), np.asarray(info["action_mask"]).tolist())
        )
    assert masks[0][0] == [True, False]
    assert masks[0] == masks[1]


@pytest.mark.parametrize("kind", ["atari", "cartpole", "native"])
def test_envpool_seed_leaves_room_for_the_per_slot_offset(kind: str) -> None:
    from perpetua.adapters.host import (
        _EnvPoolAtariBackend,  # pyright: ignore[reportPrivateUsage]
        _EnvPoolBackend,  # pyright: ignore[reportPrivateUsage]
    )

    seeds: list[int] = []

    class Pool:
        def reset(self) -> Any:
            return np.zeros((4, 4, 84, 84), np.uint8), {"env_id": np.arange(4)}

    def make_gymnasium(*args: Any, seed: int, **kwargs: Any) -> Pool:
        del args, kwargs
        seeds.append(seed)
        return Pool()

    def native_pool(num_envs: int, num_threads: int, episode_steps: int, seed: int) -> Pool:
        del num_envs, num_threads, episode_steps
        seeds.append(seed)
        return Pool()

    backend: Any = object.__new__(_EnvPoolBackend if kind != "atari" else _EnvPoolAtariBackend)
    backend.num_slots = 4
    backend.num_threads = 1
    backend.episode_steps = 10
    backend.task = "Breakout-v5"
    backend.overlap = False
    backend.checkpointable = kind == "native"
    backend.module = type("FakeEnvPool", (), {"make_gymnasium": staticmethod(make_gymnasium)})
    backend.native = type("FakeNative", (), {"CartPolePool": staticmethod(native_pool)})
    # The native pool accepts seeds up to INT_MAX - num_envs + 1.
    limit = 2**31 - 4
    for seed in (0, 12345, limit, limit + 1, 2**31 - 1):
        backend.init(seed)
    assert seeds == [0, 12345, limit, 0, 2]


def test_host_actions_reach_the_simulator_c_contiguous() -> None:
    from perpetua.adapters.host import CPULifetime

    class Slots(_Counter):
        num_slots = 2
        action_space = gym.spaces.Discrete(2)

    env: Any = CPULifetime(cast(Any, Slots()), env_id="fake")
    strided = np.asarray([1, 9, 0, 9], dtype=np.int32)[::2]
    assert not strided.flags.c_contiguous
    actions = env.env._actions(strided)
    assert actions.flags.c_contiguous
    assert actions.dtype == np.int32
    assert actions.tolist() == [1, 0]
