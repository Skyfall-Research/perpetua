"""Buffer donation: same results, in-place updates, and a consumed input carry."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import hashlib
import platform
import sys
from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import (
    DQN,
    PPO,
    SAC,
    ComputePPORunner,
    DQNConfig,
    DQNRunner,
    HostDQNRunner,
    HostPPORunner,
    HostSACRunner,
    PPORunner,
    SACConfig,
    SACRunner,
    dqn,
    sac,
)


def leaves(tree: object) -> list[jax.Array]:
    return [leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)]


def assert_same(left: object, right: object) -> None:
    for a, b in zip(leaves(left), leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def grid_runner(donate: bool) -> tuple[Any, Any]:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(2, 2), (0, 0)], 4))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=8, donate=donate)
    return runner, pp.runtime.init_lifetime(env, seed=3, window_size=4)


def ppo_runner(donate: bool) -> tuple[Any, Any]:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    runner = PPORunner(env, PPO(action_size=5), donate=donate)
    return runner, runner.init(seed=3)


def dqn_runner(donate: bool) -> tuple[Any, Any]:
    env = pp.make("ContinualGrid-v0", size=4)
    config = DQNConfig(rollout_steps=16, buffer_capacity=256, batch_size=16, gradient_steps=2)
    runner = DQNRunner(env, DQN(action_size=5, config=config), donate=donate)
    return runner, runner.init(seed=3, window_size=8)


def sac_runner(donate: bool) -> tuple[Any, Any]:
    env = pp.make("gymnax/Pendulum-v1")
    agent = SAC(
        action_low=env.action_space.low[0],
        action_high=env.action_space.high[0],
        config=SACConfig(rollout_steps=16, buffer_capacity=512, batch_size=16, gradient_steps=2),
    )
    runner = SACRunner(env, agent, donate=donate)
    return runner, runner.init(seed=3, window_size=8)


@pytest.mark.parametrize("build", [grid_runner, ppo_runner, dqn_runner, sac_runner])
def test_donation_keeps_results_and_consumes_the_input(
    build: Callable[[bool], tuple[Any, Any]],
) -> None:
    plain, start = build(False)
    expected = start
    for _ in range(3):
        expected = plain(expected)
    donating, initial = build(True)
    carry = initial
    for _ in range(3):
        carry = donating(carry)
    assert_same(expected, carry)
    # Leaves the first call did not have to copy were donated and are gone.
    assert any(leaf.is_deleted() for leaf in leaves(initial))
    assert not any(leaf.is_deleted() for leaf in leaves(carry))
    assert donating.metadata == plain.metadata


def _grid() -> Any:
    return pp.make("ContinualGrid-v0", size=3)


def _pendulum_sac() -> tuple[Any, SAC]:
    env = pp.make("gymnax/Pendulum-v1")
    low, high = env.action_space.low[0], env.action_space.high[0]
    return env, SAC(action_low=low, action_high=high)


def _torch_runner() -> pp.runtime.ChunkRunner[Any]:
    from perpetua.torch_bridge import TorchPPO, TorchRunner

    env = _grid()
    return TorchRunner(env, TorchPPO(env.observation_space, env.action_space))


# Typed as the protocol, so pyright checks each class against it as well.
CHUNK_RUNNERS: dict[str, Callable[[bool], pp.runtime.ChunkRunner[Any]]] = {
    "Runner": lambda donate: pp.runtime.Runner(_grid(), pp.policies.random, donate=donate),
    "PPORunner": lambda donate: PPORunner(_grid(), PPO(action_size=5), donate=donate),
    "DQNRunner": lambda donate: DQNRunner(_grid(), DQN(action_size=5), donate=donate),
    "SACRunner": lambda donate: SACRunner(*_pendulum_sac(), donate=donate),
    "ProbedRunner": lambda donate: pp.probes.ProbedRunner(
        PPORunner(_grid(), PPO(action_size=5), donate=donate)
    ),
    "HostRunner": lambda _: pp.runtime.HostRunner(_grid(), pp.policies.random),
    "HostPPORunner": lambda _: HostPPORunner(_grid(), PPO(action_size=5)),
    "HostDQNRunner": lambda _: HostDQNRunner(_grid(), DQN(action_size=5)),
    "HostSACRunner": lambda _: HostSACRunner(*_pendulum_sac()),
    "TorchRunner": lambda _: _torch_runner(),
}
# These take the chunk's timing trace as a second argument, so they are not typed as the protocol.
TIMED_RUNNERS: dict[str, Callable[[bool], Any]] = {
    "ComputeRunner": lambda _: pp.compute.ComputeRunner(_grid(), pp.policies.random),
    "ComputePPORunner": lambda _: ComputePPORunner(_grid(), PPO(action_size=5)),
}
DONATING = ("Runner", "PPORunner", "DQNRunner", "SACRunner", "ProbedRunner")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=[pytest.mark.torch] * (name == "TorchRunner"))
        for name in (*CHUNK_RUNNERS, *TIMED_RUNNERS)
    ],
)
def test_every_runner_reports_whether_it_consumes_its_input(name: str) -> None:
    build: Callable[[bool], Any] = {**CHUNK_RUNNERS, **TIMED_RUNNERS}[name]
    runner: Any = build(False)
    assert runner.donates is False
    assert isinstance(runner.metadata, Mapping)
    with pytest.raises(AttributeError):
        runner.donates = True
    assert callable(runner)
    assert build(True).donates is (name in DONATING)


def _host_values(leaf: jax.Array) -> np.ndarray:
    """A host copy of a leaf's values (a key's data)."""
    if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jax.random.key_data(leaf)
    return np.array(leaf)


def _digest(tree: object) -> str:
    """SHA-256 over every array leaf's dtype, shape and bytes."""
    digest = hashlib.sha256()
    for leaf in leaves(tree):
        array = _host_values(leaf)
        digest.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def _host_dqn() -> tuple[Any, Any]:
    env = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=5))
    config = DQNConfig(rollout_steps=8, buffer_capacity=12, batch_size=4, gradient_steps=2)
    return env, HostDQNRunner(env, DQN(action_size=2, config=config))


def _host_sac() -> tuple[Any, Any]:
    env = GymnasiumLifetime(gym.make("Pendulum-v1", max_episode_steps=5))
    space = env.action_space
    config = SACConfig(rollout_steps=8, buffer_capacity=12, batch_size=4, gradient_steps=2)
    agent = SAC(action_low=space.low[0], action_high=space.high[0], config=config)
    return env, HostSACRunner(env, agent)


def _eager_dqn_chunk(runner: Any, carry: Any) -> Any:
    """One HostDQNRunner call with every replay insert uncompiled and copying."""
    for _ in range(runner.agent.config.rollout_steps):
        carry = dqn._collect(  # pyright: ignore[reportPrivateUsage]
            runner.env,
            runner.agent,
            carry,
            act=runner._act,
            reference=runner.reference,
            update=runner._metrics,
        )
    for _ in range(runner.agent.config.gradient_steps):
        key, step_key = jax.random.split(carry.optimizer_key)
        agent, statistics = runner._update(carry.agent, carry.replay, step_key)
        carry = carry._replace(optimizer_key=key, agent=agent, statistics=statistics)
    return carry._replace(updates=carry.updates.increment())


def _eager_sac_chunk(runner: Any, carry: Any) -> Any:
    """One HostSACRunner call with every replay insert uncompiled and copying."""
    for _ in range(runner.agent.config.rollout_steps):
        carry = sac._collect(  # pyright: ignore[reportPrivateUsage]
            runner.env,
            carry,
            runner._act,
            runner.agent,
            reference=runner.reference,
            update=runner._metrics,
        )
    for _ in range(runner.agent.config.gradient_steps):
        key, step_key = jax.random.split(carry.optimizer_key)
        agent, statistics = runner._update(carry.agent, carry.replay, step_key)
        carry = carry._replace(optimizer_key=key, agent=agent, statistics=statistics)
    return carry._replace(updates=carry.updates.increment())


# Final carries of three calls from seed 3, recorded on macOS arm64 (CPU) from
# the host runners that inserted without compiling, before they wrote in place.
HOST_RUNS: dict[str, tuple[Callable[[], tuple[Any, Any]], Callable[[Any, Any], Any], str]] = {
    "dqn": (
        _host_dqn,
        _eager_dqn_chunk,
        "6718c390b28cf54d9a679e01c1f11388643d56b493a29a99e44c0116f60d611d",
    ),
    "sac": (
        _host_sac,
        _eager_sac_chunk,
        "85d625e33a49ed1e4aad017b0ee40bab46f39b04cf2fa5fb055773312e2b4a0a",
    ),
}
RECORDED_ON = ("darwin", "arm64", "cpu", False)


@pytest.mark.parametrize("name", sorted(HOST_RUNS))
def test_host_replay_runners_write_in_place_and_keep_their_results(name: str) -> None:
    build, eager_chunk, recorded = HOST_RUNS[name]
    env, runner = build()
    twin_env, twin = build()
    try:
        carry = runner.init(seed=3, window_size=8)
        expected = twin.init(seed=3, window_size=8)
        for _ in range(3):
            before = [_host_values(leaf) for leaf in leaves(carry)]
            previous, carry = carry, runner(carry)
            expected = eager_chunk(twin, expected)
            # The carry passed in stays readable and unchanged; its replay was copied once.
            assert not any(leaf.is_deleted() for leaf in leaves(previous))
            for old, leaf in zip(before, leaves(previous), strict=True):
                np.testing.assert_array_equal(old, _host_values(leaf))
            assert not any(leaf.is_deleted() for leaf in leaves(carry))
            assert all(
                new is not old
                for new, old in zip(leaves(carry.replay), leaves(previous.replay), strict=True)
            )
            assert_same(
                carry._replace(lifetime=carry.lifetime._replace(state=None)),
                expected._replace(
                    lifetime=expected.lifetime._replace(state=None), healthy=carry.healthy
                ),
            )
        assert bool(carry.healthy)
        assert int(carry.replay.filled) == 12
        here = (sys.platform, platform.machine(), jax.default_backend(), jax.config.jax_enable_x64)
        if here == RECORDED_ON:
            final = carry._replace(lifetime=carry.lifetime._replace(state=None))
            assert _digest(final) == recorded
    finally:
        env.close()
        twin_env.close()


def test_host_replay_inserts_donate_only_the_runners_own_copy() -> None:
    env, runner = _host_dqn()
    try:
        carry = runner.init(seed=0, window_size=8)
        transition = (
            carry.pending.obs,
            carry.pending.mask,
            jnp.ones(1, dtype=jnp.bool_),
            carry.pending.action,
            jnp.zeros(1),
            carry.lifetime.obs,
            carry.lifetime.info["action_mask"],
        )
        discount = jnp.ones(1)
        copied = runner._insert(carry.replay, transition, discount)
        assert not any(leaf.is_deleted() for leaf in leaves(carry.replay))
        written = runner._insert_in_place(copied, transition, discount)
        # Every buffer of the intermediate replay was reused, so a tick's cost
        # does not grow with the capacity.
        assert all(leaf.is_deleted() for leaf in leaves(copied))
        assert int(written.cursor) == 2
    finally:
        env.close()
