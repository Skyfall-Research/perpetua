"""The host lift must accept Box actions and run SAC on a CPU Pendulum."""

# Optional simulator/JAX interfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import SAC, HostSACRunner, SACConfig


def test_gymnasium_host_lift_accepts_box_actions() -> None:
    env = GymnasiumLifetime(gym.make("Pendulum-v1", max_episode_steps=3))
    twin = GymnasiumLifetime(gym.make("Pendulum-v1", max_episode_steps=3))
    try:
        assert env.metadata.get("checkpointable") is False
        configuration = env.metadata.get("configuration")
        assert isinstance(configuration, dict)
        backend = configuration["backend"]
        assert isinstance(backend, dict)
        assert backend["action_encoding"] == "box-(1,)"
        space = env.action_space
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (1, 1)
        low = cast("NDArray[np.float32]", space.low)
        high = cast("NDArray[np.float32]", space.high)
        np.testing.assert_array_equal(low, [[-2.0]])
        np.testing.assert_array_equal(high, [[2.0]])
        initial = pp.runtime.init_lifetime(env, seed=0)
        assert initial.obs.shape == (1, 3)
        assert initial.obs.dtype == jnp.float32
        np.testing.assert_array_equal(initial.info["action_mask"], [[True]])
        bounded = pp.runtime.tick(env, lambda obs, info, key: jnp.asarray([[2.0]]), initial)
        assert bounded.state.t.to_int() == 1
        # Truncation respawns on the same tick without resetting the clock.
        current = bounded
        for _ in range(2):
            current = pp.runtime.tick(env, lambda obs, info, key: jnp.asarray([[0.0]]), current)
        assert current.state.t.to_int() == 3
        assert bool(jnp.all(current.info["respawn"]))

        # Actions are clipped into the declared bounds, not rejected.
        twin_initial = pp.runtime.init_lifetime(twin, seed=0)
        with pytest.raises(ValueError, match=r"CPU continuous actions must be a float array"):
            twin.step(twin_initial.state, jnp.asarray([0], dtype=jnp.int32), jax.random.key(0))
        with pytest.raises(ValueError, match=r"CPU continuous actions must be a float array"):
            twin.step(twin_initial.state, jnp.asarray([[0.5, 0.5]]), jax.random.key(0))
        with pytest.raises(ValueError, match=r"CPU continuous actions must be finite\."):
            twin.step(twin_initial.state, jnp.asarray([[np.nan]]), jax.random.key(0))
        clipped = pp.runtime.tick(twin, lambda obs, info, key: jnp.asarray([[99.0]]), twin_initial)
        np.testing.assert_array_equal(
            np.asarray(bounded.state.env.obs), np.asarray(clipped.state.env.obs)
        )
    finally:
        env.close()
        twin.close()


def test_host_sac_trains_on_a_cpu_pendulum_lifetime() -> None:
    env = GymnasiumLifetime(gym.make("Pendulum-v1", max_episode_steps=3))
    try:
        space = env.action_space
        assert isinstance(space, gym.spaces.Box)
        agent = SAC(
            action_low=space.low[0],
            action_high=space.high[0],
            config=SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1),
        )
        trainer = HostSACRunner(env, agent)
        first = trainer(trainer.init(seed=0, window_size=4))
        assert bool(first.healthy)
        assert first.lifetime.state.t.to_int() == 4
        assert int(first.replay.cursor) == 4
        second = trainer(first)
        assert bool(second.healthy)
        assert second.lifetime.state.t.to_int() == 8
        assert env.metadata.get("checkpointable") is False
        # Mutable single-use ownership: a consumed state cannot be replayed.
        with pytest.raises(ValueError, match=r"CPU state is stale or belongs to another"):
            trainer(first)
    finally:
        env.close()
