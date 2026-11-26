"""A transition whose slot goes inactive is stored when the slot acts again."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportPrivateUsage=false

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import DQN, IQL, SAC, DQNConfig, DQNRunner, SACConfig, SACRunner
from perpetua.agents._common import bridge_step, init_bridge

GAMMA = 0.9


class Blinking:
    """Wrap a world so its slots are inactive, with zero observations, on every third tick."""

    def __init__(self, env: Any) -> None:
        self.env = env
        for name in ("num_agents", "observation_space", "action_space", "oracle_fields"):
            setattr(self, name, getattr(env, name))
        self.metadata = env.metadata

    def _blink(self, obs: jax.Array, info: Any) -> tuple[jax.Array, Any]:
        on = info["t"].lo % 3 != 1
        active = info["active_mask"] & on
        return jnp.where(on, obs, 0.0), {**info, "active_mask": active}

    def init(self, key: jax.Array) -> tuple[Any, jax.Array, Any]:
        state, obs, info = self.env.init(key)
        return (state, *self._blink(obs, info))

    def step(
        self, state: Any, action: Any, key: jax.Array
    ) -> tuple[Any, jax.Array, jax.Array, Any]:
        state, obs, reward, info = self.env.step(state, action, key)
        # A known reward stream: tick t pays t + 1.
        reward = jnp.broadcast_to(info["t"].lo.astype(jnp.float32), reward.shape)
        return (state, *self._blink(obs, info)[:1], reward, self._blink(obs, info)[1])

    def observe(self, state: Any) -> tuple[jax.Array, Any]:
        return self._blink(*self.env.observe(state))

    def privileged(self, state: Any) -> dict[str, jax.Array]:
        return self.env.privileged(state)


def test_bridge_accumulates_discounted_reward_until_the_slot_acts_again() -> None:
    gamma = jnp.float32(GAMMA)
    on, off = jnp.asarray([True]), jnp.asarray([False])
    bridge = init_bridge(1)
    # Acts at tick 0 and is inactive at ticks 1 and 2; active again at tick 3.
    bridge, complete = bridge_step(bridge, on, off, jnp.asarray([1.0]), gamma)
    assert not bool(complete[0])
    assert bool(bridge.open[0])
    bridge, complete = bridge_step(bridge, off, off, jnp.asarray([2.0]), gamma)
    assert not bool(complete[0])
    bridge, complete = bridge_step(bridge, off, on, jnp.asarray([4.0]), gamma)
    assert bool(complete[0])
    assert not bool(bridge.open[0])
    np.testing.assert_allclose(bridge.reward, [1.0 + GAMMA * 2.0 + GAMMA**2 * 4.0], rtol=1e-6)
    # The target multiplies by gamma once more: three ticks were spanned.
    np.testing.assert_allclose(bridge.discount, [GAMMA**2], rtol=1e-6)


def test_an_ordinary_transition_passes_through_the_bridge_unchanged() -> None:
    reward = jnp.asarray([0.25, -3.0])
    always = jnp.asarray([True, True])
    bridge, complete = bridge_step(init_bridge(2), always, always, reward, jnp.float32(GAMMA))
    assert np.array_equal(bridge.reward, reward)
    assert np.array_equal(bridge.discount, [1.0, 1.0])
    assert bool(jnp.all(complete))
    assert not bool(jnp.any(bridge.open))


def test_a_slot_that_never_acts_stores_nothing() -> None:
    never = jnp.asarray([False])
    bridge, complete = bridge_step(
        init_bridge(1), never, never, jnp.asarray([5.0]), jnp.float32(GAMMA)
    )
    assert not bool(complete[0])
    assert not bool(bridge.open[0])


def blinking_grid() -> Any:
    return Blinking(pp.make("ContinualGrid-v0", size=4, num_agents=2))


@pytest.mark.parametrize("learner", [DQN, IQL])
def test_dqn_replay_never_holds_a_placeholder_next_observation(learner: Any) -> None:
    config = DQNConfig(
        rollout_steps=12, buffer_capacity=64, batch_size=8, gradient_steps=1, gamma=GAMMA
    )
    trainer = DQNRunner(blinking_grid(), learner(action_size=5, config=config))
    state = trainer(trainer.init(seed=0))
    replay = state.replay
    stored = np.asarray(replay.active[:12])
    next_obs = np.asarray(replay.next_obs[:12])
    # Ticks 0, 3, 6, 9 lead into an inactive tick, so their transitions are
    # stored one tick later, when the slot acts again: at rows 1, 4, 7, 10.
    assert stored[:, 0].tolist() == [False, True, True] * 4
    assert np.all(np.any(next_obs[stored] != 0.0, axis=-1))
    bridged = stored[:, 0] & (np.arange(12) % 3 == 1)
    np.testing.assert_allclose(np.asarray(replay.bridge[:12])[bridged], GAMMA, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(replay.bridge[:12])[~bridged], 1.0)
    # Tick t pays t + 1: the transition from tick 3 holds 4 + gamma * 5.
    np.testing.assert_allclose(np.asarray(replay.reward[4]), 4.0 + GAMMA * 5.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(replay.reward[5]), 6.0, rtol=1e-6)
    assert bool(state.healthy)


def test_sac_replay_never_holds_a_placeholder_next_observation() -> None:
    env: Any = Blinking(pp.make("gymnax/Pendulum-v1"))
    low = np.asarray(env.action_space.low).reshape(-1)
    high = np.asarray(env.action_space.high).reshape(-1)
    config = SACConfig(
        rollout_steps=12, buffer_capacity=64, batch_size=8, gradient_steps=1, gamma=GAMMA
    )
    trainer = SACRunner(env, SAC(action_low=low, action_high=high, config=config))
    state = trainer(trainer.init(seed=0))
    replay = state.replay
    stored = np.asarray(replay.active[:12])[:, 0]
    assert stored.tolist() == [False, True, True] * 4
    assert np.all(np.any(np.asarray(replay.next_obs[:12])[stored] != 0.0, axis=-1))
    np.testing.assert_allclose(np.asarray(replay.reward[4]), 4.0 + GAMMA * 5.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(replay.bridge[4]), GAMMA, rtol=1e-6)
    assert bool(state.healthy)
