"""Learner runners share one per-tick body, so regret reaches every algorithm."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import DQN, PPO, SAC, DQNConfig, PPOConfig, SACConfig
from perpetua.agents.dqn import DQNRunner, HostDQNRunner
from perpetua.agents.ppo import HostPPORunner, PPORunner
from perpetua.agents.sac import SACRunner


def stay(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
    """Reference that never moves; on a stationary grid it never collects."""
    del info, key
    return jnp.zeros(obs.shape[0], dtype=jnp.int32)


def zero_torque(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
    """Continuous reference applying no torque."""
    del info, key
    return jnp.zeros((obs.shape[0], 1), dtype=jnp.float32)


def without_regret(metrics: pp.metrics.Metrics) -> pp.metrics.Metrics:
    zero = jnp.zeros_like(metrics.regret_total)
    return metrics._replace(
        regret_total=zero,
        regret_compensation=zero,
        regret_window=jnp.zeros_like(metrics.regret_window),
        regret_window_total=zero,
    )


def test_ppo_regret_is_exact_and_leaves_training_untouched() -> None:
    env = pp.make("ContinualGrid-v0", size=2)
    config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    plain = PPORunner(env, PPO(action_size=5, config=config))
    scored = PPORunner(env, PPO(action_size=5, config=config), reference=stay)
    a, b = plain.init(seed=1), scored.init(seed=1)
    for _ in range(4):
        a, b = plain(a), scored(b)
    # The shadow step never perturbs learning: parameters, optimizer state,
    # world state and every non-regret metric are bit-identical.
    assert_same(a.agent, b.agent)
    assert_same(a.lifetime, b.lifetime._replace(metrics=without_regret(b.lifetime.metrics)))
    # Against a reference earning exactly zero, regret is minus the learner's total.
    assert float(b.lifetime.metrics.total[0]) > 0.0
    np.testing.assert_array_equal(
        np.asarray(b.lifetime.metrics.regret_total), -np.asarray(b.lifetime.metrics.total)
    )
    assert str(scored.metadata["regret_reference"]).endswith("stay")
    assert "regret_reference" not in plain.metadata


def test_dqn_regret_is_exact_and_leaves_training_untouched() -> None:
    env = pp.make("ContinualGrid-v0", size=2)
    plain = DQNRunner(env, DQN(action_size=5, config=DQNConfig(hidden_size=8)))
    scored = DQNRunner(env, DQN(action_size=5, config=DQNConfig(hidden_size=8)), reference=stay)
    a, b = plain.init(seed=2), scored.init(seed=2)
    for _ in range(3):
        a, b = plain(a), scored(b)
    assert_same((a.agent, a.replay), (b.agent, b.replay))
    assert_same(a.lifetime.metrics, without_regret(b.lifetime.metrics))
    assert float(b.lifetime.metrics.total[0]) > 0.0
    np.testing.assert_array_equal(
        np.asarray(b.lifetime.metrics.regret_total), -np.asarray(b.lifetime.metrics.total)
    )


def test_sac_regret_reaches_the_continuous_learner() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    low, high = env.action_space.low[0], env.action_space.high[0]
    config = SACConfig(hidden_size=8)
    plain = SACRunner(env, SAC(action_low=low, action_high=high, config=config))
    scored = SACRunner(
        env, SAC(action_low=low, action_high=high, config=config), reference=zero_torque
    )
    a, b = plain.init(seed=3), scored.init(seed=3)
    for _ in range(2):
        a, b = plain(a), scored(b)
    assert_same((a.agent, a.replay), (b.agent, b.replay))
    assert_same(a.lifetime.metrics, without_regret(b.lifetime.metrics))
    regret = np.asarray(b.lifetime.metrics.regret_total)
    assert np.all(np.isfinite(regret))
    assert float(np.abs(regret).sum()) > 0.0
    assert bool(b.lifetime.metrics.healthy)


def test_host_learners_shadow_step_snapshot_backends_exactly() -> None:
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    first = GymnasiumLifetime(gym.make("CartPole-v1"))
    second = GymnasiumLifetime(gym.make("CartPole-v1"))
    try:
        plain = HostPPORunner(first, PPO(action_size=2, config=config))
        scored = HostPPORunner(
            second, PPO(action_size=2, config=config), reference=pp.policies.random
        )
        a, b = plain(plain.init(seed=4)), scored(scored.init(seed=4))
        assert_same(a.agent, b.agent)
        np.testing.assert_array_equal(np.asarray(a.lifetime.obs), np.asarray(b.lifetime.obs))
        assert np.all(np.isfinite(np.asarray(b.lifetime.metrics.regret_total)))
    finally:
        first.close()
        second.close()


def test_opaque_host_backends_refuse_a_reference_before_stepping() -> None:
    env = GymnasiumLifetime(gym.make("Acrobot-v1"))
    try:
        assert env.metadata.get("checkpointable") is False
        with pytest.raises(ValueError, match="shadow step"):
            pp.runtime.HostRunner(env, pp.policies.random, ticks=2, reference=pp.policies.random)
        with pytest.raises(ValueError, match="shadow step"):
            HostDQNRunner(env, DQN(action_size=3), reference=pp.policies.random)
        # Without a reference the same backend still runs.
        out = pp.runtime.HostRunner(env, pp.policies.random, ticks=2)(
            pp.runtime.init_lifetime(env, seed=0)
        )
        assert out.state.t.to_int() == 2
    finally:
        env.close()
