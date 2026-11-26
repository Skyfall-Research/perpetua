"""CPU bridges must not masquerade as stateless, checkpointable JAX simulators."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="EnvPool CPU bridge is validated on Linux; macOS needs external Qt.",
)


def test_envpool_host_runner_batches_simulators_and_reward_drift(tmp_path: Path) -> None:
    env = pp.make(
        "envpool/CartPole-v1",
        num_envs=4,
        num_threads=2,
        drift=pp.drift.Cyclic([(1.0,), (0.5,)], period=2),
    )
    try:
        runner = pp.runtime.HostRunner(env, pp.policies.random, ticks=8)
        state = pp.runtime.init_lifetime(env, seed=0)
        state = runner(state)
        assert state.obs.shape == (4, 4)
        assert state.state.t.to_int() == 8
        assert state.metrics.ticks.to_int() == 8
        assert bool(state.metrics.healthy)
        assert env.metadata.get("checkpointable") is False
        with pytest.raises(NotImplementedError, match="checkpoint"):
            pp.checkpoint.save(tmp_path / "unsupported", state, metadata=runner.metadata)
        assert not (tmp_path / "unsupported").exists()
    finally:
        env.close()


def test_cpu_state_cannot_be_forked_or_used_by_jax_runner() -> None:
    env = pp.make("envpool/CartPole-v1", num_envs=2, num_threads=1)
    try:
        with pytest.raises(ValueError, match="HostRunner"):
            pp.runtime.Runner(env, pp.policies.random)
        state = pp.runtime.init_lifetime(env)
        current = pp.runtime.tick(env, pp.policies.random, state)
        with pytest.raises(ValueError, match="stale"):
            pp.runtime.tick(env, pp.policies.random, state)
        with pytest.raises(ValueError, match="outside the backend action space"):
            env.step(current.state, jnp.asarray([9, 9]), current.env_key)
    finally:
        env.close()


def test_cpu_ppo_reuses_the_same_optimizer_and_loss() -> None:
    from perpetua.agents.ppo import PPO, HostPPORunner, PPOConfig, PPORunner

    env = pp.make("envpool/CartPole-v1", num_envs=4, num_threads=2)
    try:
        agent = PPO(
            action_size=2, config=PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
        )
        with pytest.raises(ValueError, match="HostPPORunner"):
            PPORunner(env, agent)
        trainer = HostPPORunner(env, agent)
        state = trainer(trainer.init(seed=0))
        assert state.lifetime.state.t.to_int() == 8
        assert state.updates.to_int() == 1
        assert state.agent.gradient_steps.to_int() == 1
        assert bool(state.healthy)
        assert np.all(np.isfinite(np.asarray(state.lifetime.metrics.mean)))
    finally:
        env.close()
