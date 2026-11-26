"""Independent Q-learning must isolate slots, skip inactive agents and resume exactly."""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from helpers import assert_same

import perpetua as pp
from perpetua.agents import IQL, DQNConfig, DQNRunner
from perpetua.agents.dqn import DQNTrainingState, init_dqn, insert
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def seeded_carry(
    env: pp.Environment[DriftEnvState[GridState]], agent: IQL, *, seed: int, ticks: int
) -> DQNTrainingState[DriftEnvState[GridState]]:
    carry = init_dqn(env, agent, seed=seed, window_size=4)
    for _ in range(ticks):
        agent_key, act_key = jax.random.split(carry.lifetime.agent_key)
        env_key, step_key = jax.random.split(carry.lifetime.env_key)
        action = agent.act(carry.agent, carry.lifetime.obs, carry.lifetime.info, act_key)
        state, obs, reward, info = env.step(carry.lifetime.state, action, step_key)
        lifetime = carry.lifetime._replace(
            state=state, obs=obs, info=info, env_key=env_key, agent_key=agent_key
        )
        transition = (
            carry.lifetime.obs,
            carry.lifetime.info["action_mask"],
            carry.lifetime.info["active_mask"],
            action,
            reward,
            obs,
            info["action_mask"],
        )
        carry = carry._replace(lifetime=lifetime, replay=insert(carry.replay, transition))
    return carry


def test_iql_isolates_slots_and_skips_inactive_updates() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = DQNConfig(rollout_steps=2, buffer_capacity=8, batch_size=2, gradient_steps=1)
    agent = IQL(action_size=5, config=config)
    carry = seeded_carry(env, agent, seed=0, ticks=4)
    assert np.asarray(carry.agent.gradient_steps.lo).shape == (2,)
    inactive = carry._replace(
        replay=carry.replay._replace(active=carry.replay.active.at[:, 1].set(False))
    )
    trained, statistics = agent.update(inactive.agent, inactive.replay, jax.random.key(3))
    assert int(trained.gradient_steps.lo[0]) == 1
    assert int(trained.gradient_steps.lo[1]) == 0
    assert np.asarray(statistics.loss).shape == (2,)
    assert_same(
        trained.params["params"]["Dense_0"]["kernel"][1],
        carry.agent.params["params"]["Dense_0"]["kernel"][1],
    )
    assert not np.array_equal(
        np.asarray(trained.params["params"]["Dense_0"]["kernel"][0]),
        np.asarray(carry.agent.params["params"]["Dense_0"]["kernel"][0]),
    )
    first_only = agent.update(inactive.agent, inactive.replay, jax.random.key(3))
    assert_same(first_only, (trained, statistics))


def test_iql_runner_checkpoints_and_continues_exactly(tmp_path: Path) -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=2)
    trainer = DQNRunner(env, IQL(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    assert bool(state.healthy)
    path = tmp_path / "iql"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert_same(trainer(state), trainer(restored))


def test_iql_example_runs_and_resumes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "train_dqn.py"
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), "--algorithm", "iql", *extra],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        return json.loads(result.stdout)

    saved = run(["--updates", "2", "--checkpoint", str(tmp_path / "iql")])
    assert saved["algorithm"] == "iql"
    assert saved["lifetime_ticks"] == 256
    assert saved["gradient_steps_per_agent"] == [8, 8]
    resumed = run(
        ["--updates", "1", "--resume", str(tmp_path / "iql"), "--checkpoint", str(tmp_path / "r")]
    )
    assert resumed["lifetime_ticks"] == 384
    assert resumed["gradient_steps_per_agent"] == [12, 12]
    assert resumed["healthy"] is True


def test_iql_learns_cooperative_grid_above_random() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = DQNConfig(
        rollout_steps=32,
        buffer_capacity=4096,
        batch_size=64,
        gradient_steps=8,
        hidden_size=64,
        epsilon=0.1,
        target_update_interval=64,
    )
    trainer = DQNRunner(env, IQL(action_size=5, config=config))
    state = trainer.init(seed=0, window_size=128)
    for _ in range(150):
        state = trainer(state)
    assert bool(state.healthy)
    learned = float(np.asarray(state.lifetime.metrics.mean).mean())
    baseline_env = pp.make("CooperativeGrid-v0", size=3)
    baseline = pp.runtime.Runner(baseline_env, pp.policies.random, ticks=150 * 32)(
        pp.runtime.init_lifetime(baseline_env, seed=0, window_size=128)
    )
    baseline_mean = float(np.asarray(baseline.metrics.mean).mean())
    assert learned > 2 * baseline_mean
    print(f"IQL mean {learned:.4f} vs random {baseline_mean:.4f}")


def test_iql_trains_through_cyclic_goal_drift() -> None:
    env = pp.make("CooperativeGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 256))
    config = DQNConfig(
        rollout_steps=32,
        buffer_capacity=4096,
        batch_size=64,
        gradient_steps=8,
        hidden_size=64,
        epsilon=0.1,
        target_update_interval=64,
    )
    trainer = DQNRunner(env, IQL(action_size=5, config=config))
    state = trainer.init(seed=0, window_size=128)
    trace: list[np.ndarray] = []
    for _ in range(150):
        state = trainer(state)
        trace.append(np.asarray(pp.evaluation.recent_rewards(state.lifetime.metrics, steps=32)))
    assert bool(state.healthy)
    # 4800 ticks with period 256: 18 switches, 192 ticks into the next cycle.
    assert int(state.lifetime.state.drift["goal"].elapsed) == 192
    assert int(state.lifetime.state.drift["goal"].phase) == 0
    rewards = np.concatenate(trace).mean(axis=1)
    learned = float(np.asarray(state.lifetime.metrics.mean).mean())
    baseline_env = pp.make("CooperativeGrid-v0", size=3)
    baseline = pp.runtime.Runner(baseline_env, pp.policies.random, ticks=4800)(
        pp.runtime.init_lifetime(baseline_env, seed=0, window_size=128)
    )
    baseline_mean = float(np.asarray(baseline.metrics.mean).mean())
    assert learned > 2 * baseline_mean
    assert rewards[-1200:].mean() > baseline_mean
    changes = np.asarray([255 + 256 * k for k in range(18)])
    ticks = np.asarray(pp.evaluation.recovery_ticks(rewards, changes, window=64, fraction=0.5))
    capped = int(np.sum(ticks >= len(rewards) - changes))
    assert float(ticks.mean()) < 160.0
    assert capped <= 3
    print(
        f"IQL through drift: mean {learned:.4f} vs random {baseline_mean:.4f}, "
        f"recovery mean {ticks.mean():.1f}, capped {capped}"
    )


def test_iql_validates_each_slots_hyperparameters_like_dqn() -> None:
    import pytest

    from perpetua.agents.dqn import DQNHyperparameters

    env = pp.make("CooperativeGrid-v0", size=3, num_agents=2)
    _, obs, _ = env.init(jax.random.key(0))
    bad = DQNHyperparameters.from_config(DQNConfig())._replace(learning_rate=jnp.asarray(-1.0))
    with pytest.raises(ValueError, match=r"Hyperparameter learning rate and maximum"):
        IQL(action_size=5).init(obs, jax.random.key(0), hyperparameters=bad)
