"""Continuing SAC on Pendulum must respect bounds, masks and checkpoints."""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import json
import math
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters.gymnax import PendulumParams, PendulumState
from perpetua.agents.sac import (
    SAC,
    SACConfig,
    SACRunner,
    SACTrainingState,
    init_sac,
    init_sac_replay,
    insert,
)
from perpetua.core import Info
from perpetua.params import DriftEnv, DriftEnvState


def uniform_torque(obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
    del obs, info
    return jax.random.uniform(key, (1, 1), minval=-2.0, maxval=2.0)


def agent_for(env: DriftEnv[PendulumState, PendulumParams], config: SACConfig) -> SAC:
    return SAC(
        action_low=env.action_space.low[0],
        action_high=env.action_space.high[0],
        config=config,
    )


def test_pendulum_public_contract_and_determinism() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    state, obs, info = env.init(jax.random.key(0))
    twin, _, _ = env.init(jax.random.key(0))
    assert_same(state, twin)
    assert obs.shape == (1, 3)
    assert obs.dtype == jnp.float32
    assert env.observation_space.contains(np.asarray(obs))
    assert env.action_space.shape == (1, 1)
    np.testing.assert_array_equal(info["action_mask"], [[True]])
    np.testing.assert_array_equal(info["active_mask"], [True])
    np.testing.assert_array_equal(info["respawn"], [False])
    assert info["t"].to_int() == 0
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert "oracle" not in info
    assert env.metadata.get("execution") == "jax"
    assert env.metadata.get("checkpointable") is True
    np.testing.assert_array_equal(
        np.asarray(obs[0]), np.asarray(env.backend.get_obs(state.env.physics))
    )
    next_state, _, reward, next_info = env.step(
        state, np.asarray([[1.5]], dtype=np.float32), jax.random.key(1)
    )
    assert reward.shape == (1,)
    assert reward.dtype == jnp.float32
    assert next_info["t"].to_int() == 1
    # The legacy 200-step truncation cannot trigger: backend age stays at 1.
    assert int(next_state.env.physics.time) == 1
    again = env.step(next_state, np.asarray([[1.5]], dtype=np.float32), jax.random.key(2))[0]
    assert int(again.env.physics.time) == 1
    with pytest.raises(ValueError, match=r"Pendulum actions must be float arrays of shape"):
        env.step(state, jnp.asarray([1.5]), jax.random.key(1))
    with pytest.raises(ValueError, match=r"Pendulum actions must be float arrays of shape"):
        env.step(state, jnp.asarray([[1]], dtype=jnp.int32), jax.random.key(1))
    # Actions are clipped into the torque bounds rather than rejected.
    clipped, _, clipped_reward, _ = env.step(
        state, np.asarray([[99.0]], dtype=np.float32), jax.random.key(1)
    )
    bounded, _, bounded_reward, _ = env.step(
        state, np.asarray([[2.0]], dtype=np.float32), jax.random.key(1)
    )
    np.testing.assert_array_equal(clipped_reward, bounded_reward)
    np.testing.assert_array_equal(clipped.env.physics.theta, bounded.env.physics.theta)


def test_pendulum_gravity_drift_changes_dynamics() -> None:
    plain = pp.make("gymnax/Pendulum-v1")
    drifting = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (5.0,)], 1))
    left = plain.init(jax.random.key(0))[0]
    right = drifting.init(jax.random.key(0))[0]
    action = np.asarray([[1.0]], dtype=np.float32)
    left = plain.step(left, action, jax.random.key(1))[0]
    right = drifting.step(right, action, jax.random.key(1))[0]
    assert float(left.env.physics.theta) == float(right.env.physics.theta)
    assert int(right.drift["g"].phase) == 1
    left = plain.step(left, action, jax.random.key(2))[0]
    right = drifting.step(right, action, jax.random.key(2))[0]
    assert int(right.drift["g"].phase) == 0
    assert not np.allclose(float(left.env.physics.theta_dot), float(right.env.physics.theta_dot))
    assert "oracle" not in plain.observe(left)[1]
    privileged = plain.privileged(left)
    assert {"phase", "drift_parameters", "drift_changed"} <= set(privileged)
    assert {"g_parameters", "g_phase", "g_changed"} <= set(privileged)


def test_pendulum_conformance_exact_checkpoint_and_batched_worlds(tmp_path: Path) -> None:
    env = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (5.0,)], 3))
    initial = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    assert_same(initial, pp.runtime.init_lifetime(env, seed=0, window_size=4))
    chunk = pp.runtime.Runner(env, uniform_torque, ticks=4)
    long = pp.runtime.Runner(env, uniform_torque, ticks=8)
    # Exactness is asserted across compiled executions; XLA fusion rounds
    # differently from unfused eager dispatch for float dynamics.
    assert_same(chunk(chunk(initial)), long(initial))
    eager = pp.runtime.tick(env, uniform_torque, initial)
    compiled = pp.runtime.Runner(env, uniform_torque, ticks=1)(initial)
    assert eager.state.t.to_int() == compiled.state.t.to_int() == 1
    np.testing.assert_allclose(
        np.asarray(eager.obs), np.asarray(compiled.obs), rtol=1e-5, atol=1e-7
    )
    path = tmp_path / "pendulum"
    pp.checkpoint.save(path, chunk(initial), metadata=env.metadata)
    restored = pp.checkpoint.restore(path, initial, expected_metadata=env.metadata)
    assert int(restored.state.drift["g"].phase) == 1
    assert_same(chunk(restored), long(initial))

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[PendulumState]]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=4)

    batched = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(chunk)(batched)
    np.testing.assert_array_equal(result.state.t.lo, [4, 4])
    assert result.obs.shape[0:2] == (2, 1)


def test_sac_actions_respect_bounds_and_inactive_slots() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    agent = agent_for(
        env, SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1)
    )
    state = env.init(jax.random.key(0))
    agent_state = agent.init(state[1], jax.random.key(1))
    obs, info = env.observe(state[0])
    for seed in range(16):
        sampled = agent.act(agent_state, obs, info, jax.random.key(seed))
        assert float(sampled.action[0, 0]) > -2.0
        assert float(sampled.action[0, 0]) < 2.0
        assert bool(jnp.isfinite(sampled.log_prob[0]))
    info["active_mask"] = jnp.asarray([False])
    inactive = agent.act(agent_state, obs, info, jax.random.key(0))
    assert float(inactive.action[0, 0]) == 0.0


def test_sac_soft_targets_follow_tau_exactly() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    for tau in (1.0, 0.5):
        agent = agent_for(
            env,
            SACConfig(rollout_steps=2, buffer_capacity=16, batch_size=2, gradient_steps=1, tau=tau),
        )
        carry = seed_replay(env, agent, seed=0, ticks=4)
        before = carry.agent.params
        updated, _ = agent.update(carry.agent, carry.replay, jax.random.key(3))
        for target, online, old in (
            (updated.params.q1_target, updated.params.q1, before.q1),
            (updated.params.q2_target, updated.params.q2, before.q2),
        ):
            for new_leaf, old_leaf, target_leaf in zip(
                jax.tree.leaves(online),
                jax.tree.leaves(old),
                jax.tree.leaves(target),
                strict=True,
            ):
                np.testing.assert_allclose(
                    np.asarray(target_leaf),
                    tau * np.asarray(new_leaf) + (1.0 - tau) * np.asarray(old_leaf),
                    rtol=1e-6,
                    atol=1e-8,
                )


def test_sac_checkpoint_restores_every_leaf_including_replay(tmp_path: Path) -> None:
    env = pp.make("gymnax/Pendulum-v1")
    agent = agent_for(
        env, SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=2)
    )
    trainer = SACRunner(env, agent)
    state = trainer(trainer.init(seed=0, window_size=4))
    path = tmp_path / "sac"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert int(restored.replay.cursor) == 4
    assert_same(trainer(state), trainer(restored))


def test_sac_checkpoint_resumes_across_a_gravity_switch(tmp_path: Path) -> None:
    env = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (5.0,)], 3))
    agent = agent_for(
        env, SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=2)
    )
    trainer = SACRunner(env, agent)
    state = trainer(trainer.init(seed=0, window_size=4))
    # One update spans 4 ticks, so the gravity switch inside it is already history.
    assert int(state.lifetime.state.drift["g"].phase) == 1
    path = tmp_path / "switch"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert int(restored.lifetime.state.drift["g"].phase) == 1
    assert_same(trainer(state), trainer(restored))


def test_sac_autotunes_alpha_toward_target_entropy() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    with pytest.raises(ValueError, match=r"autotune_alpha must be a boolean"):
        SACConfig(autotune_alpha=cast(bool, 1))
    with pytest.raises(ValueError, match=r"target_entropy must be finite when provided"):
        SACConfig(target_entropy=float("inf"))
    agent = agent_for(
        env,
        SACConfig(
            rollout_steps=4,
            buffer_capacity=64,
            batch_size=8,
            gradient_steps=4,
            autotune_alpha=True,
        ),
    )
    carry = seed_replay(env, agent, seed=0, ticks=16)
    initial_alpha = float(jnp.exp(carry.agent.log_alpha))
    assert initial_alpha == pytest.approx(0.2)
    statistics = carry.statistics
    for step in range(4):
        updated, statistics = agent.update(carry.agent, carry.replay, jax.random.key(step))
        carry = carry._replace(agent=updated, statistics=statistics)
    final_alpha = float(statistics.alpha)
    assert final_alpha != initial_alpha
    assert float(jnp.exp(carry.agent.log_alpha)) == pytest.approx(final_alpha)
    assert "autotune_alpha" in json.dumps(agent.configuration)


def test_default_target_entropy_follows_the_action_bounds() -> None:
    assert SAC(action_low=[-1.0, -1.0], action_high=[1.0, 1.0]).target_entropy == -2.0
    assert SAC(action_low=[-0.1], action_high=[0.1]).target_entropy == pytest.approx(
        -1.0 + math.log(0.1), abs=1e-6
    )
    unit = SAC(action_low=[0.0] * 5, action_high=[1.0] * 5)
    assert unit.target_entropy == pytest.approx(-5.0 + 5 * math.log(0.5), abs=1e-5)
    explicit = SAC(action_low=[-0.1], action_high=[0.1], config=SACConfig(target_entropy=-0.5))
    assert explicit.target_entropy == -0.5
    state = SAC(action_low=[-0.1], action_high=[0.1]).init(jnp.zeros((1, 3)), jax.random.key(0))
    assert float(state.hyperparameters.target_entropy) == pytest.approx(-1.0 + math.log(0.1))


def test_autotune_lowers_alpha_on_a_narrow_box() -> None:
    # Entropy cannot exceed log(0.2) here, so the old raw-unit target of -1.0
    # sat above what the policy could reach and pushed alpha up on every step.
    agent = SAC(
        action_low=[-0.1],
        action_high=[0.1],
        config=SACConfig(rollout_steps=16, batch_size=8, buffer_capacity=16, autotune_alpha=True),
    )
    obs = jnp.zeros((1, 3))
    state = agent.init(obs, jax.random.key(0))
    replay = init_sac_replay(num_agents=1, obs_dim=3, action_dim=1, capacity=16)
    for _ in range(16):
        replay = insert(replay, (obs, jnp.zeros((1, 1)), jnp.zeros(1), obs, jnp.asarray([True])))
    updated, _ = agent.update(state, replay, jax.random.key(1))
    assert float(updated.log_alpha) < float(state.log_alpha)


def test_sac_initialises_and_updates_under_debug_nans() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=2)
    agent = agent_for(env, config)
    with jax.debug_nans(True):
        trainer = SACRunner(env, agent)
        state = trainer(trainer.init(seed=0, window_size=4))
    assert bool(state.healthy)


def zeros(leaf: Any) -> Any:
    return jnp.zeros_like(leaf)


def test_sac_reward_rate_settles_at_the_entropy_augmented_reward() -> None:
    # With every stored reward 1.0 the rate is 1 + gamma * alpha * entropy, not 1.
    rates: dict[float, tuple[float, float]] = {}
    for alpha in (0.0, 0.2):
        agent = SAC(
            action_low=[-1.0],
            action_high=[1.0],
            config=SACConfig(
                rollout_steps=16,
                reward_centering=True,
                batch_size=16,
                buffer_capacity=16,
                alpha=alpha,
                gamma=0.9,
            ),
        )
        obs = jnp.zeros((1, 3))
        state = agent.init(obs, jax.random.key(0))
        zero = jax.tree.map(zeros, state.params)
        state = state._replace(
            params=zero,
            hyperparameters=state.hyperparameters._replace(
                learning_rate=jnp.asarray(1e-12),
                tau=jnp.asarray(1.0),
                centering_step_size=jnp.asarray(0.1),
            ),
        )
        replay = init_sac_replay(num_agents=1, obs_dim=3, action_dim=1, capacity=16)
        for _ in range(16):
            replay = insert(replay, (obs, jnp.zeros((1, 1)), jnp.ones(1), obs, jnp.asarray([True])))
        update = cast(Any, jax.jit(agent.update))
        statistics = None
        for step in range(200):
            state, statistics = update(state, replay, jax.random.key(step))
        assert statistics is not None
        rates[alpha] = (float(state.reward_rate), float(statistics.entropy))
    assert rates[0.0][0] == pytest.approx(1.0, abs=1e-3)
    assert rates[0.2][0] == pytest.approx(1.0 + 0.9 * 0.2 * rates[0.2][1], abs=0.02)
    assert rates[0.2][0] > 1.05


def test_sac_acts_and_updates_under_strict_rank_promotion() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=2)
    trainer = SACRunner(env, agent_for(env, config))
    with jax.numpy_rank_promotion("raise"):
        state = trainer(trainer.init(seed=0, window_size=4))
    assert bool(state.healthy)


def test_sac_fixed_alpha_stays_exact() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    agent = agent_for(
        env,
        SACConfig(rollout_steps=4, buffer_capacity=64, batch_size=8, gradient_steps=1),
    )
    carry = seed_replay(env, agent, seed=0, ticks=16)
    log_alpha = np.asarray(carry.agent.log_alpha)
    updated, statistics = agent.update(carry.agent, carry.replay, jax.random.key(3))
    assert float(statistics.alpha) == float(np.float32(agent.config.alpha))
    np.testing.assert_array_equal(np.asarray(updated.log_alpha), log_alpha)
    assert float(updated.log_alpha) == pytest.approx(math.log(agent.config.alpha), abs=1e-6)


def test_sac_autotune_checkpoint_round_trip(tmp_path: Path) -> None:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(
        rollout_steps=4, buffer_capacity=64, batch_size=8, gradient_steps=2, autotune_alpha=True
    )
    trainer = SACRunner(env, agent_for(env, config))
    state = trainer(trainer.init(seed=0, window_size=4))
    path = tmp_path / "autotune"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert_same(trainer(state), trainer(restored))


def test_invalid_sac_configuration_and_bounds() -> None:
    with pytest.raises(ValueError, match=r"rollout_steps must be a positive integer"):
        SACConfig(rollout_steps=0)
    with pytest.raises(ValueError, match=r"buffer_capacity must be a positive integer"):
        SACConfig(buffer_capacity=0)
    with pytest.raises(ValueError, match=r"batch_size must be a positive integer"):
        SACConfig(batch_size=0)
    with pytest.raises(ValueError, match=r"batch_size must not exceed buffer_capacity"):
        SACConfig(batch_size=5, buffer_capacity=4)
    with pytest.raises(ValueError, match=r"gradient_steps must be a positive integer"):
        SACConfig(gradient_steps=0)
    with pytest.raises(ValueError, match=r"Gamma must be in \[0, 1\]"):
        SACConfig(gamma=1.5)
    with pytest.raises(ValueError, match=r"Tau must be in \(0, 1\]"):
        SACConfig(tau=0.0)
    with pytest.raises(ValueError, match=r"Tau must be in \(0, 1\]"):
        SACConfig(tau=1.5)
    with pytest.raises(ValueError, match=r"Alpha must be nonnegative"):
        SACConfig(alpha=-1.0)
    with pytest.raises(ValueError, match=r"Learning rate and maximum gradient norm must be"):
        SACConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match=r"Learning rate and maximum gradient norm must be"):
        SACConfig(max_grad_norm=0.0)
    with pytest.raises(ValueError, match=r"SAC action bounds must be finite and strictly"):
        SAC(action_low=[2.0], action_high=[-2.0])
    with pytest.raises(ValueError, match=r"SAC action bounds must be equally shaped nonempty"):
        SAC(action_low=[-2.0, -2.0], action_high=[2.0])
    env = pp.make("gymnax/Pendulum-v1")
    mismatched = SAC(action_low=[-1.0], action_high=[1.0])
    with pytest.raises(ValueError, match=r"SAC action bounds do not match the environment"):
        init_sac(env, mismatched, seed=0, window_size=4)
    with pytest.raises(ValueError, match=r"capacity must be a positive integer"):
        init_sac_replay(num_agents=1, obs_dim=3, action_dim=1, capacity=0)


@pytest.mark.parametrize("autotune", [False, True])
def test_sac_learns_pendulum_above_random(autotune: bool) -> None:
    env = pp.make("gymnax/Pendulum-v1")
    agent = agent_for(
        env,
        SACConfig(
            rollout_steps=128,
            buffer_capacity=65536,
            batch_size=256,
            gradient_steps=128,
            autotune_alpha=autotune,
        ),
    )
    trainer = SACRunner(env, agent)
    state = trainer.init(seed=0, window_size=1024)
    for _ in range(150):
        state = trainer(state)
    assert bool(state.healthy)
    window = float(np.asarray(state.lifetime.metrics.window_mean)[0])
    reference = pp.runtime.Runner(env, uniform_torque, ticks=128 * 150)(
        pp.runtime.init_lifetime(env, seed=0, window_size=1024)
    )
    random_mean = float(np.asarray(reference.metrics.mean)[0])
    assert window > -1.0
    assert window > random_mean + 2.0
    print(
        f"SAC(autotune={autotune}) window {window:.3f} vs random {random_mean:.3f} "
        f"alpha {float(state.statistics.alpha):.4f}"
    )


def seed_replay(
    env: DriftEnv[PendulumState, PendulumParams], agent: SAC, *, seed: int, ticks: int
) -> SACTrainingState[DriftEnvState[PendulumState]]:
    """Insert real transitions without running the compiled runner."""
    carry = init_sac(env, agent, seed=seed, window_size=4)
    for _ in range(ticks):
        agent_key, act_key = jax.random.split(carry.lifetime.agent_key)
        env_key, step_key = jax.random.split(carry.lifetime.env_key)
        sampled = agent.act(carry.agent, carry.lifetime.obs, carry.lifetime.info, act_key)
        state, obs, reward, info = env.step(carry.lifetime.state, sampled.action, step_key)
        lifetime = carry.lifetime._replace(
            state=state, obs=obs, info=info, env_key=env_key, agent_key=agent_key
        )
        transition = (
            carry.lifetime.obs,
            sampled.action,
            reward,
            obs,
            carry.lifetime.info["active_mask"],
        )
        carry = carry._replace(lifetime=lifetime, replay=insert(carry.replay, transition))
    return carry


def test_sac_trains_through_gravity_drift() -> None:
    env = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (5.0,)], 1024))
    agent = agent_for(
        env,
        SACConfig(
            rollout_steps=128,
            buffer_capacity=65536,
            batch_size=256,
            gradient_steps=128,
        ),
    )
    trainer = SACRunner(env, agent)
    state = trainer.init(seed=0, window_size=1024)
    trace: list[np.ndarray] = []
    for _ in range(150):
        state = trainer(state)
        trace.append(np.asarray(pp.evaluation.recent_rewards(state.lifetime.metrics, steps=128)))
    assert bool(state.healthy)
    # 19200 ticks with period 1024: 18 switches, 768 ticks into the next cycle.
    assert int(state.lifetime.state.drift["g"].elapsed) == 768
    assert int(state.lifetime.state.drift["g"].phase) == 0
    rewards = np.concatenate(trace)[:, 0]
    window = float(np.asarray(state.lifetime.metrics.window_mean)[0])
    reference = pp.runtime.Runner(env, uniform_torque, ticks=128 * 150)(
        pp.runtime.init_lifetime(env, seed=0, window_size=1024)
    )
    random_mean = float(np.asarray(reference.metrics.mean)[0])
    assert window > -1.0
    assert window > random_mean + 2.0
    assert rewards[-4800:].mean() > random_mean + 2.0
    changes = np.asarray([1023 + 1024 * k for k in range(18)])
    ticks = np.asarray(pp.evaluation.recovery_ticks(rewards, changes, window=128, fraction=0.5))
    capped = int(np.sum(ticks >= len(rewards) - changes))
    # One or two switches can take thousands of ticks to recover from, and which
    # ones depends on the platform's rounding: this seed's mean is 80 on macOS,
    # 472 on Linux and 700 on an H200. The median does not move with them.
    assert float(np.median(ticks)) < 400.0
    assert capped <= 2
    print(
        f"SAC through drift: window {window:.3f} vs random {random_mean:.3f}, "
        f"recovery mean {ticks.mean():.1f}, capped {capped}"
    )


def test_twin_critics_start_and_stay_independent() -> None:
    """Clipped double-Q needs two different critics; identical twins make min() a no-op."""
    env = pp.make("gymnax/Pendulum-v1")
    box = env.action_space
    config = SACConfig(
        rollout_steps=8, batch_size=8, gradient_steps=2, hidden_size=8, buffer_capacity=64
    )
    trainer = SACRunner(env, SAC(action_low=box.low[0], action_high=box.high[0], config=config))
    state = trainer.init(seed=0)

    def same(left: object, right: object) -> bool:
        return all(
            bool(np.array_equal(a, b))
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        )

    params = state.agent.params
    assert not same(params.q1, params.q2)
    assert same(params.q1, params.q1_target)
    assert same(params.q2, params.q2_target)
    for _ in range(3):
        state = trainer(state)
    assert not same(state.agent.params.q1, state.agent.params.q2)


def test_glorot_uniform_initialises_every_dense_layer_within_its_bound() -> None:
    low, high = np.asarray([-1.0, -1.0], np.float32), np.asarray([1.0, 1.0], np.float32)
    obs = jnp.zeros((1, 5), jnp.float32)
    default = SAC(action_low=low, action_high=high, config=SACConfig(hidden_size=16))
    glorot = SAC(
        action_low=low,
        action_high=high,
        config=SACConfig(hidden_size=16, kernel_init="glorot_uniform"),
    )
    key = jax.random.key(0)
    orthogonal_leaves = jax.tree.leaves(default.init(obs, key).params)
    glorot_leaves = jax.tree.leaves(glorot.init(obs, key).params)
    kernels = [leaf for leaf in glorot_leaves if leaf.ndim == 2]
    assert len(kernels) == 5 * 3 + 1  # three Dense layers in each network, two actor heads
    for kernel in kernels:
        bound = math.sqrt(6.0 / (kernel.shape[0] + kernel.shape[1]))
        assert float(jnp.max(jnp.abs(kernel))) <= bound
    assert any(
        a.shape == b.shape and not bool(jnp.array_equal(a, b))
        for a, b in zip(orthogonal_leaves, glorot_leaves, strict=True)
    )
    with pytest.raises(ValueError, match="kernel_init must be"):
        SACConfig(kernel_init="normal")  # pyright: ignore[reportArgumentType]


def test_sac_runs_under_x64_without_changing_carry_dtypes() -> None:
    with jax.enable_x64(True):
        env = pp.make("gymnax/Pendulum-v1")
        agent = agent_for(
            env,
            SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, random_steps=2),
        )
        trainer = SACRunner(env, agent)
        state = trainer(trainer.init(seed=0, window_size=4))
        assert state.pending.action.dtype == jnp.float32


def test_sac_bounds_must_be_one_slots_and_errors_say_so() -> None:
    low, high = np.full((4, 5), -1.0), np.full((4, 5), 1.0)
    with pytest.raises(ValueError, match=r"one slot's bounds.*shape \(4, 5\).*low\[0\]"):
        SAC(action_low=low, action_high=high)
    SAC(action_low=low[:1], action_high=high[:1])  # one slot written as (1, d) is fine
    env = pp.make("gymnax/Pendulum-v1")
    wrong = SAC(action_low=np.full(3, -2.0), action_high=np.full(3, 2.0))
    with pytest.raises(ValueError, match=r"action_dim=3.*shape \(1, 1\).*low\[0\]"):
        init_sac(env, wrong, seed=0, window_size=4)
    other = SAC(action_low=[-1.0], action_high=[1.0])
    with pytest.raises(ValueError, match=r"do not match.*low=\[-1.0\].*low=\[-2.0\]"):
        init_sac(env, other, seed=0, window_size=4)
    assert "one slot" in (SAC.__doc__ or "")


def test_reward_centering_settings_are_validated_at_construction() -> None:
    config: Any = SACConfig
    for value in ("no", 1, None):
        with pytest.raises(ValueError, match=r"reward_centering must be a boolean"):
            config(reward_centering=value)
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"centering_step_size must be finite"):
            config(centering_step_size=value)
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"centering_step_size must be in \[0, 1\]"):
            config(centering_step_size=value)
    assert config(reward_centering=True, centering_step_size=1.0).reward_centering is True


def test_non_finite_target_entropy_overrides_are_refused() -> None:
    agent = SAC(action_low=[-0.1], action_high=[0.1])
    obs, key = jnp.zeros((1, 3)), jax.random.key(0)
    defaults = agent.init(obs, key).hyperparameters
    unset = agent.init(obs, key, defaults._replace(target_entropy=jnp.asarray(jnp.inf)))
    assert float(unset.hyperparameters.target_entropy) == pytest.approx(-1.0 + math.log(0.1))
    for value in (jnp.nan, -jnp.inf):
        with pytest.raises(ValueError, match=r"hyperparameter target_entropy must be finite"):
            agent.init(obs, key, defaults._replace(target_entropy=jnp.asarray(value)))


def test_probe_feeds_the_critic_the_bounded_mean_action() -> None:
    agent = SAC(action_low=[0.0, 2.0], action_high=[1.0, 6.0], config=SACConfig(hidden_size=8))
    obs = jax.random.normal(jax.random.key(0), (16, 3))
    state = agent.init(obs, jax.random.key(1))
    actor = cast(Any, state.params.actor)
    actor["params"]["Dense_2"]["bias"] = jnp.asarray([4.0, -4.0])
    mean = cast(Any, agent.policy.apply(actor, obs))[0]
    bounded = jnp.tanh(mean) * jnp.asarray([0.5, 2.0]) + jnp.asarray([0.5, 4.0])
    assert float(jnp.max(jnp.abs(bounded - mean))) > 1.0
    hidden = cast(Any, agent.critic.apply(state.params.q1, obs, bounded, probe=True))[1]
    report = agent.probe(state, obs)
    np.testing.assert_array_equal(
        np.asarray(report["rank_critic"]), np.asarray(pp.probes.effective_rank(hidden))
    )
    raw = cast(Any, agent.critic.apply(state.params.q1, obs, mean, probe=True))[1]
    assert float(report["rank_critic"]) != float(pp.probes.effective_rank(raw))


def test_a_rollout_longer_than_the_replay_is_refused_as_for_dqn() -> None:
    from perpetua.agents.dqn import DQNConfig

    message = r"rollout_steps \(16\) must not exceed buffer_capacity \(8\).*oldest 8 ticks"
    for config in (SACConfig, DQNConfig):
        with pytest.raises(ValueError, match=message):
            config(rollout_steps=16, buffer_capacity=8, batch_size=4)
    SACConfig(rollout_steps=8, buffer_capacity=8, batch_size=4)
