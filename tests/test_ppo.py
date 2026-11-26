"""PPO must learn online without episodic bootstrapping or partial checkpoints."""

# JAX/Flax transformation annotations are partial.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState

if TYPE_CHECKING:
    from perpetua.agents.ppo import PPORunner, TrainingState


@pytest.fixture
def trainer() -> "PPORunner[DriftEnvState[GridState]]":
    from perpetua.agents.ppo import PPO, PPOConfig, PPORunner

    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    config = PPOConfig(rollout_steps=8, epochs=2, minibatches=2, hidden_size=16)
    return PPORunner(env, PPO(action_size=5, config=config))


def test_gae_bootstraps_across_continuing_rollout_boundary() -> None:
    from perpetua.agents.ppo import compute_gae

    advantages, targets = compute_gae(
        rewards=jnp.asarray([[1.0], [2.0]]),
        values=jnp.asarray([[0.5], [0.75]]),
        last_value=jnp.asarray([1.0]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [[2.723], [2.15]], rtol=1e-6)
    np.testing.assert_allclose(targets, [[3.223], [2.9]], rtol=1e-6)


def test_gae_bridges_inactive_ticks_and_ignores_their_values() -> None:
    from perpetua.agents.ppo import compute_gae

    rewards = jnp.asarray([[1.0], [0.25], [2.0]])
    last_value = jnp.asarray([1.0])
    active = jnp.asarray([[True], [False], [True]])

    def run(inactive_value: float, mask: jax.Array | None) -> tuple[jax.Array, jax.Array]:
        return compute_gae(
            rewards=rewards,
            values=jnp.asarray([[0.5], [inactive_value], [0.75]]),
            last_value=last_value,
            gamma=0.9,
            gae_lambda=0.8,
            active=mask,
        )

    advantages, targets = run(123.0, active)
    other, _ = run(-7.0, active)
    np.testing.assert_array_equal(advantages[0], other[0])
    last = 2.0 + 0.9 * 1.0 - 0.75
    # Tick 0 bootstraps from tick 2 across the inactive tick, collecting its reward.
    bridged = 0.25 + 0.9 * 0.75
    first = (1.0 + 0.9 * bridged - 0.5) + (0.9 * 0.8) ** 2 * last
    np.testing.assert_allclose(advantages[:, 0], [first, (0.9 * 0.8) * last, last], rtol=1e-6)
    np.testing.assert_allclose(targets[0], first + 0.5, rtol=1e-6)

    everyone, _ = run(0.6, jnp.ones((3, 1), dtype=jnp.bool_))
    unmasked, _ = run(0.6, None)
    np.testing.assert_array_equal(everyone, unmasked)
    with pytest.raises(ValueError, match="active must be boolean"):
        run(0.6, jnp.ones((3, 1)))


def test_ppo_updates_parameters_optimizer_and_lifetime(
    trainer: "PPORunner[DriftEnvState[GridState]]",
) -> None:
    initial = trainer.init(seed=0)
    trained = trainer(initial)
    assert trained.lifetime.state.t.to_int() == 8
    assert trained.lifetime.metrics.ticks.to_int() == 8
    assert trained.updates.to_int() == 1
    assert trained.agent.gradient_steps.to_int() == 4
    assert bool(trained.healthy)
    before = cast(list[jax.Array], jax.tree.leaves(initial.agent.params))
    after = cast(list[jax.Array], jax.tree.leaves(trained.agent.params))
    assert any(not np.array_equal(x, y) for x, y in zip(before, after, strict=True))
    moments = cast(list[jax.Array], jax.tree.leaves(trained.agent.opt_state))
    assert any(np.any(np.asarray(value) != 0) for value in moments)


def test_ppo_policy_samples_only_valid_actions(
    trainer: "PPORunner[DriftEnvState[GridState]]",
) -> None:
    state = trainer.init(seed=0)
    agent = trainer.agent
    obs, info = state.lifetime.obs, state.lifetime.info

    def sample(key: jax.Array) -> jax.Array:
        return agent.act(state.agent, obs, info, key).action

    actions = np.asarray(jax.vmap(sample)(jax.random.split(jax.random.key(0), 32)))
    masks = np.asarray(info["action_mask"])
    assert np.all(masks[np.arange(4)[None, :], actions])


def test_ppo_does_not_reset_at_drift() -> None:
    from perpetua.agents.ppo import PPO, PPOConfig, PPORunner

    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 3))
    config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    state = trainer(trainer(trainer.init(seed=0)))
    assert state.lifetime.state.t.to_int() == 16
    assert state.updates.to_int() == 2
    assert state.agent.gradient_steps.to_int() == 2
    assert int(state.lifetime.state.drift["goal"].phase) == (16 // 3) % 2
    assert "oracle" not in state.lifetime.info


def test_ppo_checkpoint_restores_optimizer_and_rng(
    trainer: "PPORunner[DriftEnvState[GridState]]", tmp_path: Path
) -> None:
    state = trainer(trainer.init(seed=1))
    path = tmp_path / "training"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    restored = pp.checkpoint.restore(
        path, trainer.init(seed=999), expected_metadata=trainer.metadata
    )
    expected, actual = trainer(state), trainer(restored)
    assert jax.tree.structure(expected) == jax.tree.structure(actual)
    for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(x), jax.random.key_data(y))
        else:
            np.testing.assert_array_equal(x, y)


def test_ppo_configuration_validation() -> None:
    from perpetua.agents.ppo import PPOConfig

    with pytest.raises(ValueError, match=r"Learning rate and maximum gradient norm must be"):
        PPOConfig(learning_rate=-1)
    with pytest.raises(ValueError, match=r"Gamma and GAE lambda must be in"):
        PPOConfig(gamma=1.1)
    with pytest.raises(ValueError, match=r"rollout_steps must be a positive integer\."):
        PPOConfig(rollout_steps=0)


def test_inactive_slots_do_not_update_parameters_or_moments(
    trainer: "PPORunner[DriftEnvState[GridState]]",
) -> None:
    state = trainer.init(seed=0)
    world = state.lifetime.state._replace(
        env=state.lifetime.state.env._replace(active=jnp.zeros(4, dtype=jnp.bool_))
    )
    obs, info = trainer.env.observe(world)
    state = state._replace(lifetime=state.lifetime._replace(state=world, obs=obs, info=info))
    result = trainer(state)
    for x, y in zip(jax.tree.leaves(state.agent), jax.tree.leaves(result.agent), strict=True):
        np.testing.assert_array_equal(x, y)
    assert result.agent.gradient_steps.to_int() == 0
    assert result.lifetime.state.t.to_int() == 8
    assert bool(result.healthy)


def test_independent_learners_can_be_vmapped(
    trainer: "PPORunner[DriftEnvState[GridState]]",
) -> None:
    def initialize(seed: jax.Array) -> "TrainingState[DriftEnvState[GridState]]":
        return trainer.init(seed=seed)

    state = jax.vmap(initialize)(jnp.asarray([1, 2]))
    result = jax.vmap(trainer)(state)
    assert result.lifetime.obs.shape == (2, 4, 4)
    np.testing.assert_array_equal(result.lifetime.state.t.lo, [8, 8])
    np.testing.assert_array_equal(result.agent.gradient_steps.lo, [4, 4])
    assert np.all(np.asarray(result.healthy))


def test_learner_configuration_mismatch_is_rejected(
    trainer: "PPORunner[DriftEnvState[GridState]]", tmp_path: Path
) -> None:
    from dataclasses import replace

    from perpetua.agents.ppo import PPO, PPORunner

    state = trainer(trainer.init(seed=0))
    path = tmp_path / "training"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    config = replace(trainer.agent.config, learning_rate=1e-3)
    other = PPORunner(trainer.env, PPO(action_size=5, config=config))
    with pytest.raises(ValueError, match="configuration"):
        pp.checkpoint.restore(path, other.init(seed=0), expected_metadata=other.metadata)


def test_gradient_update_counter_exceeds_ten_billion(
    trainer: "PPORunner[DriftEnvState[GridState]]",
) -> None:
    state = trainer.init(seed=0)
    state = state._replace(
        agent=state.agent._replace(gradient_steps=pp.Clock.from_int(10_000_000_000))
    )
    assert trainer(state).agent.gradient_steps.to_int() == 10_000_000_004


def test_stationary_ppo_learns_above_random_reference() -> None:
    from perpetua.agents.ppo import PPO, PPORunner

    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    trainer = PPORunner(env, PPO(action_size=5))
    random_runner = pp.runtime.Runner(env, pp.policies.random, ticks=128)
    learned = trainer.init(seed=0)
    reference = pp.runtime.init_lifetime(env, seed=0)
    for _ in range(64):
        learned = trainer(learned)
        reference = random_runner(reference)
    assert bool(learned.healthy)
    assert learned.lifetime.state.t.to_int() == reference.state.t.to_int() == 8192
    assert float(np.asarray(learned.lifetime.metrics.mean).mean()) > (
        float(np.asarray(reference.metrics.mean).mean()) + 0.1
    )


def test_unknown_torso_is_refused() -> None:
    from perpetua.agents import DQNConfig, PPOConfig

    with pytest.raises(ValueError, match="torso"):
        PPOConfig(torso="CNN")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="torso"):
        DQNConfig(torso="bogus")  # pyright: ignore[reportArgumentType]


def test_action_size_may_come_from_the_action_space_and_mismatches_name_both_numbers() -> None:
    from perpetua.agents.ppo import IPPO, PPO, PPOConfig, PPORunner

    env = pp.make("ContinualGrid-v0", size=3, num_agents=2)
    size: Any = env.action_space.nvec[0]
    assert type(size).__name__ == "int64"
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=2, hidden_size=8)
    PPORunner(env, PPO(action_size=size, config=config)).init(seed=0)
    for agent in (PPO(action_size=3, config=config), IPPO(action_size=3, config=config)):
        name = type(agent).__name__
        with pytest.raises(
            ValueError, match=rf"{name} was built with action_size=3.*has 5 actions"
        ):
            PPORunner(env, agent).init(seed=0)


def test_ppo_and_ippo_refuse_a_continuous_environment_naming_sac() -> None:
    from perpetua.agents.ppo import IPPO, PPO, PPORunner

    env = pp.make("gymnax/Pendulum-v1")
    for agent in (PPO(action_size=1), IPPO(action_size=1)):
        with pytest.raises(ValueError, match=r"needs a discrete action space.*pp\.agents\.SAC"):
            PPORunner(env, agent).init(seed=0)


def test_the_mlp_torso_refuses_pixel_observations_and_names_the_cnn_torso() -> None:
    from perpetua.agents.ppo import IPPO, PPO, PPOConfig, PPORunner

    env = pp.make("gymnax/Breakout-MinAtar")
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    for agent in (PPO(action_size=3, config=config), IPPO(action_size=3, config=config)):
        with pytest.raises(ValueError, match=r"MLP torso needs \(agents, features\).*torso='cnn'"):
            PPORunner(env, agent).init(seed=0)
    cnn = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8, torso="cnn")
    PPORunner(env, PPO(action_size=3, config=cnn)).init(seed=0)


def test_minibatch_divisibility_errors_name_the_counts_and_the_pooling() -> None:
    from perpetua.agents.ppo import IPPO, PPO, PPOConfig, PPORunner

    env = pp.make("ContinualGrid-v0", size=3, num_agents=3)
    config = PPOConfig(rollout_steps=5, epochs=1, minibatches=2, hidden_size=8)
    with pytest.raises(ValueError, match=r"samples \(15: rollout_steps=5 x 3 agents, pooled\)"):
        PPORunner(env, PPO(action_size=5, config=config))
    with pytest.raises(
        ValueError, match=r"samples \(5: rollout_steps=5 per agent\).*2 minibatches"
    ):
        PPORunner(env, IPPO(action_size=5, config=config))


def test_reward_centering_settings_are_validated_at_construction() -> None:
    from perpetua.agents.ppo import PPOConfig

    config = cast(Any, PPOConfig)
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


def test_one_sample_minibatches_are_refused_and_the_defaults_are_not() -> None:
    from perpetua.agents.ppo import IPPO, PPO, PPOConfig, PPORunner, Rollout

    def config(steps: int, minibatches: int) -> PPOConfig:
        return PPOConfig(rollout_steps=steps, epochs=1, minibatches=minibatches, hidden_size=8)

    one = pp.make("ContinualGrid-v0", size=3)
    four = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    with pytest.raises(
        ValueError,
        match=r"samples \(8: rollout_steps=8 x 1 agents, pooled\) split into 8 minibatches "
        r"leave 1 sample per minibatch.*at least 2",
    ):
        PPORunner(one, PPO(action_size=5, config=config(8, 8)))
    with pytest.raises(ValueError, match=r"rollout_steps=2 x 4 agents, pooled.*leave 1 sample"):
        PPORunner(four, PPO(action_size=5, config=config(2, 8)))
    # IPPO splits each agent's own samples, so four slots do not help.
    with pytest.raises(ValueError, match=r"samples \(8: rollout_steps=8 per agent\).*leave 1"):
        PPORunner(four, IPPO(action_size=5, config=config(8, 8)))
    with pytest.raises(ValueError, match=r"rollout_steps=4 per agent.*leave 1 sample"):
        PPORunner(four, IPPO(action_size=5, config=PPOConfig(rollout_steps=4)))
    PPORunner(four, PPO(action_size=5, config=config(2, 4)))
    PPORunner(four, IPPO(action_size=5, config=config(8, 4)))
    PPORunner(one, PPO(action_size=5))
    PPORunner(four, IPPO(action_size=5))

    # The update guards itself, for a rollout that did not come from a runner.
    agent = PPO(action_size=5, config=config(8, 8))
    obs = jnp.zeros((8, 1, 4))
    state = agent.init(obs[0], jax.random.key(0))
    zeros = jnp.zeros((8, 1))
    rollout = Rollout(
        obs,
        jnp.ones((8, 1, 5), dtype=jnp.bool_),
        jnp.ones((8, 1), dtype=jnp.bool_),
        jnp.zeros((8, 1), dtype=jnp.int32),
        zeros,
        zeros,
        zeros,
    )
    with pytest.raises(ValueError, match=r"samples \(8\) split into 8 minibatches leave 1 sample"):
        agent.update(state, rollout, jnp.zeros(1), jax.random.key(1))


def test_a_learner_runner_inside_jit_is_refused_with_the_remedy(
    trainer: "PPORunner[DriftEnvState[GridState]]",
) -> None:
    def initialize(seed: jax.Array) -> "TrainingState[DriftEnvState[GridState]]":
        return trainer.init(seed=seed)

    with pytest.raises(ValueError, match=r"outermost compiled call.*jax\.vmap\(runner\)"):
        jax.jit(trainer)(trainer.init(seed=0))
    batch = jax.vmap(initialize)(jnp.asarray([1, 2]))
    with pytest.raises(ValueError, match=r"outermost compiled call"):
        jax.jit(jax.vmap(trainer))(batch)
    assert np.all(np.asarray(jax.vmap(trainer)(batch).healthy))
