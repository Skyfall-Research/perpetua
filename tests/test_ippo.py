"""Independent PPO must keep each agent's parameters, moments and credit separate."""

# JAX transformations are partially annotated.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.agents import IPPO, PPO, PPOConfig, PPORunner
from perpetua.agents.ppo import Rollout, TrainingState
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def test_ippo_updates_independent_models_without_changing_tree_shapes(tmp_path: Path) -> None:
    env = pp.make("CooperativeGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 3))
    config = PPOConfig(rollout_steps=8, epochs=2, minibatches=2, hidden_size=8)
    trainer = PPORunner(env, IPPO(action_size=5, config=config))
    initial = trainer.init(seed=0, window_size=4)
    assert initial.statistics.loss.shape == (2,)
    assert initial.agent.gradient_steps.lo.shape == (2,)
    assert any(not np.array_equal(x[0], x[1]) for x in jax.tree.leaves(initial.agent.params))
    trained = trainer(initial)
    np.testing.assert_array_equal(trained.agent.gradient_steps.lo, [4, 4])
    assert trained.lifetime.state.t.to_int() == 8
    assert trained.updates.to_int() == 1
    assert bool(trained.healthy)
    before, after = jax.tree.leaves(initial), jax.tree.leaves(trained)
    assert jax.tree.structure(initial) == jax.tree.structure(trained)
    assert all(a.shape == b.shape for a, b in zip(before, after, strict=True))
    path = tmp_path / "ippo"
    pp.checkpoint.save(path, trained, metadata=trainer.metadata)
    restored = pp.checkpoint.restore(
        path, trainer.init(seed=999, window_size=4), expected_metadata=trainer.metadata
    )
    assert_same(trainer(trained), trainer(restored))
    assert cast(dict[str, object], trainer.metadata["learner"])["algorithm"] == "ippo"
    other = PPORunner(env, PPO(action_size=5, config=config))
    with pytest.raises(ValueError, match="configuration"):
        pp.checkpoint.restore(path, other.init(window_size=4), expected_metadata=other.metadata)


def test_inactive_agent_does_not_change_its_parameters_or_optimizer() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    trainer = PPORunner(
        env,
        IPPO(
            action_size=5, config=PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
        ),
    )
    state = trainer.init(seed=0)
    world = state.lifetime.state._replace(
        env=state.lifetime.state.env._replace(active=jnp.asarray([True, False]))
    )
    obs, info = env.observe(world)
    state = state._replace(lifetime=state.lifetime._replace(state=world, obs=obs, info=info))
    trained = trainer(state)
    for a, b in zip(jax.tree.leaves(state.agent), jax.tree.leaves(trained.agent), strict=True):
        np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(trained.agent.gradient_steps.lo, [1, 0])
    assert trained.lifetime.state.t.to_int() == 8


def test_changing_one_agents_rewards_cannot_change_another_optimizer() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    _, obs, info = env.init(jax.random.key(0))
    agent = IPPO(
        action_size=5, config=PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    )
    state = agent.init(obs, jax.random.key(1))
    sampled = agent.act(state, obs, info, jax.random.key(2))
    rollout = Rollout(
        jnp.broadcast_to(obs, (8, *obs.shape)),
        jnp.broadcast_to(info["action_mask"], (8, 2, 5)),
        jnp.ones((8, 2), dtype=jnp.bool_),
        jnp.broadcast_to(sampled.action, (8, 2)),
        jnp.broadcast_to(sampled.value, (8, 2)),
        jnp.zeros((8, 2)),
        jnp.broadcast_to(sampled.log_prob, (8, 2)),
    )
    changed = rollout._replace(reward=rollout.reward.at[:, 0].set(jnp.arange(8, dtype=jnp.float32)))
    a, key_a, _ = agent.update(state, rollout, sampled.value, jax.random.key(3))
    b, key_b, _ = agent.update(state, changed, sampled.value, jax.random.key(3))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x[1], y[1])
    assert any(
        not np.array_equal(x[0], y[0])
        for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params), strict=True)
    )
    assert_same(key_a, key_b)


def test_ippo_samples_only_valid_actions_and_vmaps_whole_lifetimes() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    trainer = PPORunner(
        env,
        IPPO(
            action_size=5, config=PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
        ),
    )
    state = trainer.init(seed=0)

    def sample(key: jax.Array) -> jax.Array:
        return trainer.agent.act(state.agent, state.lifetime.obs, state.lifetime.info, key).action

    def initialize(seed: jax.Array) -> TrainingState[DriftEnvState[GridState]]:
        return trainer.init(seed=seed)

    actions = jax.vmap(sample)(jax.random.split(jax.random.key(0), 32))
    assert np.all(
        np.asarray(state.lifetime.info["action_mask"])[np.arange(2)[None, :], np.asarray(actions)]
    )
    batch = jax.vmap(initialize)(jnp.asarray([0, 1]))
    trained = jax.vmap(trainer)(batch)
    np.testing.assert_array_equal(trained.agent.gradient_steps.lo, [[1, 1], [1, 1]])
    np.testing.assert_array_equal(trained.lifetime.state.t.lo, [4, 4])


def test_ippo_minibatches_are_per_agent_not_pooled() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = PPOConfig(rollout_steps=3, epochs=1, minibatches=2, hidden_size=8)
    with pytest.raises(ValueError, match="minibatches"):
        PPORunner(env, IPPO(action_size=5, config=config))


def test_independent_gradient_clocks_cross_ten_billion_ticks() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    agent = IPPO(
        action_size=5, config=PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    )
    trainer = PPORunner(env, agent)
    state = trainer.init(seed=0)
    clock = pp.Clock.from_int(10_000_000_000)
    state = state._replace(
        agent=state.agent._replace(
            gradient_steps=pp.Clock(
                jnp.broadcast_to(clock.hi, (2,)), jnp.broadcast_to(clock.lo, (2,))
            )
        )
    )
    trained = trainer(state)
    for i in range(2):
        assert (
            pp.Clock(
                trained.agent.gradient_steps.hi[i], trained.agent.gradient_steps.lo[i]
            ).to_int()
            == 10_000_000_001
        )


def test_ippo_learns_coordination_above_a_seeded_random_reference() -> None:
    env = pp.make("CooperativeGrid-v0", size=2)
    trainer = PPORunner(env, IPPO(action_size=5))
    reference = pp.runtime.Runner(env, pp.policies.random, ticks=128)
    learned = trainer.init(seed=0)
    random = pp.runtime.init_lifetime(env, seed=0)
    for _ in range(64):
        learned, random = trainer(learned), reference(random)
    assert learned.lifetime.state.t.to_int() == random.state.t.to_int() == 8192
    assert bool(learned.healthy)
    assert float(np.asarray(learned.lifetime.metrics.mean).mean()) > (
        float(np.asarray(random.metrics.mean).mean()) + 0.1
    )


def test_ippo_validates_each_slots_hyperparameters_like_ppo() -> None:
    from perpetua.agents.ppo import PPOHyperparameters

    env = pp.make("CooperativeGrid-v0", size=3, num_agents=2)
    _, obs, _ = env.init(jax.random.key(0))
    bad = PPOHyperparameters.from_config(PPOConfig())._replace(learning_rate=jnp.asarray(-1.0))
    with pytest.raises(ValueError, match=r"Hyperparameter learning rate and maximum"):
        PPO(action_size=5).init(obs, jax.random.key(0), hyperparameters=bad)
    with pytest.raises(ValueError, match=r"Hyperparameter learning rate and maximum"):
        IPPO(action_size=5).init(obs, jax.random.key(0), hyperparameters=bad)
    per_slot = bad._replace(learning_rate=jnp.asarray([3e-4, -1.0]))
    with pytest.raises(ValueError, match=r"Hyperparameter learning rate and maximum"):
        IPPO(action_size=5).init(obs, jax.random.key(0), hyperparameters=per_slot)
