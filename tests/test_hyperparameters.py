"""Traced hyperparameters: sweeps are data, vmappable and checkpointed."""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import math
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.agents import (
    DQN,
    IQL,
    PPO,
    SAC,
    DQNConfig,
    DQNRunner,
    PPOConfig,
    PPORunner,
    SACConfig,
    SACRunner,
)
from perpetua.agents.dqn import DQNHyperparameters, DQNTrainingState
from perpetua.agents.ppo import PPOHyperparameters, TrainingState
from perpetua.agents.sac import SACHyperparameters, init_sac


def _ppo_hyperparameters(config: PPOConfig, **overrides: float) -> PPOHyperparameters:
    base = PPOHyperparameters.from_config(config)
    values = {name: getattr(base, name) for name in base._fields}
    values.update(
        {name: jnp.asarray(value, dtype=jnp.float32) for name, value in overrides.items()}
    )
    return PPOHyperparameters(**values)


def _stack(first: Any, second: Any) -> Any:
    return jnp.stack([first, second])


def test_ppo_hyperparameter_sweep_is_data_not_code() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=5, config=config))

    def initialize(hyperparameters: PPOHyperparameters) -> TrainingState[Any]:
        return trainer.init(seed=0, window_size=4, hyperparameters=hyperparameters)

    def advance(state: TrainingState[Any]) -> TrainingState[Any]:
        return trainer(state)

    init_batch = jax.jit(jax.vmap(initialize))
    run_batch = jax.vmap(advance)
    sweep = cast(
        PPOHyperparameters,
        jax.tree.map(
            _stack,
            _ppo_hyperparameters(config, learning_rate=1e-3),
            _ppo_hyperparameters(config, learning_rate=3e-3),
        ),
    )
    initial = cast(TrainingState[Any], init_batch(sweep))
    states = cast(TrainingState[Any], jax.block_until_ready(run_batch(initial)))
    with pp.measurement.count_compilations() as counter:
        states = cast(TrainingState[Any], jax.block_until_ready(run_batch(states)))
        # A genuinely new batch of values must reuse the same compiled program.
        fresh = cast(
            PPOHyperparameters,
            jax.tree.map(
                _stack,
                _ppo_hyperparameters(config, learning_rate=5e-3),
                _ppo_hyperparameters(config, learning_rate=1e-2),
            ),
        )
        fresh_states = cast(TrainingState[Any], init_batch(fresh))
        states = cast(TrainingState[Any], jax.block_until_ready(run_batch(fresh_states)))
    assert counter.count == 0
    assert bool(jnp.all(states.healthy))
    slow = np.asarray(states.agent.params["params"]["Dense_0"]["kernel"][0])
    fast = np.asarray(states.agent.params["params"]["Dense_0"]["kernel"][1])
    assert not np.array_equal(slow, fast)
    np.testing.assert_allclose(np.asarray(states.agent.hyperparameters.learning_rate), [5e-3, 1e-2])
    np.testing.assert_allclose(
        np.asarray(initial.agent.hyperparameters.learning_rate), [1e-3, 3e-3]
    )


def test_ppo_hyperparameter_override_and_checkpoint(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    override = _ppo_hyperparameters(config, entropy_coefficient=0.05, gamma=0.9)
    state = trainer(trainer.init(seed=0, window_size=4, hyperparameters=override))
    assert float(state.agent.hyperparameters.entropy_coefficient) == pytest.approx(0.05)
    assert float(state.agent.hyperparameters.gamma) == pytest.approx(0.9)
    learner = trainer.metadata["learner"]
    assert isinstance(learner, dict)
    assert learner["config"]["entropy_coefficient"] == 0.01
    path = tmp_path / "sweep"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert float(restored.agent.hyperparameters.gamma) == pytest.approx(0.9)
    assert_same(trainer(state), trainer(restored))


def test_dqn_hyperparameters_override_epsilon_and_lr() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(rollout_steps=2, buffer_capacity=8, batch_size=2, gradient_steps=1)
    agent = DQN(action_size=5, config=config)
    state = agent.init(
        env.init(jax.random.key(0))[1],
        jax.random.key(1),
        DQNHyperparameters.from_config(config)._replace(
            epsilon=jnp.asarray(1.0), learning_rate=jnp.asarray(1e-2)
        ),
    )
    assert float(state.hyperparameters.epsilon) == 1.0
    assert float(state.hyperparameters.learning_rate) == pytest.approx(1e-2)
    info = env.observe(env.init(jax.random.key(0))[0])[1]
    obs = env.init(jax.random.key(0))[1]
    chosen = {int(agent.act(state, obs, info, jax.random.key(seed))[0]) for seed in range(64)}
    assert len(chosen) > 1


@pytest.mark.parametrize("value", [1e-3, np.float64(1e-3), np.float32(1e-3), 1])
def test_dqn_override_of_any_numeric_type_compiles_once_and_restores(
    value: object, tmp_path: Path
) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=4, gradient_steps=2)
    base = DQNHyperparameters.from_config(config)
    override = base._replace(learning_rate=cast(Any, value))
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    initial = trainer.init(seed=0, window_size=4, hyperparameters=override)
    leaf = initial.agent.hyperparameters.learning_rate
    assert isinstance(leaf, jax.Array)
    assert leaf.dtype == jnp.float32
    state = trainer(initial)
    with pp.measurement.count_compilations() as compiles:
        state = trainer(state)
    assert compiles.count == 0
    pp.checkpoint.save(tmp_path / "override", state, metadata=trainer.metadata)
    template = trainer.init(seed=0, window_size=4, hyperparameters=override)
    restored = pp.checkpoint.restore(
        tmp_path / "override", template, expected_metadata=trainer.metadata
    )
    assert_same(state, restored)


def test_ppo_sac_and_iql_overrides_are_converted_to_float32_arrays() -> None:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=2)
    obs = env.init(jax.random.key(0))[1]
    ppo = PPO(action_size=5, config=PPOConfig(hidden_size=8))
    ppo_override = PPOHyperparameters.from_config(ppo.config)._replace(
        learning_rate=cast(Any, 1e-3)
    )
    leaf = ppo.init(obs, jax.random.key(1), ppo_override).hyperparameters.learning_rate
    assert isinstance(leaf, jax.Array)
    assert leaf.dtype == jnp.float32
    sac = SAC(action_low=[-1.0], action_high=[1.0], config=SACConfig(hidden_size=8))
    sac_override = SACHyperparameters.from_config(sac.config)._replace(alpha=cast(Any, 0.5))
    leaf = sac.init(obs, jax.random.key(1), sac_override).hyperparameters.alpha
    assert isinstance(leaf, jax.Array)
    assert leaf.dtype == jnp.float32
    dqn_config = DQNConfig(hidden_size=8)
    iql = IQL(action_size=5, config=dqn_config)
    iql_override = DQNHyperparameters.from_config(dqn_config)._replace(
        learning_rate=cast(Any, np.asarray([1e-3, 2e-3]))
    )
    leaf = iql.init(obs, jax.random.key(1), iql_override).hyperparameters.learning_rate
    assert isinstance(leaf, jax.Array)
    assert leaf.dtype == jnp.float32


def test_iql_hyperparameters_broadcast_per_slot() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = DQNConfig(rollout_steps=2, buffer_capacity=8, batch_size=2, gradient_steps=1)
    agent = IQL(action_size=5, config=config)
    obs = env.init(jax.random.key(0))[1]
    scalar = DQNHyperparameters.from_config(config)._replace(
        epsilon=jnp.asarray(0.25), learning_rate=jnp.asarray(5e-4)
    )
    broadcast = agent.init(obs, jax.random.key(1), scalar)
    assert np.asarray(broadcast.hyperparameters.epsilon).shape == (2,)
    np.testing.assert_allclose(np.asarray(broadcast.hyperparameters.epsilon), [0.25, 0.25])
    per_slot = cast(
        DQNHyperparameters,
        jax.tree.map(
            _stack,
            DQNHyperparameters.from_config(config)._replace(epsilon=jnp.asarray(0.1)),
            DQNHyperparameters.from_config(config)._replace(epsilon=jnp.asarray(0.9)),
        ),
    )
    states = agent.init(obs, jax.random.key(1), per_slot)
    np.testing.assert_allclose(np.asarray(states.hyperparameters.epsilon), [0.1, 0.9])
    assert_same(
        agent.init(obs, jax.random.key(1)).hyperparameters,
        DQNHyperparameters.from_config(config),
    )


def test_sac_hyperparameters_override_tau() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1)
    agent = SAC(
        action_low=env.action_space.low[0], action_high=env.action_space.high[0], config=config
    )
    override = SACHyperparameters.from_config(config)._replace(
        tau=jnp.asarray(0.5), gamma=jnp.asarray(0.95)
    )
    state = agent.init(env.init(jax.random.key(0))[1], jax.random.key(1), override)
    assert float(state.hyperparameters.tau) == pytest.approx(0.5)
    assert float(state.hyperparameters.gamma) == pytest.approx(0.95)
    default = agent.init(env.init(jax.random.key(0))[1], jax.random.key(1))
    assert float(default.hyperparameters.tau) == pytest.approx(config.tau)


def test_runner_init_hyperparameter_sweep_across_seeds() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1)
    trainer = DQNRunner(env, DQN(action_size=5, config=config))

    def initialize(seed: jax.Array, epsilon: jax.Array) -> DQNTrainingState[Any]:
        return trainer.init(
            seed=seed,
            window_size=4,
            hyperparameters=DQNHyperparameters.from_config(config)._replace(epsilon=epsilon),
        )

    seeds = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    epsilons = jnp.asarray([0.0, 0.0, 0.5, 0.5], dtype=jnp.float32)
    states = cast(DQNTrainingState[Any], jax.jit(jax.vmap(initialize))(seeds, epsilons))
    assert np.asarray(states.agent.hyperparameters.epsilon).shape == (4,)

    def advance(state: DQNTrainingState[Any]) -> DQNTrainingState[Any]:
        return trainer(state)

    run = jax.vmap(advance)
    states = cast(DQNTrainingState[Any], jax.block_until_ready(run(run(states))))
    assert bool(jnp.all(states.healthy))
    np.testing.assert_array_equal(np.asarray(states.updates.lo), 2)
    assert not np.array_equal(
        np.asarray(states.agent.params["params"]["Dense_0"]["kernel"][0]),
        np.asarray(states.agent.params["params"]["Dense_0"]["kernel"][2]),
    )


def test_hyperparameter_overrides_are_validated() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    with pytest.raises(ValueError, match=r"Hyperparameter learning rate and maximum"):
        trainer.init(
            seed=0, window_size=4, hyperparameters=_ppo_hyperparameters(config, learning_rate=0.0)
        )
    with pytest.raises(ValueError, match=r"Hyperparameter clip_epsilon must be in"):
        trainer.init(
            seed=0, window_size=4, hyperparameters=_ppo_hyperparameters(config, clip_epsilon=1.5)
        )
    dqn_config = DQNConfig(rollout_steps=2, buffer_capacity=8, batch_size=2, gradient_steps=1)
    dqn = DQN(action_size=5, config=dqn_config)
    obs = env.init(jax.random.key(0))[1]
    with pytest.raises(ValueError, match=r"Hyperparameter epsilon must be in"):
        dqn.init(
            obs,
            jax.random.key(1),
            DQNHyperparameters.from_config(dqn_config)._replace(epsilon=jnp.asarray(2.0)),
        )
    sac_env = pp.make("gymnax/Pendulum-v1")
    sac_config = SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1)
    sac = SAC(
        action_low=sac_env.action_space.low[0],
        action_high=sac_env.action_space.high[0],
        config=sac_config,
    )
    with pytest.raises(ValueError, match=r"Hyperparameter tau must be in"):
        sac.init(
            sac_env.init(jax.random.key(0))[1],
            jax.random.key(1),
            SACHyperparameters.from_config(sac_config)._replace(tau=jnp.asarray(0.0)),
        )


def test_sac_fixed_alpha_override_is_used() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(rollout_steps=4, buffer_capacity=32, batch_size=8, gradient_steps=1)
    agent = SAC(
        action_low=env.action_space.low[0], action_high=env.action_space.high[0], config=config
    )
    override = SACHyperparameters.from_config(config)._replace(alpha=jnp.asarray(0.5))
    state = agent.init(env.init(jax.random.key(0))[1], jax.random.key(1), override)
    carry = init_sac(env, agent, seed=0, window_size=4, hyperparameters=override)
    for _ in range(2):
        carry = carry._replace(agent=agent.update(carry.agent, carry.replay, jax.random.key(2))[0])
    _, statistics = agent.update(carry.agent, carry.replay, jax.random.key(3))
    assert float(statistics.alpha) == pytest.approx(0.5)
    assert float(state.hyperparameters.alpha) == pytest.approx(0.5)
    # The default target follows the action bounds (+-2 here): -1 + log(2).
    assert float(state.hyperparameters.target_entropy) == pytest.approx(-1.0 + math.log(2.0))


def test_numpy_integers_in_learner_configurations_are_stored_as_python_ints() -> None:
    import json

    i64: Any = np.int64
    ppo = PPOConfig(rollout_steps=i64(32), hidden_size=i64(16), reset_interval=i64(5))
    dqn = DQNConfig(
        rollout_steps=i64(32), buffer_capacity=i64(64), batch_size=i64(8), reset_last_layers=i64(1)
    )
    sac = SACConfig(
        rollout_steps=i64(32),
        buffer_capacity=i64(64),
        batch_size=i64(8),
        hidden_layers=i64(2),
        random_steps=i64(0),
        update_after=i64(4),
    )
    for config in (ppo, dqn, sac):
        for value in vars(config).values():
            assert type(value) is not np.int64
    assert ppo.reset_interval == 5
    assert json.dumps(PPO(action_size=5, config=ppo).configuration)
    with pytest.raises(ValueError, match=r"rollout_steps must be a positive integer\. Got float"):
        PPOConfig(rollout_steps=cast(Any, 32.0))
    with pytest.raises(ValueError, match=r"Got bool True"):
        DQNConfig(batch_size=cast(Any, True))


def _learner(kind: str, **floats: Any) -> tuple[Any, Any]:
    if kind == "ppo":
        config = PPOConfig(rollout_steps=8, epochs=1, minibatches=2, hidden_size=8, **floats)
        env = pp.make("ContinualGrid-v0", size=3)
        return PPORunner(env, PPO(action_size=5, config=config)), config
    if kind == "dqn":
        config = DQNConfig(
            rollout_steps=8,
            buffer_capacity=16,
            batch_size=4,
            gradient_steps=1,
            hidden_size=8,
            **floats,
        )
        env = pp.make("ContinualGrid-v0", size=3)
        return DQNRunner(env, DQN(action_size=5, config=config)), config
    config = SACConfig(
        rollout_steps=8, buffer_capacity=16, batch_size=4, gradient_steps=1, hidden_size=8, **floats
    )
    env = pp.make("gymnax/Pendulum-v1")
    space = env.action_space
    agent = SAC(action_low=space.low[0], action_high=space.high[0], config=config)
    return SACRunner(env, agent), config


@pytest.mark.parametrize("kind", ["ppo", "dqn", "sac"])
def test_numpy_and_jax_floats_in_a_config_train_and_checkpoint(kind: str, tmp_path: Path) -> None:
    f32: Any = np.float32
    runner, config = _learner(kind, learning_rate=f32(1e-3), gamma=jnp.asarray(0.9))
    plain, reference = _learner(kind, learning_rate=float(f32(1e-3)), gamma=float(f32(0.9)))
    assert config == reference
    for value in cast(dict[str, Any], vars(config)).values():
        assert type(value) in (int, float, bool, str, type(None))
    assert runner.metadata == plain.metadata
    state = runner(runner.init(seed=0, window_size=8))
    assert bool(state.healthy)
    pp.checkpoint.save(tmp_path / "checkpoint", state, metadata=runner.metadata)
    restored = pp.checkpoint.restore(
        tmp_path / "checkpoint",
        runner.init(seed=9, window_size=8),
        expected_metadata=runner.metadata,
    )
    assert_same(state, restored)
    assert_same(runner(state), runner(restored))


def test_float_config_fields_refuse_booleans_strings_and_arrays() -> None:
    for config, name in (
        (PPOConfig, "learning_rate"),
        (PPOConfig, "gamma"),
        (PPOConfig, "centering_step_size"),
        (DQNConfig, "epsilon"),
        (DQNConfig, "gamma"),
        (SACConfig, "tau"),
        (SACConfig, "target_entropy"),
    ):
        for bad in (True, np.bool_(True), "0.5", jnp.asarray([0.5]), jnp.asarray(True)):
            with pytest.raises(ValueError, match=rf"{name} must be a real number\. Got "):
                config(**{name: cast(Any, bad)})
    assert PPOConfig(max_grad_norm=cast(Any, 1)).max_grad_norm == 1.0
    assert type(PPOConfig(max_grad_norm=cast(Any, np.int64(1))).max_grad_norm) is float
    assert type(SACConfig(target_entropy=cast(Any, np.float32(-1))).target_entropy) is float
    assert SACConfig().target_entropy is None
    assert type(PPOConfig(pixel_scale=cast(Any, np.float32(255))).pixel_scale) is float
    # The traced copies stay arrays, so a sweep over them still vmaps.
    traced = PPOHyperparameters.from_config(PPOConfig(learning_rate=cast(Any, np.float32(1e-3))))
    assert all(isinstance(leaf, jax.Array) and leaf.dtype == jnp.float32 for leaf in traced)
    unset = SACHyperparameters.from_config(SACConfig())
    assert math.isinf(float(unset.target_entropy))
