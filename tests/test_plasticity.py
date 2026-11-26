"""Plasticity options must be exact mechanisms: CReLU, scheduled resets, L2."""

# JAX/Flax/Optax expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.agents import DQN, IQL, SAC, DQNConfig, DQNRunner, SACConfig, SACRunner
from perpetua.agents.dqn import DQNHyperparameters, init_replay, insert, reset_dqn_layers
from perpetua.agents.ppo import IPPO, PPO, PPOConfig, PPORunner, crelu, reset_state_layers
from perpetua.agents.sac import SACHyperparameters, init_sac_replay
from perpetua.agents.sac import insert as insert_sac


def test_crelu_activation_math() -> None:
    values = jnp.asarray([-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(crelu(values), [0.0, 0.0, 2.0, 1.0, 0.0, 0.0])


def test_plasticity_config_validation() -> None:
    with pytest.raises(ValueError, match=r"activation must be 'tanh' or 'crelu'"):
        PPOConfig(activation="relu")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"reset_interval must be None or a positive integer"):
        PPOConfig(reset_interval=0)
    with pytest.raises(ValueError, match=r"reset_interval must be None or a positive integer"):
        PPOConfig(reset_interval=-3)
    with pytest.raises(ValueError, match=r"reset_interval must be None or a positive integer"):
        PPOConfig(reset_interval=cast(int, 1.5))  # deliberately wrong runtime type
    with pytest.raises(ValueError, match=r"reset_last_layers must be a nonnegative integer"):
        PPOConfig(reset_last_layers=-1)
    with pytest.raises(ValueError, match=r"l2_coefficient must be nonnegative"):
        PPOConfig(l2_coefficient=-1e-3)
    with pytest.raises(ValueError, match=r"l2_coefficient must be finite"):
        PPOConfig(l2_coefficient=float("inf"))


def test_reset_state_layers_replaces_only_top_layers() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    agent = PPO(action_size=5, config=PPOConfig(hidden_size=8))
    state = agent.init(obs, jax.random.key(1))
    trained = state._replace(params={"params": state.params["params"]}, reset_age=jnp.int32(3))
    fresh = agent.init(obs, jax.random.key(2))
    reset = reset_state_layers(trained, fresh, 2)
    names = sorted(trained.params["params"])
    assert names[-2:] == ["Dense_4", "Dense_5"]
    for name in names:
        target = (
            fresh.params["params"][name] if name in names[-2:] else trained.params["params"][name]
        )
        np.testing.assert_array_equal(
            np.asarray(reset.params["params"][name]["kernel"]),
            np.asarray(target["kernel"]),
        )
    assert reset.gradient_steps.to_int() == trained.gradient_steps.to_int()
    assert int(reset.reset_age) == 0
    reset_names = {"Dense_4", "Dense_5"}
    leaf_paths = cast(
        list[tuple[tuple[object, ...], jax.Array]],
        jax.tree.leaves_with_path(reset.opt_state),
    )
    for (path, value), fresh_leaf, old_leaf in zip(
        leaf_paths,
        jax.tree.leaves(fresh.opt_state),
        jax.tree.leaves(trained.opt_state),
        strict=True,
    ):
        labels: set[Any] = set()
        for raw in path:
            step = cast(Any, raw)
            label = getattr(step, "key", None)
            if label is None:
                label = getattr(step, "name", step)
            labels.add(label)
        if labels & reset_names:
            np.testing.assert_array_equal(np.asarray(value), np.asarray(fresh_leaf))
        else:
            np.testing.assert_array_equal(np.asarray(value), np.asarray(old_leaf))
    with pytest.raises(ValueError, match=r"reset_last_layers exceeds the network depth"):
        reset_state_layers(trained, fresh, 99)
    untouched = reset_state_layers(trained, fresh, 0)
    np.testing.assert_array_equal(
        np.asarray(untouched.params["params"]["Dense_5"]["kernel"]),
        np.asarray(trained.params["params"]["Dense_5"]["kernel"]),
    )


def test_reset_schedule_freshens_heads_and_keeps_the_trunk() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8, reset_interval=2)
    agent = PPO(action_size=5, config=config)
    trainer = PPORunner(env, agent)
    state = trainer.init(seed=0, window_size=4)
    first = trainer(state)
    assert int(first.agent.reset_age) == 1
    second = trainer(first)
    assert int(second.agent.reset_age) == 0
    trunk_before = np.asarray(first.agent.params["params"]["Dense_0"]["kernel"])
    trunk_after = np.asarray(second.agent.params["params"]["Dense_0"]["kernel"])
    assert not np.array_equal(trunk_before, trunk_after)
    heads_before = np.asarray(first.agent.params["params"]["Dense_5"]["kernel"])
    heads_after = np.asarray(second.agent.params["params"]["Dense_5"]["kernel"])
    assert not np.array_equal(heads_before, heads_after)
    twin = trainer.init(seed=0, window_size=4)
    assert_same(trainer(trainer(twin)), second)


def test_ppo_plasticity_options_train_and_checkpoint_exactly(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = PPOConfig(
        rollout_steps=4,
        epochs=1,
        minibatches=1,
        hidden_size=8,
        activation="crelu",
        l2_coefficient=1e-4,
        reset_interval=1,
    )
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    for _ in range(2):
        state = trainer(state)
    assert bool(state.healthy)
    assert int(state.agent.reset_age) == 0
    learner = trainer.metadata["learner"]
    assert isinstance(learner, dict)
    assert learner["config"]["activation"] == "crelu"
    path = tmp_path / "plasticity"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert_same(trainer(state), trainer(restored))


def test_ippo_plasticity_options_keep_per_agent_reset_ages() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = PPOConfig(
        rollout_steps=4,
        epochs=1,
        minibatches=1,
        hidden_size=8,
        activation="crelu",
        reset_interval=3,
    )
    trainer = PPORunner(env, IPPO(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    assert bool(state.healthy)
    assert np.asarray(state.agent.reset_age).shape == (2,)
    assert np.all(np.asarray(state.agent.reset_age) == 1)


def test_crelu_reset_learner_improves_above_random() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    config = PPOConfig(
        rollout_steps=32,
        hidden_size=64,
        activation="crelu",
        l2_coefficient=1e-5,
        reset_interval=16,
    )
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    state = trainer.init(seed=0, window_size=128)
    for _ in range(96):
        state = trainer(state)
    assert bool(state.healthy)
    learned = float(np.asarray(state.lifetime.metrics.mean)[0])
    baseline_env = pp.make("ContinualGrid-v0", size=4)
    baseline = pp.runtime.Runner(baseline_env, pp.policies.random, ticks=96 * 32)(
        pp.runtime.init_lifetime(baseline_env, seed=0, window_size=128)
    )
    baseline_mean = float(np.asarray(baseline.metrics.mean)[0])
    assert learned > 2 * baseline_mean
    print(f"plasticity PPO mean {learned:.4f} vs random {baseline_mean:.4f}")


def test_offpolicy_plasticity_config_validation() -> None:
    with pytest.raises(ValueError, match=r"activation must be 'tanh' or 'crelu'"):
        DQNConfig(activation="relu")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"reset_interval must be None or a positive integer"):
        DQNConfig(reset_interval=0)
    with pytest.raises(ValueError, match=r"reset_interval must be None or a positive integer"):
        DQNConfig(reset_interval=cast(int, 1.5))
    with pytest.raises(ValueError, match=r"reset_last_layers must be a nonnegative integer"):
        DQNConfig(reset_last_layers=-1)
    with pytest.raises(ValueError, match=r"l2_coefficient must be nonnegative"):
        DQNConfig(l2_coefficient=-1e-3)
    with pytest.raises(ValueError, match=r"l2_coefficient must be finite"):
        DQNConfig(l2_coefficient=float("inf"))
    with pytest.raises(ValueError, match=r"activation must be 'tanh', 'crelu' or 'leaky_relu'"):
        SACConfig(activation="relu")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"reset_interval must be None or a positive integer"):
        SACConfig(reset_interval=0)
    with pytest.raises(ValueError, match=r"reset_last_layers must be a nonnegative integer"):
        SACConfig(reset_last_layers=-1)
    with pytest.raises(ValueError, match=r"l2_coefficient must be nonnegative"):
        SACConfig(l2_coefficient=-1e-3)


def test_crelu_widens_offpolicy_hidden_layers() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    plain = DQN(action_size=5, config=DQNConfig(hidden_size=8))
    wide = DQN(action_size=5, config=DQNConfig(hidden_size=8, activation="crelu"))
    plain_state = plain.init(obs, jax.random.key(1))
    wide_state = wide.init(obs, jax.random.key(1))
    assert np.asarray(plain_state.params["params"]["Dense_1"]["kernel"]).shape == (8, 8)
    assert np.asarray(wide_state.params["params"]["Dense_1"]["kernel"]).shape == (16, 8)
    pendulum = pp.make("gymnax/Pendulum-v1")
    wide_sac = SAC(
        action_low=pendulum.action_space.low[0],
        action_high=pendulum.action_space.high[0],
        config=SACConfig(hidden_size=8, activation="crelu"),
    )
    sac_state = wide_sac.init(pendulum.init(jax.random.key(0))[1], jax.random.key(1))
    assert np.asarray(sac_state.params.actor["params"]["Dense_1"]["kernel"]).shape == (16, 8)
    assert np.asarray(sac_state.params.q1["params"]["Dense_1"]["kernel"]).shape == (16, 8)


def test_dqn_reset_replaces_only_top_layers() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    agent = DQN(action_size=5, config=DQNConfig(hidden_size=8))
    state = agent.init(obs, jax.random.key(1))
    fresh = agent.init(obs, jax.random.key(2))

    def bump_one(leaf: Any) -> Any:
        return jnp.asarray(leaf) + 1.0

    def bump_two(leaf: Any) -> Any:
        return jnp.asarray(leaf) + 2.0

    trained = state._replace(
        params=jax.tree.map(bump_one, state.params),
        target=jax.tree.map(bump_two, state.target),
        reset_age=jnp.int32(5),
    )
    reset = reset_dqn_layers(trained, fresh, 2)
    names = sorted(trained.params["params"])
    assert names == ["Dense_0", "Dense_1", "Dense_2"]
    for name in names:
        resetting = name in names[-2:]
        expected = fresh.params["params"][name] if resetting else trained.params["params"][name]
        np.testing.assert_array_equal(
            np.asarray(reset.params["params"][name]["kernel"]), np.asarray(expected["kernel"])
        )
        expected_target = (
            fresh.params["params"][name] if resetting else trained.target["params"][name]
        )
        np.testing.assert_array_equal(
            np.asarray(reset.target["params"][name]["kernel"]),
            np.asarray(expected_target["kernel"]),
        )
    assert int(reset.reset_age) == 0
    assert int(reset.sync_age) == int(trained.sync_age)
    assert reset.gradient_steps.to_int() == trained.gradient_steps.to_int()
    with pytest.raises(ValueError, match=r"reset_last_layers exceeds the network depth"):
        reset_dqn_layers(trained, fresh, 99)


def test_dqn_plasticity_options_train_and_checkpoint_exactly(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(
        rollout_steps=4,
        buffer_capacity=16,
        batch_size=2,
        gradient_steps=1,
        hidden_size=8,
        activation="crelu",
        l2_coefficient=1e-4,
        reset_interval=2,
    )
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    assert int(state.agent.reset_age) == 1
    first = state
    state = trainer(state)
    assert int(state.agent.reset_age) == 0
    assert not np.array_equal(
        np.asarray(first.agent.params["params"]["Dense_2"]["kernel"]),
        np.asarray(state.agent.params["params"]["Dense_2"]["kernel"]),
    )
    assert bool(state.healthy)
    twin = trainer.init(seed=0, window_size=4)
    assert_same(trainer(trainer(twin)), state)
    learner = trainer.metadata["learner"]
    assert isinstance(learner, dict)
    assert learner["config"]["activation"] == "crelu"
    path = tmp_path / "dqn-plasticity"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert_same(trainer(state), trainer(restored))


def test_iql_plasticity_keeps_per_slot_reset_ages() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    config = DQNConfig(
        rollout_steps=4,
        buffer_capacity=16,
        batch_size=2,
        gradient_steps=1,
        hidden_size=8,
        activation="crelu",
        reset_interval=3,
    )
    trainer = DQNRunner(env, IQL(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    assert bool(state.healthy)
    assert np.asarray(state.agent.reset_age).shape == (2,)
    assert np.all(np.asarray(state.agent.reset_age) == 1)


def test_sac_plasticity_options_train_and_checkpoint_exactly(tmp_path: Path) -> None:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(
        rollout_steps=4,
        buffer_capacity=16,
        batch_size=2,
        gradient_steps=1,
        hidden_size=8,
        activation="crelu",
        l2_coefficient=1e-4,
        reset_interval=2,
    )
    agent = SAC(
        action_low=env.action_space.low[0], action_high=env.action_space.high[0], config=config
    )
    trainer = SACRunner(env, agent)
    state = trainer(trainer.init(seed=0, window_size=4))
    assert int(state.agent.reset_age) == 1
    first = state
    state = trainer(state)
    assert int(state.agent.reset_age) == 0
    assert not np.array_equal(
        np.asarray(first.agent.params.actor["params"]["Dense_3"]["kernel"]),
        np.asarray(state.agent.params.actor["params"]["Dense_3"]["kernel"]),
    )
    assert bool(state.healthy)
    path = tmp_path / "sac-plasticity"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert_same(trainer(state), trainer(restored))


def test_l2_penalty_enters_offpolicy_losses() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    plain = DQN(action_size=5, config=DQNConfig(hidden_size=8))
    regularized = DQN(action_size=5, config=DQNConfig(hidden_size=8, l2_coefficient=1e-3))
    state = plain.init(obs, jax.random.key(1))
    replay = init_replay(num_agents=obs.shape[0], obs_dim=obs.shape[1], action_size=5, capacity=4)
    transition = (
        obs,
        jnp.ones((obs.shape[0], 5), dtype=jnp.bool_),
        jnp.ones(obs.shape[0], dtype=jnp.bool_),
        jnp.zeros(obs.shape[0], dtype=jnp.int32),
        jnp.ones(obs.shape[0], dtype=jnp.float32),
        obs,
        jnp.ones((obs.shape[0], 5), dtype=jnp.bool_),
    )
    for _ in range(4):
        replay = insert(replay, transition)
    # Hyperparameters live in the state now: give the regularized agent its
    # own configuration values while sharing the same parameters.
    regularized_state = state._replace(
        hyperparameters=DQNHyperparameters.from_config(regularized.config)
    )
    _, plain_stats = plain.update(state, replay, jax.random.key(3))
    _, l2_stats = regularized.update(regularized_state, replay, jax.random.key(3))
    assert float(l2_stats.loss) > float(plain_stats.loss)

    pendulum = pp.make("gymnax/Pendulum-v1")
    plain_sac = SAC(
        action_low=pendulum.action_space.low[0],
        action_high=pendulum.action_space.high[0],
        config=SACConfig(rollout_steps=4, buffer_capacity=16, batch_size=2, gradient_steps=1),
    )
    l2_sac = SAC(
        action_low=pendulum.action_space.low[0],
        action_high=pendulum.action_space.high[0],
        config=SACConfig(
            rollout_steps=4,
            buffer_capacity=16,
            batch_size=2,
            gradient_steps=1,
            l2_coefficient=1e-3,
        ),
    )
    sac_obs = pendulum.init(jax.random.key(0))[1]
    sac_state = plain_sac.init(sac_obs, jax.random.key(1))
    sac_replay = init_sac_replay(num_agents=1, obs_dim=3, action_dim=1, capacity=4)
    sac_transition = (
        sac_obs,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        sac_obs,
        jnp.ones((1,), dtype=jnp.bool_),
    )
    for _ in range(4):
        sac_replay = insert_sac(sac_replay, sac_transition)
    sac_regularized_state = sac_state._replace(
        hyperparameters=SACHyperparameters.from_config(l2_sac.config)
    )
    _, sac_plain_stats = plain_sac.update(sac_state, sac_replay, jax.random.key(3))
    _, sac_l2_stats = l2_sac.update(sac_regularized_state, sac_replay, jax.random.key(3))
    assert float(sac_l2_stats.critic_loss) > float(sac_plain_stats.critic_loss)
    assert float(sac_l2_stats.actor_loss) > float(sac_plain_stats.actor_loss)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SACConfig(), {"Dense_1", "Dense_2", "Dense_3"}),
        (SACConfig(hidden_layers=9), {"Dense_8", "Dense_9", "Dense_10"}),
        (SACConfig(hidden_layers=4, layer_norm=True), {"Dense_3", "Dense_4", "Dense_5"}),
    ],
)
def test_reset_selects_layers_by_depth_not_name(config: SACConfig, expected: set[str]) -> None:
    """ "Last N layers" means the deepest N, even past Dense_9 or next to a LayerNorm."""
    from perpetua.agents.plasticity import (
        _top_layer_names,  # pyright: ignore[reportPrivateUsage]
    )

    env = pp.make("gymnax/Pendulum-v1")
    box = env.action_space
    _, obs, _ = env.init(jax.random.key(0))
    state = SAC(action_low=box.low[0], action_high=box.high[0], config=config).init(
        obs, jax.random.key(0)
    )
    # The actor's mean and log-std heads are one depth: 2 is the last hidden
    # layer and both heads, the same depth as in a critic.
    assert _top_layer_names(state.params.actor, 2, heads=2) == expected
    critic = _top_layer_names(state.params.q1, 2)
    assert len(critic) == 2
    assert critic < expected
    assert _top_layer_names(state.params.actor, 0, heads=2) == set()
    # A LayerNorm is reset together with the Dense layer it normalises.
    if config.layer_norm:
        assert "LayerNorm_0" in _top_layer_names(
            state.params.actor, config.hidden_layers + 1, heads=2
        )
    with pytest.raises(ValueError, match="exceeds the network depth"):
        _top_layer_names(state.params.actor, config.hidden_layers + 2, heads=2)


@pytest.mark.parametrize(
    ("count", "actor", "critic"),
    [
        (1, {"Dense_2", "Dense_3"}, {"Dense_2"}),
        (2, {"Dense_1", "Dense_2", "Dense_3"}, {"Dense_1", "Dense_2"}),
    ],
)
def test_sac_reset_reaches_the_same_depth_in_actor_and_critics(
    count: int, actor: set[str], critic: set[str]
) -> None:
    env = pp.make("gymnax/Pendulum-v1")
    agent = SAC(
        action_low=env.action_space.low[0],
        action_high=env.action_space.high[0],
        config=SACConfig(batch_size=2, hidden_size=8, reset_interval=1, reset_last_layers=count),
    )
    obs = env.init(jax.random.key(0))[1]
    state = agent.init(obs, jax.random.key(1))
    replay = init_sac_replay(num_agents=1, obs_dim=3, action_dim=1, capacity=4)
    transition = (
        obs,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        obs,
        jnp.ones((1,), dtype=jnp.bool_),
    )
    for _ in range(4):
        replay = insert_sac(replay, transition)
    key = jax.random.key(42)
    reset, _ = agent.update(state, replay, key)
    fresh = agent.init(obs, jax.random.split(key, 4)[3])
    for network, expected in (("actor", actor), ("q1", critic), ("q2", critic)):
        layers = getattr(reset.params, network)["params"]
        replaced = {
            name
            for name, layer in layers.items()
            if np.array_equal(
                np.asarray(layer["kernel"]),
                np.asarray(getattr(fresh.params, network)["params"][name]["kernel"]),
            )
        }
        assert replaced == expected, network


def _toy_params(value: float) -> Any:
    leaf = jnp.full((3,), value)
    return {
        "params": {
            "CNNTorso_0": {"Dense_0": {"kernel": leaf}},
            "Dense_0": {"kernel": leaf},
            "Dense_1": {"kernel": leaf},
        }
    }


def test_reset_layers_restart_their_adam_step_count() -> None:
    import optax  # pyright: ignore[reportMissingTypeStubs]

    from perpetua.agents.plasticity import adam, graft_top_layers

    params = _toy_params(1.0)
    gradient = _toy_params(0.5)
    plain, layered = optax.scale_by_adam(eps=1e-5), adam(1)
    plain_state, state = plain.init(params), layered.init(params)
    update: Any = None
    expected: Any = None
    for _ in range(1000):
        expected, plain_state = plain.update(gradient, plain_state, params)
        update, state = layered.update(gradient, state, params)
    assert_same(update, expected)  # until a reset, exactly plain Adam
    params, state = graft_top_layers(params, state, _toy_params(7.0), layered.init(params), 1)
    for _ in range(3):
        update, state = layered.update(gradient, state, params)
        # A shared step count would treat the zeroed moments as long-run
        # averages and give the reset layer a step of about 3x.
        np.testing.assert_allclose(
            np.asarray(update["params"]["Dense_1"]["kernel"]), 1.0, rtol=1e-3
        )
        np.testing.assert_allclose(
            np.asarray(update["params"]["Dense_0"]["kernel"]), 1.0, rtol=1e-3
        )


def test_resetting_every_head_layer_keeps_the_cnn_torso() -> None:
    from perpetua.agents.plasticity import adam, graft_top_layers

    params, fresh = _toy_params(1.0), _toy_params(7.0)
    optimizer = adam(2)
    reset, _ = graft_top_layers(params, optimizer.init(params), fresh, optimizer.init(fresh), 2)
    head: Any = reset["params"]
    assert float(head["Dense_0"]["kernel"][0]) == 7.0
    assert float(head["Dense_1"]["kernel"][0]) == 7.0
    assert float(head["CNNTorso_0"]["Dense_0"]["kernel"][0]) == 1.0


def test_dqn_reset_key_is_not_the_replay_sampling_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from perpetua.agents import dqn as dqn_module

    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    agent = DQN(action_size=5, config=DQNConfig(hidden_size=8, batch_size=2, reset_interval=1))
    state = agent.init(obs, jax.random.key(1))
    replay = init_replay(num_agents=obs.shape[0], obs_dim=obs.shape[1], action_size=5, capacity=4)
    transition = (
        obs,
        jnp.ones((obs.shape[0], 5), dtype=jnp.bool_),
        jnp.ones(obs.shape[0], dtype=jnp.bool_),
        jnp.zeros(obs.shape[0], dtype=jnp.int32),
        jnp.ones(obs.shape[0], dtype=jnp.float32),
        obs,
        jnp.ones((obs.shape[0], 5), dtype=jnp.bool_),
    )
    for _ in range(4):
        replay = insert(replay, transition)
    seen: list[jax.Array] = []
    original = dqn_module.sample

    def recording(replay: Any, key: jax.Array, batch_size: int) -> Any:
        seen.append(key)
        return original(replay, key, batch_size)

    monkeypatch.setattr(dqn_module, "sample", recording)
    key = jax.random.key(7)
    reset, _ = agent.update(state, replay, key)
    sampling_key, reset_key = jax.random.split(key)
    np.testing.assert_array_equal(jax.random.key_data(seen[0]), jax.random.key_data(sampling_key))
    fresh = agent.init(obs, reset_key)
    np.testing.assert_array_equal(
        np.asarray(reset.params["params"]["Dense_2"]["kernel"]),
        np.asarray(fresh.params["params"]["Dense_2"]["kernel"]),
    )
    # The sampling key randint splits internally is not the reset key.
    for child in jax.random.split(seen[0]):
        assert not np.array_equal(jax.random.key_data(child), jax.random.key_data(reset_key))


def test_sac_reset_key_is_not_a_key_the_update_consumed() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    agent = SAC(
        action_low=env.action_space.low[0],
        action_high=env.action_space.high[0],
        config=SACConfig(batch_size=2, hidden_size=8, reset_interval=1, reset_last_layers=2),
    )
    obs = env.init(jax.random.key(0))[1]
    state = agent.init(obs, jax.random.key(1))
    replay = init_sac_replay(num_agents=1, obs_dim=3, action_dim=1, capacity=4)
    transition = (
        obs,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        obs,
        jnp.ones((1,), dtype=jnp.bool_),
    )
    for _ in range(4):
        replay = insert_sac(replay, transition)
    key = jax.random.key(42)
    reset, _ = agent.update(state, replay, key)
    sample_key, next_key, actor_key, reset_key = jax.random.split(key, 4)
    consumed = [jax.random.key_data(k) for k in (sample_key, next_key, actor_key)]
    assert not any(np.array_equal(jax.random.key_data(reset_key), k) for k in consumed)
    fresh = agent.init(obs, reset_key)
    np.testing.assert_array_equal(
        np.asarray(reset.params.actor["params"]["Dense_3"]["kernel"]),
        np.asarray(fresh.params.actor["params"]["Dense_3"]["kernel"]),
    )
    # Without a reset the 3-way split of the same key is untouched.
    assert [k.tolist() for k in consumed] == [
        jax.random.key_data(k).tolist() for k in jax.random.split(key, 3)
    ]


def test_dqn_reset_runs_clean_under_the_key_reuse_checker() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    config = DQNConfig(
        rollout_steps=4, buffer_capacity=16, batch_size=2, hidden_size=8, reset_interval=1
    )
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer.init(seed=0, window_size=4)
    # Native environment init splits its key more than once by design, so only
    # the training chunk runs under the checker.
    with jax.debug_key_reuse(True):
        state = trainer(state)
    assert bool(state.healthy)
