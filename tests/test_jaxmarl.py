"""JaxMARL MPE simple_spread must become a continuing native Perpetua lifetime."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.adapters.jaxmarl import JaxMARLState
from perpetua.params import DriftEnvState


def test_jaxmarl_simple_spread_public_contract_and_determinism() -> None:
    env = pp.make("jaxmarl/MPE_simple_spread-v3")
    state, obs, info = env.init(jax.random.key(0))
    twin, _, _ = env.init(jax.random.key(0))
    assert_same(state, twin)
    assert obs.shape == (3, 18)
    assert obs.dtype == jnp.float32
    assert env.observation_space.contains(np.asarray(obs))
    assert env.action_space.shape == (3,)
    np.testing.assert_array_equal(info["action_mask"], np.ones((3, 5), dtype=bool))
    np.testing.assert_array_equal(info["active_mask"], [True, True, True])
    np.testing.assert_array_equal(info["respawn"], [False, False, False])
    assert info["t"].to_int() == 0
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert "oracle" not in info
    assert env.metadata["env_id"] == "jaxmarl/MPE_simple_spread-v3"
    assert env.metadata.get("execution") == "jax"
    assert env.metadata.get("checkpointable") is True
    configuration = env.metadata.get("configuration")
    assert isinstance(configuration, dict)
    assert configuration["backend"] == "jaxmarl-0.2.0"
    assert not hasattr(env, "reset")


def test_reward_multiplier_drift_scales_rewards_without_touching_the_world() -> None:
    plain = pp.make("jaxmarl/MPE_simple_spread-v3")
    drifting = pp.make("jaxmarl/MPE_simple_spread-v3", drift=pp.drift.Cyclic([(2.0,), (0.5,)], 1))
    left, _, _ = plain.init(jax.random.key(0))
    right, _, _ = drifting.init(jax.random.key(0))
    for multiplier in (2.0, 0.5, 2.0):
        actions = jnp.zeros(3, dtype=jnp.int32)
        left, _, left_reward, left_info = plain.step(left, actions, jax.random.key(1))
        right, _, right_reward, right_info = drifting.step(right, actions, jax.random.key(1))
        np.testing.assert_array_equal(right_reward, left_reward * np.float32(multiplier))
        np.testing.assert_array_equal(
            np.asarray(left.env.world.p_pos), np.asarray(right.env.world.p_pos)
        )
        assert left_info["t"].to_int() == right_info["t"].to_int()


def test_legacy_episode_limit_respawns_on_the_same_tick_without_clock_reset() -> None:
    env = pp.make("jaxmarl/MPE_simple_spread-v3", max_steps=2)
    state, obs, info = env.init(jax.random.key(0))
    actions = jnp.zeros(3, dtype=jnp.int32)
    for _ in range(2):
        state, obs, reward, info = env.step(state, actions, jax.random.key(1))
        del obs, reward
        assert not np.any(np.asarray(info["respawn"]))
        assert int(state.env.world.step) > 0
    state, _, reward, info = env.step(state, actions, jax.random.key(2))
    assert np.all(np.asarray(info["respawn"]))
    assert int(state.env.world.step) == 0
    assert info["t"].to_int() == 3
    assert np.all(np.isfinite(np.asarray(reward)))
    assert not bool(state.env.world.done)


def test_jaxmarl_conformance_exact_checkpoint_and_batched_worlds(tmp_path: Path) -> None:
    env = pp.make("jaxmarl/MPE_simple_spread-v3", drift=pp.drift.Cyclic([(1.0,), (0.5,)], 3))
    initial = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    assert_same(initial, pp.runtime.init_lifetime(env, seed=0, window_size=4))
    eager = pp.runtime.tick(env, pp.policies.random, initial)
    compiled = pp.runtime.Runner(env, pp.policies.random, ticks=1)(initial)
    assert_same(eager, compiled)
    chunk = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    long = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    assert_same(chunk(chunk(initial)), long(initial))
    path = tmp_path / "jaxmarl"
    pp.checkpoint.save(path, chunk(initial), metadata=env.metadata)
    restored = pp.checkpoint.restore(path, initial, expected_metadata=env.metadata)
    assert int(restored.state.drift["reward_multiplier"].phase) == 1
    assert_same(chunk(restored), long(initial))

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[JaxMARLState]]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=4)

    batched = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(chunk)(batched)
    np.testing.assert_array_equal(result.state.t.lo, [4, 4])
    assert result.obs.shape[0:2] == (2, 3)


def test_ippo_trains_and_checkpoints_the_jaxmarl_lifetime(tmp_path: Path) -> None:
    from perpetua.agents import IPPO, PPOConfig, PPORunner

    env = pp.make("jaxmarl/MPE_simple_spread-v3", drift=pp.drift.Cyclic([(1.0,), (0.5,)], 4))
    trainer = PPORunner(
        env,
        IPPO(
            action_size=5, config=PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
        ),
    )
    state = trainer(trainer.init(seed=0))
    assert bool(state.healthy)
    assert state.updates.to_int() == 1
    template = trainer.init(seed=999, window_size=128)
    path = tmp_path / "training"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(trainer(state), trainer(restored))


def test_jaxmarl_drift_is_oracle_exposed_and_hidden_by_default() -> None:
    base = pp.make("jaxmarl/MPE_simple_spread-v3")
    _, _, plain_info = base.init(jax.random.key(0))
    assert "oracle" not in plain_info
    env = pp.Oracle(base, expose=["phase"])
    state, _, _ = env.init(jax.random.key(0))
    _, _, _, info = env.step(state, jnp.zeros(3, dtype=jnp.int32), jax.random.key(1))
    assert "oracle" in info
    assert set(info["oracle"]) == {"phase"}
    assert env.metadata["oracle_exposure"] == ("phase",)
    with pytest.raises(ValueError, match="oracle field"):
        pp.Oracle(base, expose=["jaxmarl_phase"])


@pytest.mark.parametrize(
    ("num_agents", "num_landmarks", "max_steps", "local_ratio"),
    [(0, 3, 25, 0.5), (3, 0, 25, 0.5), (3, 3, 0, 0.5), (3, 3, 25, 1.5)],
)
def test_invalid_jaxmarl_configuration(
    num_agents: int, num_landmarks: int, max_steps: int, local_ratio: float
) -> None:
    with pytest.raises(ValueError, match=r"JaxMARL simple_spread"):
        pp.make(
            "jaxmarl/MPE_simple_spread-v3",
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            max_steps=max_steps,
            local_ratio=local_ratio,
        )


def test_jaxmarl_drift_parameter_shape_is_validated() -> None:
    env = pp.make("jaxmarl/MPE_simple_spread-v3", drift=pp.drift.Cyclic([(1.0, 2.0)], 2))
    with pytest.raises(ValueError, match="shape"):
        env.init(jax.random.key(0))


def test_scripted_slots_are_uncontrollable_but_world_visible() -> None:
    env = pp.make("jaxmarl/MPE_simple_spread-v3", num_scripted=1)
    state, _, _ = env.init(jax.random.key(0))
    base = state.env.world
    base = base.replace(
        p_pos=jnp.asarray(
            [[0.0, 0.0], [1.0, 1.0], [0.5, -0.5], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
    )
    state = state._replace(env=state.env._replace(world=base))
    obs, info = env.observe(state)
    assert info["active_mask"].tolist() == [True, True, False]
    assert info["action_mask"][2].tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(np.asarray(obs[2]), np.zeros(env.observation_space.shape[1]))
    # Learner 0 sees the scripted slot as an ordinary peer (relative position).
    other_block = 4 + 2 * env.backend.num_landmarks
    np.testing.assert_allclose(
        np.asarray(obs[0][other_block + 2 : other_block + 4]),
        np.asarray([0.5, -0.5]),
        atol=1e-6,
    )
    first = env.step(state, jnp.asarray([0, 0, 0]), jax.random.key(1))
    second = env.step(state, jnp.asarray([0, 0, 4]), jax.random.key(1))
    assert_same(first, second)


def test_scripted_drift_direction_changes_the_global_reward() -> None:
    def measure(behavior: str, strength: float, ticks: int = 256) -> float:
        # A one-phase cycle holds the scripted strength constant.
        env = pp.make(
            "jaxmarl/MPE_simple_spread-v3",
            num_scripted=1,
            scripted_behavior=behavior,
            scripted_drift=pp.drift.Cyclic([(strength,)], 1),
        )
        carry = pp.runtime.init_lifetime(env, seed=0, window_size=ticks)
        final = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(carry)
        return float(np.asarray(final.metrics.mean)[0])

    helpful = measure("partner", 1.0)
    wandering = measure("partner", 0.0)
    hostile = measure("opponent", 1.0)
    assert helpful > wandering
    assert hostile < wandering
    print(f"partner {helpful:.4f} vs wandering {wandering:.4f} vs opponent {hostile:.4f}")


def test_scripted_variant_checkpoints_and_exposes_oracle_fields(tmp_path: Path) -> None:
    from perpetua.agents import IPPO, PPOConfig, PPORunner

    env = pp.make(
        "jaxmarl/MPE_simple_spread-v3",
        num_scripted=1,
        scripted_drift=pp.drift.Cyclic([(1.0,), (0.0,)], 3),
    )
    config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, IPPO(action_size=5, config=config))
    state = trainer(trainer.init(seed=0, window_size=4))
    assert bool(state.healthy)
    steps = state.agent.gradient_steps
    assert [int(steps.lo[i]) for i in range(3)] == [1, 1, 0]
    path = tmp_path / "scripted"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    template = trainer.init(seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=trainer.metadata)
    assert_same(state, restored)
    assert int(restored.lifetime.state.drift["scripted"].phase) == 1
    assert_same(trainer(state), trainer(restored))
    base = pp.make("jaxmarl/MPE_simple_spread-v3", num_scripted=1)
    oracle = pp.Oracle(base, expose=["scripted_parameters", "scripted_phase"])
    ostate, _, _ = oracle.init(jax.random.key(0))
    _, _, _, oinfo = oracle.step(ostate, jnp.zeros(3, dtype=jnp.int32), jax.random.key(1))
    assert "oracle" in oinfo
    assert set(oinfo["oracle"]) == {"scripted_parameters", "scripted_phase"}
    np.testing.assert_array_equal(oinfo["oracle"]["scripted_parameters"], [1.0])
    with pytest.raises(ValueError, match=r"JaxMARL num_scripted must be in"):
        pp.make("jaxmarl/MPE_simple_spread-v3", num_scripted=3)
    with pytest.raises(ValueError, match=r"JaxMARL scripted_behavior must be 'partner' or"):
        pp.make("jaxmarl/MPE_simple_spread-v3", num_scripted=1, scripted_behavior="neutral")
    bad = pp.make(
        "jaxmarl/MPE_simple_spread-v3",
        num_scripted=1,
        scripted_drift=pp.drift.Cyclic([(1.0, 0.5)], 2),
    )
    with pytest.raises(ValueError, match="shape|match"):
        bad.init(jax.random.key(0))


def test_scripted_drift_is_refused_without_scripted_slots() -> None:
    cycle = pp.drift.Cyclic([(1.0,), (0.0,)], 5)
    refused = pytest.raises(ValueError, match=r"no scripted slots.*pass num_scripted >= 1")
    with refused:
        pp.make("jaxmarl/MPE_simple_spread-v3", scripted_drift=cycle)
    with refused:
        pp.make("jaxmarl/MPE_simple_spread-v3", drift={"scripted": cycle})
    with refused:
        pp.make("jaxmarl/MPE_simple_spread-v3").with_drift({"scripted": cycle})
    plain = cast(Any, pp.make("jaxmarl/MPE_simple_spread-v3"))
    assert "scripted" not in plain.env.drift_fields
    scripted = cast(Any, pp.make("jaxmarl/MPE_simple_spread-v3", num_scripted=1))
    assert "scripted" in scripted.env.drift_fields
    pp.make("jaxmarl/MPE_simple_spread-v3", num_scripted=1, scripted_drift=cycle)
