"""Cooperative agents must influence one another inside a continuing native world."""

# JAX transformations are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def test_cooperative_grid_spaces_unique_spawns_and_public_contract() -> None:
    env = pp.make("CooperativeGrid-v0", size=3, num_agents=2)
    state, obs, info = env.init(jax.random.key(0))
    assert obs.shape == (2, 7)
    assert obs.dtype == jnp.float32
    assert env.observation_space.contains(np.asarray(obs))
    assert info["action_mask"].shape == (2, 5)
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert not hasattr(env, "reset")
    assert len(np.unique(np.asarray(state.env.positions), axis=0)) == 2
    np.testing.assert_array_equal(obs[:, 2:4], [[1.0, 1.0], [0.0, 0.0]])
    assert not np.any(np.all(np.asarray(state.env.positions) == [2, 2], axis=-1))
    assert not np.any(np.all(np.asarray(state.env.positions) == [0, 0], axis=-1))


def test_partner_action_changes_the_other_agents_transition() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(
        env=state.env._replace(positions=jnp.asarray([[1, 0], [1, 2]], dtype=jnp.int32))
    )
    contested, _, reward, _ = env.step(state, jnp.asarray([4, 3]), jax.random.key(1))
    uncontested, _, _, _ = env.step(state, jnp.asarray([4, 0]), jax.random.key(1))
    np.testing.assert_array_equal(contested.env.positions, state.env.positions)
    np.testing.assert_array_equal(uncontested.env.positions, [[1, 1], [1, 2]])
    np.testing.assert_array_equal(reward, [0.0, 0.0])


def test_occupied_cells_and_inactive_slots_are_masked() -> None:
    env = pp.make("CooperativeGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(
        env=state.env._replace(positions=jnp.asarray([[1, 0], [1, 1]], dtype=jnp.int32))
    )
    _, info = env.observe(state)
    assert not bool(info["action_mask"][0, 4])
    assert not bool(info["action_mask"][1, 3])
    actual, _, _, _ = env.step(state, jnp.asarray([4, 3]), jax.random.key(1))
    np.testing.assert_array_equal(actual.env.positions, state.env.positions)
    state = state._replace(env=state.env._replace(active=jnp.asarray([True, False])))
    obs, info = env.observe(state)
    np.testing.assert_array_equal(info["action_mask"][1], [True, False, False, False, False])
    np.testing.assert_array_equal(obs[1], np.zeros(7))
    np.testing.assert_array_equal(obs[0, 4:], [0.0, 0.0, 0.0])
    actual, _, reward, _ = env.step(state, jnp.asarray([4, 99]), jax.random.key(1))
    np.testing.assert_array_equal(actual.env.positions, [[1, 1], [1, 1]])
    np.testing.assert_array_equal(reward, [0.0, 0.0])


def test_joint_reward_respawn_and_old_parameters_on_a_switching_tick() -> None:
    env = pp.make("CooperativeGrid-v0", size=3, drift=pp.drift.Cyclic([(2.0, 2.0), (0.0, 0.0)], 1))
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(
        env=state.env._replace(positions=jnp.asarray([[2, 1], [0, 1]], dtype=jnp.int32))
    )
    _, _, alone_reward, alone_info = env.step(state, jnp.asarray([4, 0]), jax.random.key(1))
    np.testing.assert_array_equal(alone_reward, [0.0, 0.0])
    assert not bool(jnp.any(alone_info["respawn"]))
    actual, obs, reward, info = env.step(state, jnp.asarray([4, 3]), jax.random.key(1))
    np.testing.assert_array_equal(reward, [1.0, 1.0])
    np.testing.assert_array_equal(info["respawn"], [True, True])
    np.testing.assert_array_equal(obs[:, 2:4], [[0.0, 0.0], [0.0, 0.5]])
    assert len(np.unique(np.asarray(actual.env.positions), axis=0)) == 2
    assert info["t"].to_int() == 1
    assert "oracle" not in info


@pytest.mark.parametrize("env_id", ["ContinualGrid-v0", "CooperativeGrid-v0"])
def test_shared_native_conformance_and_exact_checkpoint(env_id: str, tmp_path: Path) -> None:
    env = pp.make(env_id, size=3, num_agents=2, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 3))
    initial = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    assert_same(initial, pp.runtime.init_lifetime(env, seed=0, window_size=4))
    eager = pp.runtime.tick(env, pp.policies.random, initial)
    compiled = pp.runtime.Runner(env, pp.policies.random, ticks=1)(initial)
    assert_same(eager, compiled)
    chunk = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    long = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    assert_same(chunk(chunk(initial)), long(initial))
    path = tmp_path / "native"
    pp.checkpoint.save(path, chunk(initial), metadata=env.metadata)
    restored = pp.checkpoint.restore(path, initial, expected_metadata=env.metadata)
    assert_same(chunk(restored), long(initial))

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[GridState]]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=4)

    batched = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(chunk)(batched)
    np.testing.assert_array_equal(result.state.t.lo, [4, 4])
    assert result.obs.shape[0:2] == (2, 2)


def test_cooperative_clock_and_explicit_oracle() -> None:
    base = pp.make("CooperativeGrid-v0", size=3)
    env = pp.Oracle(base, expose=["phase"])
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(t=pp.Clock.from_int(10_000_000_000))
    step = cast(
        Callable[
            [DriftEnvState[GridState], jax.Array, jax.Array],
            tuple[DriftEnvState[GridState], jax.Array, jax.Array, pp.Info],
        ],
        jax.jit(env.step),
    )
    _, _, _, info = step(state, jnp.asarray([0, 0]), jax.random.key(1))
    assert info["t"].to_int() == 10_000_000_001
    assert "oracle" in info
    assert set(info["oracle"]) == {"phase"}
    assert env.metadata["oracle_exposure"] == ("phase",)


@pytest.mark.parametrize(("size", "num_agents"), [(1, 2), (3, 1), (2, 3), (65, 2), (8, 33)])
def test_invalid_cooperative_configuration(size: int, num_agents: int) -> None:
    with pytest.raises(ValueError, match=r"Cooperative"):
        pp.make("CooperativeGrid-v0", size=size, num_agents=num_agents)
