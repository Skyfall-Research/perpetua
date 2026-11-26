"""Scripted partners with hidden drifting cooperativeness stay uncontrollable."""

# JAX transformations are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.envs.partner_grid import PartnerState
from perpetua.params import DriftEnvState


def placed(
    state: DriftEnvState[PartnerState], positions: Sequence[Sequence[int]]
) -> DriftEnvState[PartnerState]:
    world = state.env.world._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    return state._replace(env=state.env._replace(world=world))


def test_partner_grid_public_contract_hides_partner_slots_and_parameters() -> None:
    env = pp.make("PartnerGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[1, 0], [0, 2]])
    obs, info = env.observe(state)
    assert obs.shape == (2, 7)
    assert obs.dtype == jnp.float32
    assert env.observation_space.contains(np.asarray(obs))
    assert info["active_mask"].tolist() == [True, False]
    assert info["action_mask"][0].tolist() == [True, True, True, False, True]
    assert info["action_mask"][1].tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(obs[1], np.zeros(7))
    np.testing.assert_array_equal(obs[0], [0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    # observe() is not a transition: space_change is present and False, so
    # the info pytree matches init/step.
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert not bool(jnp.any(cast(jax.Array, info.get("space_change"))))
    assert "oracle" not in info
    assert env.metadata["env_id"] == "PartnerGrid-v0"
    assert not hasattr(env, "reset")


def test_learner_cannot_control_the_partner_slot() -> None:
    env = pp.make("PartnerGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[1, 0], [0, 2]])
    first, _, _, _ = env.step(state, jnp.asarray([4, 99]), jax.random.key(1))
    second, _, _, _ = env.step(state, jnp.asarray([4, 0]), jax.random.key(1))
    np.testing.assert_array_equal(first.env.world.positions, [[1, 1], [0, 1]])
    assert_same(first, second)


def test_partner_blocks_movement_and_stays_world_active() -> None:
    env = pp.make("PartnerGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[0, 0], [0, 1]])
    _, info = env.observe(state)
    assert not bool(info["action_mask"][0, 4])
    next_state, _, reward, _ = env.step(state, jnp.asarray([4, 0]), jax.random.key(1))
    np.testing.assert_array_equal(next_state.env.world.positions, [[0, 0], [0, 1]])
    np.testing.assert_array_equal(reward, [0.0, 0.0])


def test_partner_beta_drives_greedy_and_random_ticks_with_old_parameters() -> None:
    env = pp.make("PartnerGrid-v0", size=3, partner_drift=pp.drift.Cyclic([[1.0], [0.0]], 1))
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[2, 1], [0, 2]])
    key = jax.random.key(7)
    for tick in range(6):
        beta_one = tick % 2 == 0
        before = int(jnp.sum(jnp.abs(state.env.world.positions[1] - jnp.asarray([0, 0]))))
        state, _, _, _ = env.step(state, jnp.asarray([0, 0]), key)
        after = int(jnp.sum(jnp.abs(state.env.world.positions[1] - jnp.asarray([0, 0]))))
        if beta_one and before > 0:
            assert after == before - 1
    assert int(state.drift["partner"].phase) == 0
    assert bool(state.field_changed["partner"])
    assert set(env.privileged(state)) == {
        "phase",
        "drift_parameters",
        "drift_changed",
        "task_id",
        "boundary",
        "goal_parameters",
        "goal_phase",
        "goal_changed",
        "partner_parameters",
        "partner_phase",
        "partner_changed",
    }
    np.testing.assert_array_equal(env.privileged(state)["partner_parameters"], [1.0])


def test_partner_grid_joint_reward_and_respawn_include_partners() -> None:
    env = pp.make("PartnerGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[2, 2], [0, 0]])
    next_state, obs, reward, info = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    np.testing.assert_array_equal(reward, [1.0, 1.0])
    np.testing.assert_array_equal(info["respawn"], [True, True])
    positions = np.asarray(next_state.env.world.positions)
    assert len(np.unique(positions, axis=0)) == 2
    assert not np.any(np.all(positions == [2, 2], axis=-1))
    assert not np.any(np.all(positions == [0, 0], axis=-1))
    np.testing.assert_array_equal(obs[:, 2:4], [[1.0, 1.0], [0.0, 0.0]])


def test_partner_draws_are_independent_of_the_world_key() -> None:
    env = pp.make("PartnerGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[1, 0], [0, 2]])
    left, _, _, _ = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    right, _, _, _ = env.step(state, jnp.asarray([0, 0]), jax.random.key(99))
    assert_same(left, right)
    placed_state = placed(state, [[2, 2], [0, 0]])
    a, _, _, _ = env.step(placed_state, jnp.asarray([0, 0]), jax.random.key(1))
    b, _, _, _ = env.step(placed_state, jnp.asarray([0, 0]), jax.random.key(2))
    assert not np.array_equal(np.asarray(a.env.world.positions), np.asarray(b.env.world.positions))


def test_partner_grid_multiple_slots_stay_masked_and_greedy() -> None:
    env = pp.make("PartnerGrid-v0", size=5, num_learners=2, num_partners=2)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[0, 0], [4, 4], [4, 0], [0, 4]])
    obs, info = env.observe(state)
    assert info["active_mask"].tolist() == [True, True, False, False]
    assert info["action_mask"][2].tolist() == [True, False, False, False, False]
    assert info["action_mask"][3].tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(obs[2], np.zeros(obs.shape[1]))
    np.testing.assert_array_equal(obs[3], np.zeros(obs.shape[1]))
    # Learners see both partners as active peers with real positions.
    np.testing.assert_array_equal(obs[0][6], 1.0)
    np.testing.assert_array_equal(obs[0][7:9], [1.0, 0.0])
    np.testing.assert_array_equal(obs[0][12], 1.0)
    next_state, _, reward, step_info = env.step(
        state, jnp.asarray([0, 0, 99, 99]), jax.random.key(1)
    )
    np.testing.assert_array_equal(next_state.env.world.positions, [[0, 0], [4, 4], [3, 0], [0, 3]])
    np.testing.assert_array_equal(reward, [0.0, 0.0, 0.0, 0.0])
    assert not np.any(np.asarray(step_info["respawn"]))


def test_partner_grid_conformance_exact_checkpoint_and_batched_worlds(tmp_path: Path) -> None:
    env = pp.make("PartnerGrid-v0", size=3, partner_drift=pp.drift.Cyclic([(1.0,), (0.0,)], 3))
    initial = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    assert_same(initial, pp.runtime.init_lifetime(env, seed=0, window_size=4))
    eager = pp.runtime.tick(env, pp.policies.random, initial)
    compiled = pp.runtime.Runner(env, pp.policies.random, ticks=1)(initial)
    assert_same(eager, compiled)
    chunk = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    long = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    assert_same(chunk(chunk(initial)), long(initial))
    path = tmp_path / "partner"
    pp.checkpoint.save(path, chunk(initial), metadata=env.metadata)
    restored = pp.checkpoint.restore(path, initial, expected_metadata=env.metadata)
    assert int(restored.state.drift["partner"].phase) == 1
    assert_same(chunk(restored), long(initial))

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[PartnerState]]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=4)

    batched = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(chunk)(batched)
    np.testing.assert_array_equal(result.state.t.lo, [4, 4])
    assert result.obs.shape[0:2] == (2, 2)


def test_partner_parameters_require_an_explicit_oracle() -> None:
    base = pp.make("PartnerGrid-v0", size=3)
    _, _, plain_info = base.init(jax.random.key(0))
    assert "oracle" not in plain_info
    env = pp.Oracle(base, expose=["partner_parameters", "partner_phase"])
    state, _, _ = env.init(jax.random.key(0))
    _, _, _, info = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    assert "oracle" in info
    assert set(info["oracle"]) == {"partner_parameters", "partner_phase"}
    np.testing.assert_array_equal(info["oracle"]["partner_parameters"], [1.0])
    assert env.metadata["oracle_exposure"] == ("partner_parameters", "partner_phase")
    with pytest.raises(ValueError, match="oracle field"):
        pp.Oracle(base, expose=["partner_beta"])


@pytest.mark.parametrize(
    ("num_learners", "num_partners", "size"),
    [(0, 1, 3), (1, 0, 3), (2, 1, 2)],
)
def test_invalid_partner_grid_configuration(
    num_learners: int, num_partners: int, size: int
) -> None:
    with pytest.raises(ValueError, match=r"must be|exceeds"):
        pp.make("PartnerGrid-v0", size=size, num_learners=num_learners, num_partners=num_partners)


def test_partner_drift_parameter_shape_is_validated() -> None:
    env = pp.make("PartnerGrid-v0", size=3, partner_drift=pp.drift.Cyclic([(1.0, 0.9)], 2))
    with pytest.raises(ValueError, match="shape"):
        env.init(jax.random.key(0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 64, "num_learners": 40}, r"num_learners \+ num_partners = 40 \+ 1 = 41.*= 32"),
        ({"size": 2, "num_learners": 2}, r"= 3 exceeds.*min\(32, size\*\*2 // 2\) = 2 for size=2"),
        ({"size": 100}, r"PartnerGrid size must be an integer in \[2, 64\]"),
    ],
)
def test_partner_grid_limits_are_reported_in_its_own_terms(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pp.make("PartnerGrid-v0", **kwargs)
    pp.make("PartnerGrid-v0", size=64, num_learners=31, num_partners=1)


def test_partner_grid_exposes_its_size_like_the_other_grids() -> None:
    assert cast(Any, pp.make("PartnerGrid-v0", size=6)).size == 6
