"""Scripted opponents with hidden drifting aggression hunt learners zero-sum."""

# JAX transformations are partially annotated.
# pyright: reportUnknownMemberType=false

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.envs.opponent_grid import OpponentState
from perpetua.params import DriftEnvState


def placed(
    state: DriftEnvState[OpponentState], positions: Sequence[Sequence[int]]
) -> DriftEnvState[OpponentState]:
    world = state.env.world._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    return state._replace(env=state.env._replace(world=world))


def test_opponent_grid_public_contract_hides_opponent_slots_and_parameters() -> None:
    env = pp.make("OpponentGrid-v0", size=3)
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
    assert env.metadata["env_id"] == "OpponentGrid-v0"
    assert not hasattr(env, "reset")


def test_learner_cannot_control_the_opponent_slot() -> None:
    env = pp.make("OpponentGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[1, 1], [0, 0]])
    first, _, reward, _ = env.step(state, jnp.asarray([4, 99]), jax.random.key(1))
    second, _, _, _ = env.step(state, jnp.asarray([4, 0]), jax.random.key(1))
    np.testing.assert_array_equal(first.env.world.positions, [[1, 2], [1, 0]])
    np.testing.assert_array_equal(reward, [0.0, 0.0])
    assert_same(first, second)


def test_adjacent_opponent_catches_interrupts_collection_zero_sum() -> None:
    env = pp.make("OpponentGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[2, 2], [2, 1]])
    next_state, obs, reward, info = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    np.testing.assert_array_equal(reward, [-1.0, 1.0])
    assert float(jnp.sum(reward)) == 0.0
    np.testing.assert_array_equal(info["respawn"], [True, False])
    assert not np.array_equal(np.asarray(next_state.env.world.positions[0]), [2, 2])
    np.testing.assert_array_equal(next_state.env.world.positions[1], [2, 1])
    np.testing.assert_array_equal(obs[1], np.zeros(7))


def test_collection_with_a_far_opponent_is_zero_sum() -> None:
    env = pp.make("OpponentGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[2, 2], [0, 0]])
    next_state, _, reward, info = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    np.testing.assert_array_equal(reward, [1.0, -1.0])
    assert float(jnp.sum(reward)) == 0.0
    np.testing.assert_array_equal(info["respawn"], [True, False])
    assert not np.array_equal(np.asarray(next_state.env.world.positions[0]), [2, 2])
    np.testing.assert_array_equal(next_state.env.world.positions[1], [1, 0])


def test_opponent_alpha_drives_hunting_and_wandering_ticks_with_old_parameters() -> None:
    env = pp.make("OpponentGrid-v0", size=5, opponent_drift=pp.drift.Cyclic([[1.0], [0.0]], 1))
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[0, 0], [4, 4]])
    key = jax.random.key(7)
    for tick in range(6):
        alpha_one = tick % 2 == 0
        before = int(jnp.sum(jnp.abs(state.env.world.positions[1] - state.env.world.positions[0])))
        state, _, _, _ = env.step(state, jnp.asarray([0, 0]), key)
        after = int(jnp.sum(jnp.abs(state.env.world.positions[1] - state.env.world.positions[0])))
        if alpha_one:
            assert after == before - 1
    assert int(state.drift["opponent"].phase) == 0
    assert bool(state.field_changed["opponent"])
    assert set(env.privileged(state)) == {
        "phase",
        "drift_parameters",
        "drift_changed",
        "task_id",
        "boundary",
        "goal_parameters",
        "goal_phase",
        "goal_changed",
        "opponent_parameters",
        "opponent_phase",
        "opponent_changed",
    }
    np.testing.assert_array_equal(env.privileged(state)["opponent_parameters"], [1.0])


def test_opponent_draws_are_independent_of_the_world_key() -> None:
    env = pp.make("OpponentGrid-v0", size=3)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[1, 0], [0, 2]])
    left, _, _, _ = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    right, _, _, _ = env.step(state, jnp.asarray([0, 0]), jax.random.key(99))
    assert_same(left, right)
    placed_state = placed(state, [[2, 2], [0, 0]])
    a, _, _, _ = env.step(placed_state, jnp.asarray([0, 0]), jax.random.key(1))
    b, _, _, _ = env.step(placed_state, jnp.asarray([0, 0]), jax.random.key(2))
    assert not np.array_equal(np.asarray(a.env.world.positions), np.asarray(b.env.world.positions))


def test_opponent_grid_multiple_slots_share_the_negated_learner_sum() -> None:
    env = pp.make("OpponentGrid-v0", size=5, num_learners=2, num_opponents=2)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[1, 1], [0, 0], [1, 2], [4, 0]])
    obs, info = env.observe(state)
    assert info["active_mask"].tolist() == [True, True, False, False]
    assert info["action_mask"][2].tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(obs[2], np.zeros(obs.shape[1]))
    next_state, _, reward, step_info = env.step(
        state, jnp.asarray([0, 0, 99, 99]), jax.random.key(1)
    )
    np.testing.assert_array_equal(reward, [-1.0, 0.0, 0.5, 0.5])
    assert float(jnp.sum(reward)) == 0.0
    np.testing.assert_array_equal(step_info["respawn"], [True, False, False, False])
    np.testing.assert_array_equal(next_state.env.world.positions[1], [0, 0])
    np.testing.assert_array_equal(next_state.env.world.positions[2], [1, 2])
    np.testing.assert_array_equal(next_state.env.world.positions[3], [3, 0])
    assert not np.array_equal(np.asarray(next_state.env.world.positions[0]), [1, 1])


def test_opponent_grid_conformance_exact_checkpoint_and_batched_worlds(tmp_path: Path) -> None:
    env = pp.make("OpponentGrid-v0", size=3, opponent_drift=pp.drift.Cyclic([(1.0,), (0.0,)], 3))
    initial = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    assert_same(initial, pp.runtime.init_lifetime(env, seed=0, window_size=4))
    eager = pp.runtime.tick(env, pp.policies.random, initial)
    compiled = pp.runtime.Runner(env, pp.policies.random, ticks=1)(initial)
    assert_same(eager, compiled)
    chunk = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    long = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    assert_same(chunk(chunk(initial)), long(initial))
    path = tmp_path / "opponent"
    pp.checkpoint.save(path, chunk(initial), metadata=env.metadata)
    restored = pp.checkpoint.restore(path, initial, expected_metadata=env.metadata)
    assert int(restored.state.drift["opponent"].phase) == 1
    assert_same(chunk(restored), long(initial))

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[OpponentState]]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=4)

    batched = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(chunk)(batched)
    np.testing.assert_array_equal(result.state.t.lo, [4, 4])
    assert result.obs.shape[0:2] == (2, 2)


def test_opponent_parameters_require_an_explicit_oracle() -> None:
    base = pp.make("OpponentGrid-v0", size=3)
    _, _, plain_info = base.init(jax.random.key(0))
    assert "oracle" not in plain_info
    env = pp.Oracle(base, expose=["opponent_parameters", "opponent_phase"])
    state, _, _ = env.init(jax.random.key(0))
    _, _, _, info = env.step(state, jnp.asarray([0, 0]), jax.random.key(1))
    assert "oracle" in info
    assert set(info["oracle"]) == {"opponent_parameters", "opponent_phase"}
    np.testing.assert_array_equal(info["oracle"]["opponent_parameters"], [1.0])
    assert env.metadata["oracle_exposure"] == ("opponent_parameters", "opponent_phase")
    with pytest.raises(ValueError, match="oracle field"):
        pp.Oracle(base, expose=["opponent_alpha"])


@pytest.mark.parametrize(
    ("num_learners", "num_opponents", "size"),
    [(0, 1, 3), (1, 0, 3), (2, 1, 2)],
)
def test_invalid_opponent_grid_configuration(
    num_learners: int, num_opponents: int, size: int
) -> None:
    with pytest.raises(ValueError, match=r"must be|exceeds"):
        pp.make(
            "OpponentGrid-v0", size=size, num_learners=num_learners, num_opponents=num_opponents
        )


def test_opponent_drift_parameter_shape_is_validated() -> None:
    env = pp.make("OpponentGrid-v0", size=3, opponent_drift=pp.drift.Cyclic([(1.0, 0.9)], 2))
    with pytest.raises(ValueError, match="shape"):
        env.init(jax.random.key(0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 2, "num_learners": 2}, r"num_learners \+ num_opponents = 2 \+ 1 = 3 exceeds"),
        ({"size": 100}, r"OpponentGrid size must be an integer in \[2, 64\]"),
        ({"size": 64, "num_opponents": 40}, r"1 \+ 40 = 41.*limit.*= 32"),
    ],
)
def test_opponent_grid_limits_are_reported_in_its_own_terms(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pp.make("OpponentGrid-v0", **kwargs)


type Life = pp.runtime.Lifetime[DriftEnvState[OpponentState]]


def trajectory_digest(env: pp.Environment[DriftEnvState[OpponentState]]) -> str:
    """SHA-256 of the positions and reward totals of three random-policy lifetimes."""

    def tick(lifetime: Life) -> Life:
        return pp.runtime.tick(env, pp.policies.random, lifetime)

    step = cast(Callable[[Life], Life], jax.jit(tick))
    digest = hashlib.sha256()
    for seed in range(3):
        lifetime = pp.runtime.init_lifetime(env, seed=seed)
        for _ in range(200):
            lifetime = step(lifetime)
            digest.update(np.asarray(lifetime.state.env.world.positions).tobytes())
            digest.update(np.asarray(lifetime.metrics.total).tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("kwargs", "recorded"),
    [
        ({"size": 5}, "bd47433bb0233d4ecd896be34936ff6c4bf03306b483929e10dcefa57094ee70"),
        (
            {"size": 4, "num_learners": 3},
            "5ba320b661541b08a9a818143a625ffcb1fad3d6de15c6954c414d22bb12acd5",
        ),
    ],
)
def test_one_opponent_lifetimes_are_unchanged(kwargs: dict[str, int], recorded: str) -> None:
    # Digests recorded before opponents spawned on distinct cells and broke
    # ties among themselves; with one opponent neither rule applies.
    assert trajectory_digest(pp.make("OpponentGrid-v0", **kwargs)) == recorded


def test_opponents_start_on_distinct_cells() -> None:
    env = pp.make("OpponentGrid-v0", size=3, num_learners=1, num_opponents=3)
    states, _, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 256))
    cells = np.asarray(states.env.world.positions[:, 1:])
    indices = cells[..., 0] * 3 + cells[..., 1]
    assert all(len(set(row)) == 3 for row in indices.tolist())
    assert set(indices.reshape(-1).tolist()) == set(range(9))


def test_opponents_proposing_one_cell_let_the_lowest_slot_move() -> None:
    env = pp.make("OpponentGrid-v0", size=5, num_learners=1, num_opponents=2)
    state, _, _ = env.init(jax.random.key(0))
    # Both opponents are one greedy step from (0, 2), from either side.
    state = placed(state, [[0, 0], [0, 3], [1, 2]])
    state, _, _, _ = env.step(state, jnp.asarray([0, 0, 0]), jax.random.key(1))
    np.testing.assert_array_equal(state.env.world.positions[1:], [[0, 2], [1, 2]])


def test_opponents_on_one_cell_separate() -> None:
    env = pp.make("OpponentGrid-v0", size=5, num_learners=1, num_opponents=2)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[0, 0], [3, 3], [3, 3]])
    state, _, _, _ = env.step(state, jnp.asarray([0, 0, 0]), jax.random.key(1))
    first, second = np.asarray(state.env.world.positions[1:])
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(second, [3, 3])
    assert int(np.abs(first).sum()) == 5


def test_a_cell_a_learner_also_proposes_stays_empty() -> None:
    from perpetua.envs.continual_grid import MOVES

    env = pp.make("OpponentGrid-v0", size=5, num_learners=1, num_opponents=2)
    state, _, _ = env.init(jax.random.key(0))
    state = placed(state, [[0, 0], [0, 2], [4, 4]])
    toward = int(np.flatnonzero((np.asarray(MOVES) == [0, 1]).all(axis=1))[0])
    moved, _, _, _ = env.step(state, jnp.asarray([toward, 0, 0]), jax.random.key(1))
    np.testing.assert_array_equal(moved.env.world.positions[:2], [[0, 0], [0, 2]])
