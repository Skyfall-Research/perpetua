"""Invariants of the native worlds over long random lifetimes under drift."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp

TICKS = 2000
GOAL = pp.drift.Cyclic([(4, 4), (0, 0), (2, 3)], period=37)
DOORS = pp.drift.Cyclic([(1, 1, 1, 1), (1, 0, 1, 0), (0, 1, 1, 1)], period=37)
CASES: dict[str, dict[str, Any]] = {
    "ContinualGrid-v0": {"num_agents": 3, "drift": {"goal": GOAL}},
    "DoorGrid-v0": {"drift": {"doors": DOORS}},
    "ForagerGrid-v0": {
        "drift": {"mass": pp.drift.Accumulate(gain=1.0, decay=0.5, low=1.0, high=8.0)}
    },
    "BeanWorld-v0": {
        "drift": {"item_rewards": pp.drift.Cyclic([(1, 0.1, -1), (-1, -0.1, 1)], period=37)}
    },
    "CooperativeGrid-v0": {"drift": {"goal": GOAL}},
    "PartnerGrid-v0": {"drift": {"goal": GOAL}},
    "OpponentGrid-v0": {"drift": {"goal": GOAL}},
}
# One +1 per collecting slot, on the tick that slot respawns.
REWARD_IS_RESPAWN = {
    "ContinualGrid-v0",
    "DoorGrid-v0",
    "ForagerGrid-v0",
    "CooperativeGrid-v0",
    "PartnerGrid-v0",
}
JOINT = {"CooperativeGrid-v0", "PartnerGrid-v0"}
# A random pair almost never completes a joint collection, so this world is
# driven by the scripted greedy policy.
POLICY = {"CooperativeGrid-v0": pp.policies.greedy}
# Observations are (row, column, goal row, goal column), scaled to [0, 1].
POSITION_AND_GOAL = {"ContinualGrid-v0", "DoorGrid-v0", "ForagerGrid-v0"}


def lifetime(env_id: str, seed: int) -> tuple[Any, dict[str, np.ndarray[Any, Any]]]:
    env = pp.Oracle(pp.make(env_id, **CASES[env_id]), expose=["boundary"])
    policy = POLICY.get(env_id, pp.policies.random)

    def step(carry: Any, key: jax.Array) -> tuple[Any, dict[str, jax.Array]]:
        state, obs, info = carry
        action_key, step_key = jax.random.split(key)
        action = policy(obs, info, action_key)
        state, obs, reward, info = env.step(state, action, step_key)
        record = {
            "obs": obs,
            "reward": reward,
            "respawn": info["respawn"],
            "has_action": jnp.any(info["action_mask"], axis=-1),
            "changed": cast(Any, info)["oracle"]["boundary"],
        }
        return (state, obs, info), record

    key = jax.random.key(seed)
    _, record = jax.lax.scan(step, env.init(key), jax.random.split(key, TICKS))
    return env, {name: np.asarray(value) for name, value in record.items()}


@pytest.mark.parametrize("env_id", CASES)
def test_native_world_invariants_hold_over_a_long_drifting_lifetime(env_id: str) -> None:
    env, seen = lifetime(env_id, seed=0)
    space = env.observation_space
    obs, reward, respawn = seen["obs"], seen["reward"], seen["respawn"]

    assert obs.dtype == np.float32
    assert reward.dtype == np.float32
    assert respawn.dtype == np.bool_
    assert np.all(obs >= space.low)
    assert np.all(obs <= space.high)
    assert np.all(np.isfinite(reward))
    assert seen["has_action"].all()

    if env_id in REWARD_IS_RESPAWN:
        assert respawn.any()
        assert np.array_equal(reward, respawn.astype(np.float32))
    if env_id in JOINT:
        assert np.all(respawn.all(axis=1) | ~respawn.any(axis=1))
        assert np.all(reward == reward[:, :1])
    if env_id in POSITION_AND_GOAL:
        # A collection respawns the agent off the goal, so an agent stands on
        # the goal only on a tick where drift has just moved the goal there.
        on_goal = np.all(obs[..., :2] == obs[..., 2:], axis=-1)
        assert not np.any(on_goal & ~seen["changed"][:, None])
    if env_id == "OpponentGrid-v0":
        assert set(np.unique(reward)) == {-1.0, 0.0, 1.0}
        assert np.all(reward.sum(axis=1) == 0)
    if env_id == "BeanWorld-v0":
        assert not respawn.any()
        assert set(np.unique(reward)) <= {np.float32(v) for v in (-1, -0.1, 0, 0.1, 1)}
    if env_id != "ForagerGrid-v0":  # Accumulate moves a value and declares no switch
        assert seen["changed"].sum() == (TICKS - 1) // 37


@pytest.mark.parametrize("env_id", sorted(CASES))
def test_native_worlds_step_under_strict_rank_promotion(env_id: str) -> None:
    # Operands are broadcast explicitly, so the strict mode that JAX users set
    # in CI to catch accidental broadcasting does not trip over the library.
    env = pp.make(env_id, **CASES[env_id])
    with jax.numpy_rank_promotion("raise"):
        runner = pp.runtime.Runner(env, pp.policies.random, ticks=8)
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
        jax.block_until_ready(carry)
