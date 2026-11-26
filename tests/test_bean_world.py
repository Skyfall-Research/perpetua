"""BeanWorld: a never-ending Jelly-Bean-World-style foraging world."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.envs.bean_world import ITEMS, BeanWorld


def inner(env: Any) -> BeanWorld:
    return cast(BeanWorld, env.env)


def test_layout_has_banded_biomes_at_the_declared_densities() -> None:
    env = pp.make("BeanWorld-v0", size=256, densities=(0.08, 0.04, 0.02))
    state, obs, info = env.init(jax.random.key(0))
    items = np.asarray(state.env.items)
    fractions = [float((items == kind).mean()) for kind in (1, 2, 3)]
    # Each density is the mean of a (1 + sin) field, so the expected share is the density.
    np.testing.assert_allclose(fractions, [0.08, 0.04, 0.02], atol=0.01)
    # Banded biomes: across 16x16 blocks the jelly-bean share runs from almost
    # none to about twice the mean density.
    blocks = (items == 1).reshape(16, 16, 16, 16).mean(axis=(1, 3))
    assert blocks.min() < 0.03
    assert blocks.max() > 0.13
    assert obs.shape == (1, 7 * 7 * len(ITEMS) + len(ITEMS))
    assert bool(info["action_mask"].all())
    assert not bool(info["respawn"].any())


def test_collecting_pays_the_item_reward_and_regrowth_is_exact() -> None:
    world = BeanWorld(size=5, view=3, scent_radius=1, densities=(0.0, 0.0, 0.0))
    params = world.derive(world.default_params._replace(regrow_ticks=jnp.asarray([3.0, 3.0, 3.0])))
    state, _, _ = world.init(jax.random.key(0), params)
    # Put a jelly bean to the right of the agent at (2, 2).
    state = state._replace(
        items=state.items.at[2, 3].set(1), positions=jnp.asarray([[2, 2]], dtype=jnp.int32)
    )
    right, left = jnp.asarray([4]), jnp.asarray([3])
    state, reward = world.step(state, right, jax.random.key(0), params)
    assert float(reward[0]) == 1.0
    rewards = []
    for tick in range(6):  # step off and back on, alternating
        state, reward = world.step(
            state, left if tick % 2 == 0 else right, jax.random.key(0), params
        )
        rewards.append(float(reward[0]))
    # Back on the cell 2, 4 and 6 ticks after collection: absent, then regrown.
    assert rewards == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_the_world_is_a_torus() -> None:
    world = BeanWorld(size=4, densities=(0.0, 0.0, 0.0))
    params = world.default_params
    state, _, _ = world.init(jax.random.key(0), params)
    state = state._replace(positions=jnp.asarray([[0, 3]], dtype=jnp.int32))
    state, _ = world.step(state, jnp.asarray([1]), jax.random.key(0), params)  # up
    state, _ = world.step(state, jnp.asarray([4]), jax.random.key(0), params)  # right
    np.testing.assert_array_equal(np.asarray(state.positions), [[3, 0]])


def test_reward_drift_is_hidden_and_changes_what_items_are_worth() -> None:
    flip = pp.drift.Cyclic([(1.0, 0.1, -1.0), (-1.0, 0.1, 1.0)], 2)
    env = pp.make("BeanWorld-v0", size=16, drift={"item_rewards": flip})
    _, _, info = env.init(jax.random.key(3))
    assert not any("reward" in key for key in info)
    oracle = pp.Oracle(env, expose=["item_rewards_parameters"])
    state, _, _ = oracle.init(jax.random.key(3))
    values: list[np.ndarray[Any, Any]] = []
    for tick in range(4):
        state, _, _, info = oracle.step(state, jnp.asarray([0]), jax.random.key(tick))
        values.append(
            np.asarray(cast(dict[str, Any], info.get("oracle"))["item_rewards_parameters"])
        )
    np.testing.assert_allclose(values[0], [1.0, 0.1, -1.0])
    np.testing.assert_allclose(values[1], [-1.0, 0.1, 1.0])


def test_scent_counts_items_by_distance() -> None:
    world = BeanWorld(size=9, view=1, scent_radius=2, densities=(0.0, 0.0, 0.0))
    params = world.default_params
    state, _, _ = world.init(jax.random.key(0), params)
    state = state._replace(
        items=state.items.at[4, 6].set(3).at[4, 5].set(1),
        positions=jnp.asarray([[4, 4]], dtype=jnp.int32),
    )
    obs, _ = world.observe(state, params)
    scent = np.asarray(obs[0, -3:])
    np.testing.assert_allclose(scent, [1 / 2, 0.0, 1 / 3], rtol=1e-6)


def test_invalid_configuration_is_refused() -> None:
    for kwargs in ({"size": 2}, {"view": 4}, {"densities": (0.3, 0.3, 0.3)}, {"scent_radius": -1}):
        with pytest.raises(ValueError):  # noqa: PT011 - several messages
            BeanWorld(**kwargs)  # pyright: ignore[reportArgumentType]


def test_untouched_items_stay_on_the_ground_at_any_clock_value() -> None:
    env: Any = pp.make("BeanWorld-v0", size=16)
    state, start, _ = env.init(jax.random.key(0))
    assert float(jnp.abs(start).sum()) > 0.0
    for tick in (2**31 - 1, 2**31, 2**31 + 99, 2**32 - 1):
        late = state._replace(env=state.env._replace(tick=jnp.asarray(tick, dtype=jnp.uint32)))
        observation, _ = env.observe(late)
        # At tick 2**31 every never-collected item used to vanish for a regrow period.
        np.testing.assert_array_equal(np.asarray(observation), np.asarray(start))
