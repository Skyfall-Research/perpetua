"""Evolving spaces: masks and events change semantics, never compiled shapes."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

import perpetua as pp


def configuration_of(env: pp.Environment[Any]) -> dict[str, Any]:
    """Read the JSON configuration recorded in an environment's metadata."""
    metadata = cast(dict[str, Any], env.metadata)
    return cast(dict[str, Any], metadata["configuration"])


def space_change(info: pp.Info) -> jax.Array:
    """Read the optional evolving-space event from public info."""
    return cast(jax.Array, info.get("space_change"))


def test_space_signatures_capture_shapes_dtypes_and_cardinality() -> None:
    env = pp.make("DoorGrid-v0", size=3)
    configuration = configuration_of(env)
    signature = cast(dict[str, Any], configuration["spaces"])
    assert signature["observation"] == {"type": "Box", "shape": [1, 4], "dtype": "float32"}
    assert signature["action"] == {
        "type": "MultiDiscrete",
        "shape": [1],
        "dtype": "int64",
        "nvec": [5],
    }
    assert configuration["evolving"] == "action-mask"


def test_space_manager_refuses_shape_or_cardinality_changes() -> None:
    manager = pp.spaces.SpaceManager(
        observation_space=gym_spaces.Box(0.0, 1.0, shape=(2, 4), dtype=np.float32),
        action_space=gym_spaces.MultiDiscrete(np.full(2, 5, dtype=np.int64)),
    )
    manager.check(
        observation_space=gym_spaces.Box(0.0, 1.0, shape=(2, 4), dtype=np.float32),
        action_space=gym_spaces.MultiDiscrete(np.full(2, 5, dtype=np.int64)),
    )
    with pytest.raises(ValueError, match=r"Space 'observation' changed shape, dtype or"):
        manager.check(
            observation_space=gym_spaces.Box(0.0, 1.0, shape=(3, 4), dtype=np.float32),
            action_space=gym_spaces.MultiDiscrete(np.full(2, 5, dtype=np.int64)),
        )
    with pytest.raises(ValueError, match=r"Space 'action' changed shape, dtype or"):
        manager.check(
            observation_space=gym_spaces.Box(0.0, 1.0, shape=(2, 4), dtype=np.float32),
            action_space=gym_spaces.MultiDiscrete(np.full(2, 6, dtype=np.int64)),
        )
    with pytest.raises(ValueError, match=r"Space names must match between configurations\."):
        pp.spaces.validate_stable({"observation": gym_spaces.Box(0.0, 1.0, (2,))}, {})


def test_mask_change_is_exact_and_validated() -> None:
    before = jnp.asarray([[True, True], [True, False]])
    after = jnp.asarray([[True, True], [False, True]])
    np.testing.assert_array_equal(np.asarray(pp.spaces.mask_change(before, after)), [False, True])
    with pytest.raises(ValueError, match=r"Masks must have matching \(agents, actions\)"):
        pp.spaces.mask_change(jnp.zeros((2,)), jnp.zeros((2,)))


def test_door_drift_masks_and_blocks_movement() -> None:
    env = pp.make("DoorGrid-v0", size=5, drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 4))
    state, _, info = env.init(jax.random.key(0))
    state = state._replace(env=state.env._replace(positions=jnp.asarray([[2, 2]])))
    _, info = env.observe(state)
    # Phase zero: every door is open and the interior cell allows all moves.
    np.testing.assert_array_equal(np.asarray(info["action_mask"]), [[True, True, True, True, True]])
    for tick in range(4):
        state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(tick))
    # The period-4 switch closes the down and right doors; the event fires once.
    np.testing.assert_array_equal(
        np.asarray(env.privileged(state)["drift_parameters"]), [1.0, 0.0, 1.0, 0.0]
    )
    np.testing.assert_array_equal(
        np.asarray(info["action_mask"]), [[True, True, False, True, False]]
    )
    assert bool(space_change(info)[0])
    np.testing.assert_array_equal(np.asarray(state.env.positions), [[2, 2]])
    # A masked move becomes stay instead of moving through the closed door.
    blocked, _, _, _ = env.step(state, jnp.asarray([2]), jax.random.key(7))
    np.testing.assert_array_equal(np.asarray(blocked.env.positions), [[2, 2]])


def test_space_change_events_fire_exactly_on_door_switches() -> None:
    env = pp.make("DoorGrid-v0", size=3, drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 4))
    state, _, info = env.init(jax.random.key(0))
    assert not bool(space_change(info)[0])
    changes = []
    for tick in range(12):
        state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(tick))
        if bool(space_change(info)[0]):
            changes.append(tick)
    assert changes == [3, 7, 11]


def test_space_change_is_per_agent_and_only_for_changed_masks() -> None:
    # Closing the "down" door changes the mask of an agent that could move down,
    # but not of an agent on the bottom row, whose down move was already masked.
    env = pp.make(
        "DoorGrid-v0", size=3, num_agents=2, drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 1]], 2)
    )
    state, _, _ = env.init(jax.random.key(0))
    placed = jnp.asarray([[0, 0], [2, 0]], dtype=jnp.int32)
    state = state._replace(env=state.env._replace(positions=placed))
    events = []
    for tick in range(4):
        state, _, _, info = env.step(state, jnp.zeros(2, dtype=jnp.int32), jax.random.key(tick))
        events.append(np.asarray(space_change(info)).tolist())
    # Switches happen on ticks 1 and 3; only the top-left agent is flagged.
    assert events == [[False, False], [True, False], [False, False], [True, False]]


def test_moving_into_a_wall_is_not_a_space_change() -> None:
    # The mask changes when the agent reaches the bottom row, but no drift moved,
    # so the parameter-driven event stays false.
    env = pp.make("DoorGrid-v0", size=4, drift=pp.drift.Stationary())
    state, _, info = env.init(jax.random.key(0))
    state = state._replace(env=state.env._replace(positions=jnp.asarray([[0, 0]])))
    masks = []
    for tick in range(5):
        state, _, _, info = env.step(state, jnp.asarray([2]), jax.random.key(tick))
        masks.append(np.asarray(info["action_mask"]).tolist())
        assert not bool(space_change(info)[0])
    assert masks[0] != masks[-1]


def test_door_parameters_stay_hidden_behind_the_masks() -> None:
    env = pp.make("DoorGrid-v0", size=3, drift=pp.drift.Cyclic([[1, 1, 1, 1], [0, 0, 0, 0]], 4))
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(env=state.env._replace(positions=jnp.asarray([[1, 1]])))
    obs_open, info_open = env.observe(state)
    closed = state._replace(drift={"doors": state.drift["doors"]._replace(value=jnp.zeros(4))})
    obs_closed, info_closed = env.observe(closed)
    # Observations are identical; the affordance change appears only as masks.
    np.testing.assert_array_equal(np.asarray(obs_open), np.asarray(obs_closed))
    assert "drift_parameters" not in info_closed
    assert "phase" not in info_closed
    np.testing.assert_array_equal(
        np.asarray(info_open["action_mask"]), [[True, True, True, True, True]]
    )
    np.testing.assert_array_equal(
        np.asarray(info_closed["action_mask"]), [[True, False, False, False, False]]
    )
    np.testing.assert_array_equal(
        np.asarray(env.privileged(closed)["drift_parameters"]), [0.0, 0.0, 0.0, 0.0]
    )


def test_door_lifetime_crosses_switches_without_recompiling() -> None:
    env = pp.make("DoorGrid-v0", size=4, drift=pp.drift.Cyclic([[1, 1, 1, 1], [0, 1, 0, 1]], 8))
    runner = pp.runtime.Runner(env, pp.policies.greedy, ticks=8)
    carry = pp.runtime.init_lifetime(env, seed=0, window_size=8)
    with pp.measurement.count_compilations() as cold:
        carry = runner(carry)
    assert cold.count >= 1
    with pp.measurement.count_compilations() as warm:
        for _ in range(4):  # ticks 8..40 include four door switches
            carry = runner(carry)
    assert warm.count == 0
    assert carry.metrics.ticks.to_int() == 40
    assert carry.obs.shape == (1, 4)
    assert carry.info["action_mask"].shape == (1, 5)
    assert space_change(carry.info).shape == (1,)
    assert bool(carry.metrics.healthy)


def test_door_grid_rebinds_drift_through_compose() -> None:
    env = pp.make("DoorGrid-v0", size=3)
    composed = pp.compose(env, pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 2))
    assert configuration_of(composed)["drift"]["type"] == "cyclic"
    state, _, info = composed.init(jax.random.key(0))
    for tick in range(2):
        state, _, _, info = composed.step(state, jnp.asarray([0]), jax.random.key(tick))
    assert bool(space_change(info)[0])
    np.testing.assert_array_equal(np.asarray(info["action_mask"][:, 2]), np.asarray([False]))


def test_ppo_trains_through_door_switches_without_recompiling() -> None:
    from perpetua.agents.ppo import PPO, PPOConfig, PPORunner

    env = pp.make("DoorGrid-v0", size=4, drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 16))
    config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=8)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    state = trainer.init(seed=0)
    with pp.measurement.count_compilations() as cold:
        state = trainer(state)
    assert cold.count >= 1
    with pp.measurement.count_compilations() as warm:
        for _ in range(5):  # 40 more ticks cross the door switches at ticks 15 and 31
            state = trainer(state)
    assert warm.count == 0
    assert state.updates.to_int() == 6
    assert state.agent.gradient_steps.to_int() == 6
    assert state.lifetime.state.t.to_int() == 48
    assert state.lifetime.obs.shape == (1, 4)
    assert state.lifetime.info["action_mask"].shape == (1, 5)
    assert space_change(state.lifetime.info).shape == (1,)
    assert bool(state.lifetime.metrics.healthy)


def test_door_grid_initial_mask_uses_the_drifted_doors() -> None:
    env = pp.make("DoorGrid-v0", size=3, drift=pp.drift.Stationary(jnp.zeros(4)))
    state, _, info = env.init(jax.random.key(0))
    np.testing.assert_array_equal(info["action_mask"], env.observe(state)[1]["action_mask"])


def test_stable_spaces_compare_discrete_sizes_and_nested_spaces() -> None:
    with pytest.raises(ValueError, match="changed shape, dtype or cardinality"):
        pp.spaces.validate_stable({"a": gym_spaces.Discrete(3)}, {"a": gym_spaces.Discrete(5)})
    small = gym_spaces.Dict({"x": gym_spaces.Box(0.0, 1.0, (2,))})
    large = gym_spaces.Dict({"x": gym_spaces.Box(0.0, 1.0, (3,))})
    with pytest.raises(ValueError, match="changed shape, dtype or cardinality"):
        pp.spaces.validate_stable({"a": small}, {"a": large})
    pp.spaces.validate_stable({"a": small}, {"a": small})
    pair = gym_spaces.Tuple((gym_spaces.Discrete(2), gym_spaces.Discrete(3)))
    assert pp.spaces.space_signature(pair)["spaces"] == [
        pp.spaces.space_signature(gym_spaces.Discrete(2)),
        pp.spaces.space_signature(gym_spaces.Discrete(3)),
    ]
