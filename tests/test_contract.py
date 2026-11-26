"""Behavioral contract for the native continuing environment."""

# JAX 0.11 transformation/device annotations are partial.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.drift import DriftState
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def test_make_and_initial_public_contract() -> None:
    env = pp.make("ContinualGrid-v0", size=4, num_agents=2)
    state, obs, info = env.init(jax.random.key(0))
    assert env.observation_space.contains(np.asarray(obs))
    assert env.action_space.contains(np.asarray([0, 0]))
    assert obs.shape == (2, 4)
    assert info["action_mask"].shape == (2, 5)
    assert info["active_mask"].shape == (2,)
    assert info["t"].to_int() == 0
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert not hasattr(env, "reset")
    assert not np.any(np.all(np.asarray(state.env.positions) == [3, 3], axis=-1))


def test_reward_and_respawn_are_world_events() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(env=state.env._replace(positions=jnp.asarray([[3, 2]], dtype=jnp.int32)))
    state, obs, reward, info = env.step(state, jnp.asarray([4]), jax.random.key(1))
    np.testing.assert_array_equal(reward, [1.0])
    np.testing.assert_array_equal(info["respawn"], [True])
    assert not np.array_equal(np.asarray(state.env.positions[0]), [3, 3])
    assert env.observation_space.contains(np.asarray(obs))
    assert info["t"].to_int() == 1
    assert "terminated" not in info
    assert "truncated" not in info


def test_masks_invalid_actions_and_inactive_slots() -> None:
    env = pp.make("ContinualGrid-v0", size=4, num_agents=2)
    state, _, _ = env.init(jax.random.key(0))
    world = state.env._replace(
        positions=jnp.asarray([[0, 0], [3, 3]], dtype=jnp.int32),
        active=jnp.asarray([True, False]),
    )
    state = state._replace(env=world)
    _, info = env.observe(state)
    np.testing.assert_array_equal(info["action_mask"][0], [True, False, True, False, True])
    np.testing.assert_array_equal(info["action_mask"][1], [True, False, False, False, False])
    next_state, _, reward, _ = env.step(state, jnp.asarray([-1, 99]), jax.random.key(1))
    np.testing.assert_array_equal(next_state.env.positions, state.env.positions)
    np.testing.assert_array_equal(reward, [0.0, 0.0])


def test_action_shape_and_dtype_are_checked() -> None:
    env = pp.make("ContinualGrid-v0")
    state, _, _ = env.init(jax.random.key(0))
    with pytest.raises(ValueError, match="shape"):
        env.step(state, jnp.asarray(0), jax.random.key(1))
    with pytest.raises(TypeError, match="integer"):
        env.step(state, jnp.asarray([1.5]), jax.random.key(1))


@pytest.mark.parametrize(("size", "num_agents"), [(1, 1), (5, 0)])
def test_bad_environment_configuration(size: int, num_agents: int) -> None:
    with pytest.raises(ValueError, match=r"must be a"):
        pp.make("ContinualGrid-v0", size=size, num_agents=num_agents)


def test_unknown_environment_has_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        pp.make("Missing-v0")


def test_unknown_environment_suggests_the_closest_ids() -> None:
    with pytest.raises(ValueError, match=r"Did you mean .*'gymnax/CartPole-v1'"):
        pp.make("CartPole-v1")


def test_gym_habits_get_errors_that_name_the_perpetua_call() -> None:
    env = pp.make("ContinualGrid-v0")
    assert not hasattr(env, "reset")
    with pytest.raises(AttributeError, match=r"no reset.*env\.init\(jax\.random\.key"):
        env.reset()  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError, match=r"pass jax\.random\.key\(0\)"):
        env.init(0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"Unknown oracle field: goal; .*'task_id'"):
        pp.Oracle(env, expose=["goal"])


def test_cyclic_drift_is_hidden_and_has_fixed_shapes() -> None:
    env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], period=2))
    state, obs, info = env.init(jax.random.key(0))
    structure = cast(object, jax.tree.structure((state, obs, info)))
    step = cast(
        Callable[
            [DriftEnvState[GridState], jax.Array, jax.Array],
            tuple[DriftEnvState[GridState], jax.Array, jax.Array, pp.Info],
        ],
        jax.jit(env.step),
    )
    for t in range(1, 7):
        state, obs, _, info = step(state, jnp.asarray([0]), jax.random.key(t))
        assert jax.tree.structure((state, obs, info)) == structure
        assert obs.shape == (1, 4)
        assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
        assert int(state.drift["goal"].phase) == (t // 2) % 2


def test_oracle_is_explicit_and_recorded() -> None:
    base = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], 1))
    env = pp.Oracle(base, expose=["phase", "boundary"])
    state, _, info = env.init(jax.random.key(0))
    assert "oracle" in info
    assert set(info["oracle"]) == {"phase", "boundary"}
    state, _, _, info = env.step(state, jnp.asarray([0]), jax.random.key(1))
    assert "oracle" in info
    assert int(info["oracle"]["phase"]) == 1
    assert bool(info["oracle"]["boundary"])
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=2)
    assert runner.metadata["oracle_exposure"] == ("phase", "boundary")
    with pytest.raises(ValueError, match="Unknown oracle field"):
        pp.Oracle(base, expose=["unknown"])


def test_clock_crosses_uint32_and_ten_billion_without_global_x64() -> None:
    x64 = jax.config.x64_enabled
    clock = pp.Clock.from_int(2**32 - 1)
    increment = cast(Callable[[pp.Clock], pp.Clock], jax.jit(pp.Clock.increment))
    assert increment(clock).to_int() == 2**32
    assert increment(pp.Clock.from_int(10_000_000_000)).to_int() == 10_000_000_001
    assert jax.config.x64_enabled == x64
    with pytest.raises(ValueError, match=r"Clock must be an integer in"):
        pp.Clock.from_int(-1)


def test_random_walk_bounds_reproducibility_and_shape() -> None:
    drift = pp.drift.RandomWalk(low=(0, 0), high=(3, 3), scale=0.5)
    initial = drift.init(jnp.asarray([1.0, 2.0]), jax.random.key(0))
    left = right = initial
    step = cast(Callable[[DriftState, jax.Array], DriftState], jax.jit(drift.step))
    for t in range(20):
        left = step(left, jax.random.key(t))
        right = step(right, jax.random.key(t))
        np.testing.assert_array_equal(left.value, right.value)
        assert left.value.shape == (2,)
        assert np.all((np.asarray(left.value) >= 0) & (np.asarray(left.value) <= 3))
    assert not np.array_equal(np.asarray(initial.value), np.asarray(left.value))


def test_drift_configuration_validation() -> None:
    with pytest.raises(ValueError, match=r"Cyclic period must be an integer in"):
        pp.drift.Cyclic([(0, 0)], period=0)
    with pytest.raises(ValueError, match=r"Random-walk bounds must be ordered and scale"):
        pp.drift.RandomWalk(low=2, high=1)
    with pytest.raises(ValueError, match=r"Random-walk bounds must be ordered and scale"):
        pp.drift.RandomWalk(low=0, high=1, scale=-1)


def test_drift_rng_is_independent_of_agent_actions() -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.RandomWalk(low=0, high=4, scale=0.3))
    left, _, _ = env.init(jax.random.key(0))
    right = left
    for t in range(8):
        left, _, _, _ = env.step(left, jnp.asarray([0]), jax.random.key(t))
        right, _, _, _ = env.step(right, jnp.asarray([4]), jax.random.key(t))
        np.testing.assert_array_equal(left.drift["goal"].value, right.drift["goal"].value)
        np.testing.assert_array_equal(
            jax.random.key_data(left.drift_key), jax.random.key_data(right.drift_key)
        )


def test_switching_tick_uses_old_goal_then_returns_new_observation() -> None:
    env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], 1))
    state, _, _ = env.init(jax.random.key(0))
    state = state._replace(env=state.env._replace(positions=jnp.asarray([[3, 2]], dtype=jnp.int32)))
    _, obs, reward, _ = env.step(state, jnp.asarray([4]), jax.random.key(1))
    np.testing.assert_array_equal(reward, [1.0])
    np.testing.assert_array_equal(obs[:, 2:], [[0.0, 0.0]])


def test_incompatible_drift_parameter_shape_is_rejected() -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(1, 2, 3)], 1))
    with pytest.raises(ValueError, match="shape"):
        env.init(jax.random.key(0))


def test_procedural_drift_redraws_on_schedule_reproducibly() -> None:
    drift = pp.drift.Procedural(low=(0, 0), high=(3, 3), period=3)
    initial = drift.init(jnp.asarray([1.0, 2.0]), jax.random.key(0))
    step = cast(Callable[[DriftState, jax.Array], DriftState], jax.jit(drift.step))
    left = right = initial
    phases = []
    for t in range(7):
        left = step(left, jax.random.key(t))
        right = step(right, jax.random.key(t))
        np.testing.assert_array_equal(left.value, right.value)
        assert left.value.shape == (2,)
        assert np.all((np.asarray(left.value) >= 0) & (np.asarray(left.value) <= 3))
        phases.append(int(left.phase))
    assert phases == [0, 0, 1, 1, 1, 2, 2]
    np.testing.assert_array_equal(np.asarray(initial.value), [1.0, 2.0])
    changed = not np.array_equal(
        np.asarray(initial.value),
        np.asarray(
            step(step(step(initial, jax.random.key(0)), jax.random.key(1)), jax.random.key(2)).value
        ),
    )
    assert changed


def test_procedural_drift_integration_and_validation() -> None:
    env = pp.make(
        "ContinualGrid-v0", size=4, drift=pp.drift.Procedural(low=(0, 0), high=(3, 3), period=3)
    )
    state, _, _ = env.init(jax.random.key(0))
    flags = []
    for t in range(6):
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(t))
        flags.append(bool(state.drift_changed))
    assert flags == [False, False, True, False, False, True]
    assert int(state.drift["goal"].phase) == 2
    assert np.all(
        (np.asarray(state.drift["goal"].value) >= 0) & (np.asarray(state.drift["goal"].value) <= 3)
    )
    with pytest.raises(ValueError, match=r"Procedural period must be an integer in"):
        pp.drift.Procedural(low=0, high=1, period=0)
    with pytest.raises(ValueError, match=r"Procedural bounds must be ordered\."):
        pp.drift.Procedural(low=2, high=1)
    with pytest.raises(ValueError, match=r"Procedural bounds must be finite\."):
        pp.drift.Procedural(low=float("nan"), high=1)
    with pytest.raises(ValueError, match=r"Incompatible shapes for broadcasting:"):
        pp.drift.Procedural(low=np.zeros(3), high=np.ones(3)).init(
            jnp.asarray([1.0, 2.0]), jax.random.key(0)
        )
    configuration = pp.drift.Procedural(low=0, high=1, period=5).configuration
    assert configuration["type"] == "procedural"
    assert configuration["distribution"] == "uniform"


@pytest.mark.parametrize("name", ["task_id", "boundary", "drift_changed"])
def test_oracle_expose_must_be_a_list_not_a_string(name: str) -> None:
    env = pp.make("ContinualGrid-v0")
    with pytest.raises(TypeError, match=rf"expose must be a list or tuple.*expose=\['{name}'\]"):
        pp.Oracle(env, expose=cast(Any, name))
    field = "boundary" if name == "drift_changed" else name
    assert pp.Oracle(env, expose=cast(Any, (field,))).expose == (field,)


def test_drift_values_must_be_processes() -> None:
    schedule_data = pp.drift.schedule([1.0, 2.0], period=2)
    with pytest.raises(
        TypeError, match=r"drift must be a pp\.drift process.*is ScheduleData.*Schedule"
    ):
        pp.make("ContinualGrid-v0", drift=cast(Any, schedule_data))
    with pytest.raises(TypeError, match=r"the process for 'goal' is list"):
        pp.make("ContinualGrid-v0", drift=cast(Any, {"goal": [1, 2], "x": [3]}))
    with pytest.raises(TypeError, match=r"drift must be a pp\.drift process.*it is str"):
        pp.make("ContinualGrid-v0", drift=cast(Any, "cyclic"))
    with pytest.raises(TypeError, match=r"process instance.*class Cyclic.*Cyclic\(\.\.\.\)"):
        pp.make("ContinualGrid-v0", drift=cast(Any, pp.drift.Cyclic))


def test_numpy_and_jax_integers_are_accepted_and_stored_as_python_ints() -> None:
    # Typed as Any: the annotations say int, the contract accepts any integer scalar.
    i64, i32, u8, f64 = cast(Any, np.int64), cast(Any, np.int32), cast(Any, np.uint8), np.float64
    cyclic = pp.drift.Cyclic([(0, 0), (1, 1)], period=i64(4))
    assert type(cyclic.period) is int
    sequence = pp.drift.Sequence([(pp.drift.Stationary(), jnp.int32(3)), (cyclic, u8(2))])
    assert sequence.ticks == (3, 2)
    assert all(type(t) is int for t in sequence.ticks)
    assert pp.drift.schedule([1.0, 2.0], period=i32(3), offset=i64(1)).period == 3
    grid = cast(Any, pp.make("ContinualGrid-v0", size=i64(4), num_agents=i32(2)))
    assert grid.num_agents == 2
    assert grid.size == 4
    runner = pp.runtime.Runner(grid, pp.policies.random, ticks=i64(3))
    assert runner(pp.runtime.init_lifetime(grid, window_size=i32(4))).state.t.to_int() == 3
    pp.metrics.init(num_agents=i64(2), window_size=i32(8))
    with pytest.raises(ValueError, match=r"Cyclic period must be an integer in .* Got float64"):
        pp.drift.Cyclic([1.0], period=cast(Any, f64(4)))
    with pytest.raises(ValueError, match=r"ticks must be a positive integer\. Got bool True"):
        pp.runtime.Runner(grid, pp.policies.random, ticks=cast(Any, True))
    with pytest.raises(ValueError, match=r"Got int 0"):
        pp.runtime.HostRunner(grid, pp.policies.random, ticks=0)


def test_float_drift_settings_are_json_ready() -> None:
    f32: Any = np.float32
    walk = pp.drift.RandomWalk(low=0, high=1, scale=f32(0.25))
    accumulate = pp.drift.Accumulate(gain=f32(0.5), decay=jnp.float32(0.5), low=f32(0), high=f32(1))
    for process in (walk, accumulate):
        assert all(type(v) in (float, list, str, bool) for v in process.configuration.values())
    env = pp.make("ForagerGrid-v0", drift=accumulate)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=2)
    carry = runner(pp.runtime.init_lifetime(env, window_size=4))
    import json

    json.dumps(runner.metadata)
    del carry


def test_drift_bounds_are_required_where_a_default_would_move_the_parameter() -> None:
    for process in (pp.drift.RandomWalk, pp.drift.Procedural, pp.drift.Accumulate):
        with pytest.raises(TypeError, match="low"):
            process()  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("values", "parameter", "hint"),
    [
        ([(9.8,), (4.9,)], (), "plain number"),
        ([9.8, 4.9], (1,), "1-tuple"),
        ([9.8, 4.9], (2,), r"array of shape \(2,\)"),
    ],
)
def test_cyclic_shape_hint_follows_the_parameter_shape(
    values: list[Any], parameter: tuple[int, ...], hint: str
) -> None:
    cyclic = pp.drift.Cyclic(values, period=2)
    with pytest.raises(ValueError, match=hint):
        cyclic.init(jnp.zeros(parameter), jax.random.key(0))


def test_a_gymnax_environment_without_driftable_fields_says_so() -> None:
    with pytest.raises(ValueError, match=r"gymnax/Pong-misc has no driftable parameters"):
        pp.make("gymnax/Pong-misc", drift=pp.drift.Stationary())
    with pytest.raises(ValueError, match=r"has no driftable parameters"):
        pp.make("gymnax/Catch-bsuite", drift={"max_steps_in_episode": pp.drift.Stationary()})
    acrobot = pp.make("gymnax/Acrobot-v1")
    with pytest.raises(ValueError, match=r"Unknown drift parameters \['nope'\]; available: \["):
        pp.compose(acrobot, {"nope": pp.drift.Stationary()})
    assert pp.make("gymnax/Pong-misc").num_agents == 1
