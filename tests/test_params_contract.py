"""Conformance for parameter environments: one suite every environment passes.

The framework owns drift state and keys, declared switch events, the lifetime
clock, Oracle exposure and metadata; an environment only supplies parameters
and dynamics. This suite is parametrised so ported environments join it as
they migrate.
"""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp


def optional(info: pp.Info, key: str) -> jax.Array:
    """Read an optional info entry (NotRequired keys need a checked lookup)."""
    return cast(jax.Array, info.get(key))


def configuration_of(env: pp.Environment[Any]) -> dict[str, Any]:
    """Read the JSON configuration recorded in an environment's metadata."""
    return cast(dict[str, Any], cast(dict[str, Any], env.metadata)["configuration"])


def door_grid(**kwargs: Any) -> pp.Environment[Any]:
    """DoorGrid with a period-4 door schedule; other parameters stay default."""
    return pp.make(
        "DoorGrid-v0",
        drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 4),
        **kwargs,
    )


def forager_grid(**kwargs: Any) -> pp.Environment[Any]:
    """Agent-driven accumulation of mass; masks never depend on it."""
    return pp.make(
        "ForagerGrid-v0",
        drift=pp.drift.Accumulate(gain=1.0, decay=0.5, low=1.0, high=8.0),
        **kwargs,
    )


def bean_world(**kwargs: Any) -> pp.Environment[Any]:
    """Item rewards swap every 4 ticks; the torus never changes a mask."""
    return pp.make(
        "BeanWorld-v0",
        drift={"item_rewards": pp.drift.Cyclic([(1.0, 0.1, -1.0), (-1.0, 0.1, 1.0)], 4)},
        **kwargs,
    )


def goal_grid(env_id: str) -> Callable[..., pp.Environment[Any]]:
    """A grid whose goal follows a period-4 schedule; masks do not depend on it."""

    def factory(**kwargs: Any) -> pp.Environment[Any]:
        return pp.make(env_id, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 4), **kwargs)

    return factory


# (name, factory, ticks within 12 where space_change fires). Goal drift never
# changes an action mask, so only DoorGrid reports space-change events.
ENVIRONMENTS: list[tuple[str, Callable[..., pp.Environment[Any]], list[int]]] = [
    ("DoorGrid-v0", door_grid, [3, 7, 11]),
    ("ContinualGrid-v0", goal_grid("ContinualGrid-v0"), []),
    ("CooperativeGrid-v0", goal_grid("CooperativeGrid-v0"), []),
    ("PartnerGrid-v0", goal_grid("PartnerGrid-v0"), []),
    ("OpponentGrid-v0", goal_grid("OpponentGrid-v0"), []),
    ("ForagerGrid-v0", forager_grid, []),
    ("BeanWorld-v0", bean_world, []),
]


def actions(env: pp.Environment[Any], value: int) -> jax.Array:
    """One integer action per agent slot."""
    return jnp.full((env.num_agents,), value % 5, dtype=jnp.int32)


@pytest.mark.parametrize(("env_id", "factory", "switches"), ENVIRONMENTS)
def test_seeded_reproducibility(
    env_id: str, factory: Callable[..., pp.Environment[Any]], switches: list[int]
) -> None:
    del env_id, switches
    first, second = factory(size=3), factory(size=3)
    a, b = first.init(jax.random.key(0)), second.init(jax.random.key(0))
    assert_same(a[0], b[0])
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
    for tick in range(6):
        a = first.step(a[0], actions(first, tick), jax.random.key(tick))
        b = second.step(b[0], actions(second, tick), jax.random.key(tick))
        assert_same(a[0], b[0])
        np.testing.assert_array_equal(np.asarray(a[2]), np.asarray(b[2]))


@pytest.mark.parametrize(("env_id", "factory", "switches"), ENVIRONMENTS)
def test_chunk_continuity_and_clock(
    env_id: str, factory: Callable[..., pp.Environment[Any]], switches: list[int]
) -> None:
    del env_id, switches
    env = factory(size=3)
    long_run = pp.runtime.Runner(env, pp.policies.random, ticks=8)(
        pp.runtime.init_lifetime(env, seed=1, window_size=4)
    )
    short = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    chunked = short(short(pp.runtime.init_lifetime(env, seed=1, window_size=4)))
    assert_same(long_run.state, chunked.state)
    assert long_run.info["t"].to_int() == chunked.info["t"].to_int() == 8
    assert long_run.state.t.to_int() == 8
    assert long_run.obs.shape == chunked.obs.shape


@pytest.mark.parametrize(("env_id", "factory", "switches"), ENVIRONMENTS)
def test_privilege_isolation_and_oracle(
    env_id: str, factory: Callable[..., pp.Environment[Any]], switches: list[int]
) -> None:
    del env_id, switches
    env = factory(size=3)
    state, _, info = env.init(jax.random.key(0))
    assert not {"phase", "drift_parameters", "drift_changed", "task_id", "boundary"} & set(info)
    assert not any("drift" in key for key in info)
    exposed = env.privileged(state)
    assert {"phase", "drift_parameters", "drift_changed"} <= set(exposed)
    fields = {name.rsplit("_", 1)[0] for name in exposed if name.endswith("_parameters")}
    fields.discard("drift")
    for field in fields:
        assert {f"{field}_parameters", f"{field}_phase", f"{field}_changed"} <= set(exposed)
    # The aggregate names report the primary (default drifting) field, so
    # single-field environments keep the names and shapes they always had.
    primary = next(
        field
        for field in fields
        if np.array_equal(
            np.asarray(exposed["drift_parameters"]), np.asarray(exposed[f"{field}_parameters"])
        )
    )
    np.testing.assert_array_equal(
        np.asarray(exposed["phase"]), np.asarray(exposed[f"{primary}_phase"])
    )
    assert env.oracle_fields[:3] == ("phase", "drift_parameters", "drift_changed")
    # The generic names from the API sketch: the active segment and the switch flag.
    np.testing.assert_array_equal(np.asarray(exposed["task_id"]), np.asarray(exposed["phase"]))
    np.testing.assert_array_equal(
        np.asarray(exposed["boundary"]), np.asarray(exposed["drift_changed"])
    )

    oracle = pp.Oracle(env, expose=["drift_parameters", "boundary"])
    _, _, o_info = oracle.init(jax.random.key(0))
    assert set(optional(o_info, "oracle")) == {"drift_parameters", "boundary"}
    # The old name still works, with a warning.
    with pytest.warns(DeprecationWarning, match="expose 'boundary'"):
        alias = pp.Oracle(env, expose=["drift_changed"])
    _, _, a_info = alias.init(jax.random.key(0))
    np.testing.assert_array_equal(
        np.asarray(optional(a_info, "oracle")["drift_changed"]),
        np.asarray(optional(o_info, "oracle")["boundary"]),
    )
    exposure = oracle.metadata["oracle_exposure"]
    assert "drift_parameters" in exposure
    assert "phase" not in exposure


@pytest.mark.parametrize(("env_id", "factory", "switches"), ENVIRONMENTS)
def test_space_change_events_and_stable_shapes(
    env_id: str, factory: Callable[..., pp.Environment[Any]], switches: list[int]
) -> None:
    del env_id
    env = factory(size=3)
    state, obs, info = env.init(jax.random.key(0))
    obs_shape, mask_shape = obs.shape, info["action_mask"].shape
    assert optional(info, "space_change").shape == (env.num_agents,)
    assert not bool(jnp.any(optional(info, "space_change")))
    events: list[int] = []
    for tick in range(12):
        state, obs, _, info = env.step(state, actions(env, 0), jax.random.key(tick))
        assert obs.shape == obs_shape
        assert info["action_mask"].shape == mask_shape
        if bool(optional(info, "space_change")[0]):
            events.append(tick)
    assert events == switches


@pytest.mark.parametrize(("env_id", "factory", "switches"), ENVIRONMENTS)
def test_metadata_and_unknown_drift_fields(
    env_id: str, factory: Callable[..., pp.Environment[Any]], switches: list[int]
) -> None:
    del switches
    env = factory(size=3)
    configuration = configuration_of(env)
    expected_type = cast("str", next(iter(cast(Any, env).drifts.values())).configuration["type"])
    assert configuration["drift"]["type"] == expected_type
    assert list(configuration["drifts"].values()) == [configuration["drift"]]
    assert env.metadata["env_id"] == env_id
    with pytest.raises(ValueError, match="Unknown drift parameters"):
        pp.make(env_id, drift={"hatch": pp.drift.Stationary()})
    field = next(iter(configuration["drifts"]))
    with pytest.raises(ValueError, match="Unknown drift parameters"):
        pp.make(env_id, drift={f"{field}.inner": pp.drift.Stationary()})


@pytest.mark.parametrize(("env_id", "factory", "switches"), ENVIRONMENTS)
def test_checkpoint_round_trip(
    env_id: str, factory: Callable[..., pp.Environment[Any]], switches: list[int], tmp_path: Any
) -> None:
    del env_id, switches
    env = factory(size=3)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    carry = runner(pp.runtime.init_lifetime(env, seed=2, window_size=4))
    path = tmp_path / "lifetime"
    pp.checkpoint.save(path, carry, metadata=runner.metadata)
    restored = pp.checkpoint.restore(
        path,
        pp.runtime.init_lifetime(env, seed=9, window_size=4),
        expected_metadata=runner.metadata,
    )
    assert_same(runner(carry), runner(restored))


@pytest.mark.parametrize("env_id", ["BeanWorld-v0", "ForagerGrid-v0", "mjx/Pendulum-v0"])
def test_shipped_worlds_never_split_the_lifetime_key_into_a_drift_key(env_id: str) -> None:
    """Whatever width a built-in world splits its init key to, no child is a drift key."""
    if env_id.startswith("mjx/"):
        pytest.importorskip("mujoco.mjx")
    seen: dict[str, jax.Array] = {}

    class Recording(pp.drift.Stationary):
        def init(self, value: jax.Array, key: jax.Array) -> pp.drift.DriftState:
            seen["drift_init"] = key
            return super().init(value, key)

    env = cast(Any, pp.make(env_id, drift=Recording()))
    original = env.env.init

    def recording_init(key: jax.Array, params: Any) -> Any:
        seen["world"] = key
        return original(key, params)

    env.env.init = recording_init
    state, _, _ = env.init(jax.random.key(3))
    drift_keys = [np.asarray(jax.random.key_data(k)) for k in (seen["drift_init"], state.drift_key)]
    for width in (2, 3, 4, 5, 8):
        for child in jax.random.split(seen["world"], width):
            data = np.asarray(jax.random.key_data(child))
            assert not any(np.array_equal(data, other) for other in drift_keys)


def test_drift_init_keys_never_reuse_the_drift_stream() -> None:
    """A combinator splits its init key; that key must not equal a stream key."""
    seen: list[jax.Array] = []

    class Recording:
        """Stationary drift that records the key it is initialised with."""

        def init(self, value: jax.Array, key: jax.Array) -> pp.drift.DriftState:
            seen.append(jax.random.key_data(key))
            return pp.drift.Stationary().init(value, key)

        def step(self, state: pp.drift.DriftState, key: jax.Array) -> pp.drift.DriftState:
            return state

        @property
        def configuration(self) -> dict[str, Any]:
            return {"type": "recording"}

    env = pp.make("PartnerGrid-v0", size=3, drift={"goal": Recording(), "partner": Recording()})
    state, _, _ = env.init(jax.random.key(0))
    first_draw = jax.random.split(state.drift_key)[1]
    assert len(seen) == 2
    # Each field gets its own init key, and neither is the stream or a draw key.
    assert not np.array_equal(np.asarray(seen[0]), np.asarray(seen[1]))
    for key_data in seen:
        assert not np.array_equal(key_data, jax.random.key_data(state.drift_key))
        assert not np.array_equal(key_data, jax.random.key_data(first_draw))
