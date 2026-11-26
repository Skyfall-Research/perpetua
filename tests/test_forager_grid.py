"""Agent-driven drift: mass from the agent's own behaviour, not the clock."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp


def light() -> pp.drift.Accumulate:
    """Accumulation with gain 1 and decay 0.5, clipped to [1, 8]."""
    return pp.drift.Accumulate(gain=1.0, decay=0.5, low=1.0, high=8.0)


def place(state: Any, row: int, column: int) -> Any:
    """Put the single agent at (row, column) through the wrapper's world state."""
    positions = jnp.asarray([[row, column]], dtype=jnp.int32)
    return state._replace(env=state.env._replace(positions=positions))


def test_accumulate_follows_the_hand_computed_recurrence() -> None:
    process = light()
    state = process.init(jnp.asarray([1.0]), jax.random.key(0))
    # mass_{t+1} = 0.5 * mass_t + 1, starting at 1.
    for want in [1.5, 1.75, 1.875, 1.9375, 1.96875]:
        state = process.step(state, jax.random.key(1), jnp.asarray([1.0]))
        np.testing.assert_allclose(np.asarray(state.value), [want], rtol=1e-6)
    # With no signal the value decays back toward the lower bound.
    for _ in range(4):
        state = process.step(state, jax.random.key(1), jnp.asarray([0.0]))
    assert float(state.value[0]) < 1.5
    with pytest.raises(ValueError, match=r"Accumulate needs an environment signal\."):
        process.step(state, jax.random.key(1), None)


def test_accumulate_clips_and_declares_band_crossings() -> None:
    process = pp.drift.Accumulate(gain=4.0, decay=0.5, low=1.0, high=4.0, bands=[2.0, 3.0])
    state = process.init(jnp.asarray([1.0]), jax.random.key(0))
    state = process.step(state, jax.random.key(1), jnp.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(state.value), [4.0])  # clipped at the upper bound
    assert int(state.phase) == 1  # several crossings on one tick count once
    state = process.step(state, jax.random.key(1), jnp.asarray([0.0]))
    np.testing.assert_allclose(np.asarray(state.value), [2.0])
    assert int(state.phase) == 2  # dropping below the 3.0 band is a crossing too


def test_mass_grows_with_collection_and_shrinks_when_idle() -> None:
    env = pp.make("ForagerGrid-v0", size=3, drift=light())
    state, _, _ = env.init(jax.random.key(0))
    # Mass 1 means every attempted move succeeds, so one down move collects.
    state = place(state, 1, 2)
    state, _, reward, _ = env.step(state, jnp.asarray([2]), jax.random.key(1))
    assert float(reward[0]) == 1.0
    np.testing.assert_allclose(np.asarray(env.privileged(state)["mass_parameters"]), [1.5])
    # A second collection from the same cell grows it: 0.5 * 1.5 + 1 = 1.75.
    state = place(state, 1, 2)
    state, _, reward, _ = env.step(state, jnp.asarray([2]), jax.random.key(2))
    np.testing.assert_allclose(np.asarray(env.privileged(state)["mass_parameters"]), [1.75])
    # Idle ticks (stay actions, no collection) decay it back toward 1.
    for tick in range(3):
        state, _, _, _ = env.step(state, jnp.asarray([0]), jax.random.key(10 + tick))
    assert float(env.privileged(state)["mass_parameters"][0]) < 1.5


def test_heavy_agents_move_less_often_with_the_same_actions() -> None:
    def attempt_rate(mass: float, attempts: int = 400) -> float:
        # Stationary drift so the mass stays at the value being measured.
        env = pp.make("ForagerGrid-v0", size=8, drift=pp.drift.Stationary())
        state, _, _ = env.init(jax.random.key(0))
        state = state._replace(
            drift={"mass": state.drift["mass"]._replace(value=jnp.asarray([mass]))}
        )
        moved = 0
        for tick in range(attempts):
            # Start every attempt from the top-left cell, so the down action is
            # always valid and only the mass decides whether it succeeds.
            state = place(state, 0, 0)
            state, _, _, _ = env.step(state, jnp.asarray([2]), jax.random.key(tick))
            moved += int(np.asarray(state.env.positions)[0, 0] == 1)
        return moved / attempts

    # Mass 1 is deterministic; at mass 8 the success probability is 1/8.
    assert attempt_rate(1.0) == 1.0
    heavy = attempt_rate(8.0)
    assert 0.06 < heavy < 0.19, f"expected ~1/8, got {heavy}"


def test_mass_is_hidden_and_exogenous_drift_still_works() -> None:
    env = pp.make("ForagerGrid-v0", size=3, drift=light())
    _, obs, info = env.init(jax.random.key(0))
    assert obs.shape == (1, 4)  # position and goal only, no mass channel
    assert not any("mass" in key for key in info)
    configuration = cast("dict[str, Any]", cast("dict[str, Any]", env.metadata)["configuration"])
    assert configuration["mass"] == "hidden; only an Oracle exposes it"
    exposed = pp.Oracle(env, expose=["mass_parameters"])
    _, _, o_info = exposed.init(jax.random.key(0))
    oracle = cast("dict[str, Any]", o_info.get("oracle"))
    assert set(oracle) == {"mass_parameters"}
    # A stationary drift is unaffected by the signal path at all.
    still = pp.make("ForagerGrid-v0", size=3, drift=pp.drift.Stationary())
    a, _, _ = still.init(jax.random.key(0))
    b, _, _ = still.init(jax.random.key(0))
    for tick in range(4):
        a, _, _, _ = still.step(a, jnp.asarray([1]), jax.random.key(tick))
        b, _, _, _ = still.step(b, jnp.asarray([1]), jax.random.key(tick))
    assert np.array_equal(np.asarray(a.env.positions), np.asarray(b.env.positions))
    assert float(a.drift["mass"].value[0]) == 1.0


@pytest.mark.parametrize("max_mass", [8, 8.0, np.float32(8.0), np.int64(8)])
def test_max_mass_accepts_any_real_number_and_stores_a_float(max_mass: Any) -> None:
    env = cast(Any, pp.make("ForagerGrid-v0", max_mass=max_mass))
    assert type(env.max_mass) is float
    assert env.max_mass == 8.0
    assert type(env.metadata["configuration"]["max_mass"]) is float


@pytest.mark.parametrize("max_mass", [float("nan"), float("inf"), 0.5, True, "8", None])
def test_max_mass_must_be_finite_and_at_least_one(max_mass: Any) -> None:
    with pytest.raises(ValueError, match="max_mass must be a finite number >= 1"):
        pp.make("ForagerGrid-v0", max_mass=max_mass)
