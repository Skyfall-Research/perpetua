"""MuJoCo control on MJX: drifting gravity and mass, compiled and continuing."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownLambdaType=false
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import platform
import sys
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp

pytest.importorskip("mujoco.mjx", reason="optional mjx extra")


def zero(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
    del obs, info, key
    return jnp.zeros((1, 1), jnp.float32)


def run(env: pp.Environment[Any], ticks: int) -> pp.runtime.Lifetime[Any]:
    return pp.runtime.Runner(env, zero, ticks=ticks)(pp.runtime.init_lifetime(env, seed=4))


@pytest.mark.parametrize(("env_id", "features"), [("mjx/Pendulum-v0", 3), ("mjx/CartPole-v0", 5)])
def test_spaces_and_a_continuing_lifetime(env_id: str, features: int) -> None:
    env = pp.make(env_id)
    _, obs, _ = env.init(jax.random.key(0))
    assert obs.shape == (1, features)
    assert env.action_space.shape == (1, 1)
    # Starts hanging down: cos(angle from upright) is about -1.
    assert float(obs[0, -3]) < -0.9
    carry = run(env, 400)
    assert carry.state.t.to_int() == 400
    assert bool(carry.metrics.healthy)
    assert not bool(carry.info["respawn"].any())


def test_zero_gravity_keeps_the_unforced_pendulum_still() -> None:
    still: Any = pp.make("mjx/Pendulum-v0", drift={"gravity": pp.drift.Cyclic([(0.0,)], 1)})
    earth: Any = pp.make("mjx/Pendulum-v0")
    state, _, _ = still.init(jax.random.key(1))
    state = state._replace(env=state.env._replace(qvel=jnp.zeros(1, jnp.float32)))
    for tick in range(20):
        state, _, _, _ = still.step(state, jnp.zeros((1, 1)), jax.random.key(tick))
    # No gravity, no velocity: the pole stays where it was put.
    start, _, _ = still.init(jax.random.key(1))
    np.testing.assert_allclose(np.asarray(state.env.qpos), np.asarray(start.env.qpos), atol=1e-5)
    falling, _, _ = earth.init(jax.random.key(1))
    falling = falling._replace(env=falling.env._replace(qpos=jnp.asarray([0.5], jnp.float32)))
    moved, _, _, _ = earth.step(falling, jnp.zeros((1, 1)), jax.random.key(0))
    assert abs(float(moved.env.qvel[0])) > 0.1


def test_heavier_pole_responds_less_to_the_same_torque() -> None:
    def spin(scale: float) -> float:
        env = pp.make(
            "mjx/Pendulum-v0",
            drift={
                "gravity": pp.drift.Cyclic([(0.0,)], 1),
                "mass_scale": pp.drift.Cyclic([(scale,)], 1),
            },
        )
        state, _, _ = env.init(jax.random.key(0))
        state = state._replace(env=state.env._replace(qvel=jnp.zeros(1, jnp.float32)))
        state, _, _, _ = env.step(state, jnp.full((1, 1), 2.0), jax.random.key(0))
        return float(state.env.qvel[0])

    light, heavy = spin(1.0), spin(4.0)
    assert light > 0
    assert heavy > 0
    np.testing.assert_allclose(light / heavy, 4.0, rtol=0.05)


def test_reward_matches_the_pendulum_cost() -> None:
    env = pp.make("mjx/Pendulum-v0")
    state, _, _ = env.init(jax.random.key(0))
    state, obs, reward, _ = env.step(state, jnp.full((1, 1), 1.0), jax.random.key(0))
    angle = float(np.arctan2(obs[0, 1], obs[0, 0]))
    expected = -(angle**2 + 0.1 * float(obs[0, 2]) ** 2 + 0.001)
    np.testing.assert_allclose(float(reward[0]), expected, rtol=1e-5)


def test_lifetimes_vmap_and_reproduce() -> None:
    env = pp.make("mjx/CartPole-v0", drift={"mass_scale": pp.drift.Cyclic([(1.0,), (2.0,)], 8)})
    runner = pp.runtime.Runner(env, zero, ticks=16)

    def lifetime(seed: jax.Array) -> pp.runtime.Lifetime[Any]:
        return runner(pp.runtime.init_lifetime(env, seed=seed))

    batch = cast(Any, jax.jit(jax.vmap(lifetime))(jnp.arange(3)))
    again = cast(Any, jax.jit(jax.vmap(lifetime))(jnp.arange(3)))
    np.testing.assert_array_equal(np.asarray(batch.obs), np.asarray(again.obs))
    assert batch.obs.shape == (3, 1, 5)
    assert not np.array_equal(np.asarray(batch.obs[0]), np.asarray(batch.obs[1]))


def test_invalid_actions_and_settings_are_refused() -> None:
    env = pp.make("mjx/Pendulum-v0")
    state, _, _ = env.init(jax.random.key(0))
    with pytest.raises(ValueError, match=r"MJX control actions must be float arrays of"):
        env.step(state, jnp.zeros((1,)), jax.random.key(0))
    with pytest.raises(ValueError, match=r"frame_skip must be a positive integer\."):
        pp.make("mjx/Pendulum-v0", frame_skip=0)


def test_a_lifetime_runs_under_x64() -> None:
    with jax.enable_x64(True):
        env = pp.make("mjx/Pendulum-v0")
        lifetime = run(env, 3)
        assert lifetime.obs.dtype == jnp.float32


def push(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
    del obs, info, key
    return jnp.ones((1, 1), jnp.float32)


def scaled_model(task: str, scale: float) -> Any:
    """The task's model with every mass scaled and its constants recomputed by MuJoCo."""
    import mujoco
    from mujoco import mjx

    from perpetua.adapters.mjx import TASKS

    model = mujoco.MjModel.from_xml_string(TASKS[task])
    model.body_mass[:] *= scale
    model.body_inertia[:] *= scale
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return mjx.put_model(model)


@pytest.mark.parametrize("task", ["pendulum", "cartpole"])
def test_mass_scale_matches_a_model_recomputed_by_mujoco(task: str) -> None:
    import dataclasses

    from perpetua.adapters.mjx import MJXControl, MJXParams

    world = MJXControl(task=task)
    params = MJXParams(jnp.asarray([9.81], jnp.float32), jnp.asarray([4.0], jnp.float32))
    ours, reference = world._physics(params), scaled_model(task, 4.0)  # pyright: ignore[reportPrivateUsage]
    compared = 0
    for field in dataclasses.fields(reference):
        expected = getattr(reference, field.name)
        if isinstance(expected, jax.Array) and expected.dtype.kind == "f" and expected.size:
            np.testing.assert_allclose(
                np.asarray(getattr(ours, field.name)),
                np.asarray(expected),
                rtol=1e-5,
                atol=1e-7,
                err_msg=field.name,
            )
            compared += 1
    assert compared > 20
    np.testing.assert_allclose(
        float(ours.stat.meaninertia), float(reference.stat.meaninertia), rtol=1e-5
    )


def test_cartpole_under_mass_drift_follows_the_recomputed_model_at_the_slider_limit() -> None:
    from mujoco import mjx

    env: Any = pp.make("mjx/CartPole-v0", drift={"mass_scale": pp.drift.Cyclic([(4.0,)], 1)})
    reference = scaled_model("cartpole", 4.0)

    def advance(data: Any) -> Any:
        data = data.replace(ctrl=jnp.ones(1, data.qpos.dtype))
        data = jax.lax.fori_loop(0, 5, lambda _, current: mjx.step(reference, current), data)
        # The environment keeps its state in float32 between decisions.
        return data.replace(
            qpos=data.qpos.astype(jnp.float32).astype(data.qpos.dtype),
            qvel=data.qvel.astype(jnp.float32).astype(data.qvel.dtype),
        )

    state, _, _ = env.init(jax.random.key(4))
    data = mjx.make_data(reference).replace(qpos=state.env.qpos, qvel=state.env.qvel)
    step, advance = jax.jit(env.step), jax.jit(advance)
    worst, furthest = 0.0, 0.0
    for tick in range(300):
        state, _, _, _ = step(state, jnp.ones((1, 1), jnp.float32), jax.random.key(tick))
        data = advance(data)
        worst = max(worst, float(jnp.max(jnp.abs(state.env.qpos - data.qpos))))
        furthest = max(furthest, float(state.env.qpos[0]))
    assert furthest > 1.99  # the cart pressed against the slider limit at 2
    # 0.02 before the solver's inverse weights were scaled with the masses.
    assert worst < 1e-3


@pytest.mark.parametrize(
    ("env_id", "drift", "qpos", "qvel", "total"),
    [
        (
            "mjx/CartPole-v0",
            None,
            ["0x1.000bdap+1", "0x1.844a4cp+1"],
            ["0x1.ad7eb2p-15", "-0x1.2ef3ep+0"],
            "-0x1.02947p+8",
        ),
        ("mjx/Pendulum-v0", None, ["0x1.abf3d2p+1"], ["-0x1.038f4ep-4"], "-0x1.437a32p+11"),
        (
            "mjx/CartPole-v0",
            {"gravity": pp.drift.Cyclic([(9.81,), (4.0,)], 16)},
            ["0x1.000b36p+1", "0x1.cb3e36p+4"],
            ["0x1.25f36cp-16", "-0x1.576a48p+0"],
            "-0x1.f38154p+5",
        ),
    ],
    ids=["cartpole", "pendulum", "cartpole-gravity-drift"],
)
def test_lifetimes_without_mass_drift_are_unchanged(
    env_id: str, drift: Any, qpos: list[str], qvel: list[str], total: str
) -> None:
    # Recorded on macOS arm64 CPU (float32) before the mass-derived constants
    # were scaled: at mass_scale 1 every one of them keeps its value. Other
    # platforms round differently, so only the recording platform is exact.
    env: Any = pp.make(env_id) if drift is None else pp.make(env_id, drift=drift)
    carry = pp.runtime.Runner(env, push, ticks=300)(pp.runtime.init_lifetime(env, seed=4))
    actual = [
        *np.asarray(carry.state.env.qpos).tolist(),
        *np.asarray(carry.state.env.qvel).tolist(),
        float(carry.metrics.total[0]),
    ]
    recorded = [float.fromhex(value) for value in (*qpos, *qvel, total)]
    here = sys.platform == "darwin" and platform.machine() == "arm64"
    if here and jax.default_backend() == "cpu" and not jax.config.jax_enable_x64:
        assert actual == recorded
    else:
        # 300 ticks with the cart at its limit: an H200 ends 0.3% away.
        np.testing.assert_allclose(actual, recorded, rtol=2e-2, atol=2e-2)


def test_unit_mass_scale_leaves_every_model_constant_untouched() -> None:
    import dataclasses

    from perpetua.adapters.mjx import MJXControl

    world = MJXControl(task="cartpole")
    scaled = world._physics(world.default_params)  # pyright: ignore[reportPrivateUsage]
    base = cast(Any, world._model)  # pyright: ignore[reportPrivateUsage]
    for field in dataclasses.fields(base):
        if field.name == "opt":
            continue
        before = getattr(base, field.name)
        if isinstance(before, jax.Array):
            np.testing.assert_array_equal(
                np.asarray(getattr(scaled, field.name)), np.asarray(before), err_msg=field.name
            )
