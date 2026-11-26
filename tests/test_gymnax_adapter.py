"""The generic gymnax adapter: any environment, safe drift, continuing time."""

# gymnax ships no type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportPrivateUsage=false

import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua import registration
from perpetua.adapters.gymnax_params import GymnaxParameters

GENERIC_IDS = tuple(f"gymnax/{name}" for name in registration._GYMNAX_GENERIC)


def test_every_registered_gymnax_environment_constructs() -> None:
    assert len(GENERIC_IDS) == 20
    for env_id in GENERIC_IDS:
        env = pp.make(env_id)
        assert env.metadata["env_id"] == env_id
        assert env.num_agents == 1
        assert (env.observation_space.shape or (0,))[0] == 1
        assert (env.action_space.shape or (0,))[0] == 1


def test_broken_upstream_gymnax_environments_are_unregistered() -> None:
    for env_id in ("gymnax/SimpleBandit-bsuite", "gymnax/PointRobot-misc"):
        with pytest.raises(ValueError, match="Unknown environment"):
            pp.make(env_id)


def test_drifting_a_precomputed_input_is_refused() -> None:
    with pytest.raises(ValueError, match="precomputes"):
        GymnaxParameters("CartPole-v1", drift_fields=("masspole",))
    with pytest.raises(ValueError, match="cannot drift"):
        GymnaxParameters("CartPole-v1", drift_fields=("nonsense",))
    # The fields the physics reads directly stay driftable, including the
    # derived ones such as total_mass.
    adapter = GymnaxParameters("CartPole-v1")
    assert "gravity" in adapter.drift_fields
    assert "total_mass" in adapter.drift_fields
    assert "masspole" not in adapter.drift_fields


def test_time_limit_is_removed_for_continuing_lifetimes() -> None:
    adapter = GymnaxParameters("CartPole-v1")
    params = adapter.derive(adapter.default_params)
    assert int(params.max_steps_in_episode) > 10**9


def test_drift_and_continuing_contract_through_the_wrapper() -> None:
    env = pp.make("gymnax/Acrobot-v1", drift={"link_mass_1": pp.drift.Cyclic([1.0, 1.2], 4)})
    state, obs, info = env.init(jax.random.key(0))
    assert obs.shape == (1, 6)
    assert info["action_mask"].shape == (1, 3)
    assert info["t"].to_int() == 0
    assert not bool(jax.tree.leaves(info["respawn"])[0])
    parameters = []
    for tick in range(6):
        state, obs, reward, info = env.step(
            state, jnp.zeros(1, dtype=jnp.int32), jax.random.key(tick)
        )
        assert obs.shape == (1, 6)
        assert reward.shape == (1,)
        parameters.append(float(env.privileged(state)["link_mass_1_parameters"]))
    assert info["t"].to_int() == 6
    assert int(state.drift["link_mass_1"].phase) == 1  # the period-4 switch fired
    assert len({round(value, 4) for value in parameters}) == 2


def test_continuous_actions_have_box_spaces() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    state, obs, _ = env.init(jax.random.key(0))
    assert env.action_space.shape == (1, 1)
    assert obs.shape == (1, 3)
    rewards = []
    for tick in range(3):
        state, obs, reward, _ = env.step(
            state, jnp.zeros((1, 1), dtype=jnp.float32), jax.random.key(tick)
        )
        rewards.append(np.asarray(reward))
    assert rewards[-1].shape == (1,)
    assert np.all(np.isfinite(rewards))


# The last two never terminate, so their counters only ever stop at the ceiling
# (the long-run soak flags them as linearly growing). DiscountingChain keeps its
# time limit as a respawn and is covered below.
@pytest.mark.parametrize(
    "env_id",
    [
        "Acrobot-v1",
        "MemoryChain-bsuite",
        "MetaMaze-misc",
        "Freeway-MinAtar",
        "Swimmer-misc",
    ],
)
def test_episode_counter_never_fires_a_time_limit_respawn(env_id: str) -> None:
    # Start the backend's episode counter just below the raised time limit
    # (about 2.1 billion ticks into a lifetime); stepping past it must not
    # fire a time-limit respawn or wrap the counter.
    env = pp.make(f"gymnax/{env_id}")
    state, _, _ = env.init(jax.random.key(0))
    time = jnp.asarray(state.env.env.time)
    # Integer counters start 4 ticks below the limit. A float32 counter (MetaMaze)
    # stops increasing at 2**24, so it starts at the largest exact value used.
    start = 2**31 - 4 if jnp.issubdtype(time.dtype, jnp.integer) else 2**31 - 256
    near = state.env.env.replace(time=jnp.asarray(start, dtype=time.dtype))
    state = state._replace(env=state.env._replace(env=near))
    space = env.action_space
    continuous = np.dtype(space.dtype).kind == "f"
    action = jnp.zeros(space.shape, dtype=jnp.float32 if continuous else jnp.int32)
    for tick in range(6):
        state, _, _, info = env.step(state, action, jax.random.key(tick))
        assert not bool(info["respawn"][0]), f"spurious respawn at tick {tick}"
        assert 0 < int(state.env.env.time) < 2**31 - 1


def test_reset_runs_only_on_termination() -> None:
    # A respawn still supplies a fresh state, exactly as gymnax's own reset.
    env = pp.make("gymnax/Catch-bsuite")
    state, _, _ = env.init(jax.random.key(0))
    respawned = 0
    for tick in range(40):
        state, obs, _, info = env.step(
            state, jnp.zeros((1,), dtype=jnp.int32), jax.random.key(tick)
        )
        if bool(info["respawn"][0]):
            respawned += 1
            assert int(state.env.env.time) == 0
            assert float(jnp.sum(obs)) == 2.0  # a fresh ball and paddle
    assert respawned >= 3


def test_generic_gymnax_ids_refuse_unknown_keyword_arguments() -> None:
    import perpetua as pp

    with pytest.raises(TypeError, match="size"):
        pp.make("gymnax/Acrobot-v1", size=3)


def test_unreadable_parameter_source_requires_explicit_drift_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    def no_source(obj: object) -> str:
        raise OSError("could not get source code")

    monkeypatch.setattr(inspect, "getsource", no_source)
    with pytest.raises(ValueError, match=r"pass drift_fields=\(\.\.\.\) explicitly"):
        GymnaxParameters("CartPole-v1")
    adapter = GymnaxParameters("CartPole-v1", drift_fields=("gravity",))
    assert adapter.drift_fields == ("gravity",)


def _rewards(env: Any, action: int, ticks: int) -> tuple[list[float], int]:
    step = jax.jit(env.step)
    state, _, _ = env.init(jax.random.key(0))
    rewards: list[float] = []
    respawns = 0
    for tick in range(ticks):
        state, _, reward, info = step(
            state, jnp.full((1,), action, jnp.int32), jax.random.key(tick)
        )
        rewards.append(float(reward[0]))
        respawns += int(info["respawn"][0])
    return rewards, respawns


def test_time_limit_is_never_a_drift_field() -> None:
    adapter = GymnaxParameters("BernoulliBandit-misc")
    assert "max_steps_in_episode" not in adapter.drift_fields
    assert adapter.default_drift_field == "reward_prob"
    with pytest.raises(ValueError, match="cannot drift"):
        GymnaxParameters("BernoulliBandit-misc", drift_fields=("max_steps_in_episode",))


def test_time_limit_only_environments_keep_it_as_a_respawn() -> None:
    # Without its limit the chain would pay its one reward once per lifetime.
    rewards, respawns = _rewards(pp.make("gymnax/DiscountingChain-bsuite"), 0, 350)
    assert [tick for tick, reward in enumerate(rewards) if reward != 0.0] == [0, 100, 200, 300]
    assert respawns == 3
    metadata = cast(Any, pp.make("gymnax/DiscountingChain-bsuite").metadata)
    assert "kept as a respawn" in json.dumps(metadata["configuration"])
    # Environments with real terminations still lose their limit.
    _, acrobot_respawns = _rewards(pp.make("gymnax/Acrobot-v1"), 1, 600)
    assert acrobot_respawns == 0  # its 500-step limit is removed


def test_drift_on_reset_time_parameters_takes_effect() -> None:
    # The Gaussian bandit samples its arm mean from mean_mu only at reset.
    default = float(GymnaxParameters("GaussianBandit-misc").default_params.mean_mu)
    plain, _ = _rewards(pp.make("gymnax/GaussianBandit-misc"), 1, 150)
    drift = {"mean_mu": pp.drift.Cyclic([default, default + 50.0], 50)}
    drifted, _ = _rewards(pp.make("gymnax/GaussianBandit-misc", drift=drift), 1, 150)
    # A reset-time parameter takes effect at the next respawn (tick 100 here),
    # not at the switch itself (tick 50).
    assert drifted[:100] == plain[:100]
    assert drifted[100:] != plain[100:]


def test_observations_are_float32_as_the_space_declares() -> None:
    env = pp.make("gymnax/FourRooms-misc")
    state, obs, _ = env.init(jax.random.key(0))
    assert obs.dtype == jnp.float32 == env.observation_space.dtype
    _, obs, _, _ = env.step(state, jnp.zeros((1,), jnp.int32), jax.random.key(1))
    assert obs.dtype == jnp.float32


@pytest.mark.parametrize("env_id", GENERIC_IDS)
def test_observations_stay_inside_the_declared_box(env_id: str) -> None:
    env = pp.make(env_id)
    space = cast(Any, env.observation_space)
    actions = cast(Any, env.action_space)

    def sample(key: jax.Array) -> jax.Array:
        if hasattr(actions, "nvec"):
            return jax.random.randint(key, (1,), 0, int(actions.nvec[0]))
        return jax.random.uniform(
            key, actions.shape, minval=jnp.asarray(actions.low), maxval=jnp.asarray(actions.high)
        )

    def step(carry: Any, key: jax.Array) -> tuple[Any, None]:
        state, low, high = carry
        action_key, step_key = jax.random.split(key)
        state, obs, _, _ = env.step(state, sample(action_key), step_key)
        return (state, jnp.minimum(low, obs), jnp.maximum(high, obs)), None

    key = jax.random.key(0)
    state, obs, _ = env.init(key)
    (_, low, high), _ = jax.lax.scan(step, (state, obs, obs), jax.random.split(key, 400))
    assert np.all(np.asarray(low) >= space.low)
    assert np.all(np.asarray(high) <= space.high)


@pytest.mark.parametrize("env_id", ["gymnax/MountainCar-v0", "gymnax/FourRooms-misc"])
def test_discrete_actions_are_checked_for_shape_and_dtype(env_id: str) -> None:
    env = pp.make(env_id)
    state, _, _ = env.init(jax.random.key(0))
    key = jax.random.key(1)
    for bad in (
        jnp.zeros((), jnp.int32),
        jnp.zeros((2,), jnp.int32),
        jnp.zeros((1, 1), jnp.int32),
        jnp.zeros((1,), jnp.float32),
        jnp.zeros((1,), jnp.bool_),
    ):
        with pytest.raises(ValueError, match=r"integer arrays of shape \(1,\).*got dtype"):
            env.step(state, bad, key)
    state, obs, _, _ = env.step(state, jnp.zeros((1,), jnp.int8), key)
    assert obs.shape == env.observation_space.shape


@pytest.mark.parametrize(
    "env_id", ["gymnax/MountainCarContinuous-v0", "gymnax/Reacher-misc", "gymnax/Swimmer-misc"]
)
def test_random_policy_on_a_box_action_environment_is_refused(env_id: str) -> None:
    env = pp.make(env_id)
    state, obs, info = env.init(jax.random.key(0))
    action = pp.policies.random(obs, info, jax.random.key(1))
    with pytest.raises(ValueError, match=r"pp\.policies\.uniform\(env\.action_space\)"):
        env.step(state, action, jax.random.key(2))
    action = pp.policies.uniform(env.action_space)(obs, info, jax.random.key(1))
    env.step(state, action, jax.random.key(2))
    with pytest.raises(ValueError, match="float arrays of shape"):
        env.step(state, jnp.zeros((1, 7), jnp.float32), jax.random.key(2))


def test_gymnax_wrappers_follow_their_declared_dtypes_under_x64() -> None:
    with jax.enable_x64(True):
        for env_id in ("gymnax/CartPole-v1", "gymnax/Pendulum-v1", "gymnax/FourRooms-misc"):
            env = pp.make(env_id)
            discrete = hasattr(env.action_space, "nvec")
            action = (
                jnp.zeros((1,), jnp.int32)
                if discrete
                else jnp.zeros(env.action_space.shape, jnp.float32)
            )
            state, obs, _ = env.init(jax.random.key(0))
            assert obs.dtype == jnp.float32, env_id
            state, obs, _, _ = env.step(state, action, jax.random.key(1))
            assert obs.dtype == jnp.float32, env_id


def test_gymnax_pendulum_makes_without_precision_warnings() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pp.make("gymnax/Pendulum-v1")
        pp.make("gymnax/CartPole-v1")


def test_a_field_the_backend_never_reads_is_not_driftable() -> None:
    # gymnax 0.0.9's Acrobot reads link_length_1 only.
    assert "link_length_2" not in GymnaxParameters("Acrobot-v1").drift_fields
    assert "link_length_1" in GymnaxParameters("Acrobot-v1").drift_fields
    refused = pytest.raises(
        ValueError, match=r"Acrobot-v1 cannot drift 'link_length_2': .*link_length_1 only"
    )
    with refused:
        pp.make("gymnax/Acrobot-v1", drift={"link_length_2": pp.drift.Cyclic([0.1, 5.0], 10)})
    with refused:
        pp.make("gymnax/Acrobot-v1", drift_fields=("link_length_2",))
    with refused:
        cast(Any, pp.make("gymnax/Acrobot-v1")).with_drift(
            {"link_length_2": pp.drift.Stationary(2.0)}
        )
    with pytest.raises(ValueError, match=r"Task 1 \(GymnaxParameters\): Acrobot-v1 cannot drift"):
        pp.tasks.switch(
            [GymnaxParameters("MountainCar-v0"), GymnaxParameters("Acrobot-v1")],
            {"tasks[1].link_length_2": pp.drift.Cyclic([0.1, 5.0], 10)},
        )


# Driftable fields that change nothing in SWEEP_TICKS ticks of random play at the
# two generic test values, with why, and a pair of parameter settings (given as
# overrides of the defaults) under which the field does change the run.
SWEEP_TICKS = 300
BINDS_ELSEWHERE: dict[tuple[str, str], tuple[str, dict[str, float], dict[str, float]]] = {
    ("Acrobot-v1", "max_vel_1"): (
        "a clip on the first joint's speed that random torques do not reach",
        {},
        {"max_vel_1": 0.5},
    ),
    ("Acrobot-v1", "max_vel_2"): (
        "a clip on the second joint's speed that random torques do not reach",
        {},
        {"max_vel_2": 0.5},
    ),
    ("MountainCar-v0", "max_position"): (
        "the right wall, which random pushes do not reach",
        {},
        {"max_position": -0.6},
    ),
    ("MountainCar-v0", "goal_position"): (
        "the goal on the right hill, which random pushes do not reach",
        {},
        {"goal_position": -0.6},
    ),
    ("MountainCar-v0", "goal_velocity"): (
        "read only once the car is at the goal position",
        {"goal_position": -0.6},
        {"goal_position": -0.6, "goal_velocity": 1.0},
    ),
    ("MountainCarContinuous-v0", "max_position"): (
        "the right wall, which random pushes do not reach",
        {},
        {"max_position": -0.6},
    ),
    ("MountainCarContinuous-v0", "goal_position"): (
        "the goal on the right hill, which random pushes do not reach",
        {},
        {"goal_position": -0.6},
    ),
    ("MountainCarContinuous-v0", "goal_velocity"): (
        "read only once the car is at the goal position",
        {"goal_position": -0.6},
        {"goal_position": -0.6, "goal_velocity": 1.0},
    ),
}


def _rollouts(env_id: str, settings: list[dict[str, float]]) -> list[list[np.ndarray]]:
    """One random-play trace (observations, rewards, respawns) per parameter setting."""
    env = cast(Any, pp.make(env_id))
    world = env.env
    base = world.default_params
    policy = pp.policies.uniform(env.action_space)
    batch = {
        name: jnp.asarray(
            [row.get(name, float(getattr(base, name))) for row in settings], dtype=jnp.float32
        )
        for name in world.drift_fields
    }

    def rollout(overrides: dict[str, jax.Array]) -> Any:
        params = world.derive(base.replace(**overrides))
        state, obs, info = world.init(jax.random.key(0), params)

        def body(carry: Any, key: jax.Array) -> Any:
            state, obs, info = carry
            act_key, step_key = jax.random.split(key)
            state, reward = world.step(state, policy(obs, info, act_key), step_key, params)
            obs, info = world.observe(state, params)
            return (state, obs, info), (obs, reward, info["respawn"])

        keys = jax.random.split(jax.random.key(1), SWEEP_TICKS)
        _, trace = jax.lax.scan(body, (state, obs, info), keys)
        return (obs, *trace)

    stacked = [np.asarray(leaf) for leaf in jax.jit(jax.vmap(rollout))(batch)]
    return [[leaf[row] for leaf in stacked] for row in range(len(settings))]


def _differ(first: list[np.ndarray], second: list[np.ndarray]) -> bool:
    return any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize(
    "env_id", [env_id for env_id in GENERIC_IDS if cast(Any, pp.make(env_id)).env.drift_fields]
)
def test_every_driftable_field_changes_the_run(env_id: str) -> None:
    world = cast(Any, pp.make(env_id)).env
    backend = env_id.split("/", 1)[1]
    base = world.default_params
    settings: list[dict[str, float]] = [{}]
    for name in world.drift_fields:
        value = float(getattr(base, name))
        settings += [{name: value * 1.5 + 0.25}, {name: value * 0.5 - 0.1}]
    listed = [name for name in world.drift_fields if (backend, name) in BINDS_ELSEWHERE]
    for name in listed:
        settings += list(BINDS_ELSEWHERE[backend, name][1:])
    traces = _rollouts(env_id, settings)
    generic = 1 + 2 * len(world.drift_fields)
    inert = [
        name
        for index, name in enumerate(world.drift_fields)
        if not _differ(traces[0], traces[2 * index + 1])
        and not _differ(traces[0], traces[2 * index + 2])
    ]
    # Only listed fields may be quiet here (Acrobot is chaotic, so whether a listed
    # clip binds can depend on the platform), and each of those is read elsewhere.
    assert set(inert) <= set(listed)
    for index, name in enumerate(listed):
        assert _differ(traces[generic + 2 * index], traces[generic + 2 * index + 1]), name
