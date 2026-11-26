"""``pp.testing.check_env`` passes correct environments and names what breaks."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces

import perpetua as pp
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv
from perpetua.testing import ContractError


class Broken:
    """Delegates to a real environment and breaks one part of the contract."""

    def __init__(self, env: Any, fault: str) -> None:
        self.env, self.fault = env, fault

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def init(self, key: jax.Array) -> Any:
        state, obs, info = self.env.init(key)
        if self.fault == "leak":
            info = {**info, "phase": jnp.asarray(0)}
        if self.fault in ("leak_boundary", "leak_task_id", "leak_alias"):
            name = {"leak_boundary": "boundary", "leak_task_id": "task_id"}.get(
                self.fault, "goal_changed"
            )
            info = {**info, name: jnp.asarray(False)}
        if self.fault == "missing":
            info = {name: value for name, value in info.items() if name != "respawn"}
        if self.fault == "observation_dtype":
            obs = obs.astype(jnp.int32)
        if self.fault == "flag_dtype":
            info = {**info, "respawn": info["respawn"].astype(jnp.int32)}
        if self.fault == "mask_shape":
            info = {**info, "action_mask": info["action_mask"][..., :-1]}
        if self.fault == "no_valid_action":
            info = {**info, "action_mask": jnp.zeros_like(info["action_mask"])}
        return state, obs, info

    def step(self, state: Any, action: Any, key: jax.Array) -> Any:
        state, obs, reward, info = self.env.step(state, action, key)
        if self.fault == "nondeterministic":
            reward = reward + jax.random.uniform(jax.random.key(id(self) % 1000))
        if self.fault == "shape":
            obs = obs[..., :-1]
        if self.fault == "leak":
            info = {**info, "phase": jnp.asarray(0)}
        if self.fault in ("leak_boundary", "leak_task_id", "leak_alias"):
            name = {"leak_boundary": "boundary", "leak_task_id": "task_id"}.get(
                self.fault, "goal_changed"
            )
            info = {**info, name: jnp.asarray(False)}
        return state, obs, reward, info


def test_registered_environments_pass() -> None:
    pp.testing.check_env(lambda: pp.make("ContinualGrid-v0", size=4))
    pp.testing.check_env(
        lambda: pp.make("DoorGrid-v0", drift=pp.drift.Cyclic([(1, 1, 1, 1), (1, 0, 1, 0)], 2))
    )


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("leak", "privilege: public info leaks"),
        ("leak_boundary", r"privilege: public info leaks \['boundary'\]"),
        ("leak_task_id", r"privilege: public info leaks \['task_id'\]"),
        ("leak_alias", r"privilege: public info leaks \['goal_changed'\]"),
        ("missing", "shapes: info lacks"),
        ("nondeterministic", "reproducibility: reward"),
        ("shape", "shapes: observation changed shape"),
        ("observation_dtype", "types: observation is int32 but the space declares"),
        ("flag_dtype", r"types: info\['respawn'\] must be boolean"),
        ("mask_shape", "shapes: action_mask must have shape"),
        ("no_valid_action", "shapes: an active agent has no valid action"),
    ],
)
def test_broken_environments_are_named(fault: str, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        pp.testing.check_env(lambda: cast(Any, Broken(pp.make("ContinualGrid-v0", size=4), fault)))


class LineParams(NamedTuple):
    goal: jax.Array


class LineState(NamedTuple):
    position: jax.Array
    respawn: jax.Array
    moved: jax.Array


class LineWorld:
    """Four cells, actions stay/left/right; ``fault`` breaks it away from action 0."""

    default_drift_field = "goal"
    num_agents = 1

    def __init__(self, fault: str) -> None:
        self.fault = fault
        self.observation_space = spaces.Box(0.0, 3.0, (1, 2), np.float32)
        self.action_space = spaces.MultiDiscrete(np.asarray([3], dtype=np.int64))
        self.metadata = {
            "env_id": "line",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {"fault": fault},
        }

    @property
    def default_params(self) -> LineParams:
        return LineParams(jnp.asarray(3.0, jnp.float32))

    def derive(self, params: LineParams) -> LineParams:
        return LineParams(jnp.clip(jnp.rint(params.goal), 0, 3))

    def init(self, key: jax.Array, params: LineParams) -> Any:
        state = LineState(jnp.int32(0), jnp.asarray(False), jnp.asarray(False))
        return (state, *self.observe(state, params))

    def step(self, state: LineState, action: Any, key: jax.Array, params: LineParams) -> Any:
        move = jnp.clip(jnp.asarray(action, jnp.int32)[0], 0, 2)
        step = jnp.where(move == 2, 1, jnp.where(move == 1, -1, 0))
        position = jnp.clip(state.position + step, 0, 3)
        reached = position == params.goal.astype(jnp.int32)
        reward = jnp.where(reached, 1.0, 0.0)
        if self.fault == "nan_reward":
            reward = jnp.where(move != 0, jnp.nan, 0.0)
        if self.fault == "inf_reward":
            reward = jnp.where(reached, jnp.inf, 0.0)
        position = jnp.where(reached, 0, position).astype(jnp.int32)
        moved = state.moved | (move != 0)
        return LineState(position, reached, moved), jnp.asarray([reward], jnp.float32)

    def observe(self, state: LineState, params: LineParams) -> Any:
        obs = jnp.stack((state.position, params.goal.astype(jnp.int32)))[None]
        mask = jnp.ones((1, 3), jnp.bool_)
        if self.fault == "mask_closes":
            mask = mask & ~state.moved
        info = {
            "action_mask": mask,
            "active_mask": jnp.ones((1,), jnp.bool_),
            "respawn": state.respawn[None],
        }
        return obs.astype(jnp.float32), info

    def privileged(self, state: LineState) -> dict[str, Any]:
        return {}


def line(fault: str) -> Any:
    return DriftEnv(cast(Any, LineWorld(fault)), pp.drift.Cyclic([3.0, 1.0], 8), env_id="line")


def test_faults_away_from_the_zero_action_are_found() -> None:
    pp.testing.check_env(lambda: line("none"), ticks=16)
    for fault, message in (
        ("nan_reward", "shapes: non-finite reward at tick"),
        ("inf_reward", "shapes: non-finite reward at tick"),
        ("mask_closes", "shapes: an active agent has no valid action at tick"),
    ):
        with pytest.raises(ContractError, match=message):
            pp.testing.check_env(lambda fault=fault: line(fault), ticks=16)


def test_runner_checks_refuse_a_non_finite_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    # With the direct checks out of the way, the random lifetimes must still object.
    def skipped(factory: Any, ticks: int) -> None:
        del factory, ticks

    monkeypatch.setattr(pp.testing, "_shapes_and_types", skipped)
    monkeypatch.setattr(pp.testing, "_reproducibility", skipped)
    with pytest.raises(ContractError, match="continuity: a uniformly random lifetime"):
        pp.testing.check_env(lambda: line("nan_reward"), ticks=16)


def test_sampled_actions_follow_the_masks() -> None:
    env = pp.make("DoorGrid-v0", drift=pp.drift.Cyclic([(1, 1, 1, 1), (1, 0, 1, 0)], 2))
    act = cast(Any, pp.testing)._sampler(env)
    _, obs, info = env.init(jax.random.key(0))
    closed = info["action_mask"].at[:, 1:].set(False)
    assert not np.any(np.asarray(act(obs, info, 0)))
    drawn = np.stack([np.asarray(act(obs, info, tick)) for tick in range(4, 68)])
    assert len(np.unique(drawn)) > 1
    assert np.array_equal(drawn, np.stack([np.asarray(act(obs, info, t)) for t in range(4, 68)]))
    assert all(bool(info["action_mask"][0, a]) for a in drawn[:, 0])
    assert not np.any(np.asarray(act(obs, {**info, "action_mask": closed}, 9)))
    idle = {**info, "active_mask": jnp.zeros_like(info["active_mask"])}
    assert not any(np.any(np.asarray(act(obs, idle, tick))) for tick in range(4, 36))
    box = pp.make("gymnax/Pendulum-v1")
    _, obs, info = box.init(jax.random.key(0))
    sample = np.asarray(cast(Any, pp.testing)._sampler(box)(obs, info, 5))
    space: Any = box.action_space
    assert sample.shape == space.shape
    assert np.any(sample != 0)
    assert np.all((sample >= space.low) & (sample <= space.high))


def test_ticks_must_be_positive() -> None:
    with pytest.raises(ValueError, match=r"ticks must be a positive integer\."):
        pp.testing.check_env(lambda: pp.make("ContinualGrid-v0"), ticks=0)


class ActionOnePays:
    """A grid whose reward is 1 exactly when the action is 1, with an optional mask."""

    def __init__(self, env: Any, allowed: tuple[int, ...]) -> None:
        self.env = env
        self.mask = jnp.zeros((1, 5), dtype=bool).at[0, jnp.asarray(allowed)].set(True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def init(self, key: jax.Array) -> Any:
        state, obs, info = self.env.init(key)
        return state, obs, {**info, "action_mask": self.mask}

    def step(self, state: Any, action: Any, key: jax.Array) -> Any:
        state, obs, _, info = self.env.step(state, action, key)
        return state, obs, (action == 1).astype(jnp.float32), {**info, "action_mask": self.mask}


def test_reward_action_sensitivity_separates_blind_and_informative_rewards() -> None:
    sensitivity = pp.testing.reward_action_sensitivity
    # CartPole pays for the pole's position, which this tick's action cannot move.
    assert sensitivity(pp.make("gymnax/CartPole-v1"), ticks=200) == 0.0
    assert sensitivity(pp.make("ContinualGrid-v0"), ticks=200) > 0.0
    assert sensitivity(pp.make("gymnax/BernoulliBandit-misc"), ticks=200) > 0.5
    # Continuous actions and more than one agent slot.
    assert sensitivity(pp.make("gymnax/Pendulum-v1"), ticks=50) == 1.0
    assert sensitivity(pp.make("OpponentGrid-v0"), ticks=200) > 0.0


def test_reward_action_sensitivity_compares_valid_actions_only() -> None:
    grid = pp.make("ContinualGrid-v0")
    sensitivity = pp.testing.reward_action_sensitivity
    assert sensitivity(cast(Any, ActionOnePays(grid, (0, 1, 2))), ticks=50) > 0.5
    # With the paying action masked, no valid action changes the reward.
    assert sensitivity(cast(Any, ActionOnePays(grid, (0, 2, 3))), ticks=50) == 0.0


def test_reward_action_sensitivity_on_host_backends() -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    snapshot = GymnasiumLifetime(gym.make("CartPole-v1"))
    opaque = GymnasiumLifetime(gym.make("Acrobot-v1"))
    try:
        assert pp.testing.reward_action_sensitivity(snapshot, ticks=20, actions=2) == 0.0
        with pytest.raises(ValueError, match="several actions from the same state"):
            pp.testing.reward_action_sensitivity(opaque, ticks=2)
    finally:
        snapshot.close()
        opaque.close()


def test_reward_action_sensitivity_checks_its_arguments() -> None:
    env = pp.make("ContinualGrid-v0")
    with pytest.raises(ValueError, match="actions must be an integer >= 2"):
        pp.testing.reward_action_sensitivity(env, actions=1)
    with pytest.raises(ValueError, match="ticks must be a positive integer"):
        pp.testing.reward_action_sensitivity(env, ticks=0)


def test_the_deprecated_drift_changed_field_is_not_required() -> None:
    def factory() -> Any:
        env: Any = Broken(pp.make("ContinualGrid-v0", size=4), "none")
        env.oracle_fields = tuple(f for f in env.env.oracle_fields if f != "drift_changed")
        return env

    pp.testing.check_env(factory)


def test_an_environment_without_boundary_is_asked_for_boundary() -> None:
    def factory() -> Any:
        env: Any = Broken(pp.make("ContinualGrid-v0", size=4), "none")
        env.oracle_fields = tuple(f for f in env.env.oracle_fields if f != "boundary")
        return env

    with pytest.raises(ContractError, match=r"oracle_fields must include .*'boundary'"):
        pp.testing.check_env(factory)


class PadParams(NamedTuple):
    goal: jax.Array


class NanState(NamedTuple):
    position: jax.Array
    padding: jax.Array
    respawn: jax.Array


@jax.tree_util.register_pytree_node_class
class ClassState:
    """A registered pytree class: static data, leaves, and no ``__eq__``."""

    def __init__(self, position: jax.Array, respawn: jax.Array, label: str = "cell") -> None:
        self.position, self.respawn, self.label = position, respawn, label

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], str]:
        return (self.position, self.respawn), self.label

    @classmethod
    def tree_unflatten(cls, label: str, children: tuple[jax.Array, jax.Array]) -> "ClassState":
        return cls(*children, label)


class Pad:
    """One agent on a line; ``state_kind`` picks the state container."""

    default_drift_field = "goal"

    def __init__(self, state_kind: str) -> None:
        self.state_kind = state_kind
        self.num_agents = 1
        self.observation_space = spaces.Box(
            np.zeros((1, 1), np.float32), np.full((1, 1), 9.0, np.float32), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(np.asarray([3], dtype=np.int64))
        self.metadata: pp.Metadata = {
            "env_id": f"pad-{state_kind}",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {"state_kind": state_kind},
        }

    @property
    def default_params(self) -> PadParams:
        return PadParams(jnp.float32(3.0))

    def _state(self, position: jax.Array, respawn: jax.Array) -> Any:
        if self.state_kind == "nan":
            return NanState(position, jnp.full((2,), jnp.nan, jnp.float32), respawn)
        return ClassState(position, respawn)

    def init(self, key: jax.Array, params: PadParams) -> tuple[Any, jax.Array, EnvInfo]:
        del key
        state = self._state(jnp.int32(0), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(
        self, state: Any, action: Any, key: jax.Array, params: PadParams
    ) -> tuple[Any, jax.Array]:
        del key, params
        position = jnp.clip(state.position + jnp.asarray(action, jnp.int32)[0] - 1, 0, 9)
        return self._state(position.astype(jnp.int32), jnp.asarray(False)), jnp.zeros(1)

    def observe(self, state: Any, params: PadParams) -> tuple[jax.Array, EnvInfo]:
        del params
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 3), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return state.position.astype(jnp.float32)[None, None], info

    def privileged(self, state: Any) -> dict[str, jax.Array]:
        del state
        return {}


@pytest.mark.parametrize("state_kind", ["nan", "class"])
def test_nan_padding_and_registered_state_classes_pass(state_kind: str) -> None:
    pp.testing.check_env(lambda: DriftEnv(cast(Any, Pad(state_kind)), env_id=f"pad-{state_kind}"))
