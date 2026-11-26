"""``pp.compose`` rebinds drift processes without rebuilding environments."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, NamedTuple, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv


def drift_type(env: pp.Environment[Any]) -> str:
    """Read the recorded drift family from environment metadata."""
    metadata = cast(dict[str, Any], env.metadata)
    configuration = cast(dict[str, Any], metadata["configuration"])
    drift = cast(dict[str, Any], configuration["drift"])
    return cast(str, drift["type"])


def test_compose_accepts_registered_ids_and_forwards_kwargs() -> None:
    drifting = cast(
        pp.Environment[Any],
        pp.compose("gymnax/CartPole-v1", pp.drift.Cyclic([(9.8,), (4.9,)], period=2)),
    )
    assert drift_type(drifting) == "cyclic"
    sized = cast(
        Any,
        pp.compose("ContinualGrid-v0", pp.drift.RandomWalk(low=0, high=2, scale=0.1), size=4),
    )
    assert sized.size == 4
    assert drift_type(sized) == "random-walk"
    with pytest.raises(ValueError, match=r"Grid size must be an integer in"):
        pp.compose("ContinualGrid-v0", pp.drift.Stationary(), size=0)


def test_compose_rebinds_constructed_environments() -> None:
    plain = pp.make("ContinualGrid-v0", size=3)
    drifting = cast(Any, pp.compose(plain, pp.drift.Cyclic([(2, 2), (0, 0)], period=2)))
    assert drift_type(drifting) == "cyclic"
    assert drift_type(plain) == "stationary"
    state = drifting.init(jax.random.key(0))[0]
    for tick in range(2):
        state = drifting.step(state, np.asarray([0]), jax.random.key(tick))[0]
    privileged = drifting.privileged(state)
    np.testing.assert_allclose(np.asarray(privileged["drift_parameters"]), [0.0, 0.0])
    pendulum = cast(
        Any,
        pp.compose(pp.make("gymnax/Pendulum-v1"), pp.drift.Cyclic([(10.0,), (5.0,)], period=1)),
    )
    state = pendulum.init(jax.random.key(0))[0]
    state = pendulum.step(state, np.asarray([[1.0]], dtype=np.float32), jax.random.key(1))[0]
    np.testing.assert_allclose(np.asarray(pendulum.privileged(state)["drift_parameters"]), [5.0])


def test_compose_preserves_companion_drifts() -> None:
    partner = pp.make("PartnerGrid-v0", size=3)
    rebound = cast(Any, pp.compose(partner, pp.drift.Cyclic([(2, 2), (0, 0)], period=4)))
    # Rebinding the goal keeps the partner drift (stationary here).
    assert "partner" not in rebound.drifts
    assert rebound.metadata["configuration"]["partner_drift"] == {"type": "stationary"}
    assert drift_type(rebound) == "cyclic"
    assert rebound.num_partners == partner.num_partners
    opponent = pp.make("OpponentGrid-v0", size=3)
    rebound_opponent = cast(
        Any, pp.compose(opponent, pp.drift.RandomWalk(low=0, high=2, scale=0.1))
    )
    assert rebound_opponent.num_opponents == opponent.num_opponents
    assert drift_type(rebound_opponent) == "random-walk"


def test_compose_errors_are_explicit() -> None:
    host = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=5))
    try:
        with pytest.raises(TypeError, match="rebinding drift"):
            pp.compose(host, pp.drift.Stationary())
        with pytest.raises(ValueError, match="registered environment id"):
            pp.compose(pp.make("ContinualGrid-v0", size=3), pp.drift.Stationary(), size=4)
    finally:
        host.close()


class TwoParams(NamedTuple):
    alpha: jax.Array
    goal: jax.Array


class TwoState(NamedTuple):
    position: jax.Array
    respawn: jax.Array


class Two:
    """Two scalar parameters of one shape and no ``default_drift_field``."""

    num_agents = 1
    observation_space = spaces.Box(
        np.zeros((1, 1), np.float32), np.full((1, 1), 9.0, np.float32), dtype=np.float32
    )
    action_space = spaces.MultiDiscrete(np.asarray([2], dtype=np.int64))
    metadata: pp.Metadata = {"env_id": "two-v0", "oracle_exposure": ()}

    @property
    def default_params(self) -> TwoParams:
        return TwoParams(jnp.float32(1.0), jnp.float32(2.0))

    def init(self, key: jax.Array, params: TwoParams) -> tuple[TwoState, jax.Array, EnvInfo]:
        del key
        state = TwoState(jnp.int32(0), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(
        self, state: TwoState, action: Any, key: jax.Array, params: TwoParams
    ) -> tuple[TwoState, jax.Array]:
        del action, key, params
        return state, jnp.zeros(1)

    def observe(self, state: TwoState, params: TwoParams) -> tuple[jax.Array, EnvInfo]:
        del params
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 2), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return state.position.astype(jnp.float32)[None, None], info

    def privileged(self, state: TwoState) -> dict[str, jax.Array]:
        del state
        return {}


def test_a_bare_process_needs_the_same_field_rule_in_compose_as_in_the_constructor() -> None:
    walk = pp.drift.Cyclic([5.0, 7.0], period=2)
    with pytest.raises(ValueError, match=r"Two needs an explicit drift mapping"):
        DriftEnv(cast(Any, Two()), walk)
    plain = DriftEnv(cast(Any, Two()))
    with pytest.raises(ValueError, match=r"Two needs an explicit drift mapping.*alpha.*goal"):
        pp.compose(plain, walk)
    with pytest.raises(ValueError, match=r"Two needs an explicit drift mapping"):
        plain.with_drift(walk)
    mapped = cast(Any, pp.compose(plain, {"goal": walk}))
    assert list(mapped.drifts) == ["goal"]


def test_compose_on_a_wrapped_environment_says_to_compose_before_wrapping() -> None:
    inner = pp.make("ContinualGrid-v0", size=3)
    drift = pp.drift.Stationary()
    oracle = pp.Oracle(inner, expose=["boundary"])
    with pytest.raises(TypeError, match=r"Oracle wraps an environment.*Compose the drift first"):
        pp.compose(oracle, drift)
    with pytest.raises(TypeError, match=r"Oracle wraps an environment"):
        cast(Any, oracle).with_drift(drift)
    onehot = pp.OneHotObservation(pp.Oracle(inner, expose=["task_id"]), field="task_id", size=2)
    with pytest.raises(TypeError, match=r"OneHotObservation wraps an environment"):
        pp.compose(onehot, drift)
    composed = pp.Oracle(pp.compose(inner, drift), expose=["boundary"])
    assert composed.metadata["oracle_exposure"] == ("boundary",)
