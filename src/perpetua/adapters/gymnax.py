"""gymnax CartPole and Pendulum as parameter environments.

These two keep hand-written wrappers rather than using the generic adapter
because they deliberately change the backend's semantics: the time limit is
discarded and CartPole's failure tick scores zero reward and respawns in place.
Pendulum's wrapper matches the generic adapter's trajectories; it exposes
gravity as a shape-(1,) field and validates the action. Everything drift-related (the process,
its RNG stream, declared switch events, the clock, Oracle fields and metadata)
belongs to ``DriftEnv``.

The generic path for the other gymnax environments is
:mod:`perpetua.adapters.gymnax_params`.
"""

# gymnax ships no type stubs, so its lazily-resolved generics stay loose.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from dataclasses import asdict
from typing import Any, NamedTuple, cast

import gymnax  # pyright: ignore[reportMissingTypeStubs]
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from perpetua.core import EnvInfo, Metadata

__all__ = [
    "CartPoleParams",
    "CartPoleState",
    "GymnaxCartPole",
    "PendulumParams",
    "PendulumState",
    "GymnaxPendulum",
]


class CartPoleParams(NamedTuple):
    """The one drifting parameter: gravity, as a float32 (1,)."""

    gravity: jax.Array


class CartPoleState(NamedTuple):
    """Serializable physics; backend episode age is not lifetime time."""

    physics: Any
    respawn: jax.Array


class GymnaxCartPole:
    """CartPole with gravity drift, failure respawns and no backend time limit."""

    num_agents = 1
    default_drift_field = "gravity"

    def __init__(self) -> None:
        self.backend, self.default_parameters = cast(Any, gymnax.make("CartPole-v1"))
        box = self.backend.observation_space(self.default_parameters)
        self.observation_space = spaces.Box(
            np.asarray(box.low, dtype=np.float32)[None, :],
            np.asarray(box.high, dtype=np.float32)[None, :],
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.asarray([2], dtype=np.int64))
        self.metadata: Metadata = {
            "env_id": "gymnax/CartPole-v1",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {
                "backend": "gymnax-0.0.9",
                "parameters": asdict(self.default_parameters),
                "drift_parameter": "gravity",
                "respawn_policy": "physical-failure",
                "backend_time_limit": "discarded",
                "failure_reward": 0.0,
            },
        }

    @property
    def default_params(self) -> CartPoleParams:
        """The backend's own gravity."""
        return CartPoleParams(jnp.asarray([self.default_parameters.gravity], dtype=jnp.float32))

    def derive(self, params: CartPoleParams) -> Any:
        """Fold the drifted gravity into the backend's parameter struct."""
        return self.default_parameters.replace(gravity=params.gravity[0])

    def init(
        self, key: jax.Array, params: CartPoleParams
    ) -> tuple[CartPoleState, jax.Array, EnvInfo]:
        """Reset once from the first of three keys split from ``key``."""
        reset_key, _, _ = jax.random.split(key, 3)
        _, physics = self.backend.reset_env(reset_key, params)
        state = CartPoleState(physics.replace(time=jnp.int32(0)), jnp.asarray([False]))
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: CartPoleState, params: Any) -> tuple[jax.Array, EnvInfo]:
        """Return flat physics observations and the common public contract."""
        del params
        # float32 whatever ``jax_enable_x64`` is, as the space declares.
        obs = jnp.asarray(self.backend.get_obs(state.physics), dtype=jnp.float32)[None, :]
        return obs, {
            "action_mask": jnp.ones((1, 2), dtype=jnp.bool_),
            "active_mask": jnp.asarray([True]),
            "respawn": state.respawn,
        }

    def step(
        self, state: CartPoleState, action: Any, key: jax.Array, params: Any
    ) -> tuple[CartPoleState, jax.Array]:
        """Step current gravity and respawn physical failures inside the tick."""
        action = jnp.asarray(action)
        if action.shape != (1,) or action.dtype.kind not in {"i", "u"}:
            raise ValueError("CartPole actions must be integer arrays of shape (1,).")
        step_key, reset_key = jax.random.split(key)
        resolved = params
        # CartPole uses this age only for its legacy time limit, not its physics.
        physics = state.physics.replace(time=jnp.int32(0))
        _, physics, reward, _, _ = self.backend.step_env(
            step_key, physics, jnp.clip(action[0], 0, 1), resolved
        )
        failed = (jnp.abs(physics.x) > resolved.x_threshold) | (
            jnp.abs(physics.theta) > resolved.theta_threshold_radians
        )
        # Instant respawn would otherwise make CartPole's +1 reward constant.
        reward = jnp.where(failed, 0.0, reward)

        def respawn(_: None) -> Any:
            _, fresh = self.backend.reset_env(reset_key, resolved)
            return fresh.replace(time=jnp.int32(0))

        def unchanged(_: None) -> Any:
            return physics

        physics = cast(Any, jax.lax.cond(failed, respawn, unchanged, None))
        return (
            CartPoleState(physics, failed[None]),
            jnp.asarray(reward, dtype=jnp.float32)[None],
        )

    def privileged(self, state: CartPoleState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}


class PendulumParams(NamedTuple):
    """The one drifting parameter: gravity ``g``, as a float32 (1,)."""

    g: jax.Array


class PendulumState(NamedTuple):
    """Serializable physics; backend episode age is not lifetime time."""

    physics: Any
    respawn: jax.Array


class GymnaxPendulum:
    """Continuing Pendulum with gravity drift, continuous actions and no time limit.

    Pendulum has no failure states, so respawns stay false. Gravity ``g`` is the
    drift parameter and torque is clipped to the action space before stepping.
    """

    num_agents = 1
    default_drift_field = "g"

    def __init__(self) -> None:
        self.backend, self.default_parameters = cast(Any, gymnax.make("Pendulum-v1"))
        box = self.backend.observation_space(self.default_parameters)
        self.observation_space = spaces.Box(
            np.asarray(box.low, dtype=np.float32)[None, :],
            np.asarray(box.high, dtype=np.float32)[None, :],
            dtype=np.float32,
        )
        action_box = self.backend.action_space(self.default_parameters)
        action_low = np.asarray(action_box.low, dtype=np.float32)
        action_high = np.asarray(action_box.high, dtype=np.float32)
        self.action_space = spaces.Box(
            action_low.reshape(1, -1), action_high.reshape(1, -1), dtype=np.float32
        )
        self.metadata: Metadata = {
            "env_id": "gymnax/Pendulum-v1",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {
                "backend": "gymnax-0.0.9",
                "parameters": asdict(self.default_parameters),
                "drift_parameter": "g",
                "respawn_policy": "none-continuing",
                "backend_time_limit": "discarded",
            },
        }

    @property
    def default_params(self) -> PendulumParams:
        """The backend's own gravity."""
        return PendulumParams(jnp.asarray([self.default_parameters.g], dtype=jnp.float32))

    def derive(self, params: PendulumParams) -> Any:
        """Fold the drifted gravity into the backend's parameter struct."""
        return self.default_parameters.replace(g=params.g[0])

    def init(self, key: jax.Array, params: Any) -> tuple[PendulumState, jax.Array, EnvInfo]:
        """Reset once from the first of three keys split from ``key``."""
        reset_key, _, _ = jax.random.split(key, 3)
        _, physics = self.backend.reset_env(reset_key, params)
        state = PendulumState(physics.replace(time=jnp.int32(0)), jnp.asarray([False]))
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: PendulumState, params: PendulumParams) -> tuple[jax.Array, EnvInfo]:
        """Return flat physics observations and the common public contract."""
        del params
        obs = jnp.asarray(self.backend.get_obs(state.physics), dtype=jnp.float32)[None, :]
        return obs, {
            "action_mask": jnp.ones((1, 1), dtype=jnp.bool_),
            "active_mask": jnp.asarray([True]),
            "respawn": state.respawn,
        }

    def step(
        self, state: PendulumState, action: Any, key: jax.Array, params: Any
    ) -> tuple[PendulumState, jax.Array]:
        """Apply clipped torque under the current gravity."""
        action = jnp.asarray(action)
        if action.shape != (1, 1) or action.dtype.kind != "f":
            raise ValueError("Pendulum actions must be float arrays of shape (1, 1).")
        resolved = params
        torque = jnp.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        # The torque is carried in the state; keep its dtype as reset made it.
        torque = torque.astype(state.physics.last_u.dtype)
        # Discard the legacy 200-step truncation; backend age is not Perpetua time.
        physics = state.physics.replace(time=jnp.int32(0))
        _, physics, reward, _, _ = self.backend.step_env(key, physics, torque, resolved)
        return (
            PendulumState(physics, jnp.asarray([False])),
            jnp.asarray(reward, dtype=jnp.float32)[None],
        )

    def privileged(self, state: PendulumState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}
