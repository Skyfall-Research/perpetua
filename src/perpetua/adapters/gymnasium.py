"""Gymnasium API view with explicitly synthetic, lifetime-preserving truncation."""

# JAX device annotations are partial.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from perpetua.core import Environment, Info
from perpetua.runtime import seed_key

__all__ = [
    "GymnasiumAdapter",
]


class GymnasiumAdapter[State](gym.Env[np.ndarray[Any, Any], Any]):
    """Single-agent Box/Discrete view; reset never restarts an initialized lifetime.

    A horizon inserts synthetic truncations for legacy tools, not world resets.
    This intentionally does not promise episodic reset/seed-checker semantics.

    The first ``reset()`` without a seed starts the lifetime from seed 0, so two
    adapters built the same way and reset without a seed run the same lifetime;
    pass ``seed=`` for another one. Any non-empty ``options`` raises
    ``ValueError``. ``info["t"]`` is the lifetime :class:`~perpetua.core.Clock`
    (call ``.to_int()``), not an integer.
    """

    def __init__(self, env: Environment[State], *, horizon: int | None = None) -> None:
        if env.num_agents != 1:
            raise ValueError("The Gymnasium view supports one agent; do not collapse MARL rewards.")
        if not isinstance(env.observation_space, gym.spaces.Box) or not isinstance(
            env.action_space, gym.spaces.MultiDiscrete
        ):
            raise ValueError("The Gymnasium view requires Box observations and discrete slots.")
        if horizon is not None and (type(horizon) is not int or horizon < 1):
            raise ValueError("horizon must be a positive integer or None.")
        self.env = env
        self.horizon = horizon
        self.observation_space = gym.spaces.Box(
            env.observation_space.low[0],
            env.observation_space.high[0],
            dtype=cast(np.dtype[Any], env.observation_space.dtype),
        )
        self.action_space = gym.spaces.Discrete(int(env.action_space.nvec[0]))
        self.metadata = {
            "render_modes": [],
            "perpetua": env.metadata.copy(),
            "synthetic_horizon": horizon,
            "lifetime_preserved_on_reset": True,
        }
        self._state: State | None = None
        self._key = jax.random.key(0)
        self._segment_steps = 0

    def _info(self, info: Info) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_mask": np.asarray(info["action_mask"])[0],
            "active_mask": bool(info["active_mask"][0]),
            "respawn": bool(info["respawn"][0]),
            "t": info["t"],
        }
        if "oracle" in info:
            result["oracle"] = {name: np.asarray(value) for name, value in info["oracle"].items()}
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Initialize once; subsequent unseeded calls expose the current world."""
        if options:
            raise ValueError("No reset options are supported by the lifetime-preserving view.")
        if self._state is not None and seed is not None:
            raise ValueError("A seed cannot restart an existing lifetime; construct a new adapter.")
        if self._state is None:
            key = seed_key(0 if seed is None else seed, "reset")
            super().reset(seed=seed)
            self._key, init_key = jax.random.split(key)
            self._state, obs, info = self.env.init(init_key)
        else:
            super().reset(seed=seed)
            obs, info = self.env.observe(self._state)
        self._segment_steps = 0
        return np.asarray(obs)[0], self._info(info)

    def step(self, action: int) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        """Advance one native tick and optionally mark a synthetic truncation."""
        if self._state is None:
            raise RuntimeError("Call reset once before stepping the Gymnasium view.")
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the Gymnasium action space.")
        self._key, step_key = jax.random.split(self._key)
        self._state, obs, reward, info = self.env.step(
            self._state, jnp.asarray([action], dtype=jnp.int32), step_key
        )
        self._segment_steps += 1
        truncated = self.horizon is not None and self._segment_steps >= self.horizon
        return np.asarray(obs)[0], float(reward[0]), False, truncated, self._info(info)

    def close(self) -> None:
        """Release a host backend when one is wrapped by this view."""
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
