"""MEAL Overcooked as a native, continual parameter environment.

MEAL is MIT-licensed and JAX-based. Its own task step auto-resets completed
worlds, so a terminal event is reported as ``respawn`` without ending the
lifetime. Observations are carried in state: MEAL's per-agent observations,
cast to float32, zero-padded to a common shape and stacked by agent.
"""

# MEAL ships no type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false

from importlib.metadata import version
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata

__all__ = [
    "MealParams",
    "MealState",
    "MealOvercooked",
]


class MealParams(NamedTuple):
    """Reward scale; task/layout changes belong to the task-switch combinator."""

    reward_multiplier: jax.Array


class MealState(NamedTuple):
    """MEAL world state, latest public observations and this tick's respawn."""

    world: Any
    obs: jax.Array
    respawn: jax.Array


class MealOvercooked:
    """One MEAL Overcooked layout with reward drift and no episode boundary."""

    default_drift_field = "reward_multiplier"

    def __init__(
        self,
        *,
        difficulty: str = "medium",
        num_agents: int = 2,
        seed: int = 0,
        backend: Any = None,
    ) -> None:
        if backend is None:
            import meal

            # make_env's procedural layout is unseeded; the sequence API makes
            # the layout itself reproducible, not just the reset state.
            backend = meal.make_sequence(
                sequence_length=1, num_agents=num_agents, difficulty=difficulty, seed=seed
            )[0]
        self.backend = backend
        self.agents = tuple(backend.agents)
        self.num_agents = len(self.agents)
        obs, _ = backend.reset(jax.random.key(0))
        shapes = [np.asarray(obs[name]).shape for name in self.agents]
        if any(len(shape) != len(shapes[0]) for shape in shapes):
            raise ValueError("MEAL observations must have matching ranks across agents.")
        self.obs_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(len(shapes[0])))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.num_agents, *self.obs_shape),
            dtype=np.float32,
        )
        self.action_counts = tuple(int(backend.action_space(name).n) for name in self.agents)
        self.action_count = max(self.action_counts)
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_agents, self.action_count, dtype=np.int64)
        )
        self.metadata: Metadata = {
            "env_id": "meal/overcooked-v0",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {
                "backend": f"meal-bench-{version('meal-bench')}",
                "difficulty": difficulty,
                "seed": seed,
                "layout": str(getattr(backend, "name", difficulty)),
                "num_agents": self.num_agents,
                "observation_shape": list(self.obs_shape),
                "action_counts": list(self.action_counts),
                "drift_parameter": "reward_multiplier",
                "respawn_policy": "meal-auto-reset",
            },
        }

    @property
    def default_params(self) -> MealParams:
        """Rewards start unscaled."""
        return MealParams(jnp.asarray([1.0], dtype=jnp.float32))

    def derive(self, params: MealParams) -> MealParams:
        """The multiplier is used as drifted."""
        return params

    def _world(self, world: Any) -> Any:
        """Canonicalize reset-time Python scalars to the step-time JAX pytree."""

        def array(leaf: Any) -> Any:
            if isinstance(leaf, (int, float, bool, np.generic, np.ndarray)):
                return jnp.asarray(leaf)
            return leaf

        return jax.tree.map(array, world)

    def _obs(self, obs: dict[str, Any]) -> jax.Array:
        """Stack and zero-pad per-agent observations to a common shape."""
        values = []
        for name in self.agents:
            value = jnp.asarray(obs[name], dtype=jnp.float32)
            widths = [(0, self.obs_shape[dim] - value.shape[dim]) for dim in range(value.ndim)]
            values.append(jnp.pad(value, widths))
        return jnp.stack(values)

    def init(self, key: jax.Array, params: MealParams) -> tuple[MealState, jax.Array, EnvInfo]:
        """Reset the selected layout once."""
        del params
        # Child 0 of a three-way split, kept so the init stream is the one earlier seeds drew.
        reset_key, _, _ = jax.random.split(key, 3)
        obs, world = self.backend.reset(reset_key)
        state = MealState(
            self._world(world), self._obs(obs), jnp.zeros(self.num_agents, dtype=jnp.bool_)
        )
        observed, info = self.observe(state, self.default_params)
        return state, observed, info

    def observe(self, state: MealState, params: MealParams) -> tuple[jax.Array, EnvInfo]:
        """Return the latest MEAL observation and common public contract."""
        del params
        mask = jnp.arange(self.action_count)[None, :] < jnp.asarray(self.action_counts)[:, None]
        return state.obs, {
            "action_mask": mask,
            "active_mask": jnp.ones(self.num_agents, dtype=jnp.bool_),
            "respawn": state.respawn,
        }

    def step(
        self, state: MealState, action: ArrayLike, key: jax.Array, params: MealParams
    ) -> tuple[MealState, jax.Array]:
        """Step MEAL once; its auto-reset is a respawn, not a lifetime reset."""
        action = jnp.asarray(action)
        if action.shape != (self.num_agents,) or action.dtype.kind not in {"i", "u"}:
            raise ValueError(f"Actions must be integers of shape ({self.num_agents},).")
        selected = jnp.clip(action, 0, jnp.asarray(self.action_counts) - 1).astype(jnp.int32)
        actions = {name: selected[index] for index, name in enumerate(self.agents)}
        obs, world, rewards, dones, _ = self.backend.step(key, state.world, actions)
        reward = (
            jnp.stack([jnp.asarray(rewards[name], dtype=jnp.float32) for name in self.agents])
            * params.reward_multiplier[0]
        )
        done = jnp.asarray(dones["__all__"], dtype=jnp.bool_)
        return MealState(
            self._world(world), self._obs(obs), jnp.repeat(done, self.num_agents)
        ), reward

    def privileged(self, state: MealState) -> dict[str, jax.Array]:
        """No world-side privileged fields; the wrapper supplies drift diagnostics."""
        del state
        return {}
