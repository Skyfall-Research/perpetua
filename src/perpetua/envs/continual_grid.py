"""A continuing resource-collection grid with a shared drifting goal."""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int

__all__ = [
    "MOVES",
    "GridParams",
    "GoalParams",
    "GridState",
    "ContinualGrid",
]

MOVES = np.asarray([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)


def masked_actions(action: ArrayLike, mask: jax.Array) -> jax.Array:
    """Map integer (agents,) actions through bool (agents, actions) masks to stay."""
    action = jnp.asarray(action)
    if action.shape != (mask.shape[0],):
        raise ValueError(f"Action shape must be ({mask.shape[0]},).")
    if action.dtype.kind not in {"i", "u"}:
        raise TypeError("Actions must have integer dtype.")
    safe = jnp.clip(action, 0, mask.shape[1] - 1).astype(jnp.int32)
    valid = (action >= 0) & (action < mask.shape[1]) & mask[jnp.arange(mask.shape[0]), safe]
    return jnp.where(valid, safe, 0)


def spawn_off_goal(key: jax.Array, count: int, size: int, goal: jax.Array) -> jax.Array:
    """Draw ``count`` cells uniformly from a ``size`` x ``size`` grid, never the goal."""
    # Sample from one cell fewer and skip over the goal, so there is no rejection loop.
    goal_index = goal[0] * size + goal[1]
    indices = jax.random.randint(key, (count,), 0, size**2 - 1)
    indices = indices + (indices >= goal_index).astype(jnp.int32)
    return jnp.stack((indices // size, indices % size), axis=-1)


class GridParams(NamedTuple):
    """Drifting goal coordinates, raw float32 (2,); ``derive`` rounds them."""

    goal: jax.Array


class GoalParams(Protocol):
    """Anything with a drifting ``goal`` field; wider tuples satisfy it.

    Declared read-only so NamedTuple fields satisfy it.
    """

    @property
    def goal(self) -> jax.Array: ...


class GridState(NamedTuple):
    """Private world state: positions, activity and this tick's respawns."""

    positions: jax.Array
    active: jax.Array
    respawn: jax.Array


class ContinualGrid:
    """Collect the goal for +1 reward and respawn inside the same lifetime.

    Observations are float32 (agents, 4): normalized row, column, goal row,
    goal column. Actions are integer (agents,): stay/up/down/left/right =
    0/1/2/3/4. Rewards are float32 (agents,). Agents share a goal but do not
    collide or compete; this is a parallel-agent contract example, not a MARL
    benchmark. Goal parameters are rounded after clipping to grid bounds by
    :meth:`derive`, so a drift smaller than half a cell never changes the goal.
    """

    default_drift_field = "goal"

    def __init__(self, *, size: int = 5, num_agents: int = 1) -> None:
        size = as_int(size, "Grid size must be an integer in [2, 46340].", minimum=2, maximum=46340)
        num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
        self.size = size
        self.num_agents = num_agents
        self.observation_space = spaces.Box(0.0, 1.0, shape=(num_agents, 4), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.full(num_agents, 5, dtype=np.int64))
        self.metadata: Metadata = {
            "env_id": "ContinualGrid-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": num_agents,
                "clock": "decision-tick",
            },
        }

    @property
    def default_params(self) -> GridParams:
        """The goal starts at the bottom-right cell."""
        return GridParams(jnp.asarray([self.size - 1, self.size - 1], dtype=jnp.float32))

    def derive(self, params: GridParams) -> GridParams:
        """Clip raw goal coordinates to the grid and round them to a cell."""
        return GridParams(jnp.rint(jnp.clip(params.goal, 0, self.size - 1)))

    def _goal(self, params: GoalParams) -> jax.Array:
        return params.goal.astype(jnp.int32)

    def _spawn(self, key: jax.Array, goal: jax.Array) -> jax.Array:
        return spawn_off_goal(key, self.num_agents, self.size, goal)

    def init(self, key: jax.Array, params: GridParams) -> tuple[GridState, jax.Array, EnvInfo]:
        """Spawn agents from the lifetime key, never on the goal."""
        spawn_key, _, _ = jax.random.split(key, 3)
        state = GridState(
            self._spawn(spawn_key, self._goal(params)),
            jnp.ones(self.num_agents, dtype=jnp.bool_),
            jnp.zeros(self.num_agents, dtype=jnp.bool_),
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: GridState, params: GoalParams) -> tuple[jax.Array, EnvInfo]:
        """Describe the current world and validity of the next action."""
        rows, columns = state.positions[:, 0], state.positions[:, 1]
        moves = jnp.stack(
            (rows > 0, rows < self.size - 1, columns > 0, columns < self.size - 1), axis=-1
        )
        mask = jnp.concatenate(
            (jnp.ones((self.num_agents, 1), dtype=jnp.bool_), moves & state.active[:, None]),
            axis=-1,
        )
        goal = jnp.broadcast_to(self._goal(params), (self.num_agents, 2))
        obs = jnp.concatenate((state.positions, goal), axis=-1).astype(jnp.float32) / (
            self.size - 1
        )
        info: EnvInfo = {
            "action_mask": mask,
            "active_mask": state.active,
            "respawn": state.respawn,
        }
        return obs, info

    def step(
        self, state: GridState, action: ArrayLike, key: jax.Array, params: GoalParams
    ) -> tuple[GridState, jax.Array]:
        """Act under current parameters, then collect or respawn.

        Invalid or masked integer actions become stay. Shape and dtype errors
        raise during tracing/eager execution. ``key`` supplies world randomness;
        the wrapper steps the drift process and re-observes with the parameters
        that follow.
        """
        _, current_info = self.observe(state, params)
        selected = masked_actions(action, current_info["action_mask"])
        positions = state.positions + jnp.asarray(MOVES)[selected]
        goal = self._goal(params)
        collected = state.active & jnp.all(positions == goal[None], axis=-1)
        positions = jnp.where(collected[:, None], self._spawn(key, goal), positions)
        return GridState(positions, state.active, collected), collected.astype(jnp.float32)

    def privileged(self, state: GridState) -> dict[str, jax.Array]:
        """ContinualGrid has no state-side privileged fields; drift provides its own."""
        del state
        return {}
