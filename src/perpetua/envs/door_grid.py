"""A continuing grid whose open doors (affordances) drift over time.

The drifting parameter is a float32 ``(4,)`` door vector for up/down/left/right
(1.0 = open, 0.0 = closed), rounded on read; :class:`~perpetua.params.DriftEnv`
drives it and owns the drift bookkeeping. A closed door masks the matching move
action and blocks movement, so the *action-space semantics* evolve while
compiled shapes stay fixed. The public ``space_change`` flag announces a door
change without exposing the hidden parameter; policies see door state only
through ``action_mask``.
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int
from perpetua.envs.continual_grid import MOVES, masked_actions, spawn_off_goal
from perpetua.spaces import SpaceManager

__all__ = [
    "DoorParams",
    "DoorState",
    "DoorGrid",
]

DOOR_ACTIONS = 4


class DoorParams(NamedTuple):
    """Drifting affordances: one float per up/down/left/right door."""

    doors: jax.Array


class DoorState(NamedTuple):
    """Agent positions and this tick's collection events."""

    positions: jax.Array
    active: jax.Array
    respawn: jax.Array


class DoorGrid:
    """Collect a fixed goal while up/down/left/right doors open and close.

    Observations are float32 (agents, 4): normalized row, column, goal row,
    goal column. The goal is fixed at the bottom-right corner and doors are the
    only drift, so the observation shape never encodes the affordance state:
    policies learn it from ``info['action_mask']``. Actions are integer
    (agents,): stay/up/down/left/right = 0/1/2/3/4; a closed door makes the
    matching move invalid (masked actions become stay, so movement is blocked).
    Rewards are float32 (agents,): +1 when an agent reaches the goal and
    respawns elsewhere.

    ``info['space_change']`` is a bool (agents,) event, true for an agent on
    the tick a door change alters its valid actions (its mask under the old and
    new door parameters, at the same position); it is derived from public
    masks only. Door parameters live in
    :attr:`default_params` and stay hidden unless an Oracle exposes
    ``drift_parameters``.
    """

    default_drift_field = "doors"
    mask_params = ("doors",)

    def __init__(self, *, size: int = 5, num_agents: int = 1) -> None:
        size = as_int(size, "Grid size must be an integer in [2, 46340].", minimum=2, maximum=46340)
        num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
        self.size = size
        self.num_agents = num_agents
        self.observation_space = spaces.Box(0.0, 1.0, shape=(num_agents, 4), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.full(num_agents, 5, dtype=np.int64))
        self.spaces = SpaceManager(
            observation_space=self.observation_space, action_space=self.action_space
        )
        self.metadata: Metadata = {
            "env_id": "DoorGrid-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": num_agents,
                "clock": "decision-tick",
                **self.spaces.configuration,
            },
        }

    @property
    def default_params(self) -> DoorParams:
        """Every door starts open."""
        return DoorParams(jnp.ones(DOOR_ACTIONS, dtype=jnp.float32))

    def _goal(self) -> jax.Array:
        return jnp.asarray([self.size - 1, self.size - 1], dtype=jnp.int32)

    def derive(self, params: DoorParams) -> DoorParams:
        """Round raw door values to the effective open/closed affordance."""
        return DoorParams(jnp.rint(jnp.clip(params.doors, 0.0, 1.0)))

    def _doors(self, params: DoorParams) -> jax.Array:
        """Read the effective door vector as bools; zero means closed."""
        return params.doors > 0.5

    def _spawn(self, key: jax.Array, goal: jax.Array) -> jax.Array:
        return spawn_off_goal(key, self.num_agents, self.size, goal)

    def init(self, key: jax.Array, params: DoorParams) -> tuple[DoorState, jax.Array, EnvInfo]:
        """Spawn agents from the lifetime key; the doors come from ``params``."""
        spawn_key, _, _ = jax.random.split(key, 3)
        state = DoorState(
            self._spawn(spawn_key, self._goal()),
            jnp.ones(self.num_agents, dtype=jnp.bool_),
            jnp.zeros(self.num_agents, dtype=jnp.bool_),
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: DoorState, params: DoorParams) -> tuple[jax.Array, EnvInfo]:
        """Describe the current world and validity of the next action."""
        rows, columns = state.positions[:, 0], state.positions[:, 1]
        moves = jnp.stack(
            (rows > 0, rows < self.size - 1, columns > 0, columns < self.size - 1), axis=-1
        )
        doors = jnp.broadcast_to(self._doors(params), (self.num_agents, DOOR_ACTIONS))
        mask = jnp.concatenate(
            (
                jnp.ones((self.num_agents, 1), dtype=jnp.bool_),
                moves & doors & state.active[:, None],
            ),
            axis=-1,
        )
        goal = jnp.broadcast_to(self._goal(), (self.num_agents, 2))
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
        self, state: DoorState, action: ArrayLike, key: jax.Array, params: DoorParams
    ) -> tuple[DoorState, jax.Array]:
        """Act under current doors, then collect or respawn.

        Invalid or masked integer actions become stay. ``key`` supplies world
        randomness; the wrapper steps the drift process and observes once with
        the parameters that follow.
        """
        _, current_info = self.observe(state, params)
        selected = masked_actions(action, current_info["action_mask"])
        positions = state.positions + jnp.asarray(MOVES)[selected]
        goal = self._goal()
        collected = state.active & jnp.all(positions == goal[None], axis=-1)
        positions = jnp.where(collected[:, None], self._spawn(key, goal), positions)
        return DoorState(positions, state.active, collected), collected.astype(jnp.float32)

    def privileged(self, state: DoorState) -> dict[str, jax.Array]:
        """DoorGrid has no state-side privileged fields; drift provides its own."""
        del state
        return {}
