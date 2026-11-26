"""A foraging grid whose agents get slower as they collect (agent-driven drift).

The drifting parameter is a per-agent ``mass``; a heavier agent's moves
succeed less often (probability ``1 / mass``). ``make("ForagerGrid-v0")``
attaches no drift, so mass stays at its default of 1. Pass an
:class:`~perpetua.drift.Accumulate` process to make collecting add mass and
mass decay over time:

.. code-block:: python

    drift = pp.drift.Accumulate(gain=4.0, decay=0.9, low=1.0, high=8.0)
    env = pp.make("ForagerGrid-v0", drift=drift)

Each tick the mass becomes ``clip(decay * mass + gain * signal, low, high)``:
``decay`` shrinks the whole value toward zero, ``low`` is only a floor, and a
collect adds ``gain``. Once the starting value has decayed to it, the mass stays at
or below ``max(low, gain / (1 - decay))``, so with ``gain=1.0, decay=0.5, low=0.0``
it never passes 2 and ``high`` is never reached.

That drift is caused by the agent's own behaviour, which an exogenous drift
process cannot express: it is fed by the environment's privileged ``signal``
through :class:`~perpetua.params.DriftEnv`.

Mass never appears in observations or info. An agent can only infer it from
how often its own moves succeed; the Oracle exposes it on request as
``mass_parameters``.
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import math
import numbers
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int
from perpetua.envs.continual_grid import MOVES, masked_actions, spawn_off_goal

__all__ = [
    "ForagerParams",
    "ForagerState",
    "ForagerGrid",
]


class ForagerParams(NamedTuple):
    """Per-agent mass (agents,), raw float32; ``derive`` clips it into bounds."""

    mass: jax.Array


class ForagerState(NamedTuple):
    """Positions, activity and this tick's respawns.

    ``move_key`` is never read (``step`` draws from its ``key`` argument); it
    keeps the checkpointed state layout.
    """

    positions: jax.Array
    active: jax.Array
    respawn: jax.Array
    move_key: jax.Array


class ForagerGrid:
    """Collect the goal for +1 and respawn; with drift attached, getting heavier and slower.

    Observations are float32 (agents, 4): normalized row, column, goal row,
    goal column, exactly as in the other grids. Mass is **hidden**: it appears
    nowhere in the observation or info, and the drift schedule (gain, decay,
    bounds) lives in the drift configuration, so an agent can only infer its
    mass from how often its moves succeed. The Oracle exposes the mass on
    request as ``mass_parameters``. Actions are integer (agents,):
    stay/up/down/left/right = 0/1/2/3/4; an attempted move succeeds with
    probability ``1 / mass`` drawn from the world's own key, and an inactive
    slot never moves. The goal is fixed at the bottom-right cell.

    Rewards are float32 (agents,): +1 on reaching the goal, with a same-tick
    respawn outside it. Masked or invalid integer actions become stay.
    """

    default_drift_field = "mass"

    def __init__(self, *, size: int = 4, num_agents: int = 1, max_mass: float = 8.0) -> None:
        size = as_int(size, "Grid size must be an integer in [2, 46340].", minimum=2, maximum=46340)
        num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
        given = cast(object, max_mass)
        if isinstance(given, bool) or not isinstance(given, numbers.Real):
            raise ValueError(f"max_mass must be a finite number >= 1. Got {max_mass!r}.")
        max_mass = float(given)
        if not math.isfinite(max_mass) or max_mass < 1.0:
            raise ValueError(f"max_mass must be a finite number >= 1. Got {max_mass!r}.")
        self.size = size
        self.num_agents = num_agents
        self.max_mass = max_mass
        self.observation_space = spaces.Box(0.0, 1.0, shape=(num_agents, 4), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.full(num_agents, 5, dtype=np.int64))
        self.metadata: Metadata = {
            "env_id": "ForagerGrid-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": num_agents,
                "max_mass": max_mass,
                "clock": "decision-tick",
                "dynamics": "move succeeds with probability 1/mass",
                "mass": "hidden; only an Oracle exposes it",
            },
        }

    @property
    def default_params(self) -> ForagerParams:
        """Agents start light: every attempted move succeeds."""
        return ForagerParams(jnp.ones(self.num_agents, dtype=jnp.float32))

    def derive(self, params: ForagerParams) -> ForagerParams:
        """Clip raw mass into [1, max_mass] before it is used."""
        return ForagerParams(jnp.clip(params.mass, 1.0, self.max_mass))

    def _goal(self) -> jax.Array:
        return jnp.asarray([self.size - 1, self.size - 1], dtype=jnp.int32)

    def _spawn(self, key: jax.Array, goal: jax.Array) -> jax.Array:
        return spawn_off_goal(key, self.num_agents, self.size, goal)

    def init(
        self, key: jax.Array, params: ForagerParams
    ) -> tuple[ForagerState, jax.Array, EnvInfo]:
        """Spawn agents from the lifetime key; ``move_key`` is stored but never read."""
        spawn_key, _, move_key = jax.random.split(key, 3)
        state = ForagerState(
            self._spawn(spawn_key, self._goal()),
            jnp.ones(self.num_agents, dtype=jnp.bool_),
            jnp.zeros(self.num_agents, dtype=jnp.bool_),
            move_key,
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: ForagerState, params: ForagerParams) -> tuple[jax.Array, EnvInfo]:
        """Describe the world and the validity of the next action."""
        rows, columns = state.positions[:, 0], state.positions[:, 1]
        moves = jnp.stack(
            (rows > 0, rows < self.size - 1, columns > 0, columns < self.size - 1), axis=-1
        )
        mask = jnp.concatenate(
            (
                jnp.ones((self.num_agents, 1), dtype=jnp.bool_),
                moves & state.active[:, None],
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
        return obs.astype(jnp.float32), info

    def step(
        self, state: ForagerState, action: ArrayLike, key: jax.Array, params: ForagerParams
    ) -> tuple[ForagerState, jax.Array]:
        """Attempt moves with success probability ``1 / mass``, then collect.

        The key splits into a movement draw and a spawn draw; drift randomness
        belongs to the wrapper, not here.
        """
        _, current_info = self.observe(state, params)
        selected = masked_actions(action, current_info["action_mask"])
        move_key, spawn_key = jax.random.split(key)
        chance = jnp.clip(1.0 / jnp.clip(params.mass, 1.0, self.max_mass), 0.0, 1.0)
        succeeds = (jax.random.uniform(move_key, (self.num_agents,)) < chance) & state.active
        positions = state.positions + jnp.asarray(MOVES)[selected] * succeeds[:, None]
        goal = self._goal()
        collected = state.active & jnp.all(positions == goal[None], axis=-1)
        positions = jnp.where(collected[:, None], self._spawn(spawn_key, goal), positions)
        next_state = ForagerState(positions, state.active, collected, move_key)
        return next_state, collected.astype(jnp.float32)

    def signal(self, state: ForagerState, action: ArrayLike, next_state: ForagerState) -> jax.Array:
        """Privileged per-agent signal: whether this agent collected this tick."""
        del state, action
        return next_state.respawn.astype(jnp.float32)

    def privileged(self, state: ForagerState) -> dict[str, jax.Array]:
        """No world-side privileged fields; drift provides the mass."""
        del state
        return {}
