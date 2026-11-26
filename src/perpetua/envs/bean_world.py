"""A never-ending foraging world in the style of Jelly Bean World.

Inspired by Jelly Bean World (Platanios, Saparov and Mitchell, ICLR 2020); it
is a new JAX implementation, not a port. A toroidal grid holds three item
types (jelly beans, bananas, onions) laid out in seeded, banded "biomes", so
what an agent finds changes as it explores. Collected items regrow after a
per-type delay. There are no episodes and no respawns: one lifetime walks the
world forever.

Two parameters drift through :class:`~perpetua.params.DriftEnv`:
``item_rewards`` (what each item is worth) and ``regrow_ticks`` (how long a
collected item takes to return). Both are hidden; only an Oracle exposes them.
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int
from perpetua.envs.continual_grid import MOVES, masked_actions

__all__ = [
    "ITEMS",
    "BeanParams",
    "BeanState",
    "BeanWorld",
]

ITEMS = ("jelly_bean", "banana", "onion")


class BeanParams(NamedTuple):
    """Reward per item type (items,) and regrowth delay in ticks (items,), float32."""

    item_rewards: jax.Array
    regrow_ticks: jax.Array


class BeanState(NamedTuple):
    """The item layout, when each cell was last emptied, positions and the clock.

    ``items`` is int8 (size, size): 0 is empty ground, 1..3 an item type.
    ``collected`` is the uint32 world tick at which each cell was last emptied
    and ``emptied`` whether it ever was. A cell that was never emptied always
    holds its item; an emptied one holds it again once
    ``tick - collected >= regrow`` in uint32 arithmetic. ``tick`` wraps after
    2**32 ticks, so a cell emptied exactly a multiple of 2**32 ticks earlier
    would briefly read as freshly emptied.
    """

    items: jax.Array
    collected: jax.Array
    positions: jax.Array
    tick: jax.Array
    emptied: jax.Array


class BeanWorld:
    """Walk a toroidal item world and collect what you step on.

    Observations are float32 (agents, view*view*3 + 3): an egocentric
    ``view``x``view`` window with one channel per item type (1 where an item is
    present now), followed by a per-type "scent", the sum over a
    ``(2*scent_radius+1)``-square of present items weighted by
    ``1/(1+manhattan distance)``. Actions are integer (agents,):
    stay/up/down/left/right = 0/1/2/3/4, all always valid on the torus.

    Stepping onto a present item collects it: the agent receives that type's
    reward and the cell regrows after ``regrow_ticks`` for that type. Agents
    that step onto the same item on the same tick each receive it. Agents never
    die or respawn.
    """

    default_drift_field = "item_rewards"
    drift_fields = ("item_rewards", "regrow_ticks")

    def __init__(
        self,
        *,
        size: int = 128,
        num_agents: int = 1,
        view: int = 7,
        scent_radius: int = 8,
        densities: tuple[float, float, float] = (0.06, 0.04, 0.04),
        biome_wavelength: float = 32.0,
    ) -> None:
        size = as_int(size, "World size must be an integer in [3, 4096].", minimum=3, maximum=4096)
        num_agents = as_int(num_agents, "num_agents must be a positive integer.", minimum=1)
        view = as_int(view, "view must be a positive odd integer.", minimum=1)
        if view % 2 == 0:
            raise ValueError(f"view must be a positive odd integer. Got int {view}.")
        scent_radius = as_int(
            scent_radius, "scent_radius must be a non-negative integer.", minimum=0
        )
        if len(densities) != len(ITEMS) or not all(math.isfinite(d) and d >= 0 for d in densities):
            raise ValueError("densities needs one finite, non-negative value per item type.")
        if sum(densities) > 0.5:
            raise ValueError("Item densities must sum to at most 0.5.")
        if not math.isfinite(biome_wavelength) or biome_wavelength <= 0:
            raise ValueError("biome_wavelength must be positive and finite.")
        self.size = size
        self.num_agents = num_agents
        self.view = view
        self.scent_radius = scent_radius
        self.densities = tuple(float(d) for d in densities)
        self.biome_wavelength = float(biome_wavelength)
        features = view * view * len(ITEMS) + len(ITEMS)
        self.observation_space = spaces.Box(
            0.0, np.inf, shape=(num_agents, features), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(np.full(num_agents, 5, dtype=np.int64))
        self._view_offsets = jnp.asarray(
            [
                (r, c)
                for r in range(-(view // 2), view // 2 + 1)
                for c in range(-(view // 2), view // 2 + 1)
            ],
            dtype=jnp.int32,
        )
        radius = scent_radius
        offsets = [(r, c) for r in range(-radius, radius + 1) for c in range(-radius, radius + 1)]
        self._scent_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self._scent_weights = jnp.asarray(
            [1.0 / (1.0 + abs(r) + abs(c)) for r, c in offsets], dtype=jnp.float32
        )
        self.metadata: Metadata = {
            "env_id": "BeanWorld-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": num_agents,
                "view": view,
                "scent_radius": scent_radius,
                "items": list(ITEMS),
                "densities": list(self.densities),
                "biome_wavelength": self.biome_wavelength,
                "topology": "torus",
                "clock": "decision-tick",
                "inspired_by": "Jelly Bean World (Platanios et al., ICLR 2020); not a port",
            },
        }

    @property
    def default_params(self) -> BeanParams:
        """Jelly beans +1, bananas +0.1, onions -1; regrowth after 200/400/100 ticks."""
        return BeanParams(
            jnp.asarray([1.0, 0.1, -1.0], dtype=jnp.float32),
            jnp.asarray([200.0, 400.0, 100.0], dtype=jnp.float32),
        )

    def derive(self, params: BeanParams) -> BeanParams:
        """Round regrowth delays to whole ticks of at least one."""
        return BeanParams(params.item_rewards, jnp.maximum(jnp.round(params.regrow_ticks), 1.0))

    def _layout(self, key: jax.Array) -> jax.Array:
        """Seeded banded biomes: each item's density varies along its own direction."""
        angle_key, phase_key, draw_key = jax.random.split(key, 3)
        angles = jax.random.uniform(angle_key, (len(ITEMS),), maxval=2 * jnp.pi)
        phases = jax.random.uniform(phase_key, (len(ITEMS),), maxval=2 * jnp.pi)
        rows, columns = jnp.meshgrid(jnp.arange(self.size), jnp.arange(self.size), indexing="ij")
        angles, phases = angles[None, None, :], phases[None, None, :]
        along = (rows[..., None] * jnp.cos(angles) + columns[..., None] * jnp.sin(angles)) * (
            2 * jnp.pi / self.biome_wavelength
        )
        intensity = jnp.asarray(self.densities)[None, None, :] * (1.0 + jnp.sin(along + phases))
        bounds = jnp.cumsum(intensity, axis=-1)
        draw = jax.random.uniform(draw_key, (self.size, self.size, 1))
        chosen = jnp.sum(draw >= bounds, axis=-1)  # 0..3; 3 means none chosen
        return jnp.where(chosen < len(ITEMS), chosen + 1, 0).astype(jnp.int8)

    def _present(self, state: BeanState, params: BeanParams, cells: jax.Array) -> jax.Array:
        """Bool (..., items): which item types lie on ``cells`` (..., 2) right now.

        Only the queried cells are read, so a tick costs O(agents * window),
        not O(size**2).
        """
        rows, columns = cells[..., 0], cells[..., 1]
        kind = state.items[rows, columns].astype(jnp.int32)
        delay = jnp.concatenate((jnp.zeros(1), params.regrow_ticks)).astype(jnp.uint32)
        waited = (state.tick - state.collected[rows, columns]) >= delay[kind]
        regrown = waited | ~state.emptied[rows, columns]
        item_ids = jnp.arange(1, len(ITEMS) + 1, dtype=jnp.int32).reshape((1,) * kind.ndim + (-1,))
        types = kind[..., None] == item_ids
        return types & regrown[..., None]

    def init(self, key: jax.Array, params: BeanParams) -> tuple[BeanState, jax.Array, EnvInfo]:
        """Lay out the world and place the agents once per lifetime."""
        layout_key, spawn_key = jax.random.split(key)
        positions = jax.random.randint(spawn_key, (self.num_agents, 2), 0, self.size)
        state = BeanState(
            self._layout(layout_key),
            jnp.zeros((self.size, self.size), dtype=jnp.uint32),
            positions.astype(jnp.int32),
            jnp.uint32(0),
            # Every item starts on the ground. A "collected long ago" tick
            # would read as freshly emptied once the clock reached it.
            jnp.zeros((self.size, self.size), dtype=jnp.bool_),
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: BeanState, params: BeanParams) -> tuple[jax.Array, EnvInfo]:
        """Egocentric item view plus per-type scent; every move is always valid."""

        def around(offsets: jax.Array) -> jax.Array:
            cells = (state.positions[:, None, :] + offsets[None]) % self.size
            return self._present(state, params, cells).astype(jnp.float32)

        view = around(self._view_offsets).reshape(self.num_agents, -1)
        scent = jnp.einsum("aoi,o->ai", around(self._scent_offsets), self._scent_weights)
        info: EnvInfo = {
            "action_mask": jnp.ones((self.num_agents, 5), dtype=jnp.bool_),
            "active_mask": jnp.ones(self.num_agents, dtype=jnp.bool_),
            "respawn": jnp.zeros(self.num_agents, dtype=jnp.bool_),
        }
        return jnp.concatenate((view, scent), axis=-1), info

    def step(
        self, state: BeanState, action: ArrayLike, key: jax.Array, params: BeanParams
    ) -> tuple[BeanState, jax.Array]:
        """Move on the torus, collect any present item, and advance the world clock."""
        del key
        selected = masked_actions(action, jnp.ones((self.num_agents, 5), dtype=jnp.bool_))
        positions = (state.positions + jnp.asarray(MOVES)[selected]) % self.size
        here = self._present(state, params, positions)  # (agents, items)
        reward = jnp.sum(here * params.item_rewards[None], axis=-1)
        took = jnp.any(here, axis=-1)
        rows = jnp.where(took, positions[:, 0], self.size)  # out of range: dropped
        collected = state.collected.at[rows, positions[:, 1]].set(state.tick + 1, mode="drop")
        emptied = state.emptied.at[rows, positions[:, 1]].set(True, mode="drop")
        next_state = BeanState(state.items, collected, positions, state.tick + 1, emptied)
        return next_state, reward.astype(jnp.float32)

    def privileged(self, state: BeanState) -> dict[str, jax.Array]:
        """No world-side privileged fields; drift provides rewards and regrowth."""
        del state
        return {}
