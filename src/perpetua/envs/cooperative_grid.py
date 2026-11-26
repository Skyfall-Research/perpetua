"""A continuing cooperative grid with real joint rewards and movement contention."""

# JAX device/transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int
from perpetua.envs.continual_grid import (
    MOVES,
    ContinualGrid,
    GoalParams,
    GridState,
    masked_actions,
)

__all__ = [
    "CooperativeGrid",
]


def check_slots(env: str, size: int, learners: int, scripted: int, role: str) -> int:
    """Validate the size and slot totals of a grid built on a cooperative world.

    The world holds ``learners + scripted`` slots, at most ``min(32, size**2 // 2)``
    of them, on a grid of side at most 64. Returns the validated size.
    """
    size = as_int(size, f"{env} size must be an integer in [2, 64].", minimum=2, maximum=64)
    total, limit = learners + scripted, min(32, size**2 // 2)
    if total > limit:
        raise ValueError(
            f"{env} num_learners + {role} = {learners} + {scripted} = {total} exceeds the "
            f"limit min(32, size**2 // 2) = {limit} for size={size}; lower the counts or "
            "raise size."
        )
    return size


class CooperativeGrid(ContinualGrid):
    """Cooperate on assigned targets while sharing a collision-constrained world.

    Float32 obs (agents, 4 + 3*(agents-1)) contain own normalized position/target,
    then each other slot's normalized position and activity flag in slot order.
    Inactive observations/peer features are zero. Actions/rewards/masks retain
    the standard grid shapes/dtypes. Drift anchors consecutive row-major targets.

    Occupied-at-start cells block movement; swaps are disallowed. Agents proposing
    the same free cell all stay. All active agents (at least two) must reach their
    assigned targets for +1 each, followed by unique respawns outside old targets.
    Collision/reward/respawn use the current parameters; the wrapper drifts and
    re-observes before returning.
    """

    def __init__(self, *, size: int = 5, num_agents: int = 2) -> None:
        size = as_int(
            size, "Cooperative grid size must be an integer in [2, 64].", minimum=2, maximum=64
        )
        num_agents = as_int(
            num_agents,
            "Cooperative num_agents must be in [2, min(32, size**2 // 2)].",
            minimum=2,
            maximum=min(32, size**2 // 2),
        )
        super().__init__(size=size, num_agents=num_agents)
        self._peers = np.asarray(
            [[j for j in range(num_agents) if j != i] for i in range(num_agents)], dtype=np.int32
        )
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(num_agents, 4 + 3 * (num_agents - 1)), dtype=np.float32
        )
        self.metadata: Metadata = {
            "env_id": "CooperativeGrid-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": num_agents,
                "clock": "decision-tick",
                "targets": "consecutive-row-major-from-drift-anchor",
                "movement": "occupied-at-start-blocks; contested-free-cell-stays",
                "reward": "all-active-agents-on-assigned-targets; at-least-two-active",
                "respawn": "unique-outside-current-targets",
            },
        }

    def _targets(self, params: GoalParams) -> jax.Array:
        goal = self._goal(params)
        indices = (goal[0] * self.size + goal[1] + jnp.arange(self.num_agents)) % self.size**2
        return jnp.stack((indices // self.size, indices % self.size), axis=-1)

    def _spawn(self, key: jax.Array, goal: jax.Array) -> jax.Array:
        # Rotating the complement avoids rejection sampling and excludes every target.
        offset = jax.random.permutation(key, self.size**2 - self.num_agents)[: self.num_agents]
        indices = (goal[0] * self.size + goal[1] + self.num_agents + offset) % self.size**2
        return jnp.stack((indices // self.size, indices % self.size), axis=-1)

    def observe(self, state: GridState, params: GoalParams) -> tuple[jax.Array, EnvInfo]:
        """Expose physical peer/target features, never drift phase or internal RNG."""
        _, info = super().observe(state, params)
        candidates = state.positions[:, None, :] + jnp.asarray(MOVES)[None, :, :]
        others = ~jnp.eye(self.num_agents, dtype=jnp.bool_)
        occupied = jnp.any(
            jnp.all(candidates[:, :, None, :] == state.positions[None, None, :, :], axis=-1)
            & others[:, None, :]
            & state.active[None, None, :],
            axis=-1,
        )
        info["action_mask"] = (info["action_mask"] & ~occupied).at[:, 0].set(True)
        own = jnp.concatenate((state.positions, self._targets(params)), axis=-1).astype(
            jnp.float32
        ) / (self.size - 1)
        peer_active = state.active[self._peers]
        peer_positions = state.positions[self._peers].astype(jnp.float32) / (self.size - 1)
        peer = jnp.concatenate(
            (jnp.where(peer_active[..., None], peer_positions, 0.0), peer_active[..., None]),
            axis=-1,
        )
        obs = jnp.concatenate((own, peer.reshape(self.num_agents, -1)), axis=-1)
        return jnp.where(state.active[:, None], obs, 0.0), info

    def _advance(
        self, state: GridState, selected: jax.Array, key: jax.Array, params: GoalParams
    ) -> GridState:
        """Resolve one simultaneous tick of chosen actions and collection."""
        proposed = state.positions + jnp.asarray(MOVES)[selected]
        contested = (
            jnp.any(
                jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
                & ~jnp.eye(self.num_agents, dtype=jnp.bool_)
                & state.active[None, :],
                axis=-1,
            )
            & state.active
        )
        positions = jnp.where(contested[:, None], state.positions, proposed)
        reached = jnp.all(positions == self._targets(params), axis=-1)
        collected = jnp.all(reached | ~state.active) & (jnp.sum(state.active) >= 2)
        respawn = collected & state.active
        positions = jnp.where(respawn[:, None], self._spawn(key, self._goal(params)), positions)
        return GridState(positions, state.active, respawn)

    def step(
        self, state: GridState, action: ArrayLike, key: jax.Array, params: GoalParams
    ) -> tuple[GridState, jax.Array]:
        """Apply simultaneous actions, resolve contention, collect jointly, then drift."""
        _, info = self.observe(state, params)
        selected = masked_actions(action, info["action_mask"])
        next_state = self._advance(state, selected, key, params)
        return next_state, next_state.respawn.astype(jnp.float32)
