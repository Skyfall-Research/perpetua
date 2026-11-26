"""A continuing cooperative grid shared with scripted partners whose policies drift."""

# This composing environment reuses the cooperative world's protected mechanics.
# JAX device/transformation annotations are partial.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int
from perpetua.envs.continual_grid import MOVES, GridState, masked_actions
from perpetua.envs.cooperative_grid import CooperativeGrid, check_slots

__all__ = [
    "PartnerParams",
    "PartnerState",
    "PartnerGrid",
]


class PartnerParams(NamedTuple):
    """Drifting goal (float32 (2,)) and hidden partner cooperativeness (P,)."""

    goal: jax.Array
    partner: jax.Array


class PartnerState(NamedTuple):
    """Private world state plus the scripted partners' own action RNG stream."""

    world: GridState
    partner_key: jax.Array


class PartnerGrid:
    """Scripted partners with drifting cooperativeness share a cooperative grid.

    The first ``num_learners`` slots are policy-controlled; the last
    ``num_partners`` slots are scripted partners. Each tick a partner moves
    beta-greedily toward its assigned target among currently valid moves, or
    samples uniformly among them. Per-partner beta parameters are hidden drift
    state on an independent RNG stream, exposed only through an Oracle.

    Partners stay world participants: they block cells, contend for free cells,
    must reach their targets for the joint +1 and respawn with everyone. They
    are not learner-controlled: their public action masks allow only the
    placeholder stay action, their observations are zero and ``active_mask``
    is False, so PPO/IPPO never train on partner slots.
    """

    default_drift_field = "goal"

    def __init__(self, *, size: int = 5, num_learners: int = 1, num_partners: int = 1) -> None:
        num_learners = as_int(
            num_learners, "PartnerGrid num_learners must be a positive integer.", minimum=1
        )
        num_partners = as_int(
            num_partners, "PartnerGrid num_partners must be a positive integer.", minimum=1
        )
        size = check_slots("PartnerGrid", size, num_learners, num_partners, "num_partners")
        self.world = CooperativeGrid(size=size, num_agents=num_learners + num_partners)
        self.size = size
        self.num_learners = num_learners
        self.num_partners = num_partners
        self.num_agents = self.world.num_agents
        self.observation_space = self.world.observation_space
        self.action_space = self.world.action_space
        control = np.zeros(self.num_agents, dtype=np.bool_)
        control[:num_learners] = True
        self._control = jnp.asarray(control)
        self._partner_slots = jnp.asarray(np.arange(num_learners, self.num_agents), dtype=jnp.int32)
        self.metadata: Metadata = {
            "env_id": "PartnerGrid-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": self.num_agents,
                "num_learners": num_learners,
                "num_partners": num_partners,
                "slots": "learners-first-partners-last",
                "partner_policy": "beta-greedy-to-target-else-uniform-valid; beta-clipped-to-[0,1]",
                "clock": "decision-tick",
                "targets": "consecutive-row-major-from-drift-anchor",
                "movement": "occupied-at-start-blocks; contested-free-cell-stays",
                "reward": "all-world-active-agents-on-assigned-targets; at-least-two-active",
                "respawn": "unique-outside-current-targets",
            },
        }

    @property
    def default_params(self) -> PartnerParams:
        """Bottom-right goal and fully cooperative partners."""
        return PartnerParams(
            jnp.asarray([self.world.size - 1, self.world.size - 1], dtype=jnp.float32),
            jnp.ones(self.num_partners, dtype=jnp.float32),
        )

    def derive(self, params: PartnerParams) -> PartnerParams:
        """Round the goal to a cell and clip partner cooperativeness to [0, 1]."""
        goal = jnp.rint(jnp.clip(params.goal, 0, self.world.size - 1))
        return PartnerParams(goal, jnp.clip(params.partner, 0.0, 1.0))

    def init(
        self, key: jax.Array, params: PartnerParams
    ) -> tuple[PartnerState, jax.Array, EnvInfo]:
        """Spawn agents from the lifetime key."""
        spawn_key, _, _, _, partner_key = jax.random.split(key, 5)
        state = PartnerState(
            GridState(
                self.world._spawn(spawn_key, self.world._goal(params)),
                jnp.ones(self.num_agents, dtype=jnp.bool_),
                jnp.zeros(self.num_agents, dtype=jnp.bool_),
            ),
            partner_key,
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: PartnerState, params: PartnerParams) -> tuple[jax.Array, EnvInfo]:
        """Show learner slots and partner peers; never partner control or parameters."""
        obs, info = self.world.observe(state.world, params)
        placeholder = jnp.zeros_like(info["action_mask"]).at[:, 0].set(True)
        info["action_mask"] = jnp.where(self._control[:, None], info["action_mask"], placeholder)
        info["active_mask"] = info["active_mask"] & self._control
        return jnp.where(self._control[:, None], obs, 0.0), info

    def _partner_actions(
        self, state: PartnerState, params: PartnerParams, mask: jax.Array, key: jax.Array
    ) -> jax.Array:
        """Sample hidden beta-greedy partner actions from the partners' RNG stream."""
        beta = jnp.clip(params.partner, 0.0, 1.0)
        slots = self._partner_slots
        valid = mask[slots]
        proposal = state.world.positions[slots][:, None, :] + jnp.asarray(MOVES)[None, :, :]
        distance = jnp.sum(
            jnp.abs(proposal - self.world._targets(params)[slots][:, None, :]), axis=-1
        )
        greedy = jnp.argmin(distance + jnp.where(valid, 0.0, jnp.inf), axis=-1)

        def draw(
            key: jax.Array, beta: jax.Array, logits: jax.Array, greedy: jax.Array
        ) -> jax.Array:
            greedy_key, uniform_key = jax.random.split(key)
            take = jax.random.bernoulli(greedy_key, p=beta)
            uniform = jax.random.categorical(uniform_key, logits)
            return jnp.where(take, greedy, uniform).astype(jnp.int32)

        logits = jnp.where(valid, 0.0, -jnp.inf)
        return jax.vmap(draw)(jax.random.split(key, self.num_partners), beta, logits, greedy)

    def step(
        self, state: PartnerState, action: ArrayLike, key: jax.Array, params: PartnerParams
    ) -> tuple[PartnerState, jax.Array]:
        """Apply learner actions to learner slots; partners act from hidden parameters."""
        _, info = self.world.observe(state.world, params)
        placeholder = jnp.zeros_like(info["action_mask"]).at[:, 0].set(True)
        public = jnp.where(self._control[:, None], info["action_mask"], placeholder)
        selected = masked_actions(action, public)
        partner_key, action_key, _ = jax.random.split(state.partner_key, 3)
        partners = self._partner_actions(state, params, info["action_mask"], action_key)
        selected = selected.at[self._partner_slots].set(partners)
        world = self.world._advance(state.world, selected, key, params)
        return PartnerState(world, partner_key), world.respawn.astype(jnp.float32)

    def privileged(self, state: PartnerState) -> dict[str, jax.Array]:
        """PartnerGrid has no state-side privileged fields; drift provides its own."""
        del state
        return {}
