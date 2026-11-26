"""A continuing shared-goal grid where scripted opponents with drifting policies hunt learners."""

# This composing environment reuses the cooperative world's protected mechanics.
# JAX device/transformation annotations are partial.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata, as_int
from perpetua.envs.continual_grid import MOVES, GridState, masked_actions, spawn_off_goal
from perpetua.envs.cooperative_grid import CooperativeGrid, check_slots

__all__ = [
    "OpponentParams",
    "OpponentState",
    "OpponentGrid",
]


class OpponentParams(NamedTuple):
    """Drifting goal (float32 (2,)) and hidden opponent aggression (O,)."""

    goal: jax.Array
    opponent: jax.Array


class OpponentState(NamedTuple):
    """Private world state plus the scripted opponents' own action RNG stream."""

    world: GridState
    opponent_key: jax.Array


class OpponentGrid:
    """Scripted opponents with drifting aggression hunt learners in a zero-sum world.

    The first ``num_learners`` slots are policy-controlled; the last
    ``num_opponents`` slots are scripted opponents. Each tick an opponent moves
    alpha-greedily toward its nearest learner among currently valid moves, or
    samples uniformly among them, where per-opponent ``alpha`` in ``[0, 1]`` is
    hidden drift state clipped from the ``opponent_drift`` family. ``alpha = 1``
    hunts relentlessly; ``alpha = 0`` wanders. The ``drift`` argument keeps its
    shared-goal meaning and defaults to a stationary goal.

    Learners collect the shared drifting goal for +1 and respawn outside it,
    as in the continuing grid. A learner within one step of any opponent
    (including co-location) is caught: collection is interrupted, the learner
    scores -1 and respawns. Rewards are strictly zero-sum: opponent slots share
    the negated learner reward sum each tick.

    Opponents stay world participants: they block occupied cells and contend
    for free cells. When only opponents propose the same free cell, the lowest
    slot among them moves and the others stay; a cell a learner also proposes
    stays empty, as between learners. Opponents start on distinct cells. They
    are not learner-controlled: their public action masks
    allow only the placeholder stay action, their observations are zero and
    their ``active_mask`` is False, so PPO/IPPO never train on them while their
    peer positions stay visible in learner observations.
    """

    default_drift_field = "goal"

    def __init__(self, *, size: int = 5, num_learners: int = 1, num_opponents: int = 1) -> None:
        num_learners = as_int(
            num_learners, "OpponentGrid num_learners must be a positive integer.", minimum=1
        )
        num_opponents = as_int(
            num_opponents, "OpponentGrid num_opponents must be a positive integer.", minimum=1
        )
        size = check_slots("OpponentGrid", size, num_learners, num_opponents, "num_opponents")
        self.world = CooperativeGrid(size=size, num_agents=num_learners + num_opponents)
        self.size = size
        self.num_learners = num_learners
        self.num_opponents = num_opponents
        self.num_agents = self.world.num_agents
        self.observation_space = self.world.observation_space
        self.action_space = self.world.action_space
        control = np.zeros(self.num_agents, dtype=np.bool_)
        control[:num_learners] = True
        self._control = jnp.asarray(control)
        self._opponent_slots = jnp.asarray(
            np.arange(num_learners, self.num_agents), dtype=jnp.int32
        )
        # Between two opponents the lower slot keeps a contested cell, so the
        # higher one does not block it.
        opponent = ~control
        self._yields = jnp.asarray(
            opponent[:, None]
            & opponent[None, :]
            & (np.arange(self.num_agents)[None, :] > np.arange(self.num_agents)[:, None])
        )
        self._peers = np.asarray(
            [[j for j in range(self.num_agents) if j != i] for i in range(self.num_agents)],
            dtype=np.int32,
        )
        self.metadata: Metadata = {
            "env_id": "OpponentGrid-v0",
            "oracle_exposure": (),
            "configuration": {
                "size": size,
                "num_agents": self.num_agents,
                "num_learners": num_learners,
                "num_opponents": num_opponents,
                "slots": "learners-first-opponents-last",
                "opponent_policy": (
                    "alpha-greedy-to-nearest-learner-else-uniform-valid; alpha-clipped-to-[0,1]"
                ),
                "clock": "decision-tick",
                "goal": "shared-drifting-goal",
                "movement": "occupied-at-start-blocks; contested-free-cell-stays"
                + ("; opponents-only-contest-lowest-slot-moves" if num_opponents > 1 else ""),
                "catch": "within-one-step-of-an-opponent-including-co-location",
                "reward": (
                    "learner-plus-one-collect-minus-one-caught;"
                    " opponents-share-negated-learner-sum-zero-sum"
                ),
                "respawn": "collected-or-caught-learners-outside-goal",
            },
        }

    def _spawn(self, key: jax.Array, goal: jax.Array) -> jax.Array:
        """Draw learner spawns outside the shared goal, without loops.

        Draws are independent, so two learners can share a cell.
        """
        return spawn_off_goal(key, self.num_learners, self.size, goal)

    @property
    def default_params(self) -> OpponentParams:
        """Bottom-right goal and fully aggressive opponents."""
        return OpponentParams(
            jnp.asarray([self.size - 1, self.size - 1], dtype=jnp.float32),
            jnp.ones(self.num_opponents, dtype=jnp.float32),
        )

    def derive(self, params: OpponentParams) -> OpponentParams:
        """Round the goal to a cell and clip opponent aggression to [0, 1]."""
        goal = jnp.rint(jnp.clip(params.goal, 0, self.size - 1))
        return OpponentParams(goal, jnp.clip(params.opponent, 0.0, 1.0))

    def init(
        self, key: jax.Array, params: OpponentParams
    ) -> tuple[OpponentState, jax.Array, EnvInfo]:
        """Spawn agents from the lifetime key.

        Opponents are drawn over every cell without replacement, so they start
        on distinct cells, and one can start on a learner or on the goal.
        """
        (
            spawn_key,
            _,
            _,
            opponent_spawn_key,
            _,
            opponent_key,
        ) = jax.random.split(key, 6)
        goal = self.world._goal(params)
        # A single opponent keeps the draw it always had.
        opponent_indices = (
            jax.random.randint(opponent_spawn_key, (1,), 0, self.size**2)
            if self.num_opponents == 1
            else jax.random.choice(
                opponent_spawn_key, self.size**2, (self.num_opponents,), replace=False
            )
        )
        opponents = jnp.stack(
            (opponent_indices // self.size, opponent_indices % self.size), axis=-1
        )
        state = OpponentState(
            GridState(
                jnp.concatenate((self._spawn(spawn_key, goal), opponents), axis=0),
                jnp.ones(self.num_agents, dtype=jnp.bool_),
                jnp.zeros(self.num_agents, dtype=jnp.bool_),
            ),
            opponent_key,
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: OpponentState, params: OpponentParams) -> tuple[jax.Array, EnvInfo]:
        """Show learner slots, the shared goal and opponent peers; never opponent control."""
        _, info = self.world.observe(state.world, params)
        mask = info["action_mask"]
        goal = jnp.broadcast_to(self.world._goal(params), (self.num_agents, 2))
        own = jnp.concatenate((state.world.positions, goal), axis=-1).astype(jnp.float32) / (
            self.size - 1
        )
        peer_active = state.world.active[self._peers]
        peer_positions = state.world.positions[self._peers].astype(jnp.float32) / (self.size - 1)
        peer = jnp.concatenate(
            (jnp.where(peer_active[..., None], peer_positions, 0.0), peer_active[..., None]),
            axis=-1,
        )
        obs = jnp.concatenate((own, peer.reshape(self.num_agents, -1)), axis=-1)
        placeholder = jnp.zeros_like(mask).at[:, 0].set(True)
        info["action_mask"] = jnp.where(self._control[:, None], mask, placeholder)
        info["active_mask"] = info["active_mask"] & self._control
        keep = self._control & state.world.active
        return jnp.where(keep[:, None], obs, 0.0), info

    def _opponent_actions(
        self, state: OpponentState, params: OpponentParams, mask: jax.Array, key: jax.Array
    ) -> jax.Array:
        """Sample hidden alpha-greedy opponent actions from the opponents' RNG stream."""
        alpha = jnp.clip(params.opponent, 0.0, 1.0)
        slots = self._opponent_slots
        valid = mask[slots]
        learners = state.world.positions[: self.num_learners]
        proposal = state.world.positions[slots][:, None, :] + jnp.asarray(MOVES)[None, :, :]
        separation = jnp.sum(jnp.abs(proposal[:, :, None, :] - learners[None, None, :, :]), axis=-1)
        nearest = jnp.min(separation, axis=-1)
        greedy = jnp.argmin(nearest + jnp.where(valid, 0.0, jnp.inf), axis=-1)

        def draw(
            key: jax.Array, alpha: jax.Array, logits: jax.Array, greedy: jax.Array
        ) -> jax.Array:
            greedy_key, uniform_key = jax.random.split(key)
            take = jax.random.bernoulli(greedy_key, p=alpha)
            uniform = jax.random.categorical(uniform_key, logits)
            return jnp.where(take, greedy, uniform).astype(jnp.int32)

        logits = jnp.where(valid, 0.0, -jnp.inf)
        return jax.vmap(draw)(jax.random.split(key, self.num_opponents), alpha, logits, greedy)

    def step(
        self, state: OpponentState, action: ArrayLike, key: jax.Array, params: OpponentParams
    ) -> tuple[OpponentState, jax.Array]:
        """Resolve hunting and collection; the wrapper advances the drifts."""
        _, info = self.world.observe(state.world, params)
        placeholder = jnp.zeros_like(info["action_mask"]).at[:, 0].set(True)
        public = jnp.where(self._control[:, None], info["action_mask"], placeholder)
        selected = masked_actions(action, public)
        opponent_key, action_key, _ = jax.random.split(state.opponent_key, 3)
        opponents = self._opponent_actions(state, params, info["action_mask"], action_key)
        selected = selected.at[self._opponent_slots].set(opponents)
        proposed = state.world.positions + jnp.asarray(MOVES)[selected]
        contested = (
            jnp.any(
                jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
                & ~jnp.eye(self.num_agents, dtype=jnp.bool_)
                & ~self._yields
                & state.world.active[None, :],
                axis=-1,
            )
            & state.world.active
        )
        positions = jnp.where(contested[:, None], state.world.positions, proposed)
        learners = positions[: self.num_learners]
        hunters = positions[self._opponent_slots]
        separation = jnp.sum(jnp.abs(learners[:, None, :] - hunters[None, :, :]), axis=-1)
        active_learners = state.world.active[: self.num_learners]
        caught = jnp.any(separation <= 1, axis=-1) & active_learners
        goal = self.world._goal(params)
        collected = (~caught) & jnp.all(learners == goal[None], axis=-1) & active_learners
        respawn = caught | collected
        spawned = self._spawn(key, goal)
        positions = positions.at[: self.num_learners].set(
            jnp.where(respawn[:, None], spawned, positions[: self.num_learners])
        )
        learner_reward = collected.astype(jnp.float32) - caught.astype(jnp.float32)
        reward = jnp.zeros(self.num_agents, dtype=jnp.float32)
        reward = reward.at[: self.num_learners].set(learner_reward)
        reward = reward.at[self._opponent_slots].set(-jnp.sum(learner_reward) / self.num_opponents)
        next_state = OpponentState(
            GridState(
                positions,
                state.world.active,
                jnp.concatenate((respawn, jnp.zeros(self.num_opponents, dtype=jnp.bool_))),
            ),
            opponent_key,
        )
        return next_state, reward

    def privileged(self, state: OpponentState) -> dict[str, jax.Array]:
        """OpponentGrid has no state-side privileged fields; drift provides its own."""
        del state
        return {}
