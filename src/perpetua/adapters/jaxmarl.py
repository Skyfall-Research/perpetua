"""Functional MPE simple_spread integration using JaxMARL's jit-able dynamics.

The adapter is a parameter environment: the reward multiplier, the scripted
slots' cooperativeness and the MPE dynamics the audit marks driftable
(``local_ratio``, ``dt``, ``mass``, ``damping``, ``contact_margin``; see
``scripts/audit_jaxmarl.py``) are named parameter fields, and ``DriftEnv`` owns
drift, its RNG stream, switch events, the clock and Oracle fields. JaxMARL keeps
dynamics on the environment object, so they are injected by binding traced
values onto a shallow copy of the backend inside the jitted step.
"""

# JaxMARL/JAX expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import copy
from importlib.metadata import version
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata

__all__ = [
    "SimpleSpreadParams",
    "JaxMARLState",
    "JaxMARLSimpleSpread",
]

# Mirrors JaxMARL's discrete MPE action decoding: stay/x-/x+/y-/y+.
_MPE_MOVES = np.asarray([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
# How far a scripted agent imagines one move takes it. An MPE body covers about
# 0.05 per tick; scoring a whole unit instead made "stay" look best whenever a
# landmark was within half a unit, so partners idled most of the time.
_SCRIPTED_REACH = 0.1


# MPE dynamics the audit found driftable for simple_spread: injecting the
# default reproduces the trajectory exactly and a perturbed value changes it.
DYNAMICS_FIELDS = ("local_ratio", "dt", "mass", "damping", "contact_margin")


class SimpleSpreadParams(NamedTuple):
    """Every drifting quantity of simple_spread, by name."""

    reward_multiplier: jax.Array  # float32 (1,)
    scripted: jax.Array  # float32 (max(num_scripted, 1),), cooperativeness in [0, 1]
    local_ratio: jax.Array  # float32 ()
    dt: jax.Array  # float32 ()
    mass: jax.Array  # float32 (entities,)
    damping: jax.Array  # float32 ()
    contact_margin: jax.Array  # float32 ()


class JaxMARLState(NamedTuple):
    """Backend world state and the scripted slots' own action RNG stream."""

    world: Any
    respawn: jax.Array
    scripted_key: jax.Array


class JaxMARLSimpleSpread:
    """MPE simple_spread as a continuing lifetime with drift.

    JaxMARL's dictionary interfaces become fixed float32/int32 agent slots.
    The legacy episode limit respawns the world on the same tick without
    resetting the Perpetua clock, learner or drift, and the transition reward
    is retained. JaxMARL MPE exposes no action masks, so every action is valid
    for learner-controlled slots; no legacy task identity or info leaks into
    public data. The
    backend's both-branch auto-reset is replaced by a conditional respawn.

    With ``num_scripted > 0`` the last slots are scripted instead of
    learner-controlled: each tick they beta-greedily move toward (partner) or
    away from (opponent) their nearest landmark among the discrete MPE moves,
    or sample uniformly among them, where the per-slot beta is hidden drift
    state on an independent RNG stream. Scripted slots keep world rewards and
    appear in learner observations as ordinary peers, but their public action
    masks allow only the placeholder stay action, their observations are zero
    and their ``active_mask`` is False, so PPO/IPPO never train on them.
    Without scripted slots ``scripted`` has nothing to act on and is not a
    driftable field: drift on it is refused, as the generic MPE adapter does.
    """

    default_drift_field = "reward_multiplier"

    def __init__(
        self,
        *,
        num_agents: int = 3,
        num_landmarks: int = 3,
        local_ratio: float = 0.5,
        max_steps: int = 25,
        num_scripted: int = 0,
        scripted_behavior: Literal["partner", "opponent"] = "partner",
    ) -> None:
        if type(num_agents) is not int or num_agents < 1:
            raise ValueError("JaxMARL simple_spread num_agents must be a positive integer.")
        if type(num_landmarks) is not int or num_landmarks < 1:
            raise ValueError("JaxMARL simple_spread num_landmarks must be a positive integer.")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("JaxMARL simple_spread max_steps must be a positive integer.")
        if type(local_ratio) is not float or not 0.0 <= local_ratio <= 1.0:
            raise ValueError("JaxMARL simple_spread local_ratio must be a float in [0, 1].")
        if type(num_scripted) is not int or not 0 <= num_scripted < num_agents:
            raise ValueError("JaxMARL num_scripted must be in [0, num_agents - 1].")
        if scripted_behavior not in {"partner", "opponent"}:
            raise ValueError("JaxMARL scripted_behavior must be 'partner' or 'opponent'.")
        from jaxmarl.environments.mpe.simple_spread import SimpleSpreadMPE

        self.backend: Any = SimpleSpreadMPE(
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            local_ratio=local_ratio,
            max_steps=max_steps,
        )
        self.agents: tuple[str, ...] = tuple(self.backend.agents)
        self.num_agents = num_agents
        self.num_scripted = num_scripted
        self.scripted_behavior = scripted_behavior
        self.drift_fields: tuple[str, ...] = (
            "reward_multiplier",
            *DYNAMICS_FIELDS,
            *(("scripted",) if num_scripted else ()),
        )
        control = np.ones(num_agents, dtype=np.bool_)
        if num_scripted:
            control[num_agents - num_scripted :] = False
        self._control = jnp.asarray(control)
        self._scripted_slots = jnp.asarray(
            np.arange(num_agents - num_scripted, num_agents), dtype=jnp.int32
        )
        boxes = [self.backend.observation_space(name) for name in self.agents]
        self._obs_dim = int(np.asarray(boxes[0].shape).reshape(-1)[0])
        action_counts = {int(self.backend.action_space(name).n) for name in self.agents}
        if any(int(np.asarray(box.shape).reshape(-1)[0]) != self._obs_dim for box in boxes):
            raise ValueError("JaxMARL simple_spread requires homogeneous observation slots.")
        if len(action_counts) != 1 or not action_counts:
            raise ValueError("JaxMARL simple_spread requires a homogeneous Discrete action space.")
        self.action_count = action_counts.pop()
        low = np.minimum(np.asarray(boxes[0].low, dtype=np.float32), 0.0)
        high = np.maximum(np.asarray(boxes[0].high, dtype=np.float32), 0.0)
        self.observation_space = spaces.Box(
            np.broadcast_to(low, (num_agents, self._obs_dim)),
            np.broadcast_to(high, (num_agents, self._obs_dim)),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(num_agents, self.action_count, dtype=np.int64)
        )
        self.metadata: Metadata = {
            "env_id": "jaxmarl/MPE_simple_spread-v3",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {
                "backend": f"jaxmarl-{version('jaxmarl')}",
                "num_agents": num_agents,
                "num_landmarks": num_landmarks,
                "local_ratio": local_ratio,
                "max_steps": max_steps,
                "drift_parameter": "reward_multiplier",
                "dynamics_fields": list(DYNAMICS_FIELDS),
                "respawn_policy": "same-tick-on-legacy-episode-limit",
                "legacy_info": "discarded",
                "action_masks": (
                    "all-valid"
                    if num_scripted == 0
                    else "learners-all-valid;scripted-placeholder-only"
                ),
                "num_scripted": num_scripted,
                "scripted_slots": "last-slots",
                "scripted_behavior": scripted_behavior,
                "scripted_policy": (
                    "beta-greedy-toward-nearest-landmark"
                    if scripted_behavior == "partner"
                    else "beta-greedy-away-from-nearest-landmark"
                ),
            },
        }

    def check_drift(self, drifts: Any) -> None:
        """Refuse drift on ``scripted`` when no slot is scripted."""
        if "scripted" in drifts and not self.num_scripted:
            raise ValueError(
                "jaxmarl/MPE_simple_spread-v3 has no scripted slots, so drift on 'scripted' "
                "(scripted_drift=) would change nothing; pass num_scripted >= 1."
            )

    @property
    def default_params(self) -> SimpleSpreadParams:
        """Unscaled rewards, fully cooperative scripted slots, backend dynamics."""
        return SimpleSpreadParams(
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.ones(max(self.num_scripted, 1), dtype=jnp.float32),
            *(
                jnp.asarray(getattr(self.backend, name), dtype=jnp.float32)
                for name in DYNAMICS_FIELDS
            ),
        )

    def derive(self, params: SimpleSpreadParams) -> SimpleSpreadParams:
        """Parameters are used as drifted; cooperativeness is clipped where read."""
        return params

    def _dynamics(self, params: SimpleSpreadParams) -> Any:
        """A shallow backend copy carrying the drifted dynamics as traced values."""
        backend = copy.copy(self.backend)
        for name in DYNAMICS_FIELDS:
            setattr(backend, name, getattr(params, name))
        return backend

    def init(
        self, key: jax.Array, params: SimpleSpreadParams
    ) -> tuple[JaxMARLState, jax.Array, EnvInfo]:
        """Reset once from the first of five keys split from ``key``; scripting uses the last."""
        reset_key, _, _, _, scripted_key = jax.random.split(key, 5)
        _, world = self.backend.reset(reset_key)
        state = JaxMARLState(world, jnp.zeros(self.num_agents, dtype=jnp.bool_), scripted_key)
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: JaxMARLState, params: SimpleSpreadParams) -> tuple[jax.Array, EnvInfo]:
        """Return stacked public observations and the common contract."""
        del params
        world_obs = self.backend.get_obs(state.world)
        obs = jnp.stack([jnp.asarray(world_obs[name], dtype=jnp.float32) for name in self.agents])
        placeholder = jnp.zeros((self.num_agents, self.action_count), dtype=jnp.bool_)
        placeholder = placeholder.at[:, 0].set(True)
        info: EnvInfo = {
            "action_mask": jnp.where(
                self._control[:, None],
                jnp.ones((self.num_agents, self.action_count), dtype=jnp.bool_),
                placeholder,
            ),
            "active_mask": self._control,
            "respawn": state.respawn,
        }
        return jnp.where(self._control[:, None], obs, 0.0), info

    def _scripted_actions(
        self, state: JaxMARLState, params: SimpleSpreadParams, key: jax.Array
    ) -> jax.Array:
        """Sample hidden beta-greedy scripted actions from the scripted stream."""
        if self.num_scripted == 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        beta = jnp.clip(params.scripted, 0.0, 1.0)
        slots = self._scripted_slots
        positions = cast(jax.Array, state.world.p_pos[slots])
        landmarks = cast(jax.Array, state.world.p_pos[self.num_agents :])
        proposal = positions[:, None, :] + _SCRIPTED_REACH * jnp.asarray(_MPE_MOVES)[None, :, :]
        separation = cast(
            jax.Array,
            jnp.linalg.norm(proposal[:, :, None, :] - landmarks[None, None, :, :], axis=-1),
        )
        nearest = jnp.min(separation, axis=-1)

        def greedy_move(scores: jax.Array) -> jax.Array:
            if self.scripted_behavior == "partner":
                return jnp.argmin(scores, axis=-1)
            return jnp.argmax(scores, axis=-1)

        greedy = greedy_move(nearest)

        def draw(draw_key: jax.Array, beta: jax.Array, greedy: jax.Array) -> jax.Array:
            greedy_key, uniform_key = jax.random.split(draw_key)
            take = jax.random.bernoulli(greedy_key, p=beta)
            uniform = jax.random.categorical(
                uniform_key, jnp.zeros((self.action_count,), dtype=jnp.float32)
            )
            return jnp.where(take, greedy, uniform).astype(jnp.int32)

        return jax.vmap(draw)(jax.random.split(key, self.num_scripted), beta, greedy)

    def step(
        self, state: JaxMARLState, action: ArrayLike, key: jax.Array, params: SimpleSpreadParams
    ) -> tuple[JaxMARLState, jax.Array]:
        """Step the world under the current dynamics and respawn legacy exhaustion."""
        action = jnp.asarray(action)
        if action.shape != (self.num_agents,) or action.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"JaxMARL simple_spread actions must be integers of shape ({self.num_agents},)."
            )
        world_key, reset_key = jax.random.split(key)
        # The third split is unused; the three-way split fixes the scripted action stream.
        scripted_key, action_key, _ = jax.random.split(state.scripted_key, 3)
        selected = jnp.clip(action, 0, self.action_count - 1).astype(jnp.int32)
        scripted = self._scripted_actions(state, params, action_key)
        selected = selected.at[self._scripted_slots].set(scripted)
        actions = {name: selected[i] for i, name in enumerate(self.agents)}
        _, world, rewards, dones, _ = self._dynamics(params).step_env(
            world_key, state.world, actions
        )
        reward = (
            jnp.stack([jnp.asarray(rewards[name], dtype=jnp.float32) for name in self.agents])
            * params.reward_multiplier[0]
        )
        done = jnp.asarray(dones["__all__"], dtype=jnp.bool_)

        def respawn(_: None) -> Any:
            _, fresh = self.backend.reset(reset_key)
            return fresh

        def unchanged(_: None) -> Any:
            return world

        world = cast(Any, jax.lax.cond(done, respawn, unchanged, None))
        return JaxMARLState(world, jnp.repeat(done, self.num_agents), scripted_key), reward

    def privileged(self, state: JaxMARLState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}
