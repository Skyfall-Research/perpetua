"""A generic JaxMARL MPE adapter: reward and dynamics drift by parameter name.

The simple_spread adapter, :class:`perpetua.adapters.jaxmarl.JaxMARLSimpleSpread`,
is separate (its scripted slots use a landmark heuristic specific to that task).
This adapter covers the other MPE tasks: it pads observations and per-agent
action counts/dimensions, and exposes a ``reward_multiplier`` parameter and the
dynamics fields the audit measured as driftable (see ``scripts/audit_jaxmarl.py``). JaxMARL keeps
dynamics on the environment object, so they are injected onto a shallow backend
copy inside the jitted step, exactly as in the simple_spread adapter.

Optional scripted slots (``num_scripted`` or ``scripted_slots``) are partners
or opponents of the learners whose hidden drifting parameter ``scripted`` sets
how often they act on purpose; see :class:`JaxMARLMPE`.
"""

# JaxMARL ships no type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false

import copy
from collections import namedtuple
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Metadata

__all__ = [
    "MPEState",
    "JaxMARLMPE",
]

# Dynamics fields measured as driftable in the audit (default injection is
# faithful and a perturbed value changes the trajectory). Environments not in
# this table still work with reward drift only.
DYNAMICS_BY_ENV: dict[str, tuple[str, ...]] = {
    "MPE_simple_v3": ("dt", "mass", "damping"),
    "MPE_simple_tag_v3": ("rad", "mass", "damping", "contact_margin"),
    "MPE_simple_world_comm_v3": ("rad",),
    "MPE_simple_speaker_listener_v4": ("dt", "mass", "damping"),
    "MPE_simple_push_v3": ("dt", "mass", "damping", "contact_margin"),
    "MPE_simple_adversary_v3": ("dt", "mass", "damping"),
    "MPE_simple_reference_v3": ("local_ratio", "dt", "mass", "damping"),
    "MPE_simple_facmac_v1": ("rad", "mass", "damping", "view_radius"),
    "MPE_simple_facmac_3a_v1": ("rad", "mass", "damping", "view_radius"),
    "MPE_simple_facmac_6a_v1": ("rad", "mass", "max_speed", "damping", "view_radius"),
    "MPE_simple_facmac_9a_v1": ("rad", "mass", "max_speed", "damping", "view_radius"),
    "MPE_simple_crypto_v3": (),
}


class MPEState(NamedTuple):
    """Backend world state and this tick's respawn flags."""

    world: Any
    respawn: jax.Array


class JaxMARLMPE:
    """One JaxMARL MPE environment as a continuing parameter environment.

    Scripted slots work on every task because their policy only uses the
    reward: each tick, a scripted slot scores each of its candidate actions
    (every discrete action; for continuous slots no-op and each full-strength
    action dimension) by the learners' summed reward over a two-tick lookahead
    (MPE moves bodies by their previous velocity, so an action first changes
    reward one tick later), holding the learners' actual actions, using the
    real step's world key and keeping other scripted slots idle. With
    probability ``beta`` it takes the best candidate for the learners
    (``partner``) or the worst (``opponent``), ties broken at random; otherwise
    a uniformly random candidate. Per-slot ``beta``
    is the drifting parameter ``scripted`` (clipped to [0, 1], default 1), so
    it is hidden unless an Oracle exposes ``scripted_parameters``. Scripted
    slots keep their world rewards and appear in learner observations as
    ordinary peers, but their observations are zero, their action masks allow
    only the placeholder (discrete action 0, no continuous dimension) and their
    ``active_mask`` is False, so learners never train on them. Actions passed
    for them are ignored.

    ``num_scripted`` takes the last controllable slots; ``scripted_slots``
    names them. FACMAC variants already move their prey with a built-in policy
    and ignore actions for them, so those slots are always inactive
    (``backend_scripted_slots`` in the metadata) and cannot be scripted again.

    Limits: the lookahead sees only the next two ticks with held actions, so an
    action whose effect is longer delayed or depends on how others react (a
    speaker's message, for example) scores like any other and is effectively
    random; sparse rewards (tag collisions) mostly tie.
    """

    default_drift_field = "reward_multiplier"

    def __init__(
        self,
        env_id: str,
        *,
        num_scripted: int = 0,
        scripted_slots: Sequence[int] | None = None,
        scripted_behavior: Literal["partner", "opponent"] = "partner",
    ) -> None:
        import jaxmarl

        self.env_id = env_id
        self.backend: Any = jaxmarl.make(env_id)
        self.agents: tuple[str, ...] = tuple(self.backend.agents)
        self.num_agents = len(self.agents)
        # Some FACMAC variants report stale observation-space widths, so use a
        # pure reset observation rather than their metadata for padding.
        _, template = self.backend.reset(jax.random.key(0))
        template_obs = self.backend.get_obs(template)
        self._obs_dim = max(
            int(np.asarray(template_obs[name].shape).reshape(-1)[0]) for name in self.agents
        )
        action_spaces = [self.backend.action_space(name) for name in self.agents]
        # JaxMARL uses its own Box class, so detect it structurally.
        self.continuous = all(
            hasattr(space, "low") and hasattr(space, "high") for space in action_spaces
        )
        if not self.continuous and not all(hasattr(space, "n") for space in action_spaces):
            raise ValueError(f"{env_id} mixes discrete and continuous actions.")
        self.action_counts = tuple(
            int(np.asarray(space.shape).reshape(-1)[0]) if self.continuous else int(space.n)
            for space in action_spaces
        )
        self.action_count = max(self.action_counts)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.num_agents, self._obs_dim), dtype=np.float32
        )
        if self.continuous:
            low = np.zeros((self.num_agents, self.action_count), dtype=np.float32)
            high = np.ones_like(low)
            for index, space in enumerate(action_spaces):
                count = self.action_counts[index]
                low[index, :count] = np.broadcast_to(space.low, (count,))
                high[index, :count] = np.broadcast_to(space.high, (count,))
            self.action_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete(
                np.full(self.num_agents, self.action_count, dtype=np.int64)
            )
        self.dynamics_fields = DYNAMICS_BY_ENV.get(env_id, ())
        # FACMAC variants move their prey with a built-in policy and discard
        # the actions passed for them, so those slots are never controllable.
        self.backend_scripted_slots: tuple[int, ...] = (
            tuple(range(int(self.backend.num_adversaries), self.num_agents))
            if hasattr(self.backend, "_prey_policy")
            else ()
        )
        slots = self._scripted_slot_indices(num_scripted, scripted_slots)
        if scripted_behavior not in {"partner", "opponent"}:
            raise ValueError("scripted_behavior must be 'partner' or 'opponent'.")
        self.scripted_slots = slots
        self.num_scripted = len(slots)
        self.scripted_behavior = scripted_behavior
        control = np.ones(self.num_agents, dtype=np.bool_)
        control[list(slots)] = False
        control[list(self.backend_scripted_slots)] = False
        self._control = jnp.asarray(control)
        self._build_candidates()
        self._params_type = namedtuple(  # type: ignore[misc]
            "MPEParams",
            ["reward_multiplier", *self.dynamics_fields, *(("scripted",) if slots else ())],
        )
        self.metadata: Metadata = {
            "env_id": f"jaxmarl/{env_id}",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {
                "backend": f"jaxmarl-{version('jaxmarl')}",
                "num_agents": self.num_agents,
                "drift_parameter": "reward_multiplier",
                "dynamics_fields": list(self.dynamics_fields),
                "respawn_policy": "same-tick-on-legacy-episode-limit",
                "legacy_info": "discarded",
                "action_masks": "own-actions-valid;scripted-placeholder-only",
                "backend_scripted_slots": list(self.backend_scripted_slots),
                "num_scripted": self.num_scripted,
                "scripted_slots": list(slots),
                "scripted_behavior": scripted_behavior,
                "scripted_policy": "beta-greedy-two-tick-learner-reward-lookahead",
            },
        }

    def _scripted_slot_indices(
        self, num_scripted: int, scripted_slots: Sequence[int] | None
    ) -> tuple[int, ...]:
        controllable = [
            slot for slot in range(self.num_agents) if slot not in self.backend_scripted_slots
        ]
        if scripted_slots is not None:
            if num_scripted:
                raise ValueError("Pass num_scripted or scripted_slots, not both.")
            slots = tuple(scripted_slots)
            if any(type(slot) is not int for slot in slots) or len(set(slots)) != len(slots):
                raise ValueError("scripted_slots must be distinct integers.")
            if any(slot not in controllable for slot in slots):
                raise ValueError(f"{self.env_id} scripted_slots must be among {controllable}.")
        else:
            if type(num_scripted) is not int or num_scripted < 0:
                raise ValueError("num_scripted must be a nonnegative integer.")
            slots = tuple(controllable[len(controllable) - num_scripted :]) if num_scripted else ()
        if len(slots) >= len(controllable):
            raise ValueError(
                f"{self.env_id} has {len(controllable)} controllable agent(s); "
                "at least one must be a learner."
            )
        return tuple(sorted(slots))

    def _build_candidates(self) -> None:
        """Padded candidate actions and their validity for each scripted slot."""
        width = self.action_count + 1 if self.continuous else self.action_count
        valid = np.zeros((max(self.num_scripted, 1), width), dtype=np.bool_)
        if self.continuous:
            high = np.asarray(self.action_space.high, dtype=np.float32)
            candidates = np.zeros((max(self.num_scripted, 1), width, self.action_count), np.float32)
            for row, slot in enumerate(self.scripted_slots):
                valid[row, 0] = True  # no-op
                for dim in range(self.action_counts[slot]):
                    candidates[row, dim + 1, dim] = high[slot, dim]
                    valid[row, dim + 1] = True
            self._neutral = jnp.zeros((self.action_count,), dtype=jnp.float32)
        else:
            candidates = np.broadcast_to(
                np.arange(width, dtype=np.int32), (max(self.num_scripted, 1), width)
            ).copy()
            for row, slot in enumerate(self.scripted_slots):
                valid[row, : self.action_counts[slot]] = True
            self._neutral = jnp.asarray(0, dtype=jnp.int32)
        self._candidates = jnp.asarray(candidates)
        self._candidate_valid = jnp.asarray(valid)

    @property
    def default_params(self) -> Any:
        """Unscaled rewards, the backend's own dynamics values and purposeful scripted slots."""
        scripted = (
            {"scripted": jnp.ones(self.num_scripted, dtype=jnp.float32)}
            if self.num_scripted
            else {}
        )
        return self._params_type(
            reward_multiplier=jnp.asarray([1.0], dtype=jnp.float32),
            **{
                name: jnp.asarray(getattr(self.backend, name), dtype=jnp.float32)
                for name in self.dynamics_fields
            },
            **scripted,
        )

    def derive(self, params: Any) -> Any:
        """Parameters are used as drifted."""
        return params

    def _dynamics(self, params: Any) -> Any:
        """A shallow backend copy carrying the drifted dynamics as traced values."""
        backend = copy.copy(self.backend)
        for name in self.dynamics_fields:
            setattr(backend, name, getattr(params, name))
        return backend

    def init(
        self, key: jax.Array, params: dict[str, jax.Array]
    ) -> tuple[MPEState, jax.Array, EnvInfo]:
        """Reset the backend once."""
        # Child 0 of a three-way split, kept so the init stream is the one earlier seeds drew.
        reset_key, _, _ = jax.random.split(key, 3)
        _, world = self.backend.reset(reset_key)
        state = MPEState(world, jnp.zeros(self.num_agents, dtype=jnp.bool_))
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: MPEState, params: Any) -> tuple[jax.Array, EnvInfo]:
        """Return stacked public observations and the common contract."""
        # The drifted backend: fields such as FACMAC's ``view_radius`` and
        # world_comm's ``rad`` are read when observations are built.
        world_obs = self._dynamics(params).get_obs(state.world)
        obs = jnp.stack(
            [
                jnp.pad(
                    jnp.asarray(world_obs[name], dtype=jnp.float32),
                    (0, self._obs_dim - world_obs[name].shape[-1]),
                )
                for name in self.agents
            ]
        )
        # Continuous masks mark the real action dimensions; padded dimensions
        # are discarded in step, just as padded discrete actions are masked.
        mask = jnp.arange(self.action_count)[None, :] < jnp.asarray(self.action_counts)[:, None]
        # Scripted slots allow only the placeholder: discrete action 0, or no
        # continuous dimension.
        placeholder = jnp.zeros_like(mask)
        if not self.continuous:
            placeholder = placeholder.at[:, 0].set(True)
        info: EnvInfo = {
            "action_mask": jnp.where(self._control[:, None], mask, placeholder),
            "active_mask": self._control,
            "respawn": state.respawn,
        }
        return jnp.where(self._control[:, None], obs, 0.0), info

    def _backend_actions(self, selected: jax.Array) -> dict[str, jax.Array]:
        if self.continuous:
            return {
                name: selected[i, : self.action_counts[i]] for i, name in enumerate(self.agents)
            }
        return {name: selected[i] for i, name in enumerate(self.agents)}

    def _scripted_actions(
        self,
        world: Any,
        selected: jax.Array,
        world_key: jax.Array,
        draw_key: jax.Array,
        params: Any,
        backend: Any,
    ) -> jax.Array:
        """Beta-greedy two-tick-lookahead actions for every scripted slot."""
        slots = jnp.asarray(self.scripted_slots, dtype=jnp.int32)
        idle = selected.at[slots].set(self._neutral)
        learners = self._control.astype(jnp.float32)

        def learner_reward(slot: jax.Array, candidate: jax.Array) -> jax.Array:
            # MPE moves bodies by their previous velocity, so an action first
            # changes reward one tick later: hold every action for two ticks.
            actions = self._backend_actions(idle.at[slot].set(candidate))
            total = jnp.asarray(0.0, dtype=jnp.float32)
            trial_world = world
            for step_key in (world_key, jax.random.fold_in(world_key, 1)):
                _, trial_world, rewards, _, _ = backend.step_env(step_key, trial_world, actions)
                reward = jnp.stack(
                    [jnp.asarray(rewards[name], dtype=jnp.float32) for name in self.agents]
                )
                total = total + jnp.sum(reward * params.reward_multiplier[0] * learners)
            return total

        def slot_scores(slot: jax.Array, options: jax.Array) -> jax.Array:
            return jax.vmap(learner_reward, in_axes=(None, 0))(slot, options)

        scores = jax.vmap(slot_scores)(slots, self._candidates)
        if self.scripted_behavior == "opponent":
            scores = -scores
        scores = jnp.where(self._candidate_valid, scores, -jnp.inf)
        best = jnp.max(scores, axis=-1, keepdims=True)
        greedy_logits = jnp.where(self._candidate_valid & (scores >= best), 0.0, -jnp.inf)
        uniform_logits = jnp.where(self._candidate_valid, 0.0, -jnp.inf)
        beta = jnp.clip(params.scripted, 0.0, 1.0)

        def draw(
            draw_key: jax.Array, beta: jax.Array, greedy: jax.Array, uniform: jax.Array
        ) -> jax.Array:
            take_key, greedy_key, uniform_key = jax.random.split(draw_key, 3)
            return jnp.where(
                jax.random.bernoulli(take_key, p=beta),
                jax.random.categorical(greedy_key, greedy),
                jax.random.categorical(uniform_key, uniform),
            )

        choices = jax.vmap(draw)(
            jax.random.split(draw_key, self.num_scripted), beta, greedy_logits, uniform_logits
        )
        return self._candidates[jnp.arange(self.num_scripted), choices]

    def step(
        self, state: MPEState, action: ArrayLike, key: jax.Array, params: Any
    ) -> tuple[MPEState, jax.Array]:
        """Step under the current dynamics and respawn legacy exhaustion."""
        action = jnp.asarray(action)
        expected = (self.num_agents, self.action_count) if self.continuous else (self.num_agents,)
        if action.shape != expected:
            raise ValueError(f"Actions must have shape {expected}.")
        world_key, reset_key, scripted_key = jax.random.split(key, 3)
        if self.continuous:
            selected = jnp.clip(
                action.astype(jnp.float32), self.action_space.low, self.action_space.high
            )
        else:
            selected = jnp.clip(action, 0, jnp.asarray(self.action_counts) - 1).astype(jnp.int32)
        backend = self._dynamics(params)
        if self.num_scripted:
            # The third child, so unscripted lifetimes keep their streams.
            scripted = self._scripted_actions(
                state.world, selected, world_key, scripted_key, params, backend
            )
            selected = selected.at[jnp.asarray(self.scripted_slots)].set(scripted)
        _, world, rewards, dones, _ = backend.step_env(
            world_key, state.world, self._backend_actions(selected)
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
        return MPEState(world, jnp.repeat(done, self.num_agents)), reward

    def privileged(self, state: MPEState) -> dict[str, jax.Array]:
        """No world-side privileged fields; the wrapper supplies drift diagnostics."""
        del state
        return {}
