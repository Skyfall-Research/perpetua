"""One parameter adapter for every gymnax environment.

gymnax already exposes exactly the shape this framework wants: a parameter
struct, ``reset(key, params)``, ``step(key, state, action, params)`` and
``get_obs(state, params)``. :class:`GymnaxParameters` wraps any of them as a
:class:`~perpetua.params.ParamsEnv`, so every gymnax environment inherits the
framework's drift bookkeeping, Oracle, clock and metadata without per-environment
code. Run ``scripts/probe_gymnax.py`` for the measured coverage table.

Continuing semantics: gymnax episodes are not lifetimes. The wrapper removes
the time limit (``max_steps_in_episode``) from the effective parameters, so an
episode only ends on a genuine termination, and reports that termination as a
``respawn`` event on the next observation. The two bandits and DiscountingChain
have no termination other than the limit and read their parameters only at
reset, so their limit is kept and reported as a respawn.

The wrapper calls ``step_env`` and runs ``reset_env`` only on termination,
under ``lax.cond``, instead of gymnax's ``step``, which computes a reset on
every tick and selects between the two. It uses gymnax's own key split, so the
results match gymnax's ``step`` up to float32 rounding in compiled code:
rewards and respawns are equal on all 20 registered ids, and observations are
bit-equal on 18 of them, while Acrobot's and Reacher's differ in the last
bits. Under ``jax.vmap`` a ``cond`` still evaluates both
branches, so batched environments that can terminate still pay the reset.

gymnax's episode counter (``state.time``) keeps counting after the time limit
is removed and would reach the raised limit after about 2.1 billion ticks,
firing a spurious time-limit respawn. The wrapper saturates it just below the
limit before each step. That changes nothing before then, and environments
that read the counter in their observations or dynamics (MemoryChain,
UmbrellaChain, DiscountingChain, MetaMaze, PointRobot and the bandits) keep
their own behaviour; their own terminations still reset it.

Actions are checked for shape and dtype when the step is traced: an integer
array of shape ``(1,)`` for a discrete backend, a float array of the action
space's shape for a Box one. Values are not checked inside ``jit``, so an
integer outside ``[0, n - 1]`` reaches the backend as it is. Measured on gymnax
0.0.9: most backends look the index up in a table, so an index past the end acts
as the last action and ``-1`` as the last action too; MountainCar scales the
push with the index (action 6 applies five times the force of action 2);
Catch, MetaMaze and the two bandits do something else again. Clip with
``jnp.clip(action, 0, n - 1)`` in the policy.
"""

# gymnax ships no type stubs, so its lazily-resolved generics stay loose.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateImportUsage=false

import dataclasses
import inspect
import re
from typing import Any, NamedTuple, cast

import gymnax  # pyright: ignore[reportMissingTypeStubs]
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from perpetua.core import EnvInfo

__all__ = [
    "GymnaxState",
    "GymnaxParameters",
]

# gymnax precomputes derived parameter fields at construction and its parameter
# structs have no ``__post_init__``, so ``replace`` on an input leaves the
# derived field stale: CartPole's ``total_mass`` stays 1.1 after ``masspole``
# moves. The adapter therefore drifts only fields that no other field is
# computed from, and refuses the rest with that explanation instead of
# silently running inconsistent physics.
# Float parameter fields the backend declares but never reads (gymnax 0.0.9).
# Drift on one would declare switches while the world stays the same, so they
# are left out of the driftable fields and refused with the reason.
_UNREAD_FIELDS: dict[str, dict[str, str]] = {
    "Acrobot-v1": {
        "link_length_2": "gymnax's Acrobot dynamics read link_length_1 only",
    },
}
_UNLIMITED_STEPS = 2**31 - 1
# After the backend's ``time + 1`` the counter stays below the raised limit.
_TIME_CEILING = _UNLIMITED_STEPS - 1
_TIME_LIMIT = "max_steps_in_episode"
# Environments whose only episode structure is the time limit: they never
# terminate otherwise, and they read their parameters (bandit arm
# probabilities, the chain's rewarding step) only at reset. Without the limit
# they would never reset again, so drift on those parameters would do nothing
# and DiscountingChain would pay its single reward once per lifetime. Their
# limit is kept and reported as a respawn.
_TIME_LIMIT_IS_THE_EPISODE = frozenset(
    {"BernoulliBandit-misc", "GaussianBandit-misc", "DiscountingChain-bsuite"}
)
# Environments whose observations leave the box gymnax declares, so the adapter
# declares them unbounded: MetaMaze's time feature grows past the removed
# limit, GaussianBandit carries an unbounded reward, and upstream's boxes for
# MNISTBandit (pixels are centred on zero) and MemoryChain (a -1 marker) are
# too narrow.
_UNBOUNDED_OBSERVATIONS = frozenset(
    {"MetaMaze-misc", "GaussianBandit-misc", "MNISTBandit-bsuite", "MemoryChain-bsuite"}
)


def _derived_inputs(params_type: type, names: list[str]) -> set[str] | None:
    """Fields that some other field's default value is computed from.

    gymnax parameter structs spell the arithmetic out in their defaults
    (``total_mass: float = masscart + masspole``), so the class source is the
    only place the relationship exists. Fields named in another field's
    default are precomputed inputs: drifting them would leave the derived
    field stale. ``None`` when the class source is unavailable.
    """
    try:
        source = inspect.getsource(params_type)
    except (OSError, TypeError):
        return None
    inputs: set[str] = set()
    for line in source.splitlines():
        stripped = line.strip()
        if "=" not in stripped or ":" not in stripped:
            continue
        name, _, default = stripped.partition("=")
        target = name.partition(":")[0].strip()
        if target not in names:
            continue
        for other in names:
            if other != target and re.search(rf"\b{other}\b", default):
                inputs.add(other)
    return inputs


class GymnaxState(NamedTuple):
    """The environment's own state, its observation and this tick's respawn.

    The observation is carried rather than recomputed through ``get_obs``,
    because gymnax's ``get_obs`` signatures vary (some need a key, some are
    unimplemented) and the backend already returns the exact observation on
    every reset and step.
    """

    env: Any
    respawn: jax.Array
    obs: jax.Array


class GymnaxParameters:
    """Wrap one gymnax environment as a parameter environment.

    ``drift_fields`` names the parameter fields the environment will accept
    drift on (the adapter refuses anything else, including any field that
    another field is precomputed from, such as CartPole's ``masscart``), so
    precomputed fields such as ``total_mass`` stay consistent. Which fields
    are precomputed is read from the parameter class source; when that source
    is unavailable, ``drift_fields`` must be given and is not checked for it.
    A field the backend declares but never reads is not driftable either,
    because drift on it would declare switches in an unchanged world: on
    gymnax 0.0.9 that is Acrobot's ``link_length_2`` (its dynamics read
    ``link_length_1`` only). Naming one in ``drift_fields`` or in a drift
    mapping raises ``ValueError`` with that reason. ``derive`` only removes
    the time limit.
    """

    def __init__(self, env_id: str, *, drift_fields: tuple[str, ...] | None = None) -> None:
        self.env_id = env_id
        self.env, params = gymnax.make(env_id)
        # Dataclasses at runtime; gymnax stubs do not model ``replace``.
        self._params: Any = params
        names = [
            field.name
            for field in dataclasses.fields(self._params)  # type: ignore[arg-type]
        ]
        self._keeps_limit = env_id in _TIME_LIMIT_IS_THE_EPISODE
        # The time limit is the adapter's to manage, never a drift field.
        floats = tuple(
            name
            for name in names
            if isinstance(getattr(self._params, name), float) and name != _TIME_LIMIT
        )
        plugged = _derived_inputs(type(self._params), names)
        if plugged is None and drift_fields is None:
            raise ValueError(
                f"Cannot read the source of {type(self._params).__name__} to find which "
                "fields are precomputed from others; pass drift_fields=(...) explicitly."
            )
        self._plugged = plugged or set()
        self._unread = _UNREAD_FIELDS.get(env_id, {})
        requested = (
            tuple(name for name in floats if name not in self._plugged and name not in self._unread)
            if drift_fields is None
            else tuple(drift_fields)
        )
        self._refuse_unread(requested)
        unknown = set(requested) - set(floats)
        if unknown:
            raise ValueError(
                f"{env_id} cannot drift {sorted(unknown)}; float fields are {list(floats)}"
            )
        precomputed = set(requested) & self._plugged
        if precomputed:
            raise ValueError(
                f"{env_id} precomputes {sorted(precomputed)} into other parameter fields, "
                "so drifting them would leave those fields stale; drift a field they are "
                "not derived from instead."
            )
        self.drift_fields = requested
        self.num_agents = 1
        obs_space = self.env.observation_space(self._params)
        shape = (1, *obs_space.shape)
        if env_id in _UNBOUNDED_OBSERVATIONS:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=np.broadcast_to(np.asarray(obs_space.low, dtype=np.float32), shape),
                high=np.broadcast_to(np.asarray(obs_space.high, dtype=np.float32), shape),
                dtype=np.float32,
            )
        action_space = self.env.action_space(self._params)
        cardinality = getattr(action_space, "n", None)
        self._discrete = cardinality is not None
        self._scalar_action = cardinality is None and tuple(
            getattr(action_space, "shape", ()) or ()
        ) == (1,)
        if cardinality is not None:
            self.action_space = spaces.MultiDiscrete(np.asarray([int(cardinality)], dtype=np.int64))
            self._mask = jnp.ones((1, int(cardinality)), dtype=jnp.bool_)
        else:
            box = cast(Any, action_space)
            # Continuous backends declare either a scalar torque or a vector.
            self.action_space = spaces.Box(
                low=np.asarray(box.low, dtype=np.float32).reshape(1, -1),
                high=np.asarray(box.high, dtype=np.float32).reshape(1, -1),
                dtype=np.float32,
            )
            self._mask = jnp.ones((1, 1), dtype=jnp.bool_)
        self.default_drift_field = self.drift_fields[0] if self.drift_fields else None
        self.metadata = {
            "env_id": f"gymnax/{env_id}",
            "oracle_exposure": (),
            "configuration": {
                "backend": env_id,
                "driftable_fields": list(self.drift_fields),
                "clock": "decision-tick",
                "episodes": (
                    "time limit kept as a respawn (the environment has no other episode end)"
                    if self._keeps_limit
                    else "time limit removed; terminations become respawns"
                ),
            },
        }

    def _refuse_unread(self, names: Any) -> None:
        for name in names:
            if name in self._unread:
                raise ValueError(
                    f"{self.env_id} cannot drift {name!r}: {self._unread[name]}, so the "
                    "drift would declare switches without changing the environment."
                )

    def check_drift(self, drifts: Any) -> None:
        """Refuse drift on a field the backend never reads, saying why."""
        self._refuse_unread(drifts)

    @property
    def default_params(self) -> Any:
        """The backend's defaults with float fields as arrays.

        gymnax stores parameter fields as Python floats, but drift needs array
        leaves (``Cyclic`` validates shapes), so the adapter presents the
        ``drift_fields`` as float32 arrays. Other float fields stay Python
        floats, backend arithmetic accepts both, and integer fields such as the
        time limit stay integers.
        """
        updates = {
            name: jnp.asarray(getattr(self._params, name), dtype=jnp.float32)
            for name in self.drift_fields
        }
        return self._params.replace(**updates)

    def derive(self, params: Any) -> Any:
        """Remove the time limit; drift never touches precomputed fields.

        Derived fields stay consistent by construction: the adapter refuses
        drift on any field that another field is computed from, so the values
        gymnax precomputed at construction remain correct.
        """
        if self._keeps_limit:
            return params
        return params.replace(max_steps_in_episode=_UNLIMITED_STEPS)

    def init(self, key: jax.Array, params: Any) -> tuple[GymnaxState, jax.Array, EnvInfo]:
        """Reset the backend once; its episode start is the lifetime start."""
        # Child 0 of a three-way split, kept so the init stream is the one earlier seeds drew.
        reset_key, _, _ = jax.random.split(key, 3)
        obs, state = self.env.reset(reset_key, params)
        info: EnvInfo = {
            "action_mask": self._mask,
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": jnp.zeros((1,), dtype=jnp.bool_),
        }
        carried = jnp.asarray(obs, dtype=jnp.float32)
        return (
            GymnaxState(state, jnp.zeros((1,), dtype=jnp.bool_), carried),
            carried[None],
            info,
        )

    def _checked_action(self, action: Any) -> jax.Array:
        """The action as an array of the declared shape and a backend-compatible dtype."""
        action = jnp.asarray(action)
        space = self.action_space
        got = f"got dtype {action.dtype} and shape {action.shape}"
        if self._discrete:
            if action.shape != (1,) or action.dtype.kind not in {"i", "u"}:
                raise ValueError(
                    f"{self.env_id} actions must be integer arrays of shape (1,), one index in "
                    f"[0, {int(self._mask.shape[1]) - 1}] (MultiDiscrete action space); {got}. "
                    "Sample with pp.policies.random or pass jnp.asarray([index], dtype=jnp.int32)."
                )
            # int32 or int64, whichever integer the backend's state uses.
            return action.astype(jax.dtypes.canonicalize_dtype(jnp.int_))
        if action.shape != space.shape or action.dtype.kind != "f":
            raise ValueError(
                f"{self.env_id} actions must be float arrays of shape {space.shape} "
                f"(the action_space Box); {got}. pp.policies.random only samples discrete "
                "actions: use pp.policies.uniform(env.action_space) for a random baseline."
            )
        return action

    def step(
        self, state: GymnaxState, action: Any, key: jax.Array, params: Any
    ) -> tuple[GymnaxState, jax.Array]:
        """One backend step; a termination is reported as a respawn event."""
        action = self._checked_action(action)
        selected = action[0]
        if self._scalar_action:
            # One-dimensional continuous backends expect a scalar torque.
            selected = jnp.squeeze(selected)
        backend = state.env
        if hasattr(backend, "time") and not self._keeps_limit:
            time = jnp.asarray(backend.time)
            # jnp.minimum with a Python int keeps the counter's dtype and weak type.
            backend = backend.replace(time=jnp.minimum(time, _TIME_CEILING - 1))
        # gymnax.Environment.step's key layout, with the reset run only on
        # termination rather than computed every tick and selected away.
        step_key, reset_key = jax.random.split(key)
        obs, next_state, reward, done, _ = self.env.step_env(step_key, backend, selected, params)

        def respawn(_: None) -> tuple[jax.Array, Any]:
            fresh_obs, fresh = self.env.reset_env(reset_key, params)
            return jnp.asarray(fresh_obs, dtype=jnp.float32), fresh

        def unchanged(_: None) -> tuple[jax.Array, Any]:
            return jnp.asarray(obs, dtype=jnp.float32), next_state

        obs, next_state = jax.lax.cond(jnp.asarray(done), respawn, unchanged, None)
        return (
            GymnaxState(next_state, jnp.asarray([done]), jnp.asarray(obs)),
            jnp.asarray(reward, dtype=jnp.float32).reshape(1),
        )

    def observe(self, state: GymnaxState, params: Any) -> tuple[jax.Array, EnvInfo]:
        """Return the observation the backend produced, without stepping."""
        del params
        info: EnvInfo = {
            "action_mask": self._mask,
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn,
        }
        return state.obs[None], info

    def privileged(self, state: GymnaxState) -> dict[str, jax.Array]:
        """GymnaxParameters has no state-side privileged fields."""
        del state
        return {}
