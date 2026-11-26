"""Parameter environments: the framework owns drift, keys, clock and Oracle.

An environment author implements this small contract:

.. code-block:: python

    class MyEnv:
        num_agents: int
        observation_space: gymnasium.Space
        action_space: gymnasium.Space
        metadata: dict          # "env_id" required unless DriftEnv(env_id=...)
        default_params: MyParams                       # pytree of float arrays
        def init(self, key, params) -> (state, obs, info)
        def step(self, state, action, key, params) -> (state, reward)
        def observe(self, state, params) -> (obs, info)  # info without the clock
        def privileged(self, state) -> dict[str, jax.Array]
        def derive(self, params) -> params             # optional
        mask_params: tuple[str, ...]                   # optional
        drift_fields: tuple[str, ...]                  # optional
        default_drift_field: str                       # optional
        def signal(self, previous_state, action, state)  # optional
        def task_id(self, params) -> jax.Array         # optional
        def after_drift(self, state, previous, current, key) -> state  # optional
        def declared_switches(self, previous, current) -> dict[str, jax.Array]  # optional
        def check_drift(self, drifts) -> None          # optional
        def check_schedule(self, field, data) -> None  # optional

``init``, ``step`` and ``observe`` receive the already-derived parameters.
``signal`` feeds endogenous drift processes, ``task_id`` replaces the
Oracle's default ``task_id`` and ``after_drift`` reacts to a parameter change
on the tick it happens (see :meth:`DriftEnv.step`). ``declared_switches``
names drifted fields that switched on this tick although their process did
not change phase (a task switch reports ``task`` when the active task
changed); the flags are added to the processes' own.


:class:`DriftEnv` wraps one such environment and owns what every environment
would otherwise re-implement: the drift state and its RNG key, declared switch
events, the lifetime clock, the Oracle fields (``phase``,
``drift_parameters``, ``task_id`` and ``boundary``, the deprecated alias
``drift_changed`` of ``boundary``, plus the environment's own privileged
fields) and checkpoint metadata. Drift addresses parameters by dotted field
name, so one mechanism drives any field of any environment.
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import inspect
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from perpetua.core import Clock, EnvInfo, Info, Metadata
from perpetua.drift import Drift, DriftState, ScheduleData, switched
from perpetua.spaces import mask_change

__all__ = [
    "ParamsEnv",
    "DriftEnvState",
    "DriftEnv",
]


class ParamsEnv[StateT, ParamsT](Protocol):
    """Environment contract with explicit parameters and no drift bookkeeping.

    ``derive`` is optional: it maps raw parameters (what drift writes) to the
    effective ones the environment consumes, which is where rounding and
    dependent fields such as ``total_mass`` belong. ``mask_params`` is
    optional: the parameters whose effective values change the action mask.
    When one moves, the wrapper flags in ``space_change`` each agent whose
    mask it changed. ``drift_fields`` is optional: when present, drift is
    refused on any other parameter (the gymnax adapter uses it to protect
    precomputed fields). ``check_drift(drifts)`` is optional: the wrapper calls it
    with the mapping from parameter names to processes whenever drift is bound
    (construction, ``with_drift``, ``pp.compose``), so the environment can
    refuse a process it cannot run by raising ``ValueError``.
    ``check_schedule(field, data)`` is optional:
    :func:`perpetua.runtime.assign_schedule` calls it with the schedule a
    lifetime is given, for the same purpose. An endogenous process
    (``Accumulate``) is refused when drift is bound unless the environment
    defines ``signal``.
    """

    @property
    def default_params(self) -> ParamsT: ...

    def init(self, key: jax.Array, params: ParamsT) -> tuple[StateT, jax.Array, EnvInfo]: ...

    def step(
        self, state: StateT, action: ArrayLike, key: jax.Array, params: ParamsT
    ) -> tuple[StateT, jax.Array]: ...

    def observe(self, state: StateT, params: ParamsT) -> tuple[jax.Array, EnvInfo]: ...

    def privileged(self, state: StateT) -> dict[str, jax.Array]: ...

    def derive(self, params: ParamsT) -> ParamsT: ...


# Keys the framework derives from the key it also hands the environment. Each
# is a fold of a large tag, so no plain split of that key (any width) that an
# environment makes can reproduce one: split(key, n)[i] equals fold_in(key, i)
# under partitionable threefry, and these tags are far beyond any n.
_DRIFT_TAG = 0x64726674  # lifetime key: drift init keys and the drift stream
_AFTER_DRIFT_TAG = 0x61667464  # step key: the after_drift hook
_REFERENCE_TAG = 0x72656672  # step key: the regret reference policy's action


class DriftEnvState[StateT](NamedTuple):
    """Lifetime state owned by the framework: dynamics, drift states and clock.

    ``drift_changed`` is the aggregate switch flag the metrics carry consumes;
    ``field_changed`` keeps per-field flags so an Oracle can expose
    ``{field}_changed`` for each drifted parameter.
    """

    env: StateT
    drift: dict[str, DriftState]
    drift_key: jax.Array
    drift_changed: jax.Array
    field_changed: dict[str, jax.Array]
    t: Clock


def _strong(tree: Any) -> Any:
    """Drop weak types from every array leaf.

    Backends often build some state leaves from Python scalars (weakly typed)
    in ``init`` but return strongly typed ones from ``step``, or the other
    way round. The values are identical, but a compiled runner would then
    compile its chunk a second time on the next call, so the world state is
    normalised to strong types on both sides. Only arrays and numeric Python
    scalars are touched; host simulators' opaque leaves pass through.
    """

    def strong(leaf: Any) -> Any:
        if not isinstance(leaf, jax.Array | bool | int | float):
            return leaf
        aval = jax.typeof(leaf)
        if getattr(aval, "weak_type", False):
            return jnp.asarray(leaf, dtype=aval.dtype)
        return leaf

    return jax.tree.map(strong, tree)


def param_paths(params: object) -> dict[str, jax.Array]:
    """Map leaf paths to parameter arrays.

    Attribute pytrees (NamedTuples, dataclasses) give dotted names such as
    ``"goal"`` or ``"outer.inner"``; dict keys and sequence indices keep their
    brackets (``"['goal']"``, ``"tasks[0].goal"``).
    """
    leaves = cast(
        "list[tuple[tuple[object, ...], jax.Array]]",
        jax.tree_util.tree_flatten_with_path(params)[0],
    )
    return {jax.tree_util.keystr(path).lstrip("."): leaf for path, leaf in leaves}


def _apply(params: object, updates: Mapping[str, jax.Array]) -> object:
    """Return ``params`` with the named leaves replaced, checking every name."""
    known = param_paths(params)
    unknown = set(updates) - set(known)
    if unknown:
        raise ValueError(f"Unknown drift parameters {sorted(unknown)}; available: {sorted(known)}")

    def replace(path: tuple[object, ...], leaf: jax.Array) -> jax.Array:
        return updates.get(jax.tree_util.keystr(path).lstrip("."), leaf)

    return jax.tree_util.tree_map_with_path(replace, params)


def _check_process(process: object, where: str) -> None:
    """Refuse anything that is not a drift process, saying what was passed."""
    if inspect.isclass(process):
        raise TypeError(
            f"drift must be a pp.drift process instance; {where} is the class "
            f"{process.__name__}. Construct it, for example {process.__name__}(...)."
        )
    if (
        callable(getattr(process, "init", None))
        and callable(getattr(process, "step", None))
        and hasattr(process, "configuration")
    ):
        return
    hint = (
        " pp.drift.schedule() returns schedule data; wrap it in pp.drift.Schedule(values, period)."
        if isinstance(process, ScheduleData)
        else ""
    )
    raise TypeError(
        f"drift must be a pp.drift process or a mapping of parameter names to processes; "
        f"{where} is {type(process).__name__}.{hint}"
    )


class DriftEnv[StateT, ParamsT]:
    """Wrap a parameter environment and own drift, switches, clock and Oracle.

    ``drift`` is a process (applied to the environment's default drifting
    field) or a mapping from dotted parameter names to processes. A bare
    process on an environment with several parameters and no
    ``default_drift_field`` raises ``ValueError``, as does a mapping that names
    an unknown parameter and any drift on an environment with no driftable
    parameter. A value that is not a drift process raises ``TypeError``.
    ``env_id`` overrides the environment's ``metadata["env_id"]``.

    Public observations and info are the environment's, plus ``info['t']``
    from the wrapper's clock and a per-agent ``info['space_change']`` flag.
    The raw drift state (values, phases, switch flags) reaches a policy only
    through an explicit :class:`~perpetua.core.Oracle`; the environment decides
    what its observation shows of the effective parameters.
    ``drift_parameters`` and ``{field}_parameters`` are raw drift values,
    before ``derive``: a ``ContinualGrid`` goal can read 6.6 there while the
    effective goal is 7.
    """

    def __init__(
        self,
        env: ParamsEnv[StateT, ParamsT],
        drift: Drift | Mapping[str, Drift] | None = None,
        *,
        env_id: str | None = None,
    ) -> None:
        self.env = env
        self._label = env_id if env_id is not None else type(env).__name__
        self.num_agents = cast(Any, env).num_agents
        self.observation_space = cast(Any, env).observation_space
        self.action_space = cast(Any, env).action_space
        params = env.default_params
        self._names = sorted(param_paths(params))
        if not self._names:
            raise ValueError(
                f"{type(env).__name__}.default_params has no parameter fields; a "
                "parameter environment needs at least one."
            )
        allowed = cast("tuple[str, ...] | None", getattr(env, "drift_fields", None))
        self._driftable: list[str] = sorted(self._names if allowed is None else allowed)
        if drift is None:
            self.drifts: dict[str, Drift] = {}
        elif isinstance(drift, Mapping):
            self.drifts = dict(drift)
        else:
            self.drifts = {self._bare_field(): drift}
        for name, process in self.drifts.items():
            _check_process(process, f"the process for {name!r}" if len(self.drifts) > 1 else "it")
        unknown = set(self.drifts) - set(self._names)
        if unknown:
            raise ValueError(
                f"Unknown drift parameters {sorted(unknown)}; "
                + (
                    f"available: {self._driftable}"
                    if self._driftable
                    else f"{self._label} has no driftable parameters"
                )
            )
        self._params = params
        # The environment's own refusal first: it can say why a field is closed.
        check = getattr(env, "check_drift", None)
        if check is not None:
            check(self.drifts)
        if allowed is not None and set(self.drifts) - set(allowed):
            self._require_driftable()
            raise ValueError(
                f"{type(env).__name__} accepts drift only on {sorted(allowed)}, "
                f"not {sorted(set(self.drifts) - set(allowed))}."
            )
        if getattr(env, "signal", None) is None:
            for name, process in self.drifts.items():
                if getattr(process, "endogenous", False):
                    raise ValueError(
                        f"{type(process).__name__} on {name!r} is driven by the environment's "
                        f"signal, and {type(env).__name__} defines no "
                        "signal(previous_state, action, state)."
                    )
        self._fields = sorted(self.drifts)
        declared = getattr(env, "default_drift_field", None)
        self._primary: str = (
            declared
            if declared in self._names
            else (self._fields[0] if self._fields else self._names[0])
        )
        inner_fields = tuple(getattr(env, "oracle_fields", ()))
        aliases = tuple(
            alias
            for field in self._names
            for alias in (f"{field}_parameters", f"{field}_phase", f"{field}_changed")
        )
        self.oracle_fields = tuple(
            dict.fromkeys(
                (
                    "phase",
                    "drift_parameters",
                    "drift_changed",
                    "task_id",
                    "boundary",
                    *aliases,
                    *inner_fields,
                )
            )
        )
        inner_metadata: Metadata = cast(Any, env).metadata
        self.metadata: Metadata = {
            **inner_metadata,
            "env_id": env_id if env_id is not None else inner_metadata["env_id"],
            "oracle_exposure": (),
            "configuration": {
                **inner_metadata.get("configuration", {}),
                **self.drift_configuration,
            },
        }

    def _require_driftable(self) -> None:
        if not self._driftable:
            raise ValueError(
                f"{self._label} has no driftable parameters, so drift cannot be attached to it."
            )

    def _bare_field(self) -> str:
        """The parameter a bare process drives: the declared default, else the only one."""
        self._require_driftable()
        field = getattr(self.env, "default_drift_field", None)
        if field is None:
            if len(self._driftable) != 1:
                raise ValueError(
                    f"{type(self.env).__name__} needs an explicit drift mapping, e.g. "
                    f"drift={{{self._driftable[0]!r}: process}}; available parameters are "
                    f"{self._driftable}."
                )
            field = self._driftable[0]
        return cast(str, field)

    @property
    def drift_configuration(self) -> dict[str, Any]:
        """JSON-ready drift configuration in the established metadata shape.

        ``drift`` is the primary field's process configuration (``{"type":
        "stationary"}`` when it does not drift), every other parameter gets a
        ``{field}_drift`` entry, and ``drifts`` maps each drifted field to its
        process. Checkpoint restores compare this configuration exactly.
        """
        stationary: dict[str, Any] = {"type": "stationary"}

        def of(name: str) -> dict[str, Any]:
            process = self.drifts.get(name)
            return dict(process.configuration) if process is not None else stationary

        configuration: dict[str, Any] = {"drift": of(self._primary)}
        for name in self._names:
            if name != self._primary:
                configuration[f"{name}_drift"] = of(name)
        configuration["drifts"] = {name: of(name) for name in self._fields}
        return configuration

    def with_drift(self, drift: Drift | Mapping[str, Drift]) -> "DriftEnv[StateT, ParamsT]":
        """Return an equivalent environment with ``drift`` rebound.

        A bare process replaces the drift of ``default_drift_field`` (or of the
        only driftable parameter) and keeps every other field's, for example a
        partner-policy drift survives rebinding the goal. It raises
        ``ValueError``, as the constructor does, when neither exists. A mapping
        replaces exactly the named fields.
        """
        updates = dict(drift) if isinstance(drift, Mapping) else {self._bare_field(): drift}
        return type(self)(self.env, {**self.drifts, **updates}, env_id=self.metadata["env_id"])

    def __getattr__(self, name: str) -> Any:
        """Forward public attributes (``size``, ``num_partners``) to the environment."""
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        if name == "reset":
            raise AttributeError(
                "Perpetua environments have no reset: start a lifetime once with "
                "env.init(jax.random.key(seed)) and keep stepping; episode ends "
                "arrive as info['respawn']."
            )
        return getattr(self.env, name)

    def _params_of(self, drift: Mapping[str, DriftState]) -> ParamsT:
        """Effective parameters: drift values folded in, then ``derive`` applied."""
        raw = _apply(self._params, {name: s.value for name, s in drift.items()})
        derive = getattr(self.env, "derive", None)
        return cast(ParamsT, raw if derive is None else derive(raw))

    def _space_change(
        self, env_state: StateT, previous: ParamsT, current: ParamsT, mask: jax.Array
    ) -> jax.Array:
        """Per-agent event: drift changed this agent's valid-action set this tick.

        The agent's mask under the previous parameters is compared with its
        mask under the current ones, both at the same world state, through
        :func:`perpetua.spaces.mask_change`. So the event fires only for agents
        whose mask the drift changed, and never because an agent moved next to
        a wall. The comparison only runs when a declared ``mask_params`` field
        moved; otherwise no agent's mask can have changed through drift.
        """
        quiet = jnp.zeros((self.num_agents,), dtype=jnp.bool_)
        fields = tuple(cast("tuple[str, ...]", getattr(self.env, "mask_params", ())))
        if not fields:
            return quiet
        before, after = param_paths(previous), param_paths(current)
        moved = jnp.stack(
            [jnp.any(jnp.ravel(before[field]) != jnp.ravel(after[field])) for field in fields]
        ).any()
        if self.metadata.get("execution") == "host" and not bool(moved):
            # Host observations are Python calls, and lax.cond traces both
            # branches. Do not inspect every task on a tick with no mask change.
            return quiet

        def compare(operand: None) -> jax.Array:
            del operand
            earlier = self.env.observe(env_state, previous)[1]
            return mask_change(earlier["action_mask"], mask)

        def unchanged(operand: None) -> jax.Array:
            del operand
            return quiet

        return cast(jax.Array, jax.lax.cond(moved, compare, unchanged, None))

    def init(self, key: jax.Array) -> tuple[DriftEnvState[StateT], jax.Array, Info]:
        """Initialize dynamics once and each drifted parameter from its default."""
        # The environment receives the lifetime seed unchanged. The drift
        # processes' init keys and the drift stream both come from a tagged
        # fold of the same seed, so no plain split of it that an environment
        # makes can equal a drift key.
        if isinstance(key, int):
            raise TypeError(
                f"init needs a JAX PRNG key, not the integer {key}; pass jax.random.key({key})."
            )
        env_key = key
        init_key, drift_key = jax.random.split(jax.random.fold_in(key, _DRIFT_TAG))
        defaults = param_paths(self._params)
        names = sorted(self.drifts)
        init_keys = [init_key] if len(names) == 1 else list(jax.random.split(init_key, len(names)))
        drift = {
            name: self.drifts[name].init(defaults[name], init_keys[index])
            for index, name in enumerate(names)
        }
        state, obs, info = self.env.init(env_key, self._params_of(drift))
        if self.metadata.get("execution") != "host":
            state = _strong(state)
        initial: Info = {
            **info,
            "t": Clock.from_int(0),
            "space_change": jnp.zeros(self.num_agents, dtype=jnp.bool_),
        }
        return (
            DriftEnvState(
                state,
                drift,
                drift_key,
                jnp.asarray(False),
                {name: jnp.asarray(False) for name in self._fields},
                Clock.from_int(0),
            ),
            obs,
            initial,
        )

    def observe(self, state: DriftEnvState[StateT]) -> tuple[jax.Array, Info]:
        """Return the environment's observation with the lifetime clock.

        ``space_change`` is False here (observing is not a transition), so the
        info pytree is identical across ``init``, ``observe`` and ``step`` and
        a lifetime carry keeps a fixed structure.
        """
        obs, info = self.env.observe(state.env, self._params_of(state.drift))
        observed: Info = {
            **info,
            "t": state.t,
            "space_change": jnp.zeros(self.num_agents, dtype=jnp.bool_),
        }
        return obs, observed

    def step(
        self, state: DriftEnvState[StateT], action: ArrayLike, key: jax.Array
    ) -> tuple[DriftEnvState[StateT], jax.Array, jax.Array, Info]:
        """Step under current parameters, drift, then observe on the same clock.

        A drift process that declares ``endogenous = True`` receives the
        environment's optional ``signal(previous_state, action, next_state)``
        as a third argument. The signal is privileged: it only reaches drift
        processes, never observations or info. Exogenous processes are called
        as ``step(state, key)``.

        An environment may define ``after_drift(state, previous, current, key)``,
        called once the new parameters are known and before observing, to react
        to a parameter change on the same tick (task switching uses it to
        restart the incoming task). ``declared_switches(previous, current)``
        may flag a drifted field as switched on this tick in addition to its
        process's own phase change.
        """
        previous = self._params_of(state.drift)
        env_state, reward = self.env.step(state.env, action, key, previous)
        if self.metadata.get("execution") != "host":
            env_state = _strong(env_state)
        drift_key, draw_key = jax.random.split(state.drift_key)
        keys = (
            [draw_key]
            if len(self._fields) == 1
            else list(jax.random.split(draw_key, len(self._fields)))
        )
        signal_fn = getattr(self.env, "signal", None)
        signal = signal_fn(state.env, action, env_state) if signal_fn is not None else None
        drift: dict[str, DriftState] = {}
        for index, name in enumerate(self._fields):
            process = self.drifts[name]
            if getattr(process, "endogenous", False):
                # Only endogenous processes take the third, privileged argument.
                drift[name] = cast(Any, process).step(state.drift[name], keys[index], signal)
            else:
                drift[name] = process.step(state.drift[name], keys[index])
        field_changed = {name: switched(state.drift[name], drift[name]) for name in self._fields}
        # The returned observation reflects the parameters after drift: drift
        # happens inside the tick, before observing.
        current = self._params_of(drift)
        declared = getattr(self.env, "declared_switches", None)
        if declared is not None:
            for name, flag in declared(previous, current).items():
                if name in field_changed:
                    field_changed[name] = field_changed[name] | flag
        changed = (
            jnp.stack(list(field_changed.values()))
            if self._fields
            else jnp.zeros((), dtype=jnp.bool_)
        )
        after_drift = getattr(self.env, "after_drift", None)
        if after_drift is not None:
            env_state = after_drift(
                env_state, previous, current, jax.random.fold_in(key, _AFTER_DRIFT_TAG)
            )
            if self.metadata.get("execution") != "host":
                env_state = _strong(env_state)
        next_state = DriftEnvState(
            env_state,
            drift,
            drift_key,
            jnp.any(changed),
            field_changed,
            state.t.increment(),
        )
        obs, info = self.env.observe(env_state, current)
        observed: Info = {
            **info,
            "t": next_state.t,
            "space_change": self._space_change(env_state, previous, current, info["action_mask"]),
        }
        return next_state, obs, reward, observed

    def privileged(self, state: DriftEnvState[StateT]) -> dict[str, jax.Array]:
        """Drift diagnostics plus the environment's own privileged fields.

        Every parameter field gets ``{field}_parameters``, ``{field}_phase`` and
        ``{field}_changed``; an undrifted field reports its default value as a
        stationary process would (phase 0, never changed). The aggregate
        ``phase`` and ``drift_parameters`` report the primary field (the
        environment's declared default drifting field, otherwise the first
        drifted or, failing that, the first parameter), so single-field
        environments keep their names, shapes and values. The aggregate
        ``drift_changed`` is true when any drifting field declared a switch
        (its process changed phase, or the environment's ``declared_switches``
        named it).
        ``task_id`` and ``boundary`` are the generic names for "which
        piecewise-stationary segment is active" and "a declared switch happened
        on this tick": ``task_id`` is the primary phase (a repeating table index for ``Cyclic``
        and ``Schedule``, a switch count for the other processes) unless the environment
        defines ``task_id(params)`` (a task switch's active index) or reports
        its own (a MORPHEUS regime), and ``boundary`` is ``drift_changed``.
        """
        defaults = param_paths(self._params)
        fields: dict[str, jax.Array] = {}
        for name in self._names:
            if name in state.drift:
                value, phase = state.drift[name].value, state.drift[name].phase
                changed = state.field_changed[name]
            else:
                value = jnp.asarray(defaults[name])
                phase, changed = jnp.uint32(0), jnp.asarray(False)
            fields[f"{name}_parameters"] = value
            fields[f"{name}_phase"] = phase
            fields[f"{name}_changed"] = changed
        primary = self._primary
        aggregate = {
            "phase": fields[f"{primary}_phase"],
            "drift_parameters": fields[f"{primary}_parameters"],
            "drift_changed": state.drift_changed,
            "task_id": (
                task_id(self._params_of(state.drift))
                if (task_id := getattr(self.env, "task_id", None)) is not None
                else fields[f"{primary}_phase"].astype(jnp.int32)
            ),
            "boundary": state.drift_changed,
        }
        return {**aggregate, **fields, **self.env.privileged(state.env)}
