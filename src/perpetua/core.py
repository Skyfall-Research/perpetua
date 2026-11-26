"""Shared, shape-stable contracts for native continual environments."""

import operator
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import (
    Any,
    Literal,
    NamedTuple,
    NoReturn,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

__all__ = [
    "Clock",
    "EnvInfo",
    "Info",
    "JSONValue",
    "Metadata",
    "Environment",
    "Oracle",
    "OneHotObservation",
]


def as_int(
    value: object, message: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Return ``value`` as a Python int, or raise ``ValueError`` saying what was received.

    Accepts Python ints and NumPy or JAX integer scalars (through
    ``operator.index``, so what ``np.prod`` or ``jnp.argmax`` return works);
    rejects bools, floats and arrays. ``minimum`` and ``maximum`` are inclusive.
    ``message`` states the requirement, for example ``"ticks must be a positive
    integer."``; the received value is appended.
    """
    received = f"{message} Got {type(value).__name__} {value!r}."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(received)
    try:
        result = operator.index(cast(Any, value))
    except TypeError:
        raise ValueError(received) from None
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(received)
    return result


@contextmanager
def requires_extra(extra: str, feature: str) -> Generator[None]:
    """Turn a missing third-party module inside the block into an install hint.

    ``extra`` is the name of Perpetua's optional extra that provides the
    module; ``feature`` says what needed it. Perpetua's own modules and other
    errors pass through unchanged.
    """
    try:
        yield
    except ModuleNotFoundError as error:
        package = (error.name or "perpetua").partition(".")[0]
        if package == "perpetua":
            raise
        raise ImportError(
            f"{feature} needs the {package!r} package, which Perpetua's {extra} extra "
            f'installs: pip install "perpetua-rl[{extra}]".',
            name=error.name,
        ) from error


class Clock(NamedTuple):
    """Unsigned 64-bit tick count represented by two scalar uint32 arrays.

    This avoids changing JAX's process-global precision policy. ``to_int`` is
    for host reporting only; ``increment`` and ``as_float`` work under JIT.
    """

    hi: jax.Array
    lo: jax.Array

    @classmethod
    def from_int(cls, value: int) -> "Clock":
        """Construct a clock from a host integer in [0, 2**64)."""
        value = as_int(
            value, "Clock must be an integer in [0, 2**64).", minimum=0, maximum=2**64 - 1
        )
        return cls(jnp.uint32(value >> 32), jnp.uint32(value & 0xFFFFFFFF))

    def increment(self) -> "Clock":
        """Advance one tick using unsigned carry arithmetic."""
        lo = self.lo + jnp.uint32(1)
        return Clock(self.hi + (lo == 0).astype(jnp.uint32), lo)

    def add(self, other: "Clock") -> "Clock":
        """Add uint64 limb counters under JIT/vmap, with modulo-2**64 arithmetic."""
        lo = self.lo + other.lo
        return Clock(self.hi + other.hi + (lo < self.lo).astype(jnp.uint32), lo)

    def to_int(self) -> int:
        """Transfer the counter to the host; do not call from compiled code."""
        return (int(self.hi) << 32) | int(self.lo)

    def as_float(self) -> jax.Array:
        """Return an approximate float32 count for streaming reward averages."""
        return self.hi.astype(jnp.float32) * 4294967296.0 + self.lo.astype(jnp.float32)


class EnvInfo(TypedDict):
    """Info a parameter environment produces; the wrapper adds the clock.

    Every entry is a boolean array:

    ``action_mask``
        Which actions are legal now. For a discrete (``MultiDiscrete``) action
        space it has shape ``(agents, actions)``, ``True`` for a legal action,
        and every active agent has at least one. A continuous (``Box``) action
        has no per-action legality: most adapters (generic gymnax, MJX, the
        host lifts) return an all-``True`` ``(agents, 1)`` placeholder, and the
        JaxMARL continuous tasks mark which of the ``(agents, action_dim)``
        dimensions are valid. Do not read a continuous environment's mask as
        legality.
    ``active_mask``
        Shape ``(agents,)``: ``True`` for a slot the learner controls on this
        tick. The observation of an inactive slot is a placeholder, so the
        bundled learners do not learn from it.
    ``respawn``
        Shape ``(agents,)``: ``True`` for a slot that respawned on this
        transition (a death or a reached goal). There is no reset and no
        done: the lifetime continues, and this is the only episode-end signal.
    ``space_change``
        Shape ``(agents,)`` event, true for an agent when drift changed that
        agent's valid-action set (its mask before and after the drift, at the
        same world state). ``DriftEnv`` always provides it; it is
        ``NotRequired`` only for hand-written environments outside
        ``DriftEnv``.
    """

    action_mask: jax.Array
    active_mask: jax.Array
    respawn: jax.Array
    space_change: NotRequired[jax.Array]


class Info(EnvInfo):
    """Agent-visible data describing the returned observation and next action.

    Adds ``t``, the lifetime :class:`Clock` at this observation, and, only
    through :class:`Oracle`, the explicitly exposed privileged fields under
    ``oracle``. Oracle data is absent unless explicitly enabled. See
    :class:`EnvInfo` for the masks and events.
    """

    t: Clock
    oracle: NotRequired[dict[str, jax.Array]]


type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


class Metadata(TypedDict):
    """Static reporting metadata; never placed in the device scan carry.

    ``env_id`` and ``oracle_exposure`` (the privileged fields an
    :class:`Oracle` shows, recorded for audit) are required. Optional keys:
    ``configuration`` (JSON-ready settings, compared exactly when a checkpoint
    is restored); ``execution`` (``"jax"`` for a functional world whose whole
    state is the carry, ``"host"`` for a mutable simulator outside it);
    ``checkpointable`` (a complete carry can be saved; defaults to True for
    ``"jax"`` and False for ``"host"``); ``forkable`` (a state can be stepped
    more than once, as a regret reference's shadow step does; defaults to True,
    False for the single-use state of a compiled bridge).
    """

    env_id: str
    oracle_exposure: tuple[str, ...]
    configuration: NotRequired[dict[str, JSONValue]]
    execution: NotRequired[Literal["jax", "host"]]
    checkpointable: NotRequired[bool]
    forkable: NotRequired[bool]


StateT = TypeVar("StateT")


class Environment(Protocol[StateT]):
    """Shared lifetime interface; metadata distinguishes JAX and stateful host execution.

    Stepping is functional and there is no reset and no done: ``init`` starts
    one lifetime, ``step`` advances it, and episode ends arrive as
    ``info["respawn"]``. See ``docs/native-api.md`` for the full guide.
    """

    @property
    def num_agents(self) -> int:
        """Number of agent slots, fixed for the lifetime."""
        ...

    @property
    def observation_space(self) -> spaces.Space[Any]:
        """Gymnasium space of the ``(agents, features)`` observation."""
        ...

    @property
    def action_space(self) -> spaces.Space[Any]:
        """``MultiDiscrete`` (one integer per agent) or ``Box`` (continuous)."""
        ...

    @property
    def metadata(self) -> Metadata:
        """Static reporting metadata: id, Oracle exposure, configuration, capabilities."""
        ...

    @property
    def oracle_fields(self) -> tuple[str, ...]:
        """Privileged field names an :class:`Oracle` may expose (never in public info)."""
        ...

    def init(self, key: jax.Array) -> tuple[StateT, jax.Array, Info]:
        """Start one lifetime from a PRNG key: ``(state, obs, info)``."""
        ...

    def step(
        self, state: StateT, action: ArrayLike, key: jax.Array
    ) -> tuple[StateT, jax.Array, jax.Array, Info]:
        """Advance one tick: ``(state, action, key) -> (state, obs, reward, info)``.

        ``action`` is an integer ``(agents,)`` vector for discrete spaces and
        the ``Box`` shape for continuous ones; ``reward`` is ``(agents,)``
        float32. There is no reset and no done flag.
        """
        ...

    def observe(self, state: StateT) -> tuple[jax.Array, Info]:
        """Observation and info of a valid state, without advancing it."""
        ...

    def privileged(self, state: StateT) -> dict[str, jax.Array]:
        """Evaluator-only diagnostics, keyed by ``oracle_fields``."""
        ...


def _refuse_rebinding(wrapper: object) -> NoReturn:
    name = type(wrapper).__name__
    raise TypeError(
        f"{name} wraps an environment, so drift cannot be rebound through it. Compose the "
        f"drift first and wrap afterwards: {name}(pp.compose(env, drift), ...) instead of "
        f"pp.compose({name}(env, ...), drift)."
    )


class Oracle[StateT]:
    """Explicitly expose selected privileged fields and record that choice.

    An Oracle changes the public info pytree, so compile a separate runner for
    each exposure configuration. It never changes environment dynamics or RNG.

    ``boundary`` is the field for "a declared switch happened on this tick".
    ``drift_changed`` is a deprecated alias with the same value.
    """

    def close(self) -> None:
        """Close the wrapped environment if it holds external resources."""
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def __init__(self, env: Environment[StateT], *, expose: list[str]) -> None:
        if isinstance(expose, str):
            raise TypeError(
                f"expose must be a list or tuple of field names, got the string {expose!r}; "
                f"write expose=[{expose!r}]."
            )
        if len(expose) != len(set(expose)):
            raise ValueError("Oracle exposure fields must be unique.")
        for field in expose:
            if field not in env.oracle_fields:
                raise ValueError(
                    f"Unknown oracle field: {field}; this environment offers "
                    f"{sorted(env.oracle_fields)}."
                )
        if "drift_changed" in expose:
            warnings.warn(
                "The Oracle field 'drift_changed' is deprecated; expose 'boundary', "
                "which holds the same value.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.env = env
        self.expose = tuple(expose)
        self.num_agents = env.num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.oracle_fields = env.oracle_fields
        self.metadata: Metadata = env.metadata.copy()
        self.metadata["oracle_exposure"] = tuple(
            dict.fromkeys((*env.metadata["oracle_exposure"], *expose))
        )

    def with_drift(self, drift: object) -> NoReturn:
        """Refuse to rebind drift through the wrapper; compose before wrapping."""
        _refuse_rebinding(self)

    def _info(self, state: StateT, info: Info) -> Info:
        privileged = self.env.privileged(state)
        result: Info = info.copy()
        result["oracle"] = {**info.get("oracle", {}), **{k: privileged[k] for k in self.expose}}
        return result

    def init(self, key: jax.Array) -> tuple[StateT, jax.Array, Info]:
        """Initialize one lifetime with the selected public oracle fields."""
        state, obs, info = self.env.init(key)
        return state, obs, self._info(state, info)

    def step(
        self, state: StateT, action: ArrayLike, key: jax.Array
    ) -> tuple[StateT, jax.Array, jax.Array, Info]:
        """Advance the wrapped dynamics without resetting at drift."""
        state, obs, reward, info = self.env.step(state, action, key)
        return state, obs, reward, self._info(state, info)

    def observe(self, state: StateT) -> tuple[jax.Array, Info]:
        """Observe a valid internal state without advancing it."""
        obs, info = self.env.observe(state)
        return obs, self._info(state, info)

    def privileged(self, state: StateT) -> dict[str, jax.Array]:
        """Return evaluator-only diagnostics, not ordinary agent observations."""
        return self.env.privileged(state)


class OneHotObservation[StateT]:
    """Append an exposed Oracle field to every observation as a one-hot code.

    For protocols that give the learner the task identity, such as Continual
    World's one-hot task input: ``OneHotObservation(pp.Oracle(env,
    expose=["task_id"]), field="task_id", size=10)``. The field must already be
    exposed by an Oracle, so the privilege is recorded in
    ``metadata["oracle_exposure"]``; the wrapper also records the appended
    encoding in ``metadata["configuration"]["observation_appends"]``.
    Observations must be ``(agents, features)``; the code is the field's value
    per agent (a scalar field is shared by every slot), and values outside
    ``[0, size)`` encode as all zeros. A fractional value inside the range is
    truncated to its index. The returned observation is float32.
    """

    def __init__(self, env: Environment[StateT], *, field: str, size: int) -> None:
        if field not in env.metadata["oracle_exposure"]:
            raise ValueError(
                f"{field!r} is privileged: expose it first with pp.Oracle(env, expose=[{field!r}])."
            )
        size = as_int(size, "size must be a positive integer.", minimum=1)
        space = env.observation_space
        if not isinstance(space, spaces.Box) or len(space.shape) != 2:
            raise ValueError("OneHotObservation needs (agents, features) Box observations.")
        box = cast(Any, space)
        low = np.asarray(box.low, dtype=np.float32)
        high = np.asarray(box.high, dtype=np.float32)
        agents = int(low.shape[0])
        self.env = env
        self.field = field
        self.size = size
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.oracle_fields = env.oracle_fields
        self.observation_space = spaces.Box(
            np.concatenate([low, np.zeros((agents, size), dtype=np.float32)], axis=1),
            np.concatenate([high, np.ones((agents, size), dtype=np.float32)], axis=1),
            dtype=np.float32,
        )
        self.metadata: Metadata = env.metadata.copy()
        configuration = dict(env.metadata.get("configuration", {}))
        appends = cast(
            list[JSONValue], list(cast(Any, configuration.get("observation_appends", [])))
        )
        appends.append({"field": field, "encoding": "one_hot", "size": size})
        configuration["observation_appends"] = appends
        self.metadata["configuration"] = configuration

    def with_drift(self, drift: object) -> NoReturn:
        """Refuse to rebind drift through the wrapper; compose before wrapping."""
        _refuse_rebinding(self)

    def _append(self, obs: jax.Array, info: Info) -> jax.Array:
        oracle: dict[str, jax.Array] = info.get("oracle", {})
        raw = oracle[self.field]
        inside = (raw >= 0) & (raw < self.size)
        value = jnp.broadcast_to(jnp.where(inside, raw.astype(jnp.int32), -1), (obs.shape[0],))
        code = jax.nn.one_hot(value, self.size, dtype=jnp.float32)
        return jnp.concatenate([obs.astype(jnp.float32), code], axis=-1)

    def init(self, key: jax.Array) -> tuple[StateT, jax.Array, Info]:
        """Initialize the wrapped lifetime and append the code to the observation."""
        state, obs, info = self.env.init(key)
        return state, self._append(obs, info), info

    def step(
        self, state: StateT, action: ArrayLike, key: jax.Array
    ) -> tuple[StateT, jax.Array, jax.Array, Info]:
        """Advance the wrapped lifetime; dynamics and RNG are unchanged."""
        state, obs, reward, info = self.env.step(state, action, key)
        return state, self._append(obs, info), reward, info

    def observe(self, state: StateT) -> tuple[jax.Array, Info]:
        """Observe without advancing, with the code appended."""
        obs, info = self.env.observe(state)
        return self._append(obs, info), info

    def privileged(self, state: StateT) -> dict[str, jax.Array]:
        """Evaluator-only diagnostics of the wrapped environment."""
        return self.env.privileged(state)

    def close(self) -> None:
        """Close the wrapped environment if it holds external resources."""
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
