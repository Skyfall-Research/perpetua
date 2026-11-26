"""Reusable exogenous drift over fixed-shape float32 parameter arrays."""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import math
from collections.abc import Sequence as Seq
from functools import partial
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from perpetua.core import JSONValue, as_int

__all__ = [
    "DriftState",
    "ScheduleTable",
    "SegmentCounter",
    "switched",
    "Drift",
    "Stationary",
    "Cyclic",
    "ScheduleData",
    "schedule",
    "stack_schedules",
    "Schedule",
    "RandomWalk",
    "Procedural",
    "Sequence",
    "Sum",
    "Accumulate",
]


class DriftState(NamedTuple):
    """Private parameter value, phase and within-phase timer.

    ``phase`` changes exactly when the process declares a switch, and that is
    all that is common to every process. For ``Cyclic`` and ``Schedule`` it is
    the index of the current table row and wraps around, so a revisited row
    has the same phase. For ``Procedural``, ``Sequence``, ``Sum`` and
    ``Accumulate`` it counts switches and never repeats. ``Stationary`` and
    ``RandomWalk`` stay at phase 0. ``elapsed`` counts ticks since the last
    switch; ``RandomWalk``, which never switches, leaves it at 0. ``Accumulate``
    counts ticks since its last band crossing, and since the start when it has
    crossed none, including when it has no bands.

    ``parts`` holds the sub-process states of a combinator (``Sequence``,
    ``Sum``), followed for ``Sequence`` by its :class:`SegmentCounter`, or a
    ``Schedule``'s :class:`ScheduleTable`. It is empty for every other process.
    """

    value: jax.Array
    phase: jax.Array
    elapsed: jax.Array
    parts: tuple["DriftState | ScheduleTable | SegmentCounter", ...] = ()


class ScheduleTable(NamedTuple):
    """A ``Schedule``'s phase values, ticks per phase and number of phases in use."""

    values: jax.Array
    period: jax.Array
    phases: jax.Array


class SegmentCounter(NamedTuple):
    """The ``Sequence`` segment that is running and the ticks spent in it."""

    segment: jax.Array
    spent: jax.Array


def switched(before: DriftState, after: DriftState) -> jax.Array:
    """Bool scalar: the drift process declared a discrete switch on this tick.

    A switch is any change of ``phase`` (``Cyclic`` and ``Procedural`` switch
    on their period). Gradual processes such as ``RandomWalk`` never switch, so
    switch-based metrics such as time-to-recover do not apply to them; raw
    parameter movement stays available to evaluators through the Oracle.
    """
    return after.phase != before.phase


class Drift(Protocol):
    """Drift contract; parameter shapes must not change during a run.

    A process may set the class attribute ``endogenous = True`` (not part of the
    protocol, so existing processes stay structural). The protocol declares the
    two-argument ``step``; the wrapper calls such a process with a third
    ``signal`` argument, the environment's privileged signal for the tick, so
    drift can depend on the agent's own behaviour (AgarCL-style). Exogenous
    processes ignore it entirely.
    """

    def init(self, value: jax.Array, key: jax.Array) -> DriftState: ...

    def step(self, state: DriftState, key: jax.Array) -> DriftState: ...

    @property
    def configuration(self) -> dict[str, JSONValue]: ...


class Stationary:
    """Hold one parameter value: the environment's initial one by default.

    ``Stationary(value)`` holds ``value`` instead, which must have the
    parameter's shape. As the last segment of a :class:`Sequence` it ends a
    schedule on a chosen value without further switches.
    """

    def __init__(self, value: ArrayLike | None = None) -> None:
        if value is None:
            self.value: jax.Array | None = None
            return
        data = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError("Stationary value must be finite.")
        self.value = jnp.asarray(data)

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Initialize private drift state; the key is unused in this process."""
        del key
        if self.value is None:
            return DriftState(value.astype(jnp.float32), jnp.uint32(0), jnp.uint32(0))
        if self.value.shape != value.shape:
            raise ValueError(
                f"Stationary value has shape {tuple(self.value.shape)}, the parameter has "
                f"shape {tuple(value.shape)}."
            )
        return DriftState(self.value, jnp.uint32(0), jnp.uint32(0))

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Return unchanged parameters without consuming randomness."""
        del key
        return state

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Host-side configuration recorded with checkpoints and results."""
        if self.value is None:
            return {"type": "stationary"}
        return {"type": "stationary", "value": cast(JSONValue, np.asarray(self.value).tolist())}


class Cyclic:
    """Cycle through parameter values after ``period`` transitions per phase.

    ``values`` has shape ``(phases, *parameter_shape)``. Phase zero is used at
    initialization; the next phase governs actions after the switching tick.
    """

    def __init__(self, values: ArrayLike | Seq[ArrayLike | Seq[float]], period: int) -> None:
        period = as_int(
            period, "Cyclic period must be an integer in [1, 2**32).", minimum=1, maximum=2**32 - 1
        )
        data = np.asarray(values, dtype=np.float32)
        if data.ndim < 1 or data.shape[0] == 0 or not np.all(np.isfinite(data)):
            raise ValueError("Cyclic values must contain finite, equally shaped phases.")
        self.values: jax.Array = jnp.asarray(data)
        self.period = period

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Select the first phase and validate its parameter shape."""
        del key
        if self.values.shape[1:] != value.shape:
            expected = tuple(value.shape)
            if expected == ():
                hint = "write each phase as a plain number, e.g. Cyclic([9.8, 12.0], period)"
            elif expected == (1,):
                hint = "write each phase as a 1-tuple, e.g. Cyclic([(9.8,), (12.0,)], period)"
            else:
                hint = f"write each phase as an array of shape {expected}"
            raise ValueError(
                "Cyclic parameter shape does not match the environment: each phase has "
                f"shape {tuple(self.values.shape[1:])}, the parameter has shape {expected}; "
                f"{hint}."
            )
        return DriftState(self.values[0], jnp.uint32(0), jnp.uint32(0))

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Advance the private phase timer entirely as device data."""
        del key
        elapsed = state.elapsed + jnp.uint32(1)
        switch = elapsed == jnp.uint32(self.period)
        phase = (state.phase + switch.astype(jnp.uint32)) % self.values.shape[0]
        return DriftState(self.values[phase], phase, jnp.where(switch, jnp.uint32(0), elapsed))

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record the complete phase table and switching period on the host."""
        return {"type": "cyclic", "values": np.asarray(self.values).tolist(), "period": self.period}


class ScheduleData(NamedTuple):
    """A piecewise-constant schedule as data, so lifetimes can each carry their own.

    ``values`` is float32 ``(phases, *parameter_shape)``; only the first ``length``
    phases are used, so schedules of different lengths stack into one batch.
    Each phase lasts ``period`` transitions, and ``offset`` is how many
    transitions into the schedule a lifetime starts. All fields may carry a
    leading batch axis for ``jax.vmap``.
    """

    values: jax.Array
    period: jax.Array
    length: jax.Array
    offset: jax.Array


def schedule(
    values: ArrayLike | Seq[ArrayLike | Seq[float]],
    period: int,
    *,
    offset: int = 0,
    phases: int | None = None,
) -> ScheduleData:
    """Build one :class:`ScheduleData`, padded to ``phases`` rows if given."""
    data = np.asarray(values, dtype=np.float32)
    if data.ndim < 1 or data.shape[0] == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Schedule values must contain finite, equally shaped phases.")
    period = as_int(
        period, "Schedule period must be an integer in [1, 2**32).", minimum=1, maximum=2**32 - 1
    )
    offset = as_int(
        offset, "Schedule offset must be an integer in [0, 2**32).", minimum=0, maximum=2**32 - 1
    )
    count = data.shape[0]
    rows = count
    if phases is not None:
        rows = as_int(
            phases,
            f"phases must be an integer at least the number of schedule values ({count}).",
            minimum=count,
        )
    padded = np.concatenate((data, np.repeat(data[-1:], rows - count, axis=0)), axis=0)
    return ScheduleData(
        jnp.asarray(padded), jnp.uint32(period), jnp.uint32(count), jnp.uint32(offset)
    )


def stack_schedules(schedules: Seq[ScheduleData]) -> ScheduleData:
    """Stack schedules (padding to the longest) into one batch for ``jax.vmap``."""
    if not schedules:
        raise ValueError("stack_schedules needs at least one schedule.")
    rows = max(int(item.values.shape[0]) for item in schedules)

    def pad(item: ScheduleData) -> jax.Array:
        extra = rows - int(item.values.shape[0])
        return jnp.concatenate((item.values, jnp.repeat(item.values[-1:], extra, axis=0)))

    return ScheduleData(
        jnp.stack([pad(item) for item in schedules]),
        jnp.stack([item.period for item in schedules]),
        jnp.stack([item.length for item in schedules]),
        jnp.stack([item.offset for item in schedules]),
    )


class Schedule:
    """Cyclic drift whose phase table lives in the drift state, not the process.

    With ``Cyclic`` every lifetime of a compiled batch follows the same table,
    because the table is part of the program. A ``Schedule`` keeps the table,
    period, phase count and start offset in its state (in ``parts``), so
    :func:`perpetua.runtime.assign_schedule` can give each vmapped lifetime its
    own schedule: one lifetime in phase 3 next to another in phase 17, all in
    one compiled program, with no recompilation when schedules change. The
    constructor's schedule is the default for lifetimes that are not assigned
    one. A change of phase is a declared switch, exactly as for ``Cyclic``.
    """

    def __init__(
        self,
        values: ArrayLike | Seq[ArrayLike | Seq[float]],
        period: int,
        *,
        offset: int = 0,
        phases: int | None = None,
    ) -> None:
        self.default = schedule(values, period, offset=offset, phases=phases)
        self.phases = int(self.default.values.shape[0])

    def start(self, data: ScheduleData) -> DriftState:
        """The drift state at a schedule's start offset (traceable, vmappable)."""
        if data.values.shape[0] != self.phases:
            raise ValueError(f"Schedules for this process must have {self.phases} phase rows.")
        expected = self.default.values.shape[1:]
        if data.values.shape[1:] != expected:
            raise ValueError(
                f"Schedule values must have shape {expected} per phase, "
                f"got {data.values.shape[1:]}."
            )
        period = jnp.maximum(jnp.asarray(data.period, jnp.uint32), jnp.uint32(1))
        count = jnp.clip(jnp.asarray(data.length, jnp.uint32), jnp.uint32(1), self.phases)
        offset = jnp.asarray(data.offset, jnp.uint32)
        phase = (offset // period) % count
        table = ScheduleTable(data.values.astype(jnp.float32), period, count)
        return DriftState(table.values[phase], phase, offset % period, (table,))

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Start the default schedule and validate its parameter shape."""
        del key
        if self.default.values.shape[1:] != value.shape:
            raise ValueError(
                "Schedule parameter shape does not match the environment: each phase has "
                f"shape {tuple(self.default.values.shape[1:])}, the parameter has shape "
                f"{tuple(value.shape)}."
            )
        return self.start(self.default)

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Advance the timer; switch to the next of ``length`` phases each ``period``."""
        del key
        table = cast(ScheduleTable, state.parts[0])
        elapsed = state.elapsed + jnp.uint32(1)
        switch = elapsed >= table.period
        phase = (state.phase + switch.astype(jnp.uint32)) % table.phases
        return DriftState(
            table.values[phase], phase, jnp.where(switch, jnp.uint32(0), elapsed), state.parts
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """The default schedule. Assigned schedules are state: checkpoints keep them."""
        return {
            "type": "schedule",
            "values": np.asarray(self.default.values).tolist(),
            "period": int(self.default.period),
            "length": int(self.default.length),
            "offset": int(self.default.offset),
            "per_lifetime": "assigned schedules live in the drift state",
        }


class RandomWalk:
    """Add independent Gaussian noise, clipped to fixed lower/upper bounds.

    ``low`` and ``high`` are required and must broadcast to the parameter
    shape. The environment's starting value is clipped into ``[low, high]`` at
    tick 0, so bounds that exclude it move the parameter before any noise is
    added. ``scale`` is the per-tick standard deviation; zero gives a
    stationary clipped process.
    """

    def __init__(
        self,
        *,
        low: ArrayLike | tuple[int, int],
        high: ArrayLike | tuple[int, int],
        scale: float = 0.1,
    ) -> None:
        scale = float(scale)
        lower, upper = np.broadcast_arrays(
            np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Random-walk bounds must be finite.")
        if (
            np.any(lower > upper)
            or not math.isfinite(scale)
            or not 0 <= scale <= np.finfo(np.float32).max
        ):
            raise ValueError("Random-walk bounds must be ordered and scale finite/nonnegative.")
        self.low: jax.Array = jnp.asarray(lower)
        self.high: jax.Array = jnp.asarray(upper)
        self.scale = scale

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Initialize the clipped value without drawing drift noise."""
        del key
        if jnp.broadcast_shapes(value.shape, self.low.shape, self.high.shape) != value.shape:
            raise ValueError("Random-walk bounds must broadcast to the parameter shape.")
        value = jnp.clip(value.astype(jnp.float32), self.low, self.high)
        return DriftState(value, jnp.uint32(0), jnp.uint32(0))

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Consume a fresh drift key and preserve the parameter shape."""
        noise = jax.random.normal(key, shape=state.value.shape, dtype=jnp.float32)
        value = jnp.clip(state.value + self.scale * noise, self.low, self.high)
        return state._replace(value=value)

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record bounds and noise scale without exposing them to the policy."""
        return {
            "type": "random-walk",
            "low": np.asarray(self.low).tolist(),
            "high": np.asarray(self.high).tolist(),
            "scale": self.scale,
        }


class Procedural:
    """Redraw parameters from a seeded uniform stream every ``period`` transitions.

    Each switch samples fresh values uniform in ``[low, high)`` on the drift
    RNG stream, matching procedural benchmark generators (for example MEAL
    layout streams) that emit a new task each episode. ``low`` and ``high``
    are required. The first phase is not a draw: it is the environment's
    starting value clipped into ``[low, high]``, so every seed shares it.
    Values are hidden unless an Oracle exposes them, and ``phase`` counts
    completed switches.
    """

    def __init__(
        self,
        *,
        low: ArrayLike | tuple[int, int],
        high: ArrayLike | tuple[int, int],
        period: int = 1,
    ) -> None:
        period = as_int(
            period,
            "Procedural period must be an integer in [1, 2**32).",
            minimum=1,
            maximum=2**32 - 1,
        )
        lower, upper = np.broadcast_arrays(
            np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Procedural bounds must be finite.")
        if np.any(lower > upper):
            raise ValueError("Procedural bounds must be ordered.")
        self.low: jax.Array = jnp.asarray(lower)
        self.high: jax.Array = jnp.asarray(upper)
        self.period = period

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Clip the initial value into bounds without drawing stream noise."""
        del key
        if jnp.broadcast_shapes(value.shape, self.low.shape, self.high.shape) != value.shape:
            raise ValueError("Procedural bounds must broadcast to the parameter shape.")
        value = jnp.clip(value.astype(jnp.float32), self.low, self.high)
        return DriftState(value, jnp.uint32(0), jnp.uint32(0))

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Advance the switch timer and redraw parameters exactly on switches."""
        elapsed = state.elapsed + jnp.uint32(1)
        switch = elapsed == jnp.uint32(self.period)
        drawn = jax.random.uniform(
            key,
            shape=state.value.shape,
            minval=self.low,
            maxval=self.high,
            dtype=jnp.float32,
        )
        return DriftState(
            jnp.where(switch, drawn, state.value),
            state.phase + switch.astype(jnp.uint32),
            jnp.where(switch, jnp.uint32(0), elapsed),
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record bounds, period and distribution without exposing values."""
        return {
            "type": "procedural",
            "low": np.asarray(self.low).tolist(),
            "high": np.asarray(self.high).tolist(),
            "period": self.period,
            "distribution": "uniform",
        }


def _layout(state: DriftState) -> object:
    """Hashable structure of a drift state: its tree plus leaf shapes/dtypes."""
    leaves = cast("list[jax.Array]", jax.tree.leaves(state))
    tree = cast(object, jax.tree.structure(state))
    return (tree, tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves))


def _refuse_endogenous(processes: Seq[Drift], combinator: str) -> None:
    for process in processes:
        if getattr(process, "endogenous", False):
            raise ValueError(
                f"{combinator} cannot run {type(process).__name__}: agent-driven drift needs "
                "the environment's signal, which combinators do not pass on. Use it as the "
                "field's own process instead."
            )


class Sequence:
    """Run drift processes one after another for fixed tick counts.

    ``segments`` is a list of ``(process, ticks)``; the last process keeps
    running after its tick budget, so a lifetime never runs out of drift.
    On entering a segment its process is initialised from the value the
    previous segment left. Whether the parameter is then continuous across the
    boundary depends on the process: ``Stationary()``, ``RandomWalk`` and
    ``Procedural`` start from that value (clipped to their bounds), while
    ``Stationary(value)``, ``Cyclic`` and ``Schedule`` start at their own value,
    so the parameter jumps at the boundary. Agent-driven processes
    (``Accumulate``) need the environment's signal, which a combinator does not
    pass on, so they are refused here. The switch counter ``phase`` advances
    on every segment boundary and on every switch the running process itself
    declares; ``elapsed`` counts ticks since the last switch. All processes must
    use the same parameter shape, but their internal states may differ, so a
    ``Sum`` segment can follow a ``Cyclic`` one.

    Only the running segment is stepped, selected with ``lax.switch``, and only
    its key is derived (``fold_in`` on the segment index). Segments whose states
    share a layout are stored stacked, so a 20-task schedule of ``Cyclic``
    segments is one stacked state and per-tick work stays flat in the number of
    segments outside ``vmap``.
    """

    def __init__(self, segments: Seq[tuple[Drift, int]]) -> None:
        if not segments:
            raise ValueError("Sequence needs at least one (process, ticks) segment.")
        self.processes = tuple(process for process, _ in segments)
        self.ticks = tuple(
            as_int(
                ticks,
                "Sequence segment ticks must be integers in [1, 2**32).",
                minimum=1,
                maximum=2**32 - 1,
            )
            for _, ticks in segments
        )
        _refuse_endogenous(self.processes, "Sequence")

    def _groups(self, value: jax.Array) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Static (group, slot) of every segment, grouping equal state layouts."""
        layouts: list[object] = []
        group_of: list[int] = []
        slot_of: list[int] = []
        sizes: list[int] = []
        for process in self.processes:
            shape = jax.eval_shape(process.init, value, jax.random.key(0))
            layout = _layout(shape)
            if layout not in layouts:
                layouts.append(layout)
                sizes.append(0)
            group = layouts.index(layout)
            group_of.append(group)
            slot_of.append(sizes[group])
            sizes[group] += 1
        return tuple(group_of), tuple(slot_of), len(layouts)

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Initialise every segment once (fixing its state layout); run the first."""
        group_of, _, groups = self._groups(value)
        keys = jax.random.split(key, len(self.processes))
        parts = [process.init(value, keys[index]) for index, process in enumerate(self.processes)]
        stacks = tuple(
            jax.tree.map(
                lambda *leaves: jnp.stack(leaves),
                *[part for index, part in enumerate(parts) if group_of[index] == group],
            )
            for group in range(groups)
        )
        segment = jnp.uint32(0)
        return DriftState(
            parts[0].value,
            jnp.uint32(0),
            jnp.uint32(0),
            (*stacks, SegmentCounter(segment, jnp.uint32(0))),
        )

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Step only the running segment, then enter the next one if it is due."""
        stacks = cast("tuple[DriftState, ...]", state.parts[:-1])
        counter = cast(SegmentCounter, state.parts[-1])
        group_of, slot_of, _ = self._groups(state.value)
        count = len(self.processes)
        segment, spent = counter.segment, counter.spent + jnp.uint32(1)

        def read(index: int, groups: Seq[DriftState]) -> DriftState:
            def take(leaf: jax.Array) -> jax.Array:
                return leaf[slot_of[index]]

            return jax.tree.map(take, groups[group_of[index]])

        def write(index: int, groups: Seq[DriftState], part: DriftState) -> tuple[DriftState, ...]:
            updated = list(groups)

            def put(leaf: jax.Array, new: jax.Array) -> jax.Array:
                return leaf.at[slot_of[index]].set(new)

            updated[group_of[index]] = jax.tree.map(put, groups[group_of[index]], part)
            return tuple(updated)

        def run(index: int) -> tuple[tuple[DriftState, ...], jax.Array, jax.Array, jax.Array]:
            running = read(index, stacks)
            advanced = self.processes[index].step(running, jax.random.fold_in(key, index))
            due = (
                spent >= jnp.uint32(self.ticks[index]) if index + 1 < count else jnp.asarray(False)
            )
            return write(index, stacks, advanced), advanced.value, switched(running, advanced), due

        stepped, value, inner_switch, entering = cast(
            "tuple[tuple[DriftState, ...], jax.Array, jax.Array, jax.Array]",
            jax.lax.switch(
                segment.astype(jnp.int32), [partial(run, index) for index in range(count)]
            ),
        )

        def enter(index: int) -> tuple[tuple[DriftState, ...], jax.Array]:
            fresh = self.processes[index].init(value, jax.random.fold_in(key, count + index))
            return write(index, stepped, fresh), fresh.value

        def stay() -> tuple[tuple[DriftState, ...], jax.Array]:
            return stepped, value

        def entered() -> tuple[tuple[DriftState, ...], jax.Array]:
            following = jnp.minimum(segment + jnp.uint32(1), jnp.uint32(count - 1))
            return cast(
                "tuple[tuple[DriftState, ...], jax.Array]",
                jax.lax.switch(
                    following.astype(jnp.int32), [partial(enter, index) for index in range(count)]
                ),
            )

        new_stacks, value = cast(
            "tuple[tuple[DriftState, ...], jax.Array]", jax.lax.cond(entering, entered, stay)
        )
        switch = inner_switch | entering
        return DriftState(
            value,
            state.phase + switch.astype(jnp.uint32),
            jnp.where(switch, jnp.uint32(0), state.elapsed + jnp.uint32(1)),
            (
                *new_stacks,
                SegmentCounter(
                    segment + entering.astype(jnp.uint32), jnp.where(entering, jnp.uint32(0), spent)
                ),
            ),
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record every segment's process and tick budget."""
        return {
            "type": "sequence",
            "segments": [
                {"process": process.configuration, "ticks": ticks}
                for process, ticks in zip(self.processes, self.ticks, strict=True)
            ],
        }


class Sum:
    """Add drift processes: a base process plus offset processes.

    The first process drifts the environment's value; every other process
    starts from zero and contributes an offset, so ``Sum([Cyclic(...),
    RandomWalk(low=-0.1, high=0.1, scale=0.01)])`` is a switching schedule with
    noise on top. ``phase`` advances whenever any process declares a switch,
    and ``elapsed`` counts ticks since the last one. All processes must use the
    same parameter shape. Agent-driven processes (``Accumulate``) are refused,
    as in :class:`Sequence`.
    """

    def __init__(self, processes: Seq[Drift]) -> None:
        if len(processes) < 2:
            raise ValueError("Sum needs a base process and at least one offset process.")
        self.processes = tuple(processes)
        _refuse_endogenous(self.processes, "Sum")

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Initialise the base from the value and every offset from zero."""
        keys = jax.random.split(key, len(self.processes))
        zero = jnp.zeros_like(jnp.asarray(value, dtype=jnp.float32))
        parts = tuple(
            process.init(value if index == 0 else zero, keys[index])
            for index, process in enumerate(self.processes)
        )
        total = sum((part.value for part in parts[1:]), parts[0].value)
        return DriftState(total, jnp.uint32(0), jnp.uint32(0), parts)

    def step(self, state: DriftState, key: jax.Array) -> DriftState:
        """Step every process with its own key and add the results."""
        keys = jax.random.split(key, len(self.processes))
        previous = cast("tuple[DriftState, ...]", state.parts)
        parts = tuple(
            process.step(previous[index], keys[index])
            for index, process in enumerate(self.processes)
        )
        switch = jnp.any(
            jnp.stack([switched(old, new) for old, new in zip(previous, parts, strict=True)])
        )
        total = sum((part.value for part in parts[1:]), parts[0].value)
        return DriftState(
            total,
            state.phase + switch.astype(jnp.uint32),
            jnp.where(switch, jnp.uint32(0), state.elapsed + jnp.uint32(1)),
            parts,
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record the base and offset processes in order."""
        return {"type": "sum", "processes": [process.configuration for process in self.processes]}


class Accumulate:
    """Agent-driven accumulation: decayed value plus a multiple of the signal.

    Declares ``endogenous = True``, so ``DriftEnv`` calls
    ``step(state, key, signal)`` where ``signal`` is the environment's
    privileged signal for the tick (for example the agents' collected reward).
    Each tick the value becomes ``clip(decay * value + gain * signal, low,
    high)``. ``low`` and ``high`` are required, and the starting value is
    clipped into them at tick 0. ``gain``/``decay`` are JSON-serialisable
    settings (stored as Python floats), so the process can be checkpointed and
    reproduced, unlike an arbitrary Python callback.

    ``bands`` are optional thresholds: crossing any band on a tick advances the
    switch counter by one (crossing several on one tick still counts once, like
    the combinators), which gives switch-based metrics such as time-to-recover
    a defined event even though the drift itself is gradual. With no bands it
    never declares a switch, like every other gradual process.
    """

    endogenous = True

    def __init__(
        self,
        *,
        gain: float = 1.0,
        decay: float = 0.99,
        low: float,
        high: float,
        bands: Seq[float] = (),
    ) -> None:
        gain, decay, low, high = (float(v) for v in (gain, decay, low, high))
        for name, value in (("gain", gain), ("decay", decay), ("low", low), ("high", high)):
            if not math.isfinite(value):
                raise ValueError(f"Accumulate {name} must be finite.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Accumulate decay must be in [0, 1).")
        if low > high:
            raise ValueError("Accumulate bounds must be ordered.")
        if any(not math.isfinite(band) for band in bands):
            raise ValueError("Accumulate bands must be finite.")
        self.gain = gain
        self.decay = decay
        self.low = low
        self.high = high
        self.bands = tuple(float(band) for band in bands)

    def init(self, value: jax.Array, key: jax.Array) -> DriftState:
        """Clip the starting value into bounds; no randomness is consumed."""
        del key
        if jnp.ndim(value) == 0:
            value = jnp.asarray(value, dtype=jnp.float32)
        return DriftState(
            jnp.clip(value.astype(jnp.float32), self.low, self.high), jnp.uint32(0), jnp.uint32(0)
        )

    def step(self, state: DriftState, key: jax.Array, signal: object = None) -> DriftState:
        """Decay, then add ``gain * signal``; the key is unused."""
        del key
        if signal is None:
            raise ValueError("Accumulate needs an environment signal.")
        contribution = jnp.asarray(signal, dtype=jnp.float32)
        if contribution.shape != state.value.shape:
            contribution = jnp.broadcast_to(contribution, state.value.shape)
        value = jnp.clip(self.decay * state.value + self.gain * contribution, self.low, self.high)
        before = jnp.asarray(self.bands) if self.bands else jnp.zeros((0,), dtype=jnp.float32)
        crossed = (
            jnp.any(
                (state.value[..., None] < before) & (value[..., None] >= before)
                | (state.value[..., None] >= before) & (value[..., None] < before)
            )
            if before.shape[0]
            else jnp.asarray(False)
        )
        return DriftState(
            value,
            state.phase + crossed.astype(jnp.uint32),
            jnp.where(crossed, jnp.uint32(0), state.elapsed + jnp.uint32(1)),
        )

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """Record every setting so the process can be reproduced."""
        return {
            "type": "accumulate",
            "endogenous": True,
            "gain": self.gain,
            "decay": self.decay,
            "low": self.low,
            "high": self.high,
            "bands": list(self.bands),
        }
