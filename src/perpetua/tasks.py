"""Switch between whole environments as a drift over a task index.

``pp.tasks.switch([env_a, env_b, ...], schedule)`` builds a single lifetime
whose drifting parameter is ``task`` (the active environment index) plus each
task's own parameters (addressed as ``tasks[0].<field>``). Every tick on
which the active task changes is a declared drift switch, whatever process
drives ``task``: a table (``Cyclic``, ``Schedule``) or a gradual process such
as ``RandomWalk`` whose rounded value moved to another index. So recovery
tracking, the Oracle's ``boundary`` and ``task_changed`` and a learner's
boundary handling follow the task the Oracle's ``task_id`` reports. A switch
the process itself declares also counts when the task stays the same, as for
any drift field: ``Cyclic([0.0, 0.0, 1.0], period)`` declares a switch between
its first two phases.

Inactive tasks are carried unchanged and **resume** where they left off, so
each task is one continuing lifetime. ``on_switch="restart"`` instead
re-initialises the task on entry and reports a respawn.

Tasks are functional JAX parameter environments, or host lifts (through
:class:`HostTaskSwitch`, ``on_switch="resume"`` only); ``DriftEnv`` wraps the
result. All tasks must share an agent count. Discrete tasks are padded to a
common action count and each task's action mask hides the actions it does not
have, so a switch raises the public ``space_change`` event for each agent whose
mask differs between the two tasks; continuous tasks must share one action
``Box``. Observations are padded with zeros to a common shape.
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import re
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.core import Tracer
from jax.typing import ArrayLike

from perpetua.core import EnvInfo, Environment, Metadata, OneHotObservation, Oracle
from perpetua.drift import Cyclic, Drift, Procedural, RandomWalk, Schedule, ScheduleData, Stationary
from perpetua.drift import Sequence as DriftSequence
from perpetua.drift import Sum as DriftSum
from perpetua.params import DriftEnv, ParamsEnv, _strong  # pyright: ignore[reportPrivateUsage]

__all__ = [
    "OnSwitch",
    "TaskParams",
    "TaskSwitchState",
    "TaskSwitch",
    "switch",
    "HostTaskSwitch",
]

#: What a task does when it becomes active again: continue, or start over.
type OnSwitch = Literal["resume", "restart"]


class TaskParams(NamedTuple):
    """Active task index plus every task's own parameters."""

    task: jax.Array  # float32 scalar, rounded to the nearest index, clipped to [0, tasks - 1]
    tasks: tuple[Any, ...]


# Task init keys come from a tagged fold, so none equals a plain split of the key
# the switch receives (DriftEnv derives its own keys by tagged folds as well).
_TASK_INIT_TAG = 0x7461736B


class TaskSwitchState(NamedTuple):
    """Every task's state, per-task respawn flags and the current task index."""

    states: tuple[Any, ...]
    respawn: jax.Array  # (tasks, agents), bool, for the task stepped this tick
    active: jax.Array  # int32 scalar


_TASK_FIELD = re.compile(r"tasks\[(\d+)\]\.?(.+)")


def _pad(value: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Zero-pad a value on the right of every dimension to ``shape``."""
    widths = [(0, shape[i] - value.shape[i]) for i in range(len(shape))]
    return jnp.pad(value, widths)


class TaskSwitch:
    """One lifetime over several parameter environments, selected by ``task``."""

    default_drift_field = "task"
    mask_params: tuple[str, ...] = ("task",)

    def __init__(
        self,
        envs: Sequence[ParamsEnv[Any, Any]],
        *,
        on_switch: OnSwitch = "resume",
        allow_host: bool = False,
    ) -> None:
        if len(envs) < 2:
            raise ValueError("Task switching needs at least two environments.")
        if on_switch not in {"resume", "restart"}:
            raise ValueError("on_switch must be 'resume' or 'restart'.")
        self.envs = tuple(envs)
        # The switch changes the mask, and so does any task's own mask parameter.
        self.mask_params = (
            "task",
            *(
                f"tasks[{index}].{field}"
                for index, env in enumerate(envs)
                for field in cast("tuple[str, ...]", getattr(env, "mask_params", ()))
            ),
        )
        self.on_switch = on_switch
        self.num_agents = cast(Any, envs[0]).num_agents
        if any(cast(Any, env).num_agents != self.num_agents for env in envs):
            raise ValueError("Task switching requires the same agent count in every task.")
        if not allow_host and any(
            cast(Any, env).metadata.get("execution") == "host" for env in envs
        ):
            raise ValueError("Task switching requires functional JAX environments, not host lifts.")
        action_spaces = [cast(Any, env).action_space for env in envs]
        # Continuous tasks (host lifts, or simulators run through the compiled
        # bridge) must share one action box; discrete tasks are padded.
        self.continuous = all(isinstance(space, spaces.Box) for space in action_spaces)
        if self.continuous:
            first = action_spaces[0]
            if any(space.shape != first.shape for space in action_spaces) or any(
                not np.allclose(space.low, first.low) or not np.allclose(space.high, first.high)
                for space in action_spaces
            ):
                raise ValueError("Continuous tasks must share action shapes and bounds.")
            counts = [int(first.shape[-1]) for _ in action_spaces]
            self.action_space = first
        else:
            counts: list[int] = []
            for index, space in enumerate(action_spaces):
                nvec = getattr(space, "nvec", None)
                if nvec is None:
                    raise ValueError(
                        "Task switching requires all tasks to be discrete or all continuous."
                    )
                sizes = np.asarray(nvec).reshape(-1)
                if np.any(sizes != sizes[0]):
                    raise ValueError(
                        f"Task switching requires every agent slot of a task to have the same "
                        f"number of actions, but task {index} ({type(envs[index]).__name__}) has "
                        f"action_space.nvec={sizes.tolist()}."
                    )
                counts.append(int(sizes[0]))
            self.action_space = spaces.MultiDiscrete(
                np.full(self.num_agents, max(counts), dtype=np.int64)
            )
        self.action_counts = tuple(counts)
        self.action_count = max(counts)
        obs_spaces = [cast(Any, env).observation_space for env in envs]
        ndim = len(obs_spaces[0].shape)
        if any(len(space.shape) != ndim for space in obs_spaces):
            raise ValueError("Task switching requires observations with matching dimensions.")
        self.obs_shape: tuple[int, ...] = tuple(
            max(space.shape[dimension] for space in obs_spaces) for dimension in range(ndim)
        )
        lows: list[jax.Array] = [
            _pad(jnp.asarray(space.low, dtype=jnp.float32), self.obs_shape) for space in obs_spaces
        ]
        highs: list[jax.Array] = [
            _pad(jnp.asarray(space.high, dtype=jnp.float32), self.obs_shape) for space in obs_spaces
        ]
        self.observation_space = spaces.Box(
            low=np.asarray(jnp.min(jnp.stack(lows), axis=0)),
            high=np.asarray(jnp.max(jnp.stack(highs), axis=0)),
            dtype=np.float32,
        )
        task_metadata: list[Metadata] = [cast(Any, env).metadata for env in envs]
        self.metadata: Metadata = {
            "env_id": "task-switch-v0",
            "oracle_exposure": (),
            "execution": "jax",
            "configuration": {
                # Each task's own configuration, in order: restore compares this, so a
                # swapped or reconfigured task is refused instead of continued.
                "tasks": [
                    {
                        "env_id": str(entry.get("env_id", type(env).__name__)),
                        "configuration": entry.get("configuration", {}),
                    }
                    for env, entry in zip(envs, task_metadata, strict=True)
                ],
                "observation_shape": list(self.obs_shape),
                "action_counts": list(self.action_counts),
                "on_switch": on_switch,
            },
        }
        # One task that cannot checkpoint or fork (a compiled simulator bridge)
        # makes the whole lifetime so.
        if any(entry.get("checkpointable") is False for entry in task_metadata):
            self.metadata["checkpointable"] = False
        if any(entry.get("forkable") is False for entry in task_metadata):
            self.metadata["forkable"] = False

    @property
    def default_params(self) -> TaskParams:
        """Start on task 0 with every task's own default parameters."""
        return TaskParams(
            jnp.asarray(0.0, dtype=jnp.float32),
            tuple(env.default_params for env in self.envs),
        )

    def check_drift(self, drifts: Mapping[str, Drift]) -> None:
        """Refuse a task schedule or per-task drift this switch cannot run.

        The ``task`` process must stay on the tasks this switch has. Drift on
        ``tasks[i].<field>`` passes through task ``i``'s own guards: its
        ``drift_fields`` and its ``check_drift``, as when the task is wrapped
        by itself. An endogenous process is refused on every field, because
        the switch does not pass a task's ``signal`` on.
        """
        _check_task_schedule(drifts.get("task"), len(self.envs))
        per_task: list[dict[str, Drift]] = [{} for _ in self.envs]
        for name, process in drifts.items():
            if getattr(process, "endogenous", False):
                raise ValueError(
                    f"{type(process).__name__} on {name!r} is driven by an environment's "
                    "signal, which a task switch does not pass on to its tasks' parameters."
                )
            match = _TASK_FIELD.fullmatch(name)
            if match is not None and int(match[1]) < len(self.envs):
                per_task[int(match[1])][match[2]] = process
        for index, (env, own) in enumerate(zip(self.envs, per_task, strict=True)):
            check = getattr(env, "check_drift", None)
            if check is not None:
                try:
                    check(own)
                except ValueError as error:
                    raise ValueError(f"Task {index} ({type(env).__name__}): {error}") from error
            allowed = cast("tuple[str, ...] | None", getattr(env, "drift_fields", None))
            if allowed is not None and set(own) - set(allowed):
                raise ValueError(
                    f"Task {index} ({type(env).__name__}) accepts drift only on "
                    f"{sorted(allowed)}, not {sorted(set(own) - set(allowed))}."
                )

    def check_schedule(self, field: str, data: ScheduleData) -> None:
        """Refuse an assigned ``task`` schedule that names a task this switch lacks.

        ``data`` may carry leading batch axes (stacked schedules); only each
        schedule's first ``length`` values are used and checked.
        """
        if field != "task":
            return
        values = np.asarray(data.values)
        length = np.asarray(data.length)[..., None]
        used = cast(np.ndarray, np.arange(values.shape[-1]) < length)
        _refuse_task_values(values[used], len(self.envs))

    def declared_switches(self, previous: TaskParams, current: TaskParams) -> dict[str, jax.Array]:
        """``task`` switched when the active task changed, whatever its process says."""
        return {"task": self._index(previous) != self._index(current)}

    def derive(self, params: TaskParams) -> TaskParams:
        """Round/clip the task index and derive each task's own parameters."""
        index = jnp.clip(jnp.rint(params.task), 0, len(self.envs) - 1).astype(jnp.float32)
        derived: list[Any] = []
        for env, inner in zip(self.envs, params.tasks, strict=True):
            derive = getattr(env, "derive", None)
            derived.append(inner if derive is None else derive(inner))
        return TaskParams(index, tuple(derived))

    def _index(self, params: TaskParams) -> jax.Array:
        return jnp.clip(jnp.rint(params.task), 0, len(self.envs) - 1).astype(jnp.int32)

    def task_id(self, params: TaskParams) -> jax.Array:
        """The Oracle's ``task_id``: the task the next observation and action belong to."""
        return self._index(params)

    def init(
        self, key: jax.Array, params: TaskParams
    ) -> tuple[TaskSwitchState, jax.Array, EnvInfo]:
        """Initialise every task once; only the active one is observed."""
        keys = jax.random.split(jax.random.fold_in(key, _TASK_INIT_TAG), len(self.envs))
        states = tuple(
            env.init(keys[index], inner)[0]
            for index, (env, inner) in enumerate(zip(self.envs, params.tasks, strict=True))
        )
        state = TaskSwitchState(
            states,
            jnp.zeros((len(self.envs), self.num_agents), dtype=jnp.bool_),
            self._index(params),
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: TaskSwitchState, params: TaskParams) -> tuple[jax.Array, EnvInfo]:
        """Describe only the active task, padded to the shared observation shape."""
        index = self._index(params)

        def observe_task(task: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            obs, task_info = self.envs[task].observe(state.states[task], params.tasks[task])
            mask = task_info["action_mask"]
            return (
                _pad(obs, self.obs_shape).astype(jnp.float32),
                mask if self.continuous else _pad(mask, (self.num_agents, self.action_count)),
                task_info["active_mask"],
                state.respawn[task],
            )

        obs, mask, active, respawn = cast(
            "tuple[jax.Array, jax.Array, jax.Array, jax.Array]",
            jax.lax.switch(index, [partial(observe_task, i) for i in range(len(self.envs))]),
        )
        info: EnvInfo = {
            "action_mask": mask,
            "active_mask": active,
            "respawn": respawn,
        }
        return obs, info

    def step(
        self, state: TaskSwitchState, action: ArrayLike, key: jax.Array, params: TaskParams
    ) -> tuple[TaskSwitchState, jax.Array]:
        """Step only the active task; inactive states are carried unchanged.

        The action belongs to the task that was active when it was chosen, so
        that task is stepped even on a switch tick. With ``on_switch="restart"``
        the incoming task is re-initialised in :meth:`after_drift`, on the same
        tick.
        """
        index = self._index(params)
        keys = jax.random.split(key, len(self.envs))

        def run(task: int) -> tuple[tuple[Any, ...], jax.Array, jax.Array]:
            env = self.envs[task]
            new_state, reward = env.step(state.states[task], action, keys[task], params.tasks[task])
            # The task's public contract, not an attribute of its state: a state
            # may keep the flag nested (PartnerGrid) or under another name.
            respawn_flag = env.observe(new_state, params.tasks[task])[1]["respawn"]
            states = tuple(
                new_state if other == task else state.states[other]
                for other in range(len(self.envs))
            )
            respawn = state.respawn.at[task].set(
                jnp.broadcast_to(jnp.asarray(respawn_flag, dtype=jnp.bool_), (self.num_agents,))
            )
            return states, reward, respawn

        # Branch on the task index; every branch returns the full state tuple,
        # so the pytree structure is identical across branches and only the
        # active task is ever stepped.
        states, reward, respawn = cast(
            "tuple[tuple[Any, ...], jax.Array, jax.Array]",
            jax.lax.switch(index, [partial(run, i) for i in range(len(self.envs))]),
        )
        return TaskSwitchState(states, respawn, index), reward

    def after_drift(
        self, state: TaskSwitchState, previous: TaskParams, current: TaskParams, key: jax.Array
    ) -> TaskSwitchState:
        """Prepare the task that becomes active on this tick.

        With ``on_switch="resume"`` the task continues where it stopped. The
        respawn flag it left on its last active tick is not reported again;
        the tick reports the respawn of the task that was just stepped, so an
        episode that ends on a switch tick is not lost. With
        ``on_switch="restart"`` the task is re-initialised and reports a respawn
        in the observation of that same tick.
        """
        index = self._index(current)
        outgoing = self._index(previous)
        entering = outgoing != index
        if self.on_switch != "restart":
            carried = state.respawn.at[index].set(state.respawn[outgoing])
            return state._replace(respawn=jnp.where(entering, carried, state.respawn), active=index)
        keys = jax.random.split(key, len(self.envs))

        def restart(task: int) -> TaskSwitchState:
            fresh = _strong(self.envs[task].init(keys[task], current.tasks[task])[0])
            states = tuple(
                fresh if other == task else state.states[other] for other in range(len(self.envs))
            )
            respawn = state.respawn.at[task].set(True)
            return TaskSwitchState(states, respawn, index)

        def enter(_: None) -> TaskSwitchState:
            return cast(
                TaskSwitchState,
                jax.lax.switch(index, [partial(restart, i) for i in range(len(self.envs))]),
            )

        def stay(_: None) -> TaskSwitchState:
            return state._replace(active=index)

        # Only switch ticks pay for the reset. Under jax.vmap a cond on a batched
        # flag still evaluates both branches.
        return cast(TaskSwitchState, jax.lax.cond(entering, enter, stay, None))

    def privileged(self, state: TaskSwitchState) -> dict[str, jax.Array]:
        """No task-side privileged fields; the wrapper supplies drift diagnostics."""
        del state
        return {}

    def close(self) -> None:
        """Close every task that holds external resources; the wrapper forwards this."""
        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()


def switch(
    envs: Sequence[ParamsEnv[Any, Any] | Environment[Any]],
    schedule: Drift | dict[str, Drift] | None = None,
    *,
    on_switch: OnSwitch = "resume",
) -> DriftEnv[TaskSwitchState, TaskParams]:
    """Compose environments with drift over the active task index.

    ``schedule`` is any drift process (or mapping of parameter names to
    processes); a bare process drives ``task``, so
    ``pp.drift.Cyclic([0.0, 1.0], 100)`` alternates two tasks every 100 ticks.
    The values are task indices counted from 0. They are rounded to the nearest
    integer, so a fractional value from a ``RandomWalk`` selects a task by
    rounding, and every tick on which the rounded index changes is a declared
    switch (``boundary`` and ``task_changed`` through the Oracle, recovery
    tracking), as is every switch the process itself declares. The schedule is
    checked when the switch is built and whenever drift is rebound: a fixed
    table (``Cyclic``, ``Schedule``, ``Stationary(value)``, also as a
    ``Sequence`` segment) must hold integer indices of existing tasks, and a
    process whose reachable values are known (the bounds of a ``RandomWalk``
    or ``Procedural``, the tables and bounds added up by a ``Sum``) must stay
    within ``[-0.5, len(envs) - 0.5]``, the values that round to a task.
    Either raises ``ValueError`` otherwise. Only a process of another kind is
    not checked; its values are clipped to ``[0, len(envs) - 1]``. Without a
    schedule the lifetime stays on task 0.

    ``envs`` are parameter environments, host lifts, or JAX-executed
    ``DriftEnv`` environments without drift of their own (``pp.make`` ids and
    simulators lifted with ``compiled=True``), which are composed through their
    worlds. Continuous tasks must share one action box.

    A task cannot bring drift of its own: a ``DriftEnv`` task may only carry
    ``Stationary()`` drift, and any other process, or a ``Stationary(value)``
    holding a value of its own, raises ``ValueError``. A task's parameters
    start from its world's defaults and drift through this switch's mapping
    instead (``{"tasks[1].gravity": process}``), under the task's own
    ``drift_fields`` and ``check_drift``. On
    resume, a re-entered task continues from where it
    stopped, and the ``respawn`` flag from its last active tick is not reported
    a second time. The tick on which a switch happens reports the ``respawn`` of
    the task that was stepped on it.
    """
    for env in envs:
        if isinstance(env, Oracle | OneHotObservation):
            name = type(env).__name__
            raise TypeError(
                f"switch() tasks must be unwrapped environments; got {name}. Pass the "
                f"environments themselves and wrap the switch instead: "
                f"pp.{name}(pp.tasks.switch([...], schedule), ...)."
            )
    if any(cast(Any, env).metadata.get("execution") == "host" for env in envs):
        if not all(cast(Any, env).metadata.get("execution") == "host" for env in envs):
            raise ValueError("Task switching cannot mix host and functional JAX environments.")
        world: TaskSwitch = HostTaskSwitch(envs, on_switch=on_switch)
    else:
        world = TaskSwitch([_world_of(env) for env in envs], on_switch=on_switch)
    return DriftEnv(world, schedule, env_id="task-switch-v0")


def _task_values(process: object) -> np.ndarray | None:
    """The task indices a process with a fixed table can ever produce, else ``None``."""
    if isinstance(process, Cyclic):
        return np.asarray(process.values)
    if isinstance(process, Schedule):
        default = process.default
        return np.asarray(default.values)[: int(np.asarray(default.length))]
    if isinstance(process, Stationary) and process.value is not None:
        return np.asarray(process.value)
    return None


def _refuse_task_values(values: np.ndarray, tasks: int) -> None:
    bad = values[(values != np.rint(values)) | (values < 0) | (values > tasks - 1)]
    if bad.size:
        raise ValueError(
            f"The task schedule holds {sorted({float(v) for v in bad.reshape(-1)})} but "
            f"the switch has {tasks} tasks, so valid values are the integers "
            f"0..{tasks - 1} (task indices count from 0). Out-of-range values would be "
            "clipped to the nearest task without any switch happening."
        )


type _Range = tuple[float, float]


def _task_range(process: object, start: _Range) -> _Range | None:
    """The interval a process's values stay in when started from ``start``, if known."""
    values = _task_values(process)
    if values is not None:
        return float(values.min()), float(values.max())
    if isinstance(process, Stationary):
        return start
    if isinstance(process, RandomWalk | Procedural):
        return float(np.min(process.low)), float(np.max(process.high))
    if isinstance(process, DriftSequence):
        low, high = np.inf, -np.inf
        for segment in process.processes:
            reach = _task_range(segment, start)
            if reach is None:
                return None
            low, high, start = min(low, reach[0]), max(high, reach[1]), reach
        return low, high
    if isinstance(process, DriftSum):
        parts = [
            _task_range(part, start if index == 0 else (0.0, 0.0))
            for index, part in enumerate(process.processes)
        ]
        if any(part is None for part in parts):
            return None
        known = cast("list[_Range]", parts)
        return sum(part[0] for part in known), sum(part[1] for part in known)
    return None


def _check_task_schedule(process: Drift | None, tasks: int) -> None:
    """Refuse a task schedule that names, or can reach, a task the switch lacks.

    Fixed tables outside a ``Sum`` must hold integer indices. Any process whose
    reachable interval is known must stay within the values that round to a
    task, ``[-0.5, tasks - 0.5]``.
    """
    if process is None:
        return
    stack: list[object] = [process]
    while stack:
        current = stack.pop()
        if isinstance(current, DriftSequence):
            stack.extend(cast(Any, current).processes)
            continue
        values = _task_values(current)
        if values is not None:
            _refuse_task_values(values, tasks)
    reach = _task_range(process, (0.0, 0.0))
    if reach is not None and (reach[0] < -0.5 or reach[1] > tasks - 0.5):
        raise ValueError(
            f"The task schedule can reach values in [{reach[0]:g}, {reach[1]:g}] but the "
            f"switch has {tasks} tasks, so its values must stay within "
            f"[-0.5, {tasks - 0.5:g}] (they are rounded to the task indices 0..{tasks - 1}). "
            "Values outside would be clipped to the nearest task without any switch happening."
        )


def _world_of(env: Any) -> Any:
    """The parameter environment inside a JAX-executed ``DriftEnv``.

    Simulators lifted with ``compiled=True`` (for example Meta-World) are
    ``DriftEnv`` instances around a traceable world; the switch composes those
    worlds directly. Their own drift must be stationary: a task's parameters
    start from its defaults and drift only through the switch's own mapping.
    """
    if not isinstance(env, DriftEnv):
        return env
    drifts = cast(dict[str, Any], env.drifts)
    if any(
        not isinstance(process, Stationary) or process.value is not None
        for process in drifts.values()
    ):
        raise ValueError(
            "Tasks given to switch() must not drift themselves or hold a value of their own: "
            "pass the undrifted task and drift its parameters through the switch's schedule "
            "mapping, e.g. {'tasks[0].<field>': process}."
        )
    return cast(Any, env).env


class HostTaskSwitch(TaskSwitch):
    """Task switching over host-controlled environments (for example Atari).

    Same resume, padding, mask and switch-event semantics as
    :class:`TaskSwitch` (restart is refused, because a host lift cannot be
    re-initialised), but inner environments are called through the ordinary
    ``Environment`` interface because host lifts are not parameter
    environments. Only the active host environment steps; the others are
    carried unchanged.
    """

    def __init__(self, envs: Sequence[Any], *, on_switch: OnSwitch = "resume") -> None:
        if on_switch == "restart":
            raise ValueError(
                "Host task switching supports on_switch='resume' only: a host lift owns one "
                "simulator per lifetime and cannot be re-initialised."
            )
        super().__init__(envs, on_switch=on_switch, allow_host=True)
        self.metadata = {
            **self.metadata,
            "execution": "host",
            "checkpointable": all(env.metadata.get("checkpointable", False) for env in envs),
        }

    @property
    def default_params(self) -> TaskParams:
        """Only the task index drifts; host lifts keep their own drift inside."""
        return TaskParams(jnp.asarray(0.0, dtype=jnp.float32), ())

    def derive(self, params: TaskParams) -> TaskParams:
        """Round and clip the task index."""
        index = jnp.clip(jnp.rint(params.task), 0, len(self.envs) - 1).astype(jnp.float32)
        return TaskParams(index, ())

    def init(
        self, key: jax.Array, params: TaskParams
    ) -> tuple[TaskSwitchState, jax.Array, EnvInfo]:
        """Initialise every host environment once."""
        keys = jax.random.split(jax.random.fold_in(key, _TASK_INIT_TAG), len(self.envs))
        states = tuple(cast(Any, env).init(keys[index])[0] for index, env in enumerate(self.envs))
        state = TaskSwitchState(
            states,
            jnp.zeros((len(self.envs), self.num_agents), dtype=jnp.bool_),
            self._index(params),
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: TaskSwitchState, params: TaskParams) -> tuple[jax.Array, EnvInfo]:
        """Describe only the active host environment."""
        index = self._index(params)
        if not isinstance(index, Tracer):
            obs, task_info = cast(Any, self.envs[index]).observe(state.states[index])
            mask = task_info["action_mask"]
            if not self.continuous:
                mask = _pad(mask, (self.num_agents, self.action_count))
            info: EnvInfo = {
                "action_mask": mask,
                "active_mask": task_info["active_mask"],
                "respawn": state.respawn[index],
            }
            return _pad(obs, self.obs_shape).astype(jnp.float32), info
        # DriftEnv traces its mask comparison under lax.cond; a traced index
        # cannot select a Python host object, so retain the stacked form there.
        obs_list: list[jax.Array] = []
        mask_list: list[jax.Array] = []
        active_list: list[jax.Array] = []
        for task, env in enumerate(self.envs):
            obs, task_info = cast(Any, env).observe(state.states[task])
            obs_list.append(_pad(obs, self.obs_shape))
            mask = task_info["action_mask"]
            mask_list.append(
                mask if self.continuous else _pad(mask, (self.num_agents, self.action_count))
            )
            active_list.append(task_info["active_mask"])

        def chosen(values: list[jax.Array]) -> jax.Array:
            return jnp.stack(values)[index]

        info_out: EnvInfo = {
            "action_mask": chosen(mask_list),
            "active_mask": chosen(active_list),
            "respawn": state.respawn[index],
        }
        return chosen(obs_list).astype(jnp.float32), info_out

    def step(
        self, state: TaskSwitchState, action: ArrayLike, key: jax.Array, params: TaskParams
    ) -> tuple[TaskSwitchState, jax.Array]:
        """Step only the active host environment; the others are carried."""
        index = self._index(params)
        keys = jax.random.split(key, len(self.envs))
        states: list[Any] = list(state.states)
        respawn = state.respawn
        new_state, _, reward, info = cast(Any, self.envs[index]).step(
            states[index], action, keys[index]
        )
        states[index] = new_state
        respawn = respawn.at[index].set(info["respawn"])
        return TaskSwitchState(tuple(states), respawn, index), reward
