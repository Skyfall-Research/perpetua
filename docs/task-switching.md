# Task switching

`pp.tasks.switch` runs several environments inside one continuing lifetime,
selecting the active one with an ordinary drift process. Every tick on which
the active task changes is a **declared drift switch**, whatever process drives
the task: the Oracle's `boundary` and `task_changed`, recovery tracking and a
learner's boundary handling follow the task that `task_id` reports. Metadata
and checkpoints behave exactly as for any other drift.

```python
import perpetua as pp
from perpetua.envs.continual_grid import ContinualGrid
from perpetua.adapters.gymnax_params import GymnaxParameters

env = pp.tasks.switch(
    [ContinualGrid(size=4), GymnaxParameters("Acrobot-v1")],
    schedule=pp.drift.Cyclic([0.0, 1.0], 100_000),  # alternate every 100k ticks
)
carry = pp.runtime.Runner(env, pp.policies.random, ticks=256)(pp.runtime.init_lifetime(env, seed=0))
```

The schedule is any drift process or mapping of parameter names to processes.
A bare process drives `task`. Its values are task indices counted from 0 and
rounded to the nearest integer. The schedule is checked when the switch is
built and whenever drift is rebound (`with_drift`, `pp.compose`):

- a fixed table (`Cyclic`, `Schedule`, `Stationary(value)`, also as a
  `Sequence` segment) must hold integer indices of existing tasks, so
  `Cyclic([1.0, 2.0], 100)` over two tasks raises instead of staying on the
  second task;
- a process whose reachable values are known must stay within
  `[-0.5, tasks - 0.5]`, the values that round to a task: the bounds of a
  `RandomWalk` or `Procedural`, and the tables and bounds a `Sum` adds up
  (`Sum([Cyclic([0.0, 7.0], 3), Stationary()])` over two tasks raises);
- `pp.runtime.assign_schedule` refuses a `task` schedule that names a missing
  task. Under `jax.vmap` the values are traced and cannot be inspected, so
  check the stacked batch first with `env.check_schedule("task", schedules)`.

Only a process of another kind (your own) is not checked; its values are
clipped to `[0, tasks - 1]`. Without a schedule the lifetime stays on task 0.

A switch is declared on every tick on which the rounded index changes. Under
`RandomWalk(low=0, high=1, scale=0.4)` over two tasks that is each time the
walk crosses 0.5 (106 times in 500 ticks in our check, each one a `boundary`).
A switch that the process itself declares also counts when the task stays the
same, as for any other drift field: `Cyclic([0.0, 0.0, 1.0], period)` declares
a switch between its first two phases, although `task_id` stays 0 and nothing
restarts. Write the table without repeated neighbours, or use a `Sequence` of
`Stationary(value)` segments with different lengths, when every `boundary`
should be a task change.

A mapping can drift the task and per-task parameters at once:

```python
schedule = {
    "task": pp.drift.Cyclic([0.0, 1.0], 1_000),
    "tasks[1].link_mass_1": pp.drift.RandomWalk(low=0.5, high=2.0, scale=0.01),
}
```

Parameter names follow JAX's pytree paths: `tasks[0].goal` for a `NamedTuple`
parameter, `tasks[1].link_mass_1` for a dataclass one. The Oracle exposes them as
`tasks[0].goal_parameters` and so on; the aggregate `drift_parameters` reports
the task index. Drift on `tasks[i].<field>` passes task `i`'s own guards when
the switch is built: a field outside the task's `drift_fields` is refused, and
so is whatever the task's `check_drift` refuses. An agent-driven process
(`Accumulate`) is refused on every field of a switch, because the switch does
not pass a task's signal on.

## Semantics

- **Resume (default).** Inactive task states are carried unchanged and
  continue on return, so each task is one continuing lifetime. Nothing resets
  when you switch.
- **Restart.** `on_switch="restart"` re-initialises the task on entry and
  reports a respawn for it. The reset runs only on switch ticks (a `lax.cond`
  on the switch flag); under `jax.vmap` a batched flag makes the cond evaluate
  both branches, as any JAX cond does. Measured on CPU (macOS, best of five
  chunks, period 64): MinAtar Breakout/Asterix restart 140k ticks/s against
  146k for resume; MEAL with two layouts 92–94k ticks/s for restart against
  50k for resume.
- **Observations** are zero-padded to a common shape (the maximum of each
  dimension). Tasks with different observation ranks are refused.
- **Actions** are discrete for native tasks, sized to the largest action count
  (pass `action_size=int(env.action_space.nvec[0])` to the learner; a smaller
  value is refused with both numbers); continuous tasks must share one action
  box, are never padded, and each task's own mask passes through unchanged.
  Each discrete task's mask disables the actions it does not have. `task` is a
  mask-affecting parameter, so a switch raises `space_change` for every agent
  whose mask differs between the two tasks (Breakout's 3 actions and
  Asterix's 5, for example). Two tasks with identical masks switch without the
  event; every change of task is still a declared drift switch (`task_changed`
  through the Oracle).
- **All tasks share an agent count.** Continuous/discrete mixes, mixed
  host/native sets and different agent counts are refused with clear errors.
- **Respawns** come from each task's public `respawn`: the tick on which a task
  ends an episode reports it, including the tick of a switch (the outgoing
  task's episode end), and a task that is entered again does not repeat its
  earlier respawn.
- **Checkpoints** record every task's own configuration in order, so restoring
  into a switch with swapped or reconfigured tasks is refused, and a switch over
  a task that cannot be checkpointed (a `compiled=True` simulator) refuses to
  save.
- The task index is hidden behind the Oracle as `task_parameters`; agents see
  only the observation and mask of the active task.

## Measured cost

`lax.switch` evaluates every branch under `jax.vmap`, so batched task-switch
lifetimes cost roughly the sum of all task steps, not just the active one.
Measured on macOS CPU with small native grids (256-tick chunks, 40 chunks,
single process):

| Tasks | Single lifetime | `vmap` over 4 lifetimes |
|---|---:|---:|
| 1 (plain grid) | 631k ticks/s | 1.81M ticks/s |
| 2 | 562k ticks/s | 1.19M ticks/s |
| 4 | 534k ticks/s | 0.73M ticks/s |

These are environment-step rates, not training throughput, and they depend on
how expensive each task is. A single lifetime pays only a small dispatch
overhead; batching multiplies it by the task count. Observation selects only
the active task: five repetitions of the two-task CartPole host switch gave a
median of **288 ticks/s** (range 284–297) on macOS. Native `lax.switch` still
traces every branch, so its compiled and vmapped cost grows with the task
count; five MinAtar repetitions gave a median of **1.40M ticks/s** (range
1.38–1.70M).

## Host switching and presets

`pp.tasks.switch` also accepts host-controlled environments (all host, never
mixed with native ones). The active host environment steps on the host and the
others are carried unchanged, with the same resume, padding, mask and
switch-event semantics. `on_switch="restart"` is refused for host environments,
because a host lift owns one simulator per lifetime. Presets include:

```python
# MinAtar, compiled JAX:
env = pp.presets.switching_minatar(period=100_000)

# Atari through EnvPool (Linux, optional GPL-2.0 dependency):
env = pp.presets.switching_atari(["Breakout-v5", "Asterix-v5"], period=100_000)

# Continual World through Meta-World (continuous host actions):
env = pp.presets.continual_world(pp.presets.CW10, period=1_000_000)
```

Atari uses `full_action_space=True` for every game, so all share 18 actions
and the compiled action shape never changes. Measured on the H200 (16 envs, 4
threads, 128-tick chunks): **about 400–430 ticks/s**, with no clear difference
between switching every tick and never switching at this scale (run-to-run
noise dominates). MinAtar switching measured about **1.4M ticks/s** on macOS
CPU. These are environment-step rates, not training throughput.

## Limits

- Tasks must be all functional JAX or all host-controlled, with the same agent
  count and matching observation rank; host/native mixes are refused.
- Discrete tasks are padded to the largest action count; continuous tasks
  (host lifts, or simulators lifted with `compiled=True`) must share action
  shapes and bounds. Restart is unavailable for every host task.
- Tasks may be `pp.make` environments or compiled lifts without drift of their
  own: the switch refuses a task that drifts by itself. A task's parameters
  start from its defaults and may drift through the switch's own mapping
  (`tasks[i].<field>`, above). `pp.presets.continual_world(...,
  compiled=True)` runs Meta-World tasks this way, so `SACRunner` compiles the
  whole lifetime (about 18× the host loop's throughput in our check).
- Restart resets only the task being entered; other tasks keep their state.
- No per-task observation dtypes beyond float32 (observations are cast).
