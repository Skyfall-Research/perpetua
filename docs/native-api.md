# Lifetimes and drift

The exact contract of a lifetime: what `init` and `step` return, array shapes
and dtypes, the clock, and the order in which a tick applies the action,
drift and observation. Examples use `ContinualGrid-v0`. Read
[concepts](introduction/concepts.md) first for the vocabulary; other
environments and simulators are covered in [adapters](adapters.md).

## Gym-familiar names, continuing semantics

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("ContinualGrid-v0", size=8)
init_key, act_key, step_key = jax.random.split(jax.random.key(0), 3)
state, obs, info = env.init(init_key)
action = pp.policies.random(obs, info, act_key)
state, obs, reward, info = env.step(state, action, step_key)
```

`step` returns the new state plus Gymnasium's familiar observation, reward and
info, rather than one combined output record; the info carries the action
mask, respawn and space-change events and the lifetime clock `t`, and never
carries termination flags, task identity or drift parameters.

Split fresh keys for subsequent decisions; never reuse a consumed key. The
environment state is internal world/evaluator data. Policies receive only `obs`
and `info`, not the complete state. A respawn is a dynamics event, not a reset.
There are no `terminated`/`truncated` flags or synthetic episode boundaries.

## Array and time contracts

For `N` fixed agent slots and grid side length `S`:

| Value | Shape/type | Meaning |
|---|---|---|
| `obs` | `(N, 4)`, float32 | Row, column, goal row, goal column, normalized by `S-1` |
| `action` | `(N,)`, JAX integer array | Stay/up/down/left/right = 0/1/2/3/4 |
| `reward` | `(N,)`, float32 | +1 for collecting the current goal, otherwise 0 |
| `info['action_mask']` | `(N, 5)`, bool | Valid actions for the returned observation |
| `info['active_mask']` | `(N,)`, bool | Which slots a policy controls; `False` marks a stay-only slot |
| `info['respawn']` | `(N,)`, bool | Collection caused a respawn during this tick |
| `info['t']` | `Clock(hi, lo)`, scalar uint32 limbs | Global lifetime ticks |
| `info['space_change']` | `(N,)`, bool | Drift changed this agent's valid-action set this tick (evolving spaces) |

Single-agent runs keep the same slot axis (`N=1`). A slot with `active_mask`
false can only stay. Whether it still takes part in the world depends on the
environment. In `ContinualGrid-v0` and `CooperativeGrid-v0` an inactive slot is
absent and receives zero reward. The scripted slots of `PartnerGrid-v0` and
`OpponentGrid-v0` report `active_mask` false but block and contest cells and
receive reward (the joint +1 for a partner, the negated learner reward for an
opponent), so mask on `active_mask` to find the policy-controlled slots, not
the slots that earn reward. Masked/out-of-range integer actions become stay;
shape and dtype errors raise during eager execution or tracing. Supply actions
in a supported JAX integer dtype before compilation.

`DoorGrid-v0` (`size`, `num_agents`, `drift`) is a grid whose drifting `(4,)`
door vector opens and closes the up/down/left/right moves; it reports
`space_change` and is the reference example of an evolving action space. The
[evolving spaces page](evolving-spaces.md) documents the contract and the
space manager.

`observation_space` and `action_space` are Gymnasium descriptors. Their NumPy
sampling utilities are host-side; use a keyed JAX policy inside compiled runs.

Clock increments are exact through ten billion ticks and beyond, up to unsigned
64-bit capacity, without changing `jax_enable_x64`. `Clock.to_int()` is only for
host reporting; batched clocks must first select an individual lifetime.

## Drift and transition order

```python
env = pp.make(
    "ContinualGrid-v0",
    size=8,
    drift=pp.drift.Cyclic([(7, 7), (0, 0)], period=127),
)
```

The elementary drift processes are `Stationary`, `Cyclic`, bounded Gaussian
`RandomWalk` and the procedural `Procedural` stream (fresh uniform draws on
`[low, high]` every `period` transitions, mirroring benchmark generators that
emit a new task each episode). They operate on fixed-shape float32 parameter
arrays, independently of agent actions. The grid interprets its two parameters
as goal coordinates, clipped to its bounds and rounded to cells.

`Accumulate` is agent-driven: it declares `endogenous = True`, so the wrapper
passes the environment's privileged `signal(previous_state, action,
next_state)` to `step(state, key, signal)`. Each tick the value becomes
`clip(decay * value + gain * signal, low, high)`; optional `bands` declare a
switch whenever the value crosses one (either direction, at most one switch per
tick). `ForagerGrid-v0` is built for it: with `drift=pp.drift.Accumulate(...)`,
collecting adds mass, mass decays, and a move succeeds with probability
`1 / mass`. `pp.make("ForagerGrid-v0")` alone attaches no drift, so mass stays
at 1. The signal is privileged and never enters
observations or info; mass stays hidden unless an Oracle exposes
`mass_parameters`.

### Different schedules in one batch

`Cyclic` keeps its phase table in the process, so every lifetime of a compiled
batch follows the same table. `Schedule` keeps the table, period, phase count
and start offset in the drift state instead, and
`pp.runtime.assign_schedule` gives each lifetime its own:

```python
goals = [(float(i % 4), float(i // 4)) for i in range(20)]
env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Schedule(goals, 5))
bank = pp.drift.stack_schedules(
    [pp.drift.schedule(goals, 5, offset=15), pp.drift.schedule(goals, 5, offset=85)]
)


def start(seed, schedule):
    return pp.runtime.assign_schedule(env, pp.runtime.init_lifetime(env, seed=seed), schedule)


batch = jax.vmap(start)(jnp.arange(2), bank)  # one lifetime in phase 3, one in phase 17
batch = jax.vmap(pp.runtime.Runner(env, pp.policies.random, ticks=256))(batch)
```

`assign_schedule` is meant for the start of a lifetime, as here. Called
mid-lifetime it replaces the parameter without raising a switch event (it
does not set `boundary`) and without running the `after_drift` hook that a
task switch runs.

Schedules may differ in values, period, length and offset; new schedules reuse
the compiled programs, and a schedule with no assignment behaves exactly like
`Cyclic` (all tested). Every schedule assigned to an environment must have as
many phase rows as the environment's own `Schedule`: `stack_schedules` pads only
to the longest schedule in the bank, so build shorter schedules with `phases=N`
(`pp.drift.schedule(values, period, phases=N)`, and `Schedule(values, period,
phases=N)` for the environment), which pads a schedule to `N` rows and uses only
the first `len(values)` of them. A bank whose rows do not match raises
`ValueError: Schedules for this process must have N phase rows`. Together with
traced hyperparameters this vmaps seeds × drift schedules × hyperparameters ×
agent slots in one compiled call. Assigned schedules are state, so checkpoints keep
them; record them with your results, because metadata holds only the default.

Two combinators build schedules from them:

```python
# Stationary for 1,000 ticks, then a period-256 cycle (keeps cycling after its budget).
schedule = pp.drift.Sequence(
    [(pp.drift.Stationary(), 1_000), (pp.drift.Cyclic([(7, 7), (0, 0)], 256), 4_096)]
)
# A switching schedule with bounded noise on top: base plus offsets.
noisy = pp.drift.Sum(
    [
        pp.drift.Cyclic([(7, 7), (0, 0)], 256),
        pp.drift.RandomWalk(low=-0.4, high=0.4, scale=0.05),
    ]
)
```

`Sequence` runs processes in order for fixed tick budgets, and the last process
keeps running after its budget. Each process is initialised from the value the
previous one left. Continuity depends on the process: `Stationary()`,
`RandomWalk` and `Procedural` start from that value (clipped to their bounds),
while `Stationary(value)`, `Cyclic` and `Schedule` start at their own value, so
the parameter jumps there. `Accumulate` is agent-driven and cannot be used
inside `Sequence` or `Sum`; construction raises `ValueError`. Segments
may have different internal states (a `Sum` after a `Cyclic`, say); only the
running segment is stepped, so long schedules stay cheap per tick. `Sum` adds offset
processes (initialised at zero) to a base process (initialised at the
environment's value). Both declare a switch whenever a segment boundary or any
running process does, so recovery tracking sees every switch.

Drift addresses parameters **by name**, so several parameters can drift at
once, each with its own process:

```python
env = pp.make(
    "PartnerGrid-v0",
    drift={
        "goal": pp.drift.Cyclic([(3, 3), (0, 0)], 128),
        "partner": pp.drift.Cyclic([[1.0], [0.0]], 64),
    },
)
```

A bare process applies to the environment's default drifting field. The Oracle
exposes `{field}_parameters`, `{field}_phase` and `{field}_changed` for every
parameter. The aggregate `drift_parameters` and `phase` report the default
field; `boundary` fires when any drifting field declares a switch
(`drift_changed` is a deprecated alias of it).

`pp.compose` attaches a drift process without rebuilding the environment:
pass a registered id (extra keyword arguments go to `make`) or an existing
environment that supports `with_drift`. Rebinding with a bare process replaces
the default field's drift and keeps every other field's; a mapping replaces
exactly the named fields. Host-lift adapters cannot rebind drift after
construction, because each owns one mutable simulator, and say so explicitly;
every functional JAX environment, including the JaxMARL adapter, can.

```python
env = pp.compose("gymnax/CartPole-v1", pp.drift.Cyclic([(9.8,), (4.9,)], period=256))
rebound = pp.compose(
    pp.make("ContinualGrid-v0", size=4), pp.drift.RandomWalk(low=0, high=3, scale=0.2)
)
```

Each transition:

1. Applies the action using the current goal and masks.
2. Awards collection reward and respawns collectors away from that current goal.
3. Advances the independent drift RNG/process and lifetime clock.
4. Returns the next observation, masks and observed respawn events.

A cyclic phase starts after `period` transitions. Goal motion is ordinary
observable world behavior; the phase, schedule parameters and switch flag
are not included in default public info.

## Privilege is explicit

```python
env = pp.Oracle(env, expose=["phase", "drift_parameters"])
```

Selected fields appear under `info['oracle']`. Every environment supports
`phase`, `drift_parameters`, `task_id` and `boundary`, plus per-parameter
`{field}_parameters`, `{field}_phase` and `{field}_changed`. `task_id` and
`boundary` mean "which piecewise-stationary segment is active" and "a declared
switch happened this tick":

```python
envs = [pp.make("ContinualGrid-v0", size=4), pp.make("gymnax/Acrobot-v1")]
schedule = pp.drift.Cyclic([0.0, 1.0], 1_000)
env = pp.Oracle(pp.tasks.switch(envs, schedule), expose=["task_id", "boundary"])
```

`task_id` is the active task index for a task switch, the regime for MORPHEUS,
and the primary drift phase otherwise. The primary field is the
environment's declared default drifting field (`default_drift_field`), even when
only other fields drift; an environment that declares none uses its first
drifted field. `phase`, `task_id` and
`drift_parameters` follow that single field, whereas `boundary` is the any-field
flag: when only another field drifts, `boundary` fires while `phase` and
`task_id` stay at 0 (measured on `PartnerGrid-v0` with a drift on `partner`
alone). To feed a one-hot task code from a field that is not the primary one,
use that field's `{field}_phase` instead of `task_id`. `boundary` is a declared switch
(a change of `phase`, see `pp.drift.switched`), not raw parameter movement:
gradual processes such as `RandomWalk` never raise it. `env.metadata` and `runner.metadata`
record the exposure tuple. Include that metadata in experiment reports, as
`examples/continual_grid.py` does. Compile a separate runner for each exposure configuration.

## Compiled chunks and metrics

```python
carry = pp.runtime.init_lifetime(env, seed=0, window_size=128)
runner = pp.runtime.Runner(env, pp.policies.random, ticks=256)
for _ in range(10):
    carry = runner(carry)  # Continue; do not initialize again.
print(carry.metrics.mean)
```

The carry holds environment/drift state, public observations/info, separate
environment/policy keys, and metric state. Drift has its own key in world state.
`jax.vmap(runner)` batches identically shaped lifetimes. A runner is compiled once
and reused; it does not retain per-tick scan outputs or precompute lifetime keys.

Treat environment, drift and runner configuration as static after compilation.
Construct a new environment/runner for a different configuration; runtime drift
values change through state, not by mutating the configured Python objects.

Every drift implements a host-side `configuration` property for reproducible
checkpoint metadata. Custom drifts must supply a JSON-compatible configuration
describing their static behavior. These reporting fields are never policy inputs.

Metrics provide lifetime `mean`, window `window_mean`, `ema`, Welford's
running `reward_variance` and `reward_std`, an exact `ticks` counter, and
cumulative `healthy` status. The bounded ring buffer costs
`O(window_size * N)` memory. Lifetime sums, including the Welford sum of
squared deviations, use compensated float32 accumulation; means and variances
retain float32 precision (the tick count they divide by is a float32 too), not
exact arbitrary-length real arithmetic.
Inactive-slot zeros count as ticks. NaNs propagate and set `healthy=False`;
callers should inspect health at chunk boundaries, as `examples/continual_grid.py` does.

`runtime.tick` also feeds the backend's hidden switch flag into the
carry (via `runtime.drift_switch`), so `carry.metrics.recovered_at`,
`since_switch` and `switch_baseline` report time-to-recover on the same clock
with no separate reward trace; the [evaluation page](evaluation.md) states the
exact rule and its relation to the post-hoc `recovery_ticks`.

`Runner` and the compiled learner runners (`PPORunner`, `DQNRunner`,
`SACRunner`) accept `donate=True`, which donates the input carry's buffers to
the compiled call so windows, replay buffers and states update in place. The
carry passed in is consumed; keep only the returned one. Results are
bit-identical to `donate=False` (tested for all four runners).

Every runner follows one written contract, the `pp.runtime.ChunkRunner`
protocol: `runner(carry)` returns the next carry, `runner.metadata` is the
configuration a checkpoint is checked against, and the read-only
`runner.donates` says whether the call consumes the carry passed in. A wrapper
forwards `donates` from the runner it wraps (`ProbedRunner` does), so
`pp.checkpoint.replay` keeps every returned carry readable for a donating
runner whether it is bare or wrapped. The compute-aware runners take the
chunk's timing trace as a second argument and never donate.

`Runner`, `HostRunner` and every learner runner accept an optional `reference` policy: every tick
then also steps that policy from the learner's pre-action state (one extra
environment step, `runtime.shadow_reward`) and the carry accumulates same-state
regret in `metrics.regret_total` and the bounded `regret_window`. Without a
reference the carry, RNG streams and cost are unchanged; see the
[evaluation page](evaluation.md) for the semantics and measured cost.

## Current limits

- Agents in `ContinualGrid-v0` do not interact; it is not a MARL benchmark
  (see [multi-agent](marl.md) for worlds where they do).
- Checkpoints are taken at returned runner boundaries only, not mid-rollout.
- Tests check bounded carry shapes and chunk equivalence, not billion-step
  memory behavior or throughput.
