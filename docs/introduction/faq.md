# FAQ and troubleshooting

Common questions first, then error messages and what to do about them.

## Questions

**How do I know when the world changed?** By default you don't: that is
privileged information, as it would be for an agent in the real world. Wrap
the environment in `pp.Oracle(env, expose=["boundary", "task_id"])` to receive
it in `info["oracle"]`; the exposure is recorded in
`env.metadata["oracle_exposure"]`, so results say what the agent was told.
For evaluation code, the schedule you wrote also gives the switch ticks:
`Cyclic(values, period)` switches at every multiple of `period` (the Oracle's
`boundary` flag arrives one step earlier; see
[evaluation](../evaluation.md#which-index-names-a-switch)).

**Where are `terminated` and `truncated`?** There are none. A death or a
reached goal is a respawn inside the lifetime, reported in `info["respawn"]`.
See [Perpetua for Gymnasium users](gym-users.md).

**What does `period` count?** World ticks (environment steps), per phase:
`Cyclic([a, b], 1000)` uses `a` for ticks 0-999 and `b` for ticks 1000-1999.

**Are results identical everywhere?** A lifetime is exactly reproducible
from its seed on the same software and backend, and `pp.checkpoint.replay`
repeats a segment exactly. Results are not guaranteed to be bit-identical
across platforms or devices (CPU against GPU, or different GPUs).

**Why is the first chunk slow?** It compiles. Build a runner once and keep
calling it; later chunks of the same shape do not compile again
(`pp.testing.check_env` checks this for an environment).

**Can I train on several CPU simulators at once?** EnvPool environments step
`num_envs` simulators in one batched call. COOM, AgarCL and Meta-World run one
simulator per lifetime (in a worker process for COOM and AgarCL), so more
lifetimes mean more processes.

## Installation and platforms

**`ImportError` when importing `perpetua.agents`.** The learners need the
`training` extra: `uv sync --locked --extra training`.

**"Install Perpetua's metaworld extra to use Meta-World tasks."** Each
optional integration is an extra; the [installation](installation.md) page
lists them.

**"Install Perpetua's cpu-bridge extra and EnvPool's runtime dependencies."**
EnvPool runs on Linux only. On Ubuntu it also needs the user-space GL
libraries `libegl1`, `libgles2` and `libopengl0`.

**"COOM runtime not found; run `python -m perpetua.adapters.coom_setup`".**
COOM and AgarCL pin dependencies that conflict with this project, so each runs
in its own runtime, built once by `python -m perpetua.adapters.coom_setup` or
`python -m perpetua.adapters.agarcl_setup`. Both ship in the installed package;
`scripts/setup_coom.py` and `scripts/setup_agarcl.py` in a checkout run the same code.

**Why does the `meal` extra pin `tfp-nightly`?** MEAL needs TensorFlow
Probability, and the newest stable release (0.25.0) does not import with JAX
0.11. A nightly CI job tries the stable release and reports when it works.

## Drift

**"Cyclic parameter shape does not match the environment: each phase has shape
(), the parameter has shape (1,)"** (raised when the lifetime is initialised). Each phase must have the parameter's
shape. For a shape-`(1,)` parameter write phases as tuples:
`Cyclic([(9.8,), (12.0,)], 1000)`. Every environment page has a working
drift example.

**"Unknown drift parameters ['g']; available: ['gravity']".** Names differ
between environments; the environment page lists the driftable parameters.

**"Tasks given to switch() must not drift themselves".** `pp.tasks.switch`
refuses a task that brings drift of its own. Build the tasks without `drift=`
and drift their fields through the switch's mapping (`tasks[i].<field>`).

## Learners

**"task_boundary_reset reads the task switch, which is privileged".** SAC's
`task_boundary_reset` acts on the task switch, so it needs
`pp.Oracle(env, expose=["boundary", ...])`; that records the privilege.

**SAC's temperature (`alpha`) climbs without stopping.** Since SAC uses the
exact tanh correction this should not happen; if it does, the usual cause is a
`target_entropy` the policy cannot reach. The default accounts for the action
bounds (see [Training](../training.md)); an explicit target must lie below
`sum(log(high - low))`. A temperature that rises and falls again is normal.

## CPU simulators and the compiled bridge

**"compiled=True runs the simulator in a host callback, which needs JAX's CPU
platform".** On a GPU host set `JAX_PLATFORMS=cuda,cpu` (not `cuda` alone).

**"CPU state is stale or forked".** A host simulator is one mutable object, so
each state can be stepped once: always continue from the state the last call
returned. To branch a lifetime, use an environment that supports snapshots
(see [CPU snapshots](../cpu-snapshots.md)).

**"compiled=True states are single-use; use snapshot=False."** Compiled
lifetimes cannot be snapshotted, vmapped or checkpointed.

**A process holds a simulator after it finishes.** Call `env.close()` on host
environments (COOM, AgarCL, EnvPool, Meta-World, MORPHEUS) when done.

(checkpoints)=
## Checkpoints

**A checkpoint does not restore.** A checkpoint restores only into the state
layout and configuration it was written with, and strict metadata checks refuse
anything else with "Checkpoint configuration does not match this experiment" or
"Checkpoint structure, shapes or dtypes do not match the template", followed by the
key or leaf that differs. Rebuild the
runner with the settings the checkpoint was saved with, and use the same
Perpetua revision. The repository's `CHANGELOG.md` lists the changes that alter
checkpoints.

## Memory and long runs

**Host memory grows during a long CPU run.** glibc keeps a memory arena per
thread, which can look like a leak. Set `MALLOC_ARENA_MAX=2` or preload
tcmalloc. In a 10⁸-tick CPU learner run the default allocator grew by 64 MiB,
while either setting kept growth within a few MiB. The GPU backend did not show
this growth.

**`CUDA_ERROR_OUT_OF_MEMORY` when the GPU is shared.** JAX preallocates most
of the GPU by default. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so it
allocates on demand, and run fewer processes per GPU.

**"pthread_create failed: Resource temporarily unavailable" with many
processes.** Each JAX process starts thread pools sized to every core of the
machine and can hit the container's thread limit. Pin each process to a few
cores, for example `taskset -c 0-3 python ...`; the pools then size to those.

**Use donation for large carries.** `donate=True` on `Runner` and the learner
runners updates replay buffers and states in place (it halved peak device
memory in our measurement); keep only the returned carry.
