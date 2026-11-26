# Reproducibility

What you can rely on when you repeat a run, and what you cannot.

## In brief

| Across | Guarantee |
|---|---|
| Runs with the same seed, software and backend | identical, tick for tick |
| A checkpoint and its continuation | identical to the uninterrupted run, on the same software and backend |
| CPU and GPU, or two different GPUs | not bit-identical |
| Versions of Perpetua | not guaranteed; the changelog lists changes that alter results |
| Seeds | different lifetimes; report several |

## Seeds

One seed reproduces a whole lifetime: the world, its drift, the policy's
sampling and the learner's shuffling each have their own random stream derived
from it. Two environments built the same way and run with the same seeds give
identical states and rewards, which [`pp.testing.check_env`](testing.md)
checks for any environment.

There is no global seed and no `reset(seed=...)`. See
[seeding](gym-users.md#seeding) for how keys are passed. A seed is an integer in
`[0, 2**32)` (JAX keys hold 32 bits unless `jax_enable_x64` is set): a larger
or negative Python integer raises instead of wrapping, so two seeds that differ
by a multiple of 2**32 can never silently give the same lifetime. Reduce a
64-bit value (`time_ns()`, a hash) with `seed % 2**32`.

## Checkpoints

`pp.checkpoint.save` stores the whole lifetime at a runner boundary: world,
drift, random keys, metrics and learner, including replay buffers. A restored
run continues exactly as the uninterrupted run would have, and
`pp.checkpoint.replay` re-runs a segment exactly. This is tested on matching
software and backends, including resuming on CPU in a fresh process. The
[training guide](../training.md#save-and-resume) has the details.

Restore refuses a checkpoint whose configuration, shapes or state layout
differ from the current code; see the [FAQ](faq.md#checkpoints). It does not
checksum the stored arrays, so a flipped byte inside a structurally valid
checkpoint file is not detected (see the
[training guide](../training.md#save-and-resume)).

A checkpoint records what wrote it: the Perpetua version and source revision,
the JAX and Orbax versions, the backend, Python and platform.
`pp.checkpoint.provenance(path)` reads the record back, and `restore` warns,
without refusing, when the running software differs from it. It warns in the
same way when the record cannot identify the code: only one side knows its
revision, or either side had uncommitted changes, so restoring in a checkout
with uncommitted changes always warns.

`pp.checkpoint.save`, `restore` and `replay` take a complete carry. A
`TorchRunner` keeps its agent's networks and optimizer in the PyTorch agent,
outside the carry, so those three refuse it; use `TorchRunner.save` and
`TorchRunner.restore`, which write and load the lifetime and the agent
together.

## 64-bit mode

Perpetua never enables `jax_enable_x64`, and the documented runs use JAX's
32-bit default. With the flag on the same seed gives a different lifetime (JAX
draws 64-bit random numbers), integer state becomes `int64`, and a checkpoint
written in the other mode is refused: the error names the leaf whose dtype
differs and says to set the flag the same way when saving and restoring. The
hand-written gymnax wrappers, the MJX lifetimes and SAC keep their float32
observations and carries under the flag (covered by the test suite); that is a
smoke check, not a promise that results match the 32-bit ones.

## Backends

Results are not bit-identical between CPU and GPU, or between different GPUs.
Small floating-point differences between backends can grow over a learning
run, so one seed may end somewhere different on each. Compare distributions
over seeds across backends, never single runs.

## Versions

A run is reproducible with the code and dependencies it used.
`pp.reporting.provenance()` returns them as a JSON-ready dict: the Perpetua
version, the source revision, the JAX version, the backend, Python and the
platform. Put it in every result you write; the example and measurement
scripts do, under the `provenance` key, and checkpoints record it.

```python
import perpetua as pp

pp.reporting.provenance()
# {"perpetua": "0.0.1a0", "revision": "<commit hash>", "dirty": False,
#  "jax": "...", "backend": "cpu", "python": "...", "platform": "Linux-x86_64"}
```

Where `revision` comes from depends on how Perpetua was installed:

| Install | `revision` | `dirty` |
|---|---|---|
| A git checkout, including `uv sync` and `pip install -e .` | the checked-out commit, read from git when the result is produced | whether tracked files have uncommitted changes now |
| From a git URL (`pip install "git+https://..."`) | the commit the installer recorded | `None` (not recorded) |
| A copy built from a clone (`pip install .`, `uv pip install .`, `uv build`) | the commit the clone was at when the wheel or sdist was built | whether tracked files had uncommitted changes at build time |
| A wheel or sdist built outside a git checkout, or a checkout on a machine without `git` | `None` | `None` |

The three sources are tried in that order, so a checkout always reports its
live state. A copy built from a clone does not follow the clone afterwards:
after `git pull`, install again. A dirty revision does not identify the code,
so commit before a run you intend to report. When `revision` is `None` only
the version is known, and until releases are published a version does not
identify a commit.
Keep the dependency lock file with the result as well: the record names JAX
but not every dependency. Across versions, the changelog's "Changes results"
section lists every change that alters a trajectory or a learner's numbers
for a given seed.

That list is checked, in part, by a test. `scripts/record_baselines.py` holds
27 short seeded runs (a random policy on one environment of each family that
installs everywhere, and PPO, IPPO, DQN, IQL and SAC with their options for a
few updates), and a fixture per platform
(`tests/fixtures/baselines-darwin-arm64.json`,
`tests/fixtures/baselines-linux-x86_64.json`) holds a digest of each as an
earlier revision computed it there: mean reward per slot, final tick, health,
checksums of the final observation, for the random policies checksums of
every action and observation, and for the learners the last losses and sums
over each parameter module. `tests/test_baselines.py` recomputes them and
compares with the fixture of the platform it runs on (CPU, 32-bit): exactly
on macOS arm64, and on Linux x86_64 with booleans and integers exact and
floats within a relative `LINUX_RTOL` (5e-5) plus an absolute 5e-7, because
two CI runners are not the same machine: recordings on two runner CPUs
differed by up to 7e-6 relatively. A change smaller than that bound, or one
to a number below about 1e-3, is caught only on macOS arm64. On a platform without a fixture the comparison is skipped,
not loosened. It covers these runs and nothing else: host simulators, the
Linux-only backends and long lifetimes are not in it.

## What is not reproducible

- **Live services.** A live MORPHEUS world keeps its clock and state on the
  service; the bundled mock is seeded and reproduces exactly.
- **EnvPool's asynchronous mode** (`overlap=True`): which half of the slots
  returns first depends on thread timing.
- **Measured compute time.** In [compute-aware time](../compute-time.md) the
  cost of a decision is a wall-clock measurement. Save the timing trace to
  repeat a measured segment.
- **Opaque simulators.** Meta-World and COOM lifetimes are seeded, but their
  simulator state cannot be saved, so they have no checkpoints. AgarCL
  checkpoints [by replay](../agarcl.md#checkpoints-by-replay). Compiled host
  lifetimes (`compiled=True`) cannot be checkpointed.

## Reporting

Report several seeds, with the interquartile mean and a confidence interval
([evaluation](../evaluation.md#reporting-with-iqm-and-confidence-intervals)).
State the backend, the Perpetua revision, any Oracle exposure
(`env.metadata["oracle_exposure"]`) and the tuning budget. The backend and the
revision are in `pp.reporting.provenance()`, Oracle exposure is
in the environment metadata for every environment, and a checkpoint records
the same provenance. The tuning budget is recorded only for preset
environments, where `pp.tuning.declared(env)` returns it; for any other
environment `declared` raises, so report the `TuningProtocol` or `TuningResult`
you used.
