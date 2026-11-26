# Optional EnvPool CartPole snapshot binding

This package builds **only** the headless CartPole and synchronous worker core
from EnvPool 1.2.7, revision `9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3`.
It does not replace the installed EnvPool wheel, copy its floating-point
observations back into physics, or approximate its RNG with action replay.

The pinned `core.patch` exposes the private base step counter and tracks worker
unwinding after state-buffer completion. Physics and reset logic remain upstream
CartPole code. This standalone binding deliberately refuses rendering, partial
batches and async send/recv; it is not a fork of the entire simulator catalogue.

## Local build (Linux)

```bash
uv pip install --python .venv/bin/python native/envpool_snapshot
.venv/bin/python examples/control_ppo.py --backend envpool-snapshot \
  --updates 2 --checkpoint checkpoints/envpool-001
.venv/bin/python examples/control_ppo.py --backend envpool-snapshot \
  --updates 1 --resume checkpoints/envpool-001
```

Requires a C++17 compiler, Git and HTTPS access to the pinned source archives.
The isolated build downloads scikit-build-core/pybind11 and builds Abseil;
no Bazel, graphics libraries, driver changes or simulator-wide rebuild is needed.
Ordinary EnvPool remains available and non-checkpointable unless `snapshot=True`
is explicitly requested in `pp.make("envpool/CartPole-v1", ...)`.

## State and timing

Native `snapshot()` returns owned numeric arrays: configuration uint32 (4,),
physics float64 (N, 4), complete MT19937 uint32 (N, 625) and counters int64 (N, 5).
Configuration contains version, slots, threads and legacy episode limit. Counters
contain seed, base step, CartPole step, done and force-reset flags. Perpetua adds
its bool (N,) pending-reset mask and retains lifetime, drift, learning and keys in
the complete checkpoint.

Only complete consumed synchronous batches are snapshot boundaries. Snapshot and
restore refuse concurrent in-flight operations, wait for final worker unwind,
and never serialize C++ pointers, queues, mutexes, buffer addresses or threads.
Empty allocation queues are reconstructed from the initialized target rather
than treated as trajectory state. Every restored slot is validated before any
physics/RNG is mutated.

As with Gymnasium snapshots, each Perpetua step restores its numeric state before
stepping, then captures the next one. This permits replay/retry/fresh-instance
resume at the cost of per-step copies of MT19937 state; no throughput gain is
claimed. Deterministic continuation requires matching software/backends.
Use one Perpetua host controller per backend; native operation locks do not
make concurrent Python restore/step pairs atomic. Parallel lifetimes need
separate pools.

## Upstream notices

EnvPool and Abseil are Apache-2.0; ThreadPool has its upstream zlib license;
concurrentqueue has BSD/Boost licensing and its semaphore has a separate zlib
license. pybind11 is BSD licensed. The build includes upstream license materials
in `share/perpetua-envpool-snapshot/licenses` in the native wheel. Modified
upstream files have explicit Perpetua modification notices in `core.patch`.

Perpetua's original code is MIT-licensed (see the top-level `LICENSE`); the
upstream components keep the licenses listed above.
