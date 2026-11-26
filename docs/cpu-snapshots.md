# Snapshot-backed Gymnasium CPU lifetimes

```{admonition} Experimental
:class: warning

This feature is kept working and tested, but it is frozen: no new work is
planned on it, and its interface may change. Build on it with that in mind.
```

## Supported backend

Headless Gymnasium **1.4.0 CartPoleEnv**, with any chain containing only exact
`TimeLimit`, `OrderEnforcing` and `PassiveEnvChecker` wrapper types. Subclasses,
unknown wrappers, render resources and unsupported RNGs are not silently treated
as checkpointable. Ordinary EnvPool remains non-checkpointable; its separate
`native/envpool_snapshot/README.md` supports complete
synchronous CartPole snapshots.

```bash
uv run --extra training python examples/control_ppo.py \
  --backend gymnasium --updates 2 --checkpoint checkpoints/cpu-001
uv run --extra training python examples/control_ppo.py \
  --backend gymnasium --updates 1 --resume checkpoints/cpu-001
```

The existing `pp.checkpoint.save` / `restore` API works unchanged on a complete
`HostPPORunner` state. A fresh backend and initialized template provide the target
structure/device; saved state, not the template seed, determines continuation.

## Snapshot contract

CPU backends declare `checkpointable` and implement `snapshot()` / `restore()`.
Unsupported backends raise `NotImplementedError`. The CartPole codec records a
versioned, numeric `CartPoleSnapshot` containing:

- Float64 physical state and every numerical CartPole dynamics parameter.
- Integration mode, reward mode and post-termination counter.
- Full PCG64 state/increment, cached uint32 flag/value and simulation seed.
- TimeLimit elapsed steps, order-enforcement and checker flags.
- Wrapper types/order and static limits/settings, checked even on direct restore.
- Checker observation-reuse diagnostic state, including its previous observation.

The two PCG64 128-bit integers are stored as uint32 limbs, not rounded floating
point or arbitrary Python pickles. Float64 physics stays in NumPy snapshots and
Orbax storage; this does not enable global JAX 64-bit mode. Public observations
retain the original float32 contract.

Wrapper layout, limits, codec and Gymnasium version are included in configuration
metadata. Snapshots also carry numeric wrapper configuration so direct backend
restore rejects incompatible layouts or limits. Restore checks version, shapes,
dtypes and flags before applying state.
The backend codec accesses known private fields because Gymnasium provides no
universal export operation; version/support checks keep this limitation explicit.

## Execution and ownership

For supported CartPole, `CPUState.snapshot` holds a small numeric snapshot and
the opaque owner is absent. Each host transition restores that snapshot **before**
advancing the simulator and captures the next one. This makes stale-state replay,
branching, retry after a failed transition and fresh-instance continuation
deterministic on the tested stack, instead of accidentally using whatever state
remains in a mutable object.

After `pp.checkpoint.restore`, the returned CPU state is authoritative; the next
host step rehydrates the simulator before running. `observe` reads the returned
state. Use the explicit backend `restore` operation if inspecting raw Gymnasium
internals before that first step.

Snapshot arrays should be treated as immutable. External mutations of the raw
Gymnasium simulator are not part of a snapshot-backed transition unless captured
explicitly. Policies own their JAX sampling keys, saved in the full lifetime;
host action-space sampling helpers are not simulation RNG and are not captured.

Use one host controller per lifted lifetime. Replay/branching is sequential;
parallel lifetimes need separate backend instances. A snapshot does not make
concurrent Python restore/step pairs on one mutable backend atomic.

For backends without a codec, opaque owner/generation checks remain in place.
Their states cannot be forked or checkpointed, and a failed external transition
invalidates the previous state. No fake resume or action-history replay is used.

## Limits

Capturing/restoring a small CPU snapshot per step prioritizes correct state/RNG
ownership over peak throughput. It is not an optimized EnvPool checkpoint or a
generic codec for every Gymnasium environment. More backends need their own
complete, versioned export/import implementations.

Checkpoint only at returned runner/update boundaries. Bitwise continuation is
tested on matching software/backends, not promised across hardware or dependency
changes. Retain the lockfile with every checkpoint; the checkpoint itself
records the source revision.
