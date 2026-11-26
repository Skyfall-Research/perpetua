# Claim 1: throughput

**Hypothesis:** end-to-end lifetime throughput is at least 10× that of Gymnasium `AsyncVectorEnv` with a PyTorch PPO. Part of the [measured claims](../measurement.md), which has the instruments and the limits that apply to every number.

## Claim-1 comparison: AsyncVectorEnv + PyTorch PPO

**Verdict: partly supported.** The JAX-native path is 21× the baseline with
one worker and 41–49× with 128, in steps per second, and reached 10⁹ steps
123× sooner with both sides run. The compiled CPU bridge is 4.3–7.0× at
matched parallelism, short of 10×; the eager host lift was 0.30–0.33×.

The baseline stack for claim 1 lives in
`scripts/measure_baseline.py`: Gymnasium `AsyncVectorEnv` workers with a
CleanRL-style PyTorch PPO learner (no `torch.compile`) on the configured
device. `examples/measure_training_throughput.py` measures the framework's
own `PPORunner` (native, vmapped over identical lifetimes) and
`HostPPORunner` (the CPU bridge) with the same PPO shape: rollout 128 per
worker or lifetime, 4 epochs × 4 minibatches. Both sides run 60-second
wall-clock windows after warm-up, synchronize at update boundaries and count
environment transitions. Run on the H200 (2026-10-20; torch 2.11.0+cu130,
JAX 0.11.2, 256 CPUs; 3 seeds for the framework side, 2 seeds for the
128-worker baseline):

| Stack | Config | steps/s | vs baseline |
|---|---|---|---|
| Baseline (PyTorch) | 1 worker | 510 | 1× |
| Baseline (PyTorch) | 128 workers | 26.5k – 27.4k | 52–54× |
| Baseline (PyTorch) | 256 workers | 37.1k – 37.6k | 73–74× |
| Baseline (PyTorch), `torch.compile` | 256 workers | **39.7k** (best baseline) | 78× |
| Lifetime native `gymnax/CartPole-v1` | batch 1 | 10.9k | **21×** |
| Lifetime native (same dynamics) | batch 128 | 1.34M | **49×** |
| Lifetime native (same dynamics) | batch 512 | 4.29M | **156×** |
| Lifetime native + cyclic gravity drift | batch 512 | 4.23M | 154× |
| Lifetime host lift (`CartPole-v1`) | 1 lifetime | 167 | **0.33×** |

How to read the table:

- The two sides batch differently. A baseline row with N workers is one PPO
  learner collecting from N environments. A lifetime row with batch N is N
  independent PPO learners, one lifetime each, run together with `jax.vmap`.
  Both count environment transitions per second, so the ratios compare
  throughput, not the time to train one agent.
- The last column does not use one denominator. Baseline rows, batch 1 and
  the host lift are relative to the one-worker baseline (510 steps/s). Batch
  128 and batch 512 are relative to the 128-worker baseline (26.5k–27.4k
  steps/s).

Re-measured on 2026-11-06 with the compiled CPU bridge (`compiled=True`: the
simulator step is a host callback inside the compiled `PPORunner`), same PPO
shape, 30-second windows, 2 seeds, IQM shown; the baseline columns are the
2026-10-20 runs above at the same parallelism:

| CPU bridge | Slots | Eager host loop | Compiled | Compiled + overlap | Baseline (same workers) | Compiled vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Gymnasium `CartPole-v1` | 1 | 154 | **2,174** | — | 510 | **4.3×** |
| EnvPool `CartPole-v1` | 128 | 23,731 | **149,512** | 93,979 | 26.5k–27.4k | **5.5×** |
| EnvPool `CartPole-v1` | 256 | 45,177 | **261,663** | 208,588 | 37.1k–37.6k | **7.0×** (6.6× vs the compiled 39.7k) |

Compiling the bridge removes the per-tick Python dispatch that made it slower
than one AsyncVectorEnv worker: 14× faster for one Gymnasium env and 5.8–6.3×
for EnvPool batches, with bit-identical results (tested). Overlapping half
batches (`overlap=True`) is *slower* here: once the tick is compiled the
simulator is no longer the bottleneck, so halving the batch per tick only adds
ticks. It stays opt-in for slow simulators and is not bit-reproducible.

Projected wall-clock to the 10⁹ steps named in claim 1 (linear extrapolation
from the 60-second windows): baseline 128 workers ≈ 10.1 h; baseline single
worker ≈ 22.7 days; native batch 512 ≈ 3.9 min; native batch 1 ≈ 25.6 h;
host bridge ≈ 69 days.

Measured end to end on 2026-11-06 (H200, one seed each,
`examples/measure_training_throughput.py --total-steps 1000000000`), PPO
training on CartPole for 10⁹ environment steps. The steps are summed over the
batch: the native row is 512 independent learners at about 1.95M steps each,
and the bridge row is one learner on 256 simulator slots.

| Stack | Wall-clock to 10⁹ steps | Mean steps/s | Final per-tick reward |
|---|---:|---:|---:|
| Native lifetime, `gymnax/CartPole-v1`, batch 512 | **222 s (3.7 min)** | 4.50M | 0.999 |
| Compiled CPU bridge, EnvPool CartPole, 256 slots | **3,338 s (55.6 min)** | 300k | 0.998 |
| Baseline AsyncVectorEnv + PyTorch PPO, 128 workers | ≈ 10.1 h (extrapolated, not run) | 27k | — |

The two lifetime rows are real runs, not extrapolations, and both learned
the task (per-tick survival ≈ 1). The baseline row was an extrapolation from
its 60-second windows until the baseline was run to 10⁹ steps on 2026-11-18
(8.5 h, below). Against the extrapolation the native path reaches 10⁹ steps about 160×
sooner and the compiled bridge about 11× sooner; at matched parallelism (the
256-worker baseline, 37.4k steps/s, ≈ 7.4 h) the 256-slot bridge is about 8×
sooner, consistent with the 7.0× throughput ratio above.

Re-measured on 2026-11-18 with five seeds (H200, torch 2.15.0+cu132, JAX
0.11.2, same PPO shape; only this job and one resident process on the GPU
throughout), and with the baseline run to 10⁹ steps
(`scripts/measure_baseline.py --num-envs 128 --total-steps 1000000000`):

| Stack | Config | steps/s (range over 5 seeds) | Wall-clock to 10⁹ steps |
|---|---|---:|---:|
| Baseline (PyTorch) | 128 workers, 60 s windows | 27.6k – 31.6k (mean 29.3k) | — |
| Baseline (PyTorch) | 128 workers, to 10⁹ steps (1 seed) | 32.7k | **30,620 s (8.5 h)** |
| Lifetime native `gymnax/CartPole-v1` | batch 128, 60 s windows | 1.31M – 1.36M (mean 1.34M) | — |
| Lifetime native (same dynamics) | batch 512, to 10⁹ steps | 3.98M – 4.15M (mean 4.02M) | **241 – 252 s** |

- At matched parallelism (128) the native path is 41–46× the baseline: 46×
  against its 60-second windows and 41× against its rate over the full run.
  The baseline was faster over the full run than in the early windows; the
  cause was not measured.
- The native batch of 512 reached 10⁹ steps 123× sooner than the 128-worker
  baseline (mean 249 s against 30,620 s), with both sides run.
- The baseline ended at a mean episode return of 500, the maximum; the
  lifetimes ended at a per-tick reward of 0.999.
- The native batch 512 runs were about 10% slower than the single 2026-11-06
  run (222 s); the cause is unknown.

Verdict against the hypothesis:

- **Supported for the JAX-native path with identical dynamics**, in steps
  per second: 21× against one baseline worker, 41–49× against the 128-worker
  baseline (49× in the first measurement, 41–46× in the five-seed
  re-measurement above; 128 independent learners against one learner with 128
  environments), and 108× comparing each stack's best configuration (lifetime batch 512 vs the
  hardened baseline at 256 workers with `torch.compile`; the earlier
  best-vs-best figure was 156× against the un-hardened 128-worker baseline).
  The same measurement on different dynamics (`ContinualGrid-v0`,
  3.96–4.03M steps/s at batch 512) lands in the same range and is not used
  for the verdict. Hardening the baseline — doubling its workers and compiling
  it — moved its best from 27.4k to 39.7k steps/s and left the verdict
  unchanged.
- **Not supported at ≥10× for the CPU bridge, but no longer slower**: the
  eager host lift was 3× *slower* than one AsyncVectorEnv worker (167 vs 510
  steps/s) because every tick made several dispatch-bound device calls. The
  compiled bridge is 4.3× faster than the baseline at one worker, 5.5× at 128
  and 7.0× at 256 (EnvPool). The remaining gap to 10× is the per-tick host
  callback; no pinned-memory path was added, because the batches are small
  and transfer was not the measured bottleneck.
- **Drift is nearly free** in the native path: enabling cyclic gravity drift
  changed 4.29M to 4.23M steps/s (−1.3%).

Learning evidence (these are throughput harnesses, not quality benchmarks):
the baseline reaches episode returns 313–404 of 500 in its 60-second windows;
the native lifetime's per-tick survival mean is 0.997–0.999 with rare
failures, and `ContinualGrid-v0` reaches ≈0.072 collection reward per tick
(≈3.8× its random baseline). The metrics differ because a lifetime never
resets.

## Throughput harness (claim 1)

`examples/measure_throughput.py` times vmapped lifetimes of identical shape
through the framework's compiled runner: environment stepping, drift, metrics
and policy are inside the loop; warm-up and device synchronization are
excluded from timed regions; each seed runs a freshly initialized batch. The
metrics window is held fixed (128) across the sweep so the chunk dimension
measures call overhead rather than window size. Recorded 2026-10-19 with JAX
0.11.2, `--ticks 2048` (Mac) / `--ticks 8192` (H200), 3 seeds × 2
repetitions; values are IQM steps/s.

The values in this table were averaged with an earlier quantile-range
definition, not the trimmed-mean IQM used everywhere else, and have not been
re-recorded.

| Config | Mac CPU (chunk 16 / 128 / 1024) | H200 CUDA (chunk 16 / 128 / 1024 / 8192) |
|---|---|---|
| `ContinualGrid-v0`, batch 1 | 0.24 / 0.63 / 0.68 M | 0.030 / 0.032 / 0.032 / 0.032 M |
| `ContinualGrid-v0`, batch 64 | 2.49 / 2.74 / 2.76 M | 2.05 / 2.18 / 2.21 / 2.22 M |
| `ContinualGrid-v0`, batch 512 | — | 15.8 / 16.9 / 17.0 / 17.0 M |
| `gymnax/Pendulum-v1`, batch 1 | 0.24 / 0.74 / 1.01 M | 0.031 / 0.033 / 0.033 / 0.034 M |
| `gymnax/Pendulum-v1`, batch 64 | 4.14 / 4.74 / 4.93 M | 2.39 / 2.58 / 2.62 / 2.65 M |
| `gymnax/Pendulum-v1`, batch 512 | — | 18.7 / 20.1 / 20.5 / 20.7 M |

Observations:

- **Single-lifetime GPU throughput is launch-bound**: ~32 k steps/s at any
  chunk size, so batching lifetimes is what uses the device. Scaling from
  batch 1 to 512 is ~530× on `ContinualGrid-v0` and ~600–620× on
  `gymnax/Pendulum-v1`, near-linear from 64 to 512 (about 7.7× for 8× the
  batch).
- **CPU behaves differently**: a single lifetime reaches 0.24–1.0 M steps/s
  (dispatch amortization favors longer chunks), while batch scaling is modest:
  at batch 64, ~4× at chunk 1024 for both environments (4.1× for the grid,
  4.9× for Pendulum) and 10× (grid) and 17× (Pendulum) at chunk 16, where the
  single lifetime is slowest.
- A per-step attribution probe on CPU Pendulum (window 128) puts the metrics
  window at ~0.17 µs/step, the policy at ~0.12 µs/step and the remainder in
  the environment step plus the carried lifetime state — so pruning the
  carried `info`/metrics cost is a real optimization target, not a rounding
  error.
