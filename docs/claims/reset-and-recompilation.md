# Claims 2 and 3: the reset path and recompilation

**Hypotheses:** removing the reset path saves compute against a gymnax-style auto-reset (claim 2), and drift never recompiles, against naive JAX task switching (claim 3). Part of the [measured claims](../measurement.md), which has the instruments and the limits that apply to every number.

## Results

**Verdicts.** Claim 3 is supported: after one compile, the same program
covered all 100 switches with zero compile events, against 102 for the naive
baseline. Claim 2 is partly supported: static FLOPs and single-lifetime
transition loops favour the lifetime step, but a batched (`vmap`) conditional
keeps part of the reset cost, and one of the 20 environments in the H200 sweep
was slower without the auto-reset when batched (see the [sweeps](#sweeps-behind-the-figures) below). Claim 2 was measured as a
transition microbenchmark, not as end-to-end training.

Conditions: 100 phases × 64 ticks (6,400 ticks, 100 drift switches), chunk
1,280; B2 loops of 8,192 ticks with a fixed zero action, best-of-3 with the
worst case in the JSON. Recorded 2026-10-19 with JAX 0.11.2; the two FLOP rows
were re-measured on 2026-11-14. The script's defaults are smaller (4 ticks per
phase, chunk 200, 4,096-tick loops), so these conditions need the flags:

```bash
uv run python examples/measure_claims.py --phases 100 --ticks-per-phase 64 \
    --chunk 1280 --b2-ticks 8192
```

| Measurement | Mac (CPU, Apple Silicon) | H200 (CUDA) |
|---|---|---|
| B1 lifetime compile events after warm-up (cold / warm) | 1 / **0** | 1 / **0** |
| B1 lifetime wall time (cold / warm, 6,400 ticks) | 0.21 s / 0.040 s | 1.22 s / 0.203 s |
| B1 naive baseline (new jitted task constant per phase) | **102 events**, 1.40 s | **102 events**, 9.05 s |
| B2 `ContinualGrid-v0` FLOPs (step / reset / autoreset) | 340 / 373 / 835 (**+146%**) | 751 / 861 / 1,864 (**+148%**) |
| B2 `ContinualGrid-v0` steps/s (lifetime vs autoreset) | 0.30 M vs 139 k (**2.2×**) | 93.5 k vs 59.0 k (**1.6×**) |
| B2 `gymnax/Pendulum-v1` FLOPs (step / reset / autoreset) | 116 / 278 / 517 (**+346%**) | 252 / 618 / 1,130 (**+348%**) |
| B2 `gymnax/Pendulum-v1` steps/s (lifetime vs autoreset) | 20.3–21.1 M vs 265–276 k (**74–79×**) | 101.7 k vs 56.1 k (**1.8×**) |
| B3 adapters (21 files) / examples (13 files) code lines | 5,262 / 1,978 | 5,262 / 1,978 |

The naive row runs 8 ticks per phase (800 ticks in all, whatever
`--ticks-per-phase` is) on a toy 4 × 4 gridworld step, with one jitted
function per phase and a Python loop over ticks. It is not `ContinualGrid-v0`
and does not do the lifetime's work, so only its compile-event count is
comparable with the lifetime's; its seconds are not. The
[Switching](#sweeps-behind-the-figures) sweep below is the like-for-like wall
time. B3 counts `src/perpetua/adapters/*.py` and `examples/*.py` with
`pp.measurement.code_lines` (physical lines, excluding blanks, comments and
docstrings), recounted on 2026-11-24; the counts do not depend on the backend.

Observations:

- Drift is data: after one cold compile the same compiled runner covered all
  100 switches with **zero** compilation events on both backends, while the
  naive new-constant-per-phase baseline compiled 102 times — ≈9 s of GPU
  stall in this run.
- The reset path is real work: an autoreset step computes the step and the
  reset and selects between them, so its static FLOPs are about the sum of
  the two (+146% for the grid, +346% for Pendulum). The measured transition
  loops are 1.6–2.2× slower with the both-branch autoreset on the grid and up
  to 74–79× slower on CPU Pendulum, where the reset's sampling and state
  construction dominate a very cheap step.
- On the GPU, tiny transitions are launch-bound, so the wall-clock penalty is
  smaller than the static FLOPs suggest; the CPU numbers show the reset's
  intrinsic cost more clearly. Both are reported.
- **Batching keeps part of the cost.** Perpetua's adapters reset only on a
  termination, under `lax.cond`. Unbatched, XLA runs only the taken branch, so
  an environment that rarely or never terminates skips the reset entirely.
  Under `jax.vmap` a conditional evaluates both branches, so batched
  environments that can terminate still pay part of the reset cost. The B2
  rows above measure the hand-written wrappers and native grids.
- **The generic gymnax adapter** calls gymnax's `step_env` and resets only on
  termination, using gymnax's key split, so the claim-2 saving also applies to
  the 20 environments behind it. Calling gymnax's `step` instead would compute
  a reset every tick. Over 512 compiled ticks the two give identical
  observations, rewards and respawns on 18 of 20 environments; Acrobot and
  Reacher first differ by one float32 ulp, from different XLA fusion (Acrobot
  is chaotic, so that grows to 6e-3). Measured on macOS CPU (best of three,
  2,048-tick scans):

  | Environment | Single lifetime, `step` → `step_env` | `vmap` over 64, `step` → `step_env` |
  |---|---:|---:|
  | `gymnax/Acrobot-v1` | 268k → 296k ticks/s | 3.44M → 3.44M ticks/s |
  | `gymnax/Breakout-MinAtar` | 164k → 247k ticks/s | 1.84M → 1.99M ticks/s |
  | `gymnax/Catch-bsuite` | 170k → 257k ticks/s | 3.51M → 3.44M ticks/s |

  The batched rates are unchanged within noise, as the both-branch rule
  predicts.

## Sweeps behind the figures

`scripts/claim_figures.py` extends B1 and B2 to per-phase and
per-environment data and adds a learner-by-benchmark grid;
`scripts/plot_claim_figures.py` draws them (it needs matplotlib:
`uv run --with matplotlib`). Recorded with JAX 0.11.2: macOS CPU (Apple
Silicon) on 2026-11-17, and the `switching` and `reset` sweeps again on the
H200 (CUDA) on 2026-11-18. The GPU is shared with another process, which
was idle before and after the sweeps.

**Switching (claim 3).** A 100-phase lifetime, 64 ticks per phase, with a
task value never seen before in every phase. The baseline builds each task
with its value as a compile-time constant and jits it, so a value seen again
would reuse its program; here every value is new, its worst case.

| Environment | Backend | Lifetime (drift as data) | One program per task value |
|---|---|---|---|
| `ContinualGrid-v0` (10 × 10) | Mac CPU | 0.22 s, **0** compile events after phase 1 | 11.3 s, **99** compile events after phase 1 |
| `mjx/Pendulum-v0` | Mac CPU | 1.1 s, **0** compile events after phase 1 | 96.8 s, **99** compile events after phase 1 |
| `ContinualGrid-v0` (10 × 10) | H200 CUDA | 1.8 s, **0** compile events after phase 1 | 126 s, **99** compile events after phase 1 |
| `mjx/Pendulum-v0` | H200 CUDA | 14.5 s, **0** compile events after phase 1 | 589 s, **99** compile events after phase 1 |

The baseline pays about the same compile in every phase (median 0.11 s for
the grid and 0.93 s for MJX Pendulum on the Mac; 1.3 s and 6.0 s on the
H200), so its cost follows the environment's compile time, not the phase
length.

**Reset path (claim 2).** The B2 comparison on gymnax environments,
2,048-tick scans, best of three, a single lifetime and `jax.vmap` over 64.
`scripts/claim_figures.py reset` iterates gymnax's registry of 24 ids, of
which 22 are registered here (`SimpleBandit-bsuite` and `PointRobot-misc` are
not): the 20 that run through the generic adapter plus the hand-written
`gymnax/CartPole-v1` and `gymnax/Pendulum-v1`. The Mac CPU figures come from a
run of that command with its defaults on 2026-11-23 and cover all 22; the
H200 figures below are from the 2026-11-18 run, which covered 20
environments (its per-environment data is not kept, so the two missing ones
cannot be named). On the Mac CPU:

- Static FLOPs of an auto-reset step are 1.05–4.5× those of a lifetime step
  on 21 environments. `gymnax/Pong-misc` is 9.2×: its state holds a
  30 × 40 × 3 frame buffer, and selecting between two copies of it accounts
  for about 3,600 of the 4,411 FLOPs.
- A single lifetime runs 1.1–2.7× faster without the auto-reset on 20
  environments; `gymnax/Pendulum-v1` (69×) and
  `gymnax/DiscountingChain-bsuite` (42×) have steps cheap enough for the
  reset to dominate.
- Under `vmap` over 64 the lifetime loop is 1.1–4.5× faster.
  `gymnax/Asterix-MinAtar` gains least (1.15× single, 1.06× batched): its
  step is about 100 times the cost of its reset.

On the H200, with `jax.vmap` over 4,096 as well:

- A single lifetime runs 1.0–1.6× faster without the auto-reset. The two
  CPU outliers are gone (`gymnax/Pendulum-v1` 1.6×,
  `gymnax/DiscountingChain-bsuite` 1.4×): a single GPU transition is
  launch-bound, as in the B2 rows above.
- Under `vmap` over 4,096 the lifetime loop is 1.1–1.7× faster on 19
  environments, and 1.2–1.7× on the same 19 over 64.
- `gymnax/Asterix-MinAtar` is slower without the auto-reset when batched
  (0.84× over 64, 0.87× over 4,096). This is unexplained and was measured
  once.
