# Claim 4: tuning cost

**Hypothesis:** tuning becomes affordable, measured as GPU-hours for a fixed sweep against per-configuration job arrays. Part of the [measured claims](../measurement.md), which has the instruments and the limits that apply to every number.

## Claim-4 measurement: tuning cost

**Verdict: supported.** A 16-configuration sweep took 15.2× fewer GPU-hours
as one vmapped job than as a job array, and sweeps of 64 to 1,024
configurations took 50–726× fewer.

`scripts/measure_tuning.py` measures claim 4: a fixed
sweep of **seeds × hyperparameters** as one vmapped compiled job versus the
per-config job-array baseline. Both modes do identical per-lifetime work
(PPO, 128-tick rollouts, 100 updates, the same drifting `ContinualGrid-v0`)
and both include compilation, because that is what the baseline pays per
configuration.

Recorded on the H200 (2026-10-20; 4 hyperparameter sets × 4 seeds = 16
configurations):

| Mode | Wall-clock | GPU-hours | Per configuration |
|---|---|---|---|
| One vmapped job (traced hyperparameters) | **10.7 s** | **0.0030** | 0.67 s |
| Per-config job array (16 separate jobs) | 163.0 s | 0.0453 | 10.2 s |
| Ratio | **15.2×** | **15.2×** | 15.2× |

The gap is dominated by compilation, which the job array pays once per
configuration (≈9.5 s each here) while the vmapped job pays it once in total;
the training work is identical. The job array's 10.2 s per configuration
scales linearly: 100 configurations would take about 17 minutes. The vmapped
job does not scale that way (the table below has 64 configurations in 12.7 s),
so no 100-configuration figure is projected for it.

Larger sweeps, recorded on the H200 on 2026-11-19 (JAX 0.11.2; 16
hyperparameter sets × 4, 16 or 64 seeds; one run of each cell, with only this
sweep and one resident process on the GPU):

| Configurations | Updates | One vmapped job | Per-config job array | Ratio |
|---:|---:|---:|---:|---:|
| 64 | 100 | 12.7 s | 668 s | 53× |
| 64 | 1,000 | 25.7 s | 1,297 s | 50× |
| 256 | 100 | 13.4 s | 2,678 s | 199× |
| 256 | 1,000 | 23.0 s | 5,100 s | 222× |
| 1,024 | 100 | 14.6 s | 10,622 s (3.0 h) | 726× |
| 1,024 | 1,000 | 35.0 s | 20,608 s (5.7 h) | 589× |

The job array costs a median of 10.4 s per job at 100 updates and 20.3 s at
1,000, at every sweep size, so its total is linear in the number of
configurations. The vmapped job went from 12.7 s to 14.6 s across a 16-fold
larger sweep at 100 updates. Each cell was run once, and the vmapped times
vary by a few seconds between runs (256 configurations at 1,000 updates
took less than 64), so the ratios are indicative to about 10%.
`scripts/plot_claim_figures.py tuning` draws the grid.

Limits: the baseline is an honest emulation of one-job-per-config on a single
device — a cluster runs those jobs in parallel, which reduces wall-clock but
not GPU-hour totals. This is one sweep of one workload on one GPU, and
hyperparameters outside the traced set (network width, rollout length, batch
size) still need separate programs.
