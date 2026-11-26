# Measured claims

Perpetua was designed around six claims about speed, cost and comparability:
that a lifetime-first design runs faster, wastes less compute, never
recompiles when the world changes, makes tuning cheaper, needs less glue code
per benchmark, and gives comparable metrics across benchmarks. Each is
treated as a hypothesis. This page has the verdicts, the instruments and the
limits; each claim's page has its results and the exact conditions.

## In brief

- A lifetime runs as one compiled program. On the JAX-native path that is
  41–49× the steps per second of a 128-worker PyTorch baseline; through the
  compiled CPU bridge it is 4.3–7.0×.
- Drift is data, so a switch never recompiles: 0 compile events over 100
  switches, against 102 for one program per task value.
- A sweep over seeds and hyperparameters runs as one job: 15.2× fewer
  GPU-hours than a job array for 16 configurations, and 50–726× fewer for 64
  to 1,024.
- One random agent runs on five benchmarks in 5 lines of glue code.
- The cross-benchmark metrics are a pipeline check, not a performance result.
- Every number has its conditions and limits on its claim's page.

## Claims and status

| # | Hypothesis | Baseline | Verdict | Details |
|---|---|---|---|---|
| 1 | End-to-end lifetime throughput is ≥10× higher | Gymnasium `AsyncVectorEnv` + PyTorch PPO | **partly supported** | [Claim 1](claims/throughput.md) |
| 2 | Removing the reset path saves compute | gymnax-style autoreset | **partly supported** | [Claims 2 and 3](claims/reset-and-recompilation.md) |
| 3 | No recompilation across drift | Naive JAX task switching | **supported** | [Claims 2 and 3](claims/reset-and-recompilation.md) |
| 4 | Tuning becomes affordable | Per-config job arrays | **supported** | [Claim 4](claims/tuning.md) |
| 5 | Integration cost drops | CW / COOM / AgarCL / MEAL / MORPHEUS repos | **supported** | [Claim 5](claims/integration.md) |
| 6 | Results become comparable | Per-paper metrics | **pipeline check only** | [Claim 6](claims/comparability.md) |

*Supported* means the measurement agrees with the hypothesis under the
conditions recorded on its page. *Partly supported* means it holds for
some configurations and not others. *Pipeline check only* means the code path
runs end to end with tiny budgets and no performance claim is made.

## Instruments

`perpetua.measurement` ships the exact, inert helpers (they wrap existing pure
functions and change no framework behavior):

- `count_compilations()` — context manager counting XLA `Compiling ...` events
  through the `jax` logger with `jax_log_compiles` enabled for the window; the
  previous flag, level and handlers are restored on exit.
- `xla_flops(fn, *args)` — XLA's static per-call FLOP estimate.
- `autoreset_step(env, state, action, key, *, limit, ticks)` — one
  gymnax-style both-branch auto-reset transition: computes the next state
  **and** a fresh reset state every tick, then selects on the legacy episode
  limit.
- `transition_scan(env, state, action, key, *, ticks, autoreset_limit=None)` —
  the same compiled loop with or without the reset path, so both run identical
  dynamics.
- `code_lines(path)` — physical, blank, comment, docstring and code line
  counts for integration-cost reports.

`examples/measure_claims.py` runs three benchmarks and prints a JSON report:
**B1** counts compilations across drift switches (claim 3), **B2** compares a
lifetime step with a reset-every-tick autoreset step (claim 2), and **B3**
counts adapter and example code lines (claim 5). Its options are
`--phases`, `--ticks-per-phase`, `--chunk`, `--b2-ticks` and
`--autoreset-limit`. `examples/measure_throughput.py` runs the claim-1
throughput harness: vmapped lifetimes of identical shape swept over chunk
sizes (ticks per compiled call) and batch sizes, with warm-up, device
synchronization, per-seed timings and IQM/mean/median bootstrap intervals
from `perpetua.reporting` (`--envs`, `--chunks`, `--batches`, `--ticks`,
`--window`, `--seeds`, `--repetitions`, `--policy`).

`examples/measure_training_throughput.py` measures end-to-end training
throughput for the framework's PPO runners (`--backend host|native`, `--env`,
`--drift`, `--batch`, `--duration`, `--seeds`), and
`scripts/measure_baseline.py` measures the PyTorch reference stack on the same
task. The baseline runs in its own virtual environment, so its PyTorch and
pinned Gymnasium versions stay separate from the project's:

```bash
python3 -m venv --system-site-packages /tmp/baseline-venv
/tmp/baseline-venv/bin/pip install "gymnasium==1.4.0"
/tmp/baseline-venv/bin/python scripts/measure_baseline.py \
  --env-id CartPole-v1 --num-envs 128 --duration 60 --seed 0
```

## Limits

- B2 is a transition-function microbenchmark, not claim 1: policy, metrics and
  host orchestration are excluded and identical in both paths.
- The measured reset branch is the adapter's `init`, which also initializes
  drift and clock state on top of a bare simulator reset; a hand-written
  simulator reset may be cheaper.
- Timings are best-of-N with the worst case reported; treat ratios as
  indicative and re-run on the target hardware before quoting.
- Claim 1's comparison uses a CleanRL-style PyTorch PPO; the hardened runs
  use 256 workers (all CPUs on this host) and `torch.compile`, whose
  compilation falls in warm-up as a tuned baseline would amortize it. The
  framework side uses the deterministic GPU compiler options, which may cost
  it throughput. A hand-tuned baseline beyond this remains possible and would
  lower the ratios.
- The baseline's 10⁹-step wall clock was run once (one seed). The native
  path was run to 10⁹ steps with five seeds and the compiled bridge with
  one.
- Reset semantics differ: the baseline resets per episode (that is the
  reset path the claim is about); a lifetime never resets.
- Claim 6 is measured as a pipeline check and one real-budget CO8 lifetime,
  on Perpetua's own learners; re-scoring published agents remains undone. The measured
  claim-1 ratios apply to this hardware, software and task pair; re-run before
  quoting elsewhere.

```{toctree}
:hidden:

claims/throughput
claims/reset-and-recompilation
claims/tuning
claims/integration
claims/comparability
```
