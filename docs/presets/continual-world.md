# Continual World

The published CW10 order on Farama Meta-World 3.1.1 (v3 task names), one task
per `period` ticks, in one host-controlled lifetime. CW20 is CW10 twice over the
same ten simulators: `MetaWorld ∘ Switch(CW10, 2, period)`.

```bash
uv sync --locked --extra metaworld --extra training
```

```python
import perpetua as pp

env = pp.presets.cw20(period=1_000_000)  # 20M-tick horizon in metadata
env.metadata["configuration"]["preset"]["horizon_ticks"]  # 20_000_000
```

- **Learner:** `SACRunner` with `continual_world(..., compiled=True)`, which
  compiles the whole lifetime (about 18× the `HostSACRunner` loop in our
  check), or `HostSACRunner` on the host-executed preset.
- **Metrics:** the shared lifetime metrics, recovery per declared switch, and
  forgetting from boundary policies re-evaluated on each task. Because the
  Continual World paper scores task *success*, the Oracle field `success`
  exposes Meta-World's flag on a single-task `MetaWorldLifetime`. The switched
  preset offers no `success` field, with or without `compiled=True`, so
  `examples/compare_benchmarks.py` and the reproduction script below score
  success on separate one-task evaluation environments. The example reports
  average performance and forgetting on success (the fraction of evaluation
  ticks with `info["success"]`).
- **Measured:** measured as a pipeline check with tiny budgets (32 training ticks per task); see [claim 6](../claims/comparability.md).
- **Limits:** v3 implementations, not numerical equivalence to the original v1
  code; no code from the unlicensed Continual World repository; opaque
  simulator state, so no checkpoints. The example does not report forward
  transfer, which needs single-task reference runs, and its success
  definition is per tick, not the paper's per-episode evaluation; the
  reproduction script below does both the paper's way. The real-budget
  numbers for this benchmark are the ones [below](#result-9-seeds-with-the-fixed-learner-not-a-reproduction).

## Reproducing the paper's fine-tuning result

`scripts/reproduce_continual_world.py` targets Table 14 of Wołczyk et al.
(2021): CW10 fine-tuning with a single head and the task id as a one-hot input,
performance 0.12 [0.12, 0.13], forgetting 0.68 [0.66, 0.70], forward transfer
0.16 [0.12, 0.19] (20 seeds, 90% intervals). It re-implements the protocol
from the paper and its published defaults (no code is copied):

- SAC from `pp.agents` with `hidden_layers=4`, `hidden_size=256`,
  `layer_norm=True`, `activation="leaky_relu"`, learning rate 1e-3, batch 128,
  50 gradient steps every 50 ticks, a 1M replay, temperature tuned from
  alpha = e towards a target standard deviation of 0.089;
- per task: `task_boundary_reset=True` (replay, optimizer and temperature
  restart), `random_steps=10_000`, `update_after=1_000`; 1M ticks per task;
- `randomize_goals=True` (a new goal configuration every 200-tick episode) and
  `mask_respawns=True` (time-limit respawns stay out of TD targets);
- the task id through `pp.OneHotObservation(pp.Oracle(env, expose=["task_id",
  "boundary"]), field="task_id", size=10)`, so the privilege the protocol
  needs is recorded in the run's metadata;
- evaluation: the current task every 20k ticks and every task at each task
  end, by the success rate of 10 stochastic episodes (success at any tick).

```bash
uv run --extra metaworld --extra training python scripts/reproduce_continual_world.py \
    cl --seed 0 --out results/cw10-seed0.jsonl
uv run --extra metaworld --extra training python scripts/reproduce_continual_world.py \
    reference --task 3 --seed 0 --out results/reference-task3-seed0.jsonl
uv run --extra metaworld --extra training python scripts/reproduce_continual_world.py \
    metrics --cl "results/cw10-seed*.jsonl" --reference "results/reference-*.jsonl"
```

`reproduce_continual_world.py metrics` reports the paper's forgetting (the
success right after training on a task minus the final success, averaged over
all ten tasks). `scripts/plot_claim_figures.py lifetime` prints and titles its
heat map with `pp.evaluation.forgetting` instead (best saved policy minus the
final one, over the first nine tasks), so the two differ on the same log; the
table below uses the paper's.

Known differences: Meta-World v3 tasks and rewards (the paper used v1),
39-dimensional observations (v1 had 12), and one temperature restarted at
each switch instead of one per task (the same for CW10, which never revisits a
task).

### Result: 9 seeds with the fixed learner, not a reproduction

Seeds 0 to 8 with the default (orthogonal) initialiser, scored against 90
single-task references (10 tasks by 9 seeds), with 90% bootstrap intervals
over the nine seeds:

| Metric | Perpetua (9 seeds) | Paper (20 seeds) |
|---|---|---|
| Performance | 0.118 [0.109, 0.128] | 0.12 [0.12, 0.13] |
| Forgetting | 0.681 [0.636, 0.723] | 0.68 [0.66, 0.70] |
| Forward transfer | 0.043 [−0.021, 0.107] | 0.16 [0.12, 0.19] |
| Forward transfer, non-trivial tasks | −0.099 [−0.184, −0.011] | not reported |

Performance and forgetting match the paper's values. Forward transfer does
not: the interval contains zero and lies below the paper's 0.16. Per seed it
ranges from −0.147 to 0.227.

What the numbers do and do not show:

- A performance near 0.1 means the last task is solved at the end and the
  other nine are not, which is what fine-tuning is expected to do. All nine
  runs end with a success rate of 1.0 on the last task.
- `stick-pull-v3` is never solved in continual training in any run (success
  0.0 at the end of its task).
- Mean forward transfer per task runs from −1.19 (`push-v3`) and −1.09
  (`peg-unplug-side-v3`) to 0.76 (`handle-press-side-v3`). `push-v3` is at 0.0
  at the end of its task in five of the nine runs.
- The first task, `hammer-v3`, has a mean forward transfer of 0.62, although
  nothing precedes it. All nine continual runs reach a success rate of 0.9 on
  it within 240,000 ticks; of its nine reference runs, two never do and one
  does only after 700,000. Nine runs of each do not show whether that is
  chance. The references ran on CPU and the continual runs on a GPU, but the
  backend does not explain it: six of the references rerun on the GPU (seeds
  0 to 5) gave one that never learns the task (seed 2, which also fails on
  CPU) and five that do, including seed 0, which fails on CPU. The same seed
  can succeed on one backend and fail on the other, so single-task training
  on `hammer-v3` fails now and then, and its reference curve is noisy.
  Without the first task, the mean over the other nine is −0.02.
- The runs are not a reproduction. The paper used 20 seeds and Meta-World v1.

#### Initialisers, 3 seeds each

Seeds 0 to 2 for each initialiser, each arm scored against its own 30
references (10 tasks by 3 seeds), so the first column differs from the table
above:

| Metric | Orthogonal init (default) | Glorot uniform init |
|---|---|---|
| Performance | 0.127 [0.113, 0.140] | 0.113 [0.103, 0.123] |
| Forgetting | 0.677 [0.593, 0.760] | 0.637 [0.570, 0.703] |
| Forward transfer | 0.066 [−0.064, 0.195] | 0.057 [−0.103, 0.217] |
| Forward transfer, non-trivial tasks | 0.036 [−0.155, 0.227] | −0.031 [−0.181, 0.120] |

Three seeds do not separate the two initialisers. The Glorot runs use
`--kernel-init glorot_uniform`.

### Control: critics initialised identically, 10 seeds

| Metric | Perpetua (10 seeds, 90% bootstrap interval) | Paper (20 seeds) |
|---|---|---|
| Performance | 0.040 [0.017, 0.067] | 0.12 [0.12, 0.13] |
| Forgetting | 0.708 [0.672, 0.744] | 0.68 [0.66, 0.70] |
| Forward transfer | −0.402 [−0.604, −0.258] | 0.16 [0.12, 0.19] |
| Forward transfer, non-trivial tasks | −0.097 [−0.160, −0.036] | not reported |

Forgetting matches the paper; performance and forward transfer do not. The
forward-transfer mean is pulled down by one seed (−1.36); the non-trivial row
leaves out tasks whose single-task reference is solved almost immediately.

**These runs used a SAC whose two critics were initialised with the same
parameters, so they stayed identical and clipped double-Q reduced to a single
critic.** The shipped learner gives the critics independent initial
parameters, so this table does not describe it. Seeds 0 to 2 of these runs, scored the same way as the table above,
give performance 0.017 [0.003, 0.030], forgetting 0.707 [0.683, 0.730] and
forward transfer −0.206 [−0.262, −0.149]; their success rates on the last
task at the end were 0.1, 0.0 and 0.4. The learner differs from the one in
these runs in other ways as well, so the comparison shows that the current
learner closes the performance gap, not that the critic initialisation alone
caused it.

See [continual benchmarks](../benchmarks.md) for setup details.
