# Claim 6: comparability

**Hypothesis:** results become comparable across benchmarks, against per-paper metrics. Part of the [measured claims](../measurement.md), which has the instruments and the limits that apply to every number.

## Comparability (claim 6)

**Verdict: pipeline check only.** The same metrics were produced on MEAL,
CW10, CO8, Atari switching, AgarCL and MORPHEUS with tiny budgets. There is
no performance claim, and published agents were not re-scored. One benchmark,
CO8, also has a [real-budget lifetime](#a-real-budget-lifetime-on-co8).

Claim 6 is narrowed to Perpetua's own learners: published agents are not
re-scored.
`examples/compare_benchmarks.py` trains PPO (discrete) or SAC (continuous) with
one configuration per action type through each benchmark's own task sequence,
saves the policy at every task boundary, and evaluates every saved policy on
fresh lifetimes rebuilt with the training seeds. Every benchmark reports the
same fields: final-policy mean reward and stratified-bootstrap IQM over
seeds × tasks, forgetting (`pp.evaluation.forgetting` per seed), and ticks to
recover after each declared switch (from the training reward trace, inside the
new task; `null` = not recovered). Benchmarks without a task sequence report no
forgetting or recovery rather than an invented value.

Run on the H200, 2026-11-02: 2 updates × 16 steps per task, 64 evaluation ticks,
seeds 0, 1, 2, recovery window 8.

| Benchmark | Learner | Tasks | Mean reward/tick | IQM [95% CI] | Forgetting | Recovered switches |
|---|---|---:|---:|---|---:|---|
| MEAL Overcooked | PPO (CNN) | 8 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 7/7 (trivial: zero reward) |
| Continual World (CW10) | SAC | 10 | 0.501 | 0.217 [0.193, 0.250] | 0.0014 | 4/9 |
| COOM (CO8) | PPO (CNN) | 8 | 0.233 | 0.083 [0.040, 0.138] | 0.0059 | 5/7 |
| Atari switching (Breakout, Asterix) | PPO (CNN) | 2 | 1.042 | 0.781 [0.781, 1.563] | 0.000 | 1/1 |
| AgarCL screen | SAC (flat) | 1 | 0.000 | 0.000 [0.000, 0.000] | n/a | n/a |
| MORPHEUS (mock, paper regimes; 2026-11-03) | PPO | 6 | −0.002 | 0.006 [−0.024, 0.032] | 0.000 | 3/5 |

What this shows: one script produces the same lifetime metrics, on one clock,
for all six benchmarks, including host and native, discrete and continuous,
pixel and vector paths. What it does not show: learning. With 32 training ticks
per task, scores are at or near random and forgetting is near zero because
little was learned. Recovery is trivially immediate when pre-switch reward is
zero (MEAL). AgarCL's default reward is per-tick mass change, so an agent that
stays at minimum mass scores exactly 0 (random play earns +1/−1 pairs). Rewards
are not comparable *across* benchmarks; the claim is that the metric
definitions and protocol are shared. Statements about forgetting or recovery
behaviour need a real-budget run; [CO8 has one](#a-real-budget-lifetime-on-co8),
and the other benchmarks do not.

The MORPHEUS row is a six-regime sequence on the mock service (the paper's
regimes, declared drift, paper reward), recorded on 2026-11-03 with the same
budget. The runner also reports the paper's six-metric protocol
(`pp.protocol`) on each training trace:

- **Per-configuration reward:** 0.123 for the baseline, against −0.04 to
  −0.18 for the other regimes.
- **Within-trace forgetting** of the baseline (first encounter minus the
  revisit): 0.162.
- **Adaptation speed:** mostly the interval cap of 32, with recovery after 3
  of 5 switches.

At this budget these numbers describe the regimes, not learning.

## A real-budget lifetime on CO8

`scripts/coom_lifetime.py` trains PPO (CNN torso, 200-step rollouts, otherwise
the default configuration) through CO8 as one lifetime on the compiled CPU
bridge, 200,000 ticks per task. At every task boundary the saved policy is
scored on all eight tasks: reward per tick over 4,000 ticks of a fresh
lifetime. The untrained policy is scored the same way before training.

Run on the H200, 2026-11-20, JAX 0.11.2, seeds 0, 1, 2 (about 4.7 hours each,
run side by side). Reward per tick, mean of the three seeds:

| Task | Untrained | Just trained | Final policy |
|---|---:|---:|---:|
| pitfall | 0.77 | 2.79 | 0.73 |
| arms_dealer | 0.05 | 1.57 | 0.94 |
| hide_and_seek | −0.32 | 0.03 | −0.24 |
| floor_is_lava | 0.05 | 1.06 | 0.01 |
| chainsaw | 0.05 | 0.36 | 0.11 |
| raise_the_roof | 0.11 | 0.36 | 0.24 |
| run_and_gun | 0.03 | 0.15 | 0.13 |
| health_gathering | 0.20 | 0.94 | 0.94 |

The just-trained policy beat the untrained one on every task in every seed.
Reward scales differ about twentyfold across tasks, so forgetting is computed
after rescaling each task per seed: 0 is the untrained policy and 1 is the
policy that has just trained on the task. `pp.evaluation.forgetting` on the
rescaled matrix gives **0.67** (0.53, 0.81 and 0.66 by seed).

Forgetting is uneven. On the rescaled scale the final policy keeps about 0.6
of arms_dealer, 0.5 of raise_the_roof and 0.8 of run_and_gun, and nothing of
pitfall or floor_is_lava. The policy saved after arms_dealer already scores
0.8 on health_gathering, six tasks before it trains there.

```bash
uv run --extra training python scripts/coom_lifetime.py --seed 0 --out co8-ppo-seed0.jsonl
uv run --with matplotlib python scripts/plot_claim_figures.py reward-lifetime \
    co8-ppo-seed*.jsonl --out lifetime-coom.png
```

Limits:

- These are rewards, rescaled here. The COOM paper reports a success score,
  which the adapter does not expose, and its per-task budget was not matched,
  so nothing in this section compares with the paper's tables.
- The learner is Perpetua's PPO, not a published agent.
- The rescaling is weakest on run_and_gun, where training added 0.11 reward
  per tick over the untrained policy.
- Three seeds, and each evaluation is 4,000 ticks.
- In the logs of this run `window_reward` holds the lifetime mean reward; the
  script writes the window mean as `window_reward` and the lifetime mean as
  `lifetime_reward`. The table and the forgetting value use neither.
