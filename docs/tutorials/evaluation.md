# Comparing runs: seeds, scores and IQM

You will compare two learner settings fairly: run several seeds, choose a
score that can show the difference you care about, and report it with a
confidence interval. The example asks whether PPO's entropy bonus matters in
the quickstart's world. Needs the `training` extra; it runs in about 20
seconds on a laptop CPU.

```python
import jax
import numpy as np
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

# The quickstart's world: the goal visits three corners, 4,096 ticks each.
goals = [(3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
period, rollout = 4096, 64
env = pp.make("ContinualGrid-v0", size=4, num_agents=4, drift=pp.drift.Cyclic(goals, period))


def run(entropy, seed):
    """Train one lifetime; return the mean reward per tick at the end of each phase."""
    config = PPOConfig(rollout_steps=rollout, entropy_coefficient=entropy)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    train_state = trainer.init(seed=seed, window_size=1024)
    scores = []
    for update in range(1, 3 * period // rollout + 1):
        train_state = trainer(train_state)
        if update * rollout % period == 0:  # last tick of a phase
            window = train_state.lifetime.metrics.window_mean  # last 1,024 ticks, per agent
            scores.append(float(np.asarray(window).mean()))
    return scores


seeds = range(8)
for entropy in (0.01, 0.1):
    scores = np.asarray([run(entropy, seed) for seed in seeds])  # (runs, phases)
    report = pp.reporting.report(scores, key=jax.random.key(0))
    iqm = report["iqm"]
    print(
        f"entropy {entropy}: per-phase mean {np.round(scores.mean(axis=0), 3).tolist()}, "
        f"IQM {float(iqm.point):.3f} [{float(iqm.low):.3f}, {float(iqm.high):.3f}]"
    )
```

This prints:

```text
entropy 0.01: per-phase mean [0.309, 0.005, 0.295], IQM 0.250 [0.233, 0.260]
entropy 0.1: per-phase mean [0.297, 0.287, 0.289], IQM 0.292 [0.289, 0.295]
```

## Reading the results

- **The score.** Each run gives one number per phase: the mean reward per tick
  over the last 1,024 ticks of that phase, when the learner has had the most
  time to adapt. The score matrix has shape `(runs, phases)`, the
  `(runs, tasks)` layout `pp.reporting.report` expects.
- **What it shows.** With the default entropy bonus (0.01), PPO learns the
  first goal, then never learns the second (0.005, below the 0.02 that random
  moves earn in this world), and learns the third again. With 0.1 it learns
  all three. The low bonus lets the policy stop exploring, so after a switch
  it may not find the new goal.
- **IQM and its interval.** `report` returns the interquartile mean (the mean
  of the middle 50% of all run-phase scores, robust to a few outliers) with a
  95% bootstrap confidence interval, plus the mean and median. Here the
  intervals do not overlap, so eight seeds separate the two settings.
- **The choice of score matters.** The mean reward over the whole lifetime
  (`train_state.lifetime.metrics.mean`) barely differs between the two
  settings in this example, because the low bonus learns the first and third
  goals faster: one number over the lifetime can hide a phase that failed.
  Per-phase scores show it.

Results come from eight seeds of one small world and are for illustration.

## Other measures

- **Recovery:** how many ticks after a switch the reward gets back to its
  level before the switch, `pp.evaluation.recovery_ticks`, as in the
  [quickstart](quickstart.md).
- **Forgetting:** how much worse the final policy is on earlier goals,
  `pp.evaluation.forgetting`, from scores of saved policies with
  `pp.evaluation.revisit` (quickstart step 4).
- **Plasticity probes and regret:** see the [evaluation guide](../evaluation.md).

Next: [average reward with PPO](reward-centering.md).
