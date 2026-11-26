# PPO in an open-ended world

You will train PPO in a world that never ends, check that it beats random
actions, and then change what the world rewards to watch the learner lose and
regain its performance. Needs the `training` extra
(`uv sync --locked --extra training`); each part runs in a few seconds on a
laptop CPU.

`BeanWorld-v0` is a seeded torus on which agents forage (four in this tutorial). There are three
kinds of item, jelly beans, bananas and onions, worth 1.0, 0.1 and −1.0 by
default, and the mix of items changes from region to region.

## Train and compare with random actions

```python
import numpy as np
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

env = pp.make("BeanWorld-v0", size=32, num_agents=4)
trainer = PPORunner(env, PPO(action_size=5, config=PPOConfig(rollout_steps=64)))
train_state = trainer.init(seed=0, window_size=512)  # metrics keep the last 512 ticks
for _ in range(60):  # 60 updates of 64 ticks each
    train_state = trainer(train_state)
ticks = train_state.lifetime.state.t.to_int()
learned = float(np.asarray(train_state.lifetime.metrics.window_mean).mean())

# Random actions for the same number of ticks, from the same seed.
random = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(
    pp.runtime.init_lifetime(env, seed=0, window_size=512)
)
baseline = float(np.asarray(random.metrics.window_mean).mean())
print(f"last 512 ticks: PPO {learned:.3f} vs random {baseline:.3f}, after {ticks} ticks")
```

This prints:

```text
last 512 ticks: PPO 0.038 vs random -0.003, after 3840 ticks
```

Both numbers are the mean reward per agent per tick over the same final
512-tick window (`window_mean`), averaged over the four agents. Random
movement nets about zero, because the good and bad items roughly cancel; PPO
collects clearly positive reward.

## Change what items are worth

Now swap the value of jelly beans and onions every 2,048 ticks. The schedule
is hidden from the agent: it only notices that what it eats now hurts.

```python
import numpy as np
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

# Item values (jelly bean, banana, onion); the two phases swap beans and onions.
flip = pp.drift.Cyclic([(1.0, 0.1, -1.0), (-1.0, 0.1, 1.0)], period=2048)
env = pp.make("BeanWorld-v0", size=32, num_agents=4, drift={"item_rewards": flip})
trainer = PPORunner(env, PPO(action_size=5, config=PPOConfig(rollout_steps=64)))
train_state = trainer.init(seed=0, window_size=256)
print("tick  window  since switch  recovered_at")
for update in range(1, 89):
    train_state = trainer(train_state)
    if update % 8 == 0:  # every 512 ticks
        metrics = train_state.lifetime.metrics
        window = float(np.asarray(metrics.window_mean).mean())
        print(
            f"{update * 64:4d}  {window:6.3f}  {int(metrics.since_switch):12d}  {int(metrics.recovered_at):12d}"
        )
```

This prints:

```text
tick  window  since switch  recovered_at
 512   0.018             0            -1
1024   0.018             0            -1
1536   0.015             0            -1
2048   0.013             0            -1
2560  -0.018           512            -1
3072  -0.013          1024            -1
3584  -0.006          1536            -1
4096  -0.013             0            -1
4608   0.012           512           256
5120   0.007          1024           256
5632   0.016          1536           256
```

## Reading the results

- **Before the first switch** (ticks 0–2,047) the windowed reward stays small
  and positive (0.013 to 0.018) and does not climb in this table;
  `since switch` stays 0 because nothing has switched yet.
- **After the first switch** (tick 2,048) beans are worth −1 and onions +1. The
  windowed reward turns negative and `recovered_at` stays −1: within this
  phase, the learner never gets back to its pre-switch level.
- **After the second switch** (tick 4,096) the values are back to the
  original ones. The old habits pay again and the windowed reward is positive
  at the next row (0.012), so `recovered_at` reads 256, its smallest possible
  value. That is a trivial recovery: the reference is the 256 ticks just
  before the second switch, which are the poor post-flip level (about −0.013),
  so any non-negative reward counts. It does not show a return to the level
  before the first switch (about 0.018); compare the windowed rewards for that.
- **`recovered_at`** is computed live, inside the carry. It counts ticks from
  the switch to the *end* of the first 256-tick window that is back within 10%
  of the pre-switch level, so its smallest possible value is the window size,
  256. The quickstart's `pp.evaluation.recovery_ticks` measures the same thing
  after the fact and counts to the *start* of that window, so it reports
  `recovered_at - 255` (unless it reports 0: the live window never includes
  the switch tick itself).

Results come from one seed and are for illustration.

## Things to try

- Give the agents the switch signal with `pp.Oracle(env, expose=["boundary"])`
  and compare how fast they recover.
- Slow the change down: `pp.drift.RandomWalk` on `item_rewards` instead of
  `Cyclic` (recovery is then undefined; use the windowed reward).
- Try the [plasticity options](../training-options.md#plasticity-options), such as
  `PPOConfig(activation="crelu")`.

Next: [a custom drift process](custom-drift.md).
