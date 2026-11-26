# Average reward with PPO

You will turn on reward centering and see what it changes. In a lifetime that
never ends, a learner's value estimates contain a large constant part: with
discount γ, a steady reward rate *r* adds about *r* / (1 − γ) to every value,
which grows without bound as γ approaches 1. Reward centering learns the
average reward per tick and subtracts it, so the critic only has to learn how
much better or worse than average each state is (Naik et al., 2024). Needs the
`training` extra; it runs in a few seconds on a laptop CPU.

```python
import jax
import numpy as np
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

env = pp.make("ContinualGrid-v0", size=4, num_agents=4)


def train(config):
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    train_state = trainer.init(seed=0, window_size=1024)
    for _ in range(192):  # 12,288 ticks
        train_state = trainer(train_state)
    lifetime = train_state.lifetime
    values = trainer.agent.act(
        train_state.agent, lifetime.obs, lifetime.info, jax.random.key(0)
    ).value
    reward = float(np.asarray(lifetime.metrics.window_mean).mean())
    rate = float(np.asarray(train_state.agent.reward_rate).mean())
    return reward, rate, float(np.abs(np.asarray(values)).mean())


print("setting           reward  reward_rate  |value|")
settings = [("off", PPOConfig(rollout_steps=64))] + [
    (
        f"on, step {step}",
        PPOConfig(rollout_steps=64, reward_centering=True, centering_step_size=step),
    )
    for step in (0.01, 0.1, 0.3)
]
for name, config in settings:
    reward, rate, value = train(config)
    print(f"{name:<16}  {reward:6.3f}  {rate:11.3f}  {value:7.2f}")
```

This prints:

```text
setting           reward  reward_rate  |value|
off                0.320        0.000    32.03
on, step 0.01      0.321        0.078    24.36
on, step 0.1       0.321        0.227     9.66
on, step 0.3       0.321        0.274     4.80
```

## Reading the results

- **reward** is the mean reward per tick over the last 1,024 ticks: about
  0.32 in every run. Centering does not change what this learner achieves in
  this small world.
- **reward_rate** is the learner's estimate of the average reward, stored in
  `train_state.agent.reward_rate`. After every update it moves by
  `centering_step_size` times the mean TD error, so a larger step size gets
  closer to the true average (about 0.32) within this short run.
- **|value|** is the critic's mean absolute value estimate. Without centering
  it is about 0.32 / (1 − 0.99) = 32. The closer `reward_rate` gets to the true
  average, the more of that constant part is removed: 4.8 with step size 0.3.

Smaller values matter most when γ is close to 1, and when the reward level
itself drifts: an uncentered critic must relearn a large constant every time
the average reward changes. `centering_step_size` is a traced
hyperparameter, so you can sweep it like the learning rate without
recompiling. DQN and SAC accept the same `reward_centering=True`.

## Things to try

- Set `gamma=0.999` in `PPOConfig` and train for longer: the constant part
  of the uncentered values, *r* / (1 − γ), becomes ten times larger.
- Combine centering with drift, as in [PPO in an open-ended world](bean-world-ppo.md),
  and compare recovery with and without it.
- [Learner options](../training-options.md#average-reward-reward-centering) covers centering under compute-aware
  time, where one decision spans several ticks.

Next: [SAC on Continual World](continual-world-sac.md).
