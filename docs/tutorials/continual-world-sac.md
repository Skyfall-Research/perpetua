# SAC on Continual World

You will train a continuous-action learner, SAC, on robot-arm tasks from the
Continual World benchmark, which runs Meta-World manipulation tasks one after
another. Here two of its tasks alternate inside one lifetime with a tiny
budget, so the page runs in seconds; a real run uses `pp.presets.cw10()` with
a million ticks per task.

Needs the `training` and `metaworld` extras (Meta-World brings MuJoCo):

```bash
uv sync --locked --extra training --extra metaworld
```

```python
import numpy as np
import perpetua as pp
from perpetua.agents import SAC, HostSACRunner, SACConfig

# reach and push alternate every 64 ticks. Meta-World simulates on the CPU,
# so this lifetime is stepped by the host (env.metadata["execution"] == "host").
env = pp.presets.continual_world(("reach-v3", "push-v3"), period=64)
try:
    print("observation", env.observation_space.shape, "actions", env.action_space)
    config = SACConfig(
        rollout_steps=16, buffer_capacity=4096, batch_size=32, gradient_steps=4, hidden_size=32
    )
    box = env.action_space  # shape (agents, 4); [0] takes the bounds of the one agent
    trainer = HostSACRunner(env, SAC(action_low=box.low[0], action_high=box.high[0], config=config))
    train_state = trainer.init(seed=0, window_size=64)
    for _ in range(16):  # 16 updates of 16 ticks each
        train_state = trainer(train_state)
    lifetime = train_state.lifetime
    task = int(np.asarray(env.privileged(lifetime.state)["task_parameters"]).reshape(-1)[0])
    print("ticks", lifetime.state.t.to_int(), "task", task)
    print("mean reward", round(float(np.asarray(lifetime.metrics.mean)[0]), 3))
    print("healthy", bool(train_state.healthy))
finally:
    env.close()  # host simulators hold resources; always close them
```

This prints:

```text
observation (1, 39) actions Box(-1.0, 1.0, (1, 4), float32)
ticks 256 task 0
mean reward 0.568
healthy True
```

Meta-World and Gymnasium may also print warnings about observation bounds;
they do not affect the run. The printed mean reward was recorded on a Mac CPU
with JAX 0.11.2 and repeats there; the simulator is stepped by MuJoCo, so
another platform can print a different value.

## Reading the results

- **Spaces.** Each observation has 39 features (arm, gripper and object
  positions); each action is 4 numbers in [−1, 1] (arm movement and gripper).
  Both keep the leading agent axis.
- **The task index** is privileged: `env.privileged(...)` is the evaluator's
  view, and the agent sees only the observation. After 256 ticks the
  schedule is back on task 0, `reach`.
- **Episodes inside the lifetime.** Meta-World tasks have a 200-step limit
  (`max_episode_steps`); reaching it is a respawn, not the end of the
  lifetime. When a task comes back, it resumes where it left off.
- **`healthy`** is the training state's health flag: it turns `False`, and
  stays `False`, once a reward, a learner parameter or a training statistic
  becomes non-finite (NaN or infinity).
- **Host or compiled.** `HostSACRunner` steps the simulator from Python.
  `pp.presets.continual_world(..., compiled=True)` instead calls the
  simulator from inside one compiled program (use `SACRunner` then), which is
  faster for long runs.
- **No `success` field on the preset.** The switched environment that
  `continual_world` returns offers no `success` Oracle field, host or
  compiled. Meta-World's success flag is exposed only by a single
  `MetaWorldLifetime(compiled=False)`, which is what the
  [Continual World preset](../presets/continual-world.md) page evaluates on.

## The benchmark's tuning budget

A lifetime has no separate tuning phase, so each preset declares how much of
it may be spent choosing hyperparameters:

```python
import perpetua as pp

env = pp.presets.cw10(period=1_000_000)
try:
    protocol = pp.tuning.declared(env)
    horizon = env.metadata["configuration"]["preset"]["horizon_ticks"]
    print(protocol, "tune on", protocol.tuning_ticks(horizon), "of", horizon, "ticks")
finally:
    env.close()
```

This prints:

```text
TuningProtocol(fraction=0.1, tuning_seeds=3, evaluation_seeds=10, selection='iqm', seed_offset=1000000) tune on 1000000 of 10000000 ticks
```

CW10 has ten tasks of a million ticks each. Its protocol allows tuning on the
first 10% of the lifetime (one million ticks) with 3 seeds, choosing the best
candidate by IQM, and then evaluating it on 10 fresh seeds. `pp.tuning.tune`
runs that protocol and reports the ticks it spent; see
[tuning](../tuning.md#declared-tuning-protocol).

## Things to try

- Measure forgetting and forward transfer across tasks: the
  [Continual World preset](../presets/continual-world.md) describes the
  paper's protocol.
- Give the learner the task identity, as the paper's single-head setting does:
  `pp.OneHotObservation(pp.Oracle(env, expose=["task_id"]), field="task_id", size=2)`.
- SAC's options for task sequences, such as `task_boundary_reset`, are in
  [learner options](../training-options.md#sac-options-for-task-sequence-protocols).

Back to the [tutorials](index.md).
