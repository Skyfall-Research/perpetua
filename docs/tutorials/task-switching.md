# Several tasks in one lifetime

You will put two different environments into one never-ending lifetime and
switch between them on a schedule, the setting of task-sequence benchmarks
such as Continual World. The agent is not told which task is active; it has to
notice from what it sees. Needs the `interop` extra for the gymnax task
(`uv sync --locked --extra interop`).

```python
import jax
import perpetua as pp

env = pp.tasks.switch(
    [pp.make("ContinualGrid-v0", size=3), pp.make("gymnax/Acrobot-v1")],
    pp.drift.Cyclic([0.0, 1.0], 4),  # task 0 for 4 ticks, then task 1, and so on
)
print("observation", env.observation_space.shape, "actions", env.action_space)

key = jax.random.key(0)
key, init_key = jax.random.split(key)
state, obs, info = env.init(init_key)
for tick in range(1, 9):
    key, act_key, step_key = jax.random.split(key, 3)
    action = pp.policies.random(obs, info, act_key)  # picks only actions the mask allows
    state, obs, reward, info = env.step(state, action, step_key)
    task = int(env.privileged(state)["task_parameters"])  # privileged: for evaluation only
    mask = info["action_mask"][0].astype(int).tolist()
    print(tick, "task", task, "mask", mask, "space_change", bool(info["space_change"][0]))
```

This prints:

```text
observation (1, 6) actions MultiDiscrete([5])
1 task 0 mask [1, 0, 1, 1, 1] space_change False
2 task 0 mask [1, 0, 1, 0, 1] space_change False
3 task 0 mask [1, 1, 1, 0, 1] space_change False
4 task 1 mask [1, 1, 1, 0, 0] space_change True
5 task 1 mask [1, 1, 1, 0, 0] space_change False
6 task 1 mask [1, 1, 1, 0, 0] space_change False
7 task 1 mask [1, 1, 1, 0, 0] space_change False
8 task 0 mask [1, 0, 1, 0, 1] space_change True
```

The grid masks on ticks 1 to 3 depend on the random walk of the sampled
actions, so they follow from the seed in this snippet.

## How the tasks share one interface

- **Observations** are zero-padded to a common shape: the grid has 4 features
  and Acrobot 6, so the switched world has 6, and the grid's last two are 0.
- **Actions** use the largest action count (the grid's 5). Acrobot has 3
  actions, so while it is active the mask disables the last two. The grid's
  own mask also changes as the agent moves next to a wall.
- **`info["space_change"]`** is true on ticks where the mask changes because
  of a switch (ticks 4 and 8 here), so a learner can react to it if your
  protocol allows.
- **Every switch is a declared switch**, like any other drift: recovery
  metrics count from it and an Oracle can expose `boundary` or `task_id`.
- **Inactive tasks are paused, not reset.** When the grid comes back at tick
  8, it continues from where it stopped. `pp.tasks.switch(..., on_switch="restart")`
  re-initialises a task on entry instead, and reports a respawn.

All tasks must have the same number of agents, and continuous tasks must share
one action box; mixes that cannot be combined raise an error.

## Things to try

- Train a learner across the switches: pass `env` to `PPORunner` as in the
  [quickstart](quickstart.md), with a longer period such as 4,096.
- Drift a task's own parameters at the same time with a schedule mapping, for
  example `{"task": pp.drift.Cyclic([0.0, 1.0], 1000), "tasks[1].link_mass_1": ...}`
  to change Acrobot's first link mass while the tasks alternate.
- The [task switching guide](../task-switching.md) covers the parameter names,
  CPU-simulator tasks and benchmark presets.

Next: [comparing runs](evaluation.md).
