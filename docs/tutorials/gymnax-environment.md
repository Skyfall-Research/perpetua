# Drifting a gymnax environment

You will take an existing JAX environment from gymnax and make its physics
change during the lifetime. Any gymnax environment works this way: Perpetua
wraps it as a lifetime (no time limit, terminations become respawns) and lets
you drift any of its environment parameters by name. Needs the `interop` extra
(`uv sync --locked --extra interop`).

```python
import jax
import perpetua as pp

# Acrobot's integration step dt is 0.2 seconds by default. This schedule
# alternates it between 0.2 and 0.1 every 4 ticks, so the simulation advances
# half as much time per step in the second phase.
env = pp.make("gymnax/Acrobot-v1", drift={"dt": pp.drift.Cyclic([0.2, 0.1], period=4)})
print("observation", env.observation_space.shape, "actions", env.action_space)

key = jax.random.key(0)
key, init_key = jax.random.split(key)
state, obs, info = env.init(init_key)
for tick in range(1, 9):
    key, act_key, step_key = jax.random.split(key, 3)
    action = pp.policies.random(obs, info, act_key)
    state, obs, reward, info = env.step(state, action, step_key)
    privileged = env.privileged(state)  # the evaluator's view; the agent never sees it
    dt, switch = float(privileged["dt_parameters"]), bool(privileged["dt_changed"])
    print(tick, "dt", round(dt, 2), "switch", switch, "reward", float(reward[0]))
```

This prints:

```text
observation (1, 6) actions MultiDiscrete([3])
1 dt 0.2 switch False reward -1.0
2 dt 0.2 switch False reward -1.0
3 dt 0.2 switch False reward -1.0
4 dt 0.1 switch True reward -1.0
5 dt 0.1 switch False reward -1.0
6 dt 0.1 switch False reward -1.0
7 dt 0.1 switch False reward -1.0
8 dt 0.2 switch True reward -1.0
```

## Reading the results

- **Spaces** keep the agent axis: Acrobot's 6 observation features become
  `(1, 6)`, and its 3 actions become `MultiDiscrete([3])`, one entry per agent.
- **Rewards** are gymnax's own: Acrobot gives −1 every step until the arm
  swings above the line. Reaching it is a respawn inside the lifetime, not the
  end of an episode, and gymnax's time limit is removed.
- **The parameter** changes after ticks 4 and 8, and each change is a declared
  switch (`dt_changed`), so recovery metrics count from it. Privileged values
  are keyed `<parameter>_parameters`, `<parameter>_phase` and
  `<parameter>_changed`; the agent sees none of them unless you wrap the
  environment in `pp.Oracle`.
- **Values must match the parameter's shape.** `dt` is a scalar, so each phase
  is a number. Some hand-written wrappers use shape-`(1,)` fields, such as
  CartPole's `gravity`, whose phases are tuples: `Cyclic([(9.8,), (15.0,)], 4096)`.
  A mismatch raises an error that states both shapes.

Each gymnax environment's page lists its driftable parameters; see the
[environment index](../environments/index.md) (the MinAtar, bsuite and control
families).

## Things to try

- Drift two parameters at once:
  `drift={"dt": ..., "link_mass_1": pp.drift.RandomWalk(low=0.5, high=2.0, scale=0.01)}`.
- Train PPO on it as in the [quickstart](quickstart.md); Acrobot has 3 actions,
  so use `PPO(action_size=3)`.
- To write a world of your own instead, follow the
  [custom environment tutorial](custom-environment/index.md).

Next: [PPO in an open-ended world](bean-world-ppo.md).
