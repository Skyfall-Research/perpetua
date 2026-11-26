# A custom drift process

You will write your own drift process: a rule for how one environment
parameter changes over the lifetime. Do this when the built-in processes
(`Cyclic`, `RandomWalk`, `Procedural`, `Sequence`, `Sum`, ...) cannot express
the change you want to study. A process is a small class with `init`, `step`
and a JSON `configuration`, and it then works with any environment.

The example moves ContinualGrid's goal smoothly up the diagonal and back down
again, a triangle wave.

```python
import jax
import jax.numpy as jnp
import perpetua as pp


class Triangle:
    """Move a parameter from low up to high and back down again; one full cycle takes `period` ticks."""

    def __init__(self, low: float, high: float, period: int) -> None:
        self.low, self.high, self.period = low, high, period

    def value_at(self, t):
        # t in [0, period): 0 -> low, period / 2 -> high, then back toward low.
        rise = 1.0 - jnp.abs(2.0 * t / self.period - 1.0)
        return self.low + (self.high - self.low) * rise

    def init(self, value, key):
        # value is the environment's initial parameter, used here only for its shape.
        start = jnp.full(value.shape, self.low, dtype=jnp.float32)
        return pp.drift.DriftState(value=start, phase=jnp.uint32(0), elapsed=jnp.uint32(0))

    def step(self, state, key):
        t = (state.elapsed + 1) % self.period  # position within the period
        value = jnp.full(state.value.shape, self.value_at(t), dtype=jnp.float32)
        # The change is gradual, so no switch is declared: phase stays as it is.
        return pp.drift.DriftState(value=value, phase=state.phase, elapsed=t)

    @property
    def configuration(self):
        return {"type": "triangle", "low": self.low, "high": self.high, "period": self.period}


env = pp.make("ContinualGrid-v0", size=5, drift=Triangle(0.0, 4.0, 8))
state, obs, info = env.init(jax.random.key(0))
for tick in range(8):
    state, obs, reward, info = env.step(state, jnp.asarray([0]), jax.random.key(tick))
    goal = env.privileged(state)["drift_parameters"]
    print(tick + 1, goal.tolist())
```

This prints the goal's (row, column) after each tick:

```text
1 [1.0, 1.0]
2 [2.0, 2.0]
3 [3.0, 3.0]
4 [4.0, 4.0]
5 [3.0, 3.0]
6 [2.0, 2.0]
7 [1.0, 1.0]
8 [0.0, 0.0]
```

## How it works

- **The state.** A process keeps a `pp.drift.DriftState` with three fields:
  `value` (the parameter's current value, with the parameter's shape),
  `phase` (an integer the process changes at each declared switch; for
  `Cyclic`, the index of the current value, which wraps) and `elapsed` (a timer the process
  may use as it likes; here, the position within the period).
- **The shape.** ContinualGrid's goal is a `(row, column)` pair, so `init`
  receives a value of shape `(2,)` and the process fills both entries with the
  same number: the goal moves along the diagonal. The grid rounds the goal to a
  cell and keeps it inside the grid, so `low` and `high` should fit `size`.
- **Switches.** Incrementing `phase` declares a switch: recovery metrics start
  counting, and an Oracle's `boundary` flag fires. A triangle wave changes
  gradually, so it never declares one. A process with real jumps, such as
  `Cyclic`, increments `phase` at each jump.
- **Privileged values.** The schedule is hidden from the agent (in this world
  it sees where the goal is now, not where it will go). Evaluators read
  parameters with `env.privileged(state)`: the default drifting parameter is
  `drift_parameters`, and every parameter is also available as
  `<name>_parameters`, here `goal_parameters`.
- **The configuration** is recorded in `env.metadata["configuration"]`, so
  results say exactly which drift produced them. Keep it JSON-serialisable.

## Things to try

- Add noise on top:
  `pp.drift.Sum([Triangle(0.0, 4.0, 64), pp.drift.RandomWalk(low=-0.5, high=0.5, scale=0.1)])`.
- Run it for a while, then hold still:
  `pp.drift.Sequence([(Triangle(0.0, 4.0, 64), 1000), (pp.drift.Stationary(), 1)])`.
- For drift driven by the agents' own behaviour, declare `endogenous = True`
  and accept `step(state, key, signal)`; `pp.drift.Accumulate` is the built-in
  example.

Next: [several tasks in one lifetime](task-switching.md).
