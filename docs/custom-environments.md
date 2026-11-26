# Environment creation

This guide shows how to write an environment, what each method is for, and how
it plugs into drift, the Oracle, runners and learners. For a step-by-step
version, follow the [custom environment tutorial](tutorials/custom-environment/index.md). If you already have a
Gymnasium or gymnax environment you usually do not need it: wrap a CPU
Gymnasium environment with `GymnasiumLifetime`, and reach any gymnax
environment directly with `pp.make("gymnax/...")` (see
[environment adapters](adapters.md)).

## The one idea: a lifetime, not an episode

A Perpetua environment is a **pure function of state and parameters**:

```
step: (state, action, key, params) -> (state, reward)
```

There is no `reset()` and no `done`. One `init` starts a lifetime and the
clock then runs forever. "Death" and "respawn" are **events in the dynamics**
(`info["respawn"]`), not interface boundaries. Change over time comes from
**drift on parameters**, so a lifetime compiles once and runs through every
phase.

You describe the world; the framework does the rest. Wrapping your environment
in `DriftEnv` adds the drift process and its RNG stream, declared switch
events, the lifetime clock, Oracle fields and checkpoint metadata, so an
environment never contains drift bookkeeping.

Four rules make an environment framework-compatible:

1. **Functions are pure.** All randomness comes from the `key` argument; no
   host state, no globals, no Python-side mutation.
2. **Everything that may drift is a parameter.** Read it from `params`, never
   from `self`, so a drift process can change it without rebuilding anything.
3. **Public data is agent-visible only.** A parameter may be visible when the
   world itself shows it (`LineWorld` shows where its goal is, as a real world
   would), but drift bookkeeping never is: no phase, no switch flag, no table
   of upcoming values. The Oracle exposes those on request.
4. **Shapes and dtypes are fixed.** Arrays may change value, never shape or
   dtype, so `jax.jit`, `lax.scan` and `vmap` compile once.

## The contract at a glance

```python
class MyEnv:
    num_agents: int
    observation_space: gymnasium.spaces.Space
    # Discrete actions: MultiDiscrete([n] * num_agents). Continuous: Box of shape
    # (num_agents, action_dim). A bare Discrete(n) has no agent axis and is refused.
    action_space: gymnasium.spaces.Space
    metadata: Metadata
    default_drift_field: str  # optional: field a bare drift process drives
    mask_params: tuple[str, ...]  # optional: fields whose changes alter action masks
    oracle_fields: tuple[str, ...]  # optional: names your privileged() returns

    @property
    def default_params(self) -> Params: ...  # pytree of float arrays
    def derive(self, params) -> Params: ...  # optional: raw -> effective
    def check_drift(self, drifts) -> None: ...  # optional: refuse unusable processes
    def init(self, key, params) -> (state, obs, info): ...
    def step(self, state, action, key, params) -> (state, reward): ...
    def observe(self, state, params) -> (obs, info): ...
    def privileged(self, state) -> dict[str, jax.Array]: ...  # world-side diagnostics
```

`info` from your environment carries three keys; `DriftEnv` adds the clock and
the space-change event, so every info a learner sees has the same five:

| key | shape | dtype | added by | meaning |
|---|---|---|---|---|
| `action_mask` | `(agents, actions)` | bool | you | which actions are legal now (discrete actions; see below for `Box`) |
| `active_mask` | `(agents,)` | bool | you | which slots the learner controls |
| `respawn` | `(agents,)` | bool | you | slots that respawned on this transition |
| `t` | `Clock` (`hi`, `lo` uint32) | uint32 | framework | lifetime clock at this observation |
| `space_change` | `(agents,)` | bool | framework | drift changed this agent's valid-action set this tick |

The action space carries the agent axis, as in every Perpetua environment:
discrete actions are `MultiDiscrete([n] * num_agents)` (`LineWorld` below uses
`MultiDiscrete([3])`) and continuous ones a `Box` of shape `(agents, dim)`. With
`spaces.Discrete(3)`, `pp.testing.check_env` fails with `actions must have shape
(1,)`, an error raised by your own `step`; it does not mention `action_space`.

A continuous (`Box`) action has no per-action legality, so `action_mask` is a
placeholder there: an all-`True` `(agents, 1)` array (what the generic gymnax,
MJX and host-lift adapters return; the JaxMARL continuous tasks instead mark the
valid `(agents, action_dim)` dimensions). Return the placeholder from your own
continuous environment and do not read the mask as legality.

`Metadata` is static host-side reporting data: `env_id` and `oracle_exposure`
are required; `configuration` (JSON), `execution` (`"jax"` or `"host"`), `checkpointable` and
`forkable` are optional. `forkable` (default `True`) says a state can be
stepped more than once, as a regret `reference=` does; set it to `False` for a
single-use state, which refuses a `reference=`. The framework adds the drift
configuration.

## A complete example: `LineWorld`

One agent on a line of length 4 earns 1 for reaching the goal and respawns at
0 on the same tick. The goal is a parameter, so any drift process can move it.
This is the full environment, taken from `examples/custom_env.py`:

```python
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

import perpetua as pp
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv


class LineParams(NamedTuple):
    """Everything that may drift. Raw values; ``derive`` makes them effective."""

    goal: jax.Array  # () float32, rounded to a cell by derive


class LineState(NamedTuple):
    """The world only: no drift, RNG stream or clock (the framework owns those)."""

    position: jax.Array  # () int32 in [0, length - 1]
    respawn: jax.Array  # () bool, this transition only


class LineWorld:
    """A continuing line world; respawns are events, never resets."""

    default_drift_field = "goal"

    def __init__(self, *, length: int = 4) -> None:
        if type(length) is not int or length < 2:
            raise ValueError("length must be an integer >= 2.")
        self.length = length
        self.num_agents = 1
        self.observation_space = spaces.Box(
            low=np.zeros((1, 2), dtype=np.float32),
            high=np.full((1, 2), float(length - 1), dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.asarray([3], dtype=np.int64))
        self.metadata: pp.Metadata = {
            "env_id": "line-world-v0",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {"length": length, "respawn_policy": "same-tick-on-goal"},
        }

    @property
    def default_params(self) -> LineParams:
        """The goal starts at the far end of the line."""
        return LineParams(jnp.asarray(float(self.length - 1), dtype=jnp.float32))

    def derive(self, params: LineParams) -> LineParams:
        """Raw drift values become an in-bounds goal cell."""
        return LineParams(jnp.clip(jnp.rint(params.goal), 0, self.length - 1))

    def init(self, key: jax.Array, params: LineParams) -> tuple[LineState, jax.Array, EnvInfo]:
        """Start one lifetime: no reset exists, so this runs exactly once."""
        del key  # deterministic start
        state = LineState(jnp.int32(0), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(
        self, state: LineState, action: Any, key: jax.Array, params: LineParams
    ) -> tuple[LineState, jax.Array]:
        """Apply one action and resolve reward and respawn under ``params``."""
        action = jnp.asarray(action, dtype=jnp.int32)
        if action.shape != (self.num_agents,):
            raise ValueError(f"actions must have shape ({self.num_agents},).")
        del key  # deterministic dynamics; the wrapper owns drift randomness
        move = jnp.clip(action[0], 0, 2) - 1  # 0 left, 1 stay, 2 right
        position = jnp.clip(state.position + move, 0, self.length - 1)
        reached = position == params.goal.astype(jnp.int32)
        reward = jnp.asarray([jnp.where(reached, 1.0, 0.0)], dtype=jnp.float32)
        position = jnp.where(reached, jnp.int32(0), position)  # same-tick respawn
        return LineState(position, jnp.asarray(reached)), reward

    def observe(self, state: LineState, params: LineParams) -> tuple[jax.Array, EnvInfo]:
        """Return the public observation and the common contract info."""
        obs = jnp.stack((state.position, params.goal.astype(jnp.int32)))[None, :]
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 3), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return obs.astype(jnp.float32), info

    def privileged(self, state: LineState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}


def make(**kwargs: Any) -> DriftEnv[LineState, LineParams]:
    """Build LineWorld with optional drift, as a registered factory would."""
    drift = kwargs.pop("drift", None)
    return DriftEnv(LineWorld(**kwargs), drift, env_id="line-world-v0")
```

Building and running it:

```python
env = make(drift=pp.drift.Cyclic([3.0, 1.0], period=8))
state, obs, info = env.init(jax.random.key(0))
state, obs, reward, info = env.step(state, jnp.asarray([2]), jax.random.key(1))
carry = pp.runtime.Runner(env, pp.policies.random, ticks=32)(
    pp.runtime.init_lifetime(env, seed=0, window_size=32)
)
```

`tests/test_custom_env.py` checks the example against the contract, including
that `LineWorld` itself contains no drift key, clock, switch logic, Oracle
fields or `with_drift`: those belong to the framework.

## Method reference

### `default_params` — pytree of float arrays

Every quantity that may change over the lifetime, at its starting value. Use a
`NamedTuple`; each field is addressable by name in `drift={"field": process}`
and gets Oracle fields `{field}_parameters`, `{field}_phase` and
`{field}_changed`. A field's shape is fixed, and drift tables follow it: a
scalar field takes a one-dimensional table such as `Cyclic([3.0, 1.0], 8)`.

### `derive(params) -> params` — optional

Map raw drift values to the effective parameters the dynamics consume:
rounding to a cell, clipping to bounds, recomputing dependent quantities. The
framework calls it before every `init`, `step` and `observe`, so rounding never
leaks into `step`. A drift too small to change the effective value raises no
`space_change` event. A declared switch still fires whenever the drift process
changes phase: with `Cyclic([1.2, 1.4], 4)` and a `derive` that rounds to cells,
the observed goal never moves, yet the Oracle's `boundary` and `goal_changed`
are true every fourth tick (measured on `LineWorld`), and recovery metrics
measure from those switches.

**Do not call `derive` again inside `init`, `step` or `observe`.** The
framework applies `derive` and passes the resulting parameters to those
methods, so deriving twice converts a value that is already effective (for
example, indexing a scalar that was already reduced to one). Hand-written
wrappers for backends that need a different parameter struct should do that
conversion in `derive` and use the result directly.

### `init(key, params) -> (state, obs, info)`

Start the lifetime once. `key` is the lifetime seed, unchanged. Return the
world state only: no drift state, no drift key, no clock.

Split `key` however you like: the framework derives its own keys (the drift
processes' init keys and the drift stream here, and in `step` the key for
`after_drift` and for a regret reference policy) from a tagged `fold_in` of
the key it also hands you, so no `jax.random.split` of any width can reproduce
one. Consume each key once: split it into every child you need before drawing
from any of them.

### `step(state, action, key, params) -> (state, reward)`

Advance the world one tick under the current parameters. `key` supplies world
randomness; the framework steps drift separately and then calls `observe`
with the parameters that follow, so `step` returns no observation. Record
per-tick events such as respawns in the state, where `observe` reports them.
`reward` is float32 with shape `(agents,)`.

### `observe(state, params) -> (obs, info)`

Describe the current world under `params`: the observation and the three info
keys above. It is also how a learner sees the world after `init` and after
every tick.

### `privileged(state) -> dict[str, jax.Array]`

World-side diagnostics an evaluator may want but a policy must not see by
default; return `{}` if there are none. Drift diagnostics come from the
framework. A key your `privileged` returns is reachable through `pp.Oracle`
only if you also list it in an `oracle_fields` class attribute (for example
`oracle_fields = ("distance",)`). Without it `pp.Oracle(env, expose=["distance"])`
raises `Unknown oracle field`, and `pp.testing.check_env` still passes, so
declare every field you return.

### `default_drift_field` and `mask_params` — optional

`default_drift_field` names the parameter a bare process drives, so
`drift=pp.drift.Cyclic(...)` works without a mapping. `mask_params` lists
parameters whose effective value changes the action mask. When one of them
moves, the framework compares each agent's mask before and after at the same
world state and raises `space_change` for the agents whose mask changed (see
[evolving spaces](evolving-spaces.md)).

### `check_drift(drifts)` — optional

`DriftEnv` calls it with the mapping from parameter names to drift processes
every time drift is bound: at construction, in `with_drift` and through
`pp.compose`. Raise `ValueError` there for a process the world cannot run; a
task switch uses it to refuse a task table that names a task it does not have.

## Attaching drift

```python
env = DriftEnv(LineWorld(), pp.drift.Cyclic([3.0, 1.0], 8))  # default field
env = DriftEnv(LineWorld(), {"goal": pp.drift.RandomWalk(low=0, high=3, scale=0.2)})
schedule = pp.drift.Sequence([(pp.drift.Stationary(), 100), (pp.drift.Cyclic([3.0, 1.0], 8), 400)])
env = DriftEnv(LineWorld(), schedule)
rebound = pp.compose(env, pp.drift.Cyclic([1.0, 3.0], 16))  # keeps other fields
```

Any process from `perpetua.drift` works, including the `Sequence` and `Sum`
combinators (see [lifetimes and drift](native-api.md)). Recovery tracking measures from a
**declared switch** (a change of `phase`); gradual processes such as `RandomWalk` never switch.

## Testing your environment

Run the contract checker on a factory that builds a fresh environment:

```text
pp.testing.check_env(lambda: make(drift=pp.drift.Cyclic([3.0, 1.0], period=8)))
```

It checks shapes and required `info` keys, privilege isolation and the Oracle,
seeded reproducibility, chunk continuity and the clock, stable state types, no
recompilation across chunks, and a checkpoint round-trip, and names the first
failure (see [testing environments](introduction/testing.md)). `jax.vmap` over
lifetimes of identical shape must also work.

## Using your environment with the framework

```python
runner = pp.runtime.Runner(env, pp.policies.random, ticks=128)
carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=128))
mean_reward = carry.metrics.mean  # on-device streaming metric

trainer = pp.agents.PPORunner(env, pp.agents.PPO(action_size=3, config=pp.agents.PPOConfig()))
train_state = trainer(trainer.init(seed=0, window_size=128))  # collect + learn, no resets
```

Discrete-action environments work with `PPO`/`IPPO` and `DQN`/`IQL`;
continuous `Box` actions work with `SAC` (see
[training and checkpoints](training.md)). Keep the state a pure pytree of
fixed-shape arrays and set `metadata["checkpointable"] = True`.

## Registration

`pp.make("...")` resolves the built-in environments listed in the
[environment index](environments/index.md). Register a factory to reach yours by id, for
configuration files and resume paths that only know a string:

```python
pp.register("line-world-v0", make)  # factory returning the wrapped environment
env = pp.make("line-world-v0", length=4, drift=pp.drift.Cyclic([3.0, 1.0], period=8))
assert env.metadata["env_id"] == "line-world-v0"
pp.unregister("line-world-v0")
```

Rules: ids must be non-empty strings and factories callable; a built-in id or
an existing registration is refused unless you pass `overwrite=True`, and
`unregister` then restores the built-in. An override factory must construct
the environment directly; calling `make` for the same id would recurse.
