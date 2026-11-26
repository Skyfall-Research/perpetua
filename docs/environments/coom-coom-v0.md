# COOM

This environment is part of the [Continual benchmarks](families/benchmarks.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("coom/COOM-v0")` |
| Install | `uv run python -m perpetua.adapters.coom_setup` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(12)` per agent |
| Observation Shape | `(1, 84, 84, 12)` |
| Observation Values | `(-inf, inf)` |
| Execution | `host` |
| Checkpointable | no |
| Backend | `coom-7929801176c6` |
| Default drift | `stationary` |
| Driftable parameters | `reward_multiplier` |
| Default drift field | `reward_multiplier` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [COOM](https://github.com/TTomilin/COOM) |

One COOM (MIT) ViZDoom scenario and task, run as a worker in COOM's own pinned runtime. `pp.presets.coom_sequence('CO8')` cycles a COOM sequence.

## Definition

### Observation Space

float32 `(agents, 84, 84, 12)`: COOM's default wrappers (resize to 84x84, rescale, running normalisation) with the four stacked RGB frames merged into channels.

### Action Space

`Discrete(12)` per agent, COOM's unified action space (`build_multi_discrete_actions`): turn (none, right, left) by move forward (no, yes) by execute (no, yes), with index `4*turn + 2*move + execute`. What execute does depends on the scenario: it presses a button, shoots, jumps or accelerates.

### Rewards

float32 `(agents,)`: the scenario's own reward, unchanged. See [COOM](https://github.com/TTomilin/COOM) for each scenario.

### Respawns

Scenario episode ends become respawns.

### Drift

`reward_multiplier`; sequences are presets.

## Arguments

| Argument | Default |
|---|---|
| `scenario` | `'pitfall'` |
| `task` | `'default'` |
| `drift` | `None` |
| `compiled` | `False` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("coom/COOM-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_multiplier` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "coom/COOM-v0",
    drift={"reward_multiplier": pp.drift.Cyclic([(1,), (0.5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
drifting.close()
env.close()  # host simulators hold an external process or library
```

## Limits

Opaque simulator state: not checkpointable. Needs `python -m perpetua.adapters.coom_setup`.

## See also

The [Benchmark setup: MEAL, Continual World and COOM](../benchmarks.md) guide.

## Privileged fields

Hidden from the agent; expose any of them with `pp.Oracle(env, expose=["task_id", ...])`, which records the choice in `metadata["oracle_exposure"]`.

| Field | Meaning |
|---|---|
| `phase` | the drift phase: an integer that changes at every declared switch (for `Cyclic`, the index of the current value) |
| `drift_parameters` | the current value of the default drifting parameter |
| `drift_changed` | deprecated alias of `boundary` |
| `task_id` | the active task: the task index under a task switch, the regime under MORPHEUS, otherwise the phase of the default drifting parameter |
| `boundary` | whether a declared switch happened on this tick (the name task-sequence protocols use, for example SAC's `task_boundary_reset`) |
| `reward_multiplier_parameters`, `reward_multiplier_phase`, `reward_multiplier_changed` | the current value, phase and switch flag of `reward_multiplier` |

## Version History

- `v0`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.coom.COOMLifetime
   :no-index:
```
