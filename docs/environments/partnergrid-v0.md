# Partner Grid

This environment is part of the [Multi-agent worlds](families/multi-agent.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("PartnerGrid-v0")` |
| Install | `uv sync --locked` |
| Actions | Discrete |
| Agents | `2` (1 learner-controlled, 1 scripted) |
| Action Shape | `(2,)` |
| Action Values | `Discrete(5)` per agent |
| Observation Shape | `(2, 7)` |
| Observation Values | `[0, 1]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `native JAX` |
| Default drift | `stationary` |
| Driftable parameters | `goal`, `partner` |
| Default drift field | `goal` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |

Cooperate with scripted partners whose cooperativeness drifts. Each partner moves beta-greedily toward its target or at random; beta is hidden.

![A frame of Partner Grid](../figures/environments/partnergrid-v0.svg)

*One frame after a few random ticks: the learner in blue, the scripted partner in grey and each agent's target in orange.*

## Definition

### Observation Space

As `CooperativeGrid-v0`; partner slots observe zeros.

### Action Space

Integer `(agents,)`: stay/up/down/left/right = 0/1/2/3/4. Masked or invalid actions become stay. Partner slots allow only stay.

### Rewards

float32 `(agents,)`: the joint +1 when every slot reaches its target.

### Initial State

Unique spawns outside the targets.

### Respawns

Everyone respawns after a joint collection.

### Drift

`goal`, and `partner` (each partner's beta in [0, 1]): pass `partner_drift=` or a mapping such as `drift={"goal": ..., "partner": ...}`.

## Arguments

| Argument | Default |
|---|---|
| `size` | `5` |
| `num_learners` | `1` |
| `num_partners` | `1` |
| `drift` | `None` |
| `partner_drift` | `None` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("PartnerGrid-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `goal` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "PartnerGrid-v0",
    drift={"goal": pp.drift.Cyclic([(4, 4), (0, 0)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
```

## Limits

`size` is at most 64, and `num_learners + num_partners` is at most `min(32, size**2 // 2)` slots.

## See also

The [Multi-agent RL](../marl.md) guide.

## Privileged fields

Hidden from the agent; expose any of them with `pp.Oracle(env, expose=["task_id", ...])`, which records the choice in `metadata["oracle_exposure"]`.

| Field | Meaning |
|---|---|
| `phase` | the drift phase: an integer that changes at every declared switch (for `Cyclic`, the index of the current value) |
| `drift_parameters` | the current value of the default drifting parameter |
| `drift_changed` | deprecated alias of `boundary` |
| `task_id` | the active task: the task index under a task switch, the regime under MORPHEUS, otherwise the phase of the default drifting parameter |
| `boundary` | whether a declared switch happened on this tick (the name task-sequence protocols use, for example SAC's `task_boundary_reset`) |
| `goal_parameters`, `goal_phase`, `goal_changed` | the current value, phase and switch flag of `goal` |
| `partner_parameters`, `partner_phase`, `partner_changed` | the current value, phase and switch flag of `partner` |

## Version History

- `v0`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.envs.partner_grid.PartnerGrid
   :no-index:
```
