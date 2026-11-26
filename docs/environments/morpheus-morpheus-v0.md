# MORPHEUS

This environment is part of the [Continual benchmarks](families/benchmarks.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("morpheus/Morpheus-v0")` |
| Install | `uv sync --locked  # the mock; add --extra morpheus for a live service` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(5)` per agent |
| Observation Shape | `(1, 13)` |
| Observation Values | `[0, inf)` |
| Execution | `host` |
| Checkpointable | no |
| Backend | `morpheus-mock` |
| Default drift | `stationary` |
| Driftable parameters | `regime`, `reward_multiplier` |
| Default drift field | `regime` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values, `chaos_active`, `regime_index`, `invalid_action` ([all](#privileged-fields)) |
| Upstream | [MORPHEUS (paper)](https://openreview.net/forum?id=31P1VAfLkJ) |

A persistent MORPHEUS enterprise world: incident tickets raised by a chaos engine, remediated and verified. The default is a seeded mock; `mock=False` and `MORPHEUS_URL` use a live deployment over HTTP.

## Definition

### Observation Space

Thirteen values: open tickets; open tickets by priority (low, medium, high, critical) and by status (new, open, in_progress, on_hold); the verifier pass fraction of this tick's resolve (0 on a tick without one); in-transit shipments; units processed since the lifetime began; elapsed simulated hours.

### Action Space

`Discrete(5)`: 0 no-op, 1 triage the highest-priority untriaged ticket, 2 resolve and verify the highest-priority in-progress (else open) ticket, 3 order 25 units for inventory, 4 deliver the oldest in-transit shipment. Actions 3 and 4 land two simulated hours after the action, in the result of the following tick; the delivered shipment leaves transit at once and its unit is counted two hours later. An action without a target is a no-op.

### Rewards

`reward='paper'` (default) is the paper's composite, `0.5·r_failure + 0.25·r_ledger + 0.25·r_throughput`, computed per tick. `'failure'`, `'verification'`, `'capital'` or a callable select others; see the guide for the definitions and how this differs from the paper's schedule.

### Respawns

None: the world keeps running.

### Drift

`regime`, pushed to the world at each switch; Oracle-only (`regime_index`, `task_id`, `chaos_active`, `invalid_action`).

## Arguments

| Argument | Default |
|---|---|
| `mock` | `True` |
| `base_url` | `None` |
| `layout` | `'process-outbound'` |
| `headers` | `None` |
| `regimes` | `6 built-in entries (see the source)` |
| `reward` | `'paper'` |
| `client` | `None` |
| `drift` | `None` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("morpheus/Morpheus-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `regime` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "morpheus/Morpheus-v0",
    drift={"regime": pp.drift.Cyclic([0, 1], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
drifting.close()
env.close()  # host simulators hold an external process or library
```

## Limits

Not checkpointable (the live clock lives on the service).

## See also

The [MORPHEUS setup and the live service](../morpheus.md) guide.

## Privileged fields

Hidden from the agent; expose any of them with `pp.Oracle(env, expose=["task_id", ...])`, which records the choice in `metadata["oracle_exposure"]`.

| Field | Meaning |
|---|---|
| `phase` | the drift phase: an integer that changes at every declared switch (for `Cyclic`, the index of the current value) |
| `drift_parameters` | the current value of the default drifting parameter |
| `drift_changed` | deprecated alias of `boundary` |
| `task_id` | the active task: the task index under a task switch, the regime under MORPHEUS, otherwise the phase of the default drifting parameter |
| `boundary` | whether a declared switch happened on this tick (the name task-sequence protocols use, for example SAC's `task_boundary_reset`) |
| `regime_parameters`, `regime_phase`, `regime_changed` | the current value, phase and switch flag of `regime` |
| `reward_multiplier_parameters`, `reward_multiplier_phase`, `reward_multiplier_changed` | the current value, phase and switch flag of `reward_multiplier` |
| `chaos_active` | whether the declared MORPHEUS regime differs from the first regime (false for the first regime and for a revisit of it) |
| `regime_index` | the index of the MORPHEUS regime declared on this tick (the world takes it before the next transition) |
| `invalid_action` | whether the live service rejected the last action's call (the seeded mock never rejects one) |

## Version History

- `v0`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.morpheus.MorpheusLifetime
   :no-index:
```
