# Simple Spread

This environment is part of the [Multi-agent worlds](families/multi-agent.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("jaxmarl/MPE_simple_spread-v3")` |
| Install | `uv sync --locked --extra jaxmarl` |
| Actions | Discrete |
| Agents | `3` |
| Action Shape | `(3,)` |
| Action Values | `Discrete(5)` per agent |
| Observation Shape | `(3, 18)` |
| Observation Values | `(-inf, inf)` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `jaxmarl-0.2.0` |
| Default drift | `stationary` |
| Driftable parameters | `contact_margin`, `damping`, `dt`, `local_ratio`, `mass`, `reward_multiplier` |
| Default drift field | `reward_multiplier` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [MPE2 Simple Spread](https://mpe2.farama.org/environments/simple_spread/) |

JaxMARL's cooperative navigation as a continuing lifetime, with optional scripted partner or opponent slots whose cooperativeness drifts.

![A frame of Simple Spread](../figures/environments/jaxmarl-mpe-simple-spread-v3.svg)

*A drawing of the state after 12 random ticks, with each entity at its own radius: agents in blue, adversaries in orange and landmarks in grey.*

The observation, action and reward definitions are upstream's: see [MPE2 Simple Spread](https://mpe2.farama.org/environments/simple_spread/). This page lists only what changes in Perpetua.

## What changes in Perpetua

### Respawns

The backend's episode end becomes an in-place respawn. JaxMARL ends an episode after `max_steps + 1` transitions (it checks the limit before the step counter increments), so with its default of 25 `info['respawn']` is true on ticks 26, 52, 78 and so on.

### Drift

`reward_multiplier`, the dynamics fields listed under Driftable parameters, and, with `num_scripted=` at least 1, `scripted` (cooperativeness in [0, 1]) via `scripted_drift=`. Without scripted slots `scripted` is not driftable.

## Arguments

| Argument | Default |
|---|---|
| `num_agents` | `3` |
| `num_landmarks` | `3` |
| `local_ratio` | `0.5` |
| `max_steps` | `25` |
| `num_scripted` | `0` |
| `scripted_behavior` | `'partner'` |
| `drift` | `None` |
| `scripted_drift` | `None` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("jaxmarl/MPE_simple_spread-v3")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_multiplier` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "jaxmarl/MPE_simple_spread-v3",
    drift={"reward_multiplier": pp.drift.Cyclic([(1,), (0.5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
```

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
| `contact_margin_parameters`, `contact_margin_phase`, `contact_margin_changed` | the current value, phase and switch flag of `contact_margin` |
| `damping_parameters`, `damping_phase`, `damping_changed` | the current value, phase and switch flag of `damping` |
| `dt_parameters`, `dt_phase`, `dt_changed` | the current value, phase and switch flag of `dt` |
| `local_ratio_parameters`, `local_ratio_phase`, `local_ratio_changed` | the current value, phase and switch flag of `local_ratio` |
| `mass_parameters`, `mass_phase`, `mass_changed` | the current value, phase and switch flag of `mass` |
| `reward_multiplier_parameters`, `reward_multiplier_phase`, `reward_multiplier_changed` | the current value, phase and switch flag of `reward_multiplier` |
| `<name>_parameters`, `<name>_phase`, `<name>_changed` for `scripted` | constants: the backend's value of each field, phase 0, never changed; `drift=` rejects these fields |

## Version History

- `v3`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.jaxmarl.JaxMARLSimpleSpread
   :no-index:
```
