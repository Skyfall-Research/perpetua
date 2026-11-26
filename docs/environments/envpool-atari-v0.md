# EnvPool Atari

This environment is part of the [CPU bridge](families/cpu-bridge.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("envpool/Atari-v0")` |
| Install | `uv sync --locked --extra cpu-bridge  # Linux` |
| Actions | Discrete |
| Agents | `8` |
| Action Shape | `(8,)` |
| Action Values | `Discrete(18)` per agent |
| Observation Shape | `(8, 84, 84, 4)` |
| Observation Values | `[0, 255]` |
| Execution | `host` |
| Checkpointable | no |
| Backend | `envpool-atari-1.2.7` |
| Default drift | `stationary` |
| Driftable parameters | `reward_multiplier` |
| Default drift field | `reward_multiplier` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [the Arcade Learning Environment](https://ale.farama.org/environments/) |

One Atari game through EnvPool with `full_action_space=True` (18 actions), so games can switch without changing shapes (`pp.presets.switching_atari`).

## Definition

### Observation Space

float32 `(envs, 84, 84, 4)`, channel-last, values 0 to 255: four stacked single-channel 84x84 frames per slot (EnvPool's preprocessing), not ALE's raw screen, which ALE's page describes. A PPO or DQN learner with the CNN torso, built with `observation_space=env.observation_space`, divides them by 255.

### Action Space

`Discrete(18)` per slot: ALE's full action set for every game, not the game's own minimal set.

### Rewards

Each game's own score change, as defined by the Arcade Learning Environment page above; the adapter applies no rescaling beyond `reward_multiplier` drift.

### Respawns

A slot whose game ends is inactive on that tick, because EnvPool ignores the action it receives next, and its respawn is reported on the following tick with the reset frame.

### Drift

`reward_multiplier`; game switching is a preset.

## Arguments

| Argument | Default |
|---|---|
| `task` | `'Breakout-v5'` |
| `num_envs` | `8` |
| `num_threads` | `4` |
| `drift` | `None` |
| `compiled` | `False` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("envpool/Atari-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_multiplier` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "envpool/Atari-v0",
    drift={"reward_multiplier": pp.drift.Cyclic([(1,), (0.5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
drifting.close()
env.close()  # host simulators hold an external process or library
```

## Limits

Linux and EnvPool only; ALE is GPL-2.0 and never vendored.

## See also

The [Environment adapters and the CPU bridge](../adapters.md) guide.

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
.. autoclass:: perpetua.adapters.host.EnvPoolAtari
   :no-index:
```
