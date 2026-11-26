# Evolving spaces: masks, events and the space manager

Perpetua lifetimes never change observation or action **shapes**: an evolving
space changes what an action *means*, through masks, and announces it with a
public event. Compiled chunks therefore stay valid for the whole lifetime while
the affordances inside them come and go.

## The contract

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `info['action_mask']` | `(agents, actions)` | bool | which actions are legal *now* (every discrete environment; a continuous `Box` action has no legality, so the mask is a placeholder, see [custom environments](custom-environments.md)) |
| `info['space_change']` | `(agents,)` | bool | drift changed this agent's valid-action set on this tick |

**One definition.** For each agent, `space_change` is true when drift changed
that agent's valid-action set on this tick. The framework compares the agent's
mask under the previous parameters with its mask under the new ones, at the
same world state, with `mask_change`. So:

- only agents whose mask actually changed are flagged; a drift that closes a
  move another agent could not make anyway leaves that agent unflagged;
- a mask that changes because the agent moved (next to a wall, say) is not a
  space change, so the event never duplicates what `action_mask` already shows;
- it is false on `init` and `observe`, and always false for environments that
  declare no `mask_params`.

Every environment wrapped by `DriftEnv` (every registered environment) always
includes it, so the info pytree has one fixed structure. `Info` keeps it
`NotRequired` only for hand-written `Environment` implementations outside
`DriftEnv`. It is derived from public masks only and never reveals hidden
parameters.

## The space manager

```python
import jax
import perpetua as pp
from perpetua.spaces import SpaceManager, mask_change, space_signature, validate_stable

env = pp.make("DoorGrid-v0")
_, _, info = env.init(jax.random.key(0))
obs_space, act_space = env.observation_space, env.action_space
manager = SpaceManager(observation_space=obs_space, action_space=act_space)
manager.configuration  # JSON-ready signature recorded in metadata/checkpoints
manager.check(observation_space=obs_space, action_space=act_space)  # refuses shape changes
manager.event(info["action_mask"], info["action_mask"])  # device-side event value
```

- `space_signature(space)` records type, shape, dtype and discrete
  cardinality — the parts that must stay fixed during a lifetime.
- `validate_stable(reference, current)`, used by `SpaceManager.check`, raises
  when any of those changed, so a drift process or wrapper can never silently
  resize a space.
- `mask_change(previous, current)` is a pure device-side
  `(agents, actions) -> (agents,)` function. `DriftEnv` uses it to compute
  `space_change`, and `SpaceManager.event` wraps it for hand-written
  environments outside `DriftEnv`, so there is one implementation.
- The manager does no per-tick host work, and the recorded signature rides in
  `metadata['configuration']['spaces']` with `"evolving": "action-mask"`.

## A drifting affordance: `DoorGrid-v0`

```python
env = pp.make("DoorGrid-v0", size=5, drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 8))
```

The drifting parameter is a float32 `(4,)` door vector for up/down/left/right
(1 = open, 0 = closed), rounded on read. A closed door masks the matching move
action and blocks movement (masked actions become stay). Observations keep the
grid shape `(agents, 4)` — normalized row, column, goal row, goal column — with
the goal fixed at the bottom-right corner and rewards of +1 on collection with
an in-lifetime respawn. Policies learn the affordance state from
`action_mask` and the `space_change` event, never from a hidden parameter; door
parameters stay behind the Oracle (`privileged`), like every other drift
parameter.

Checks in `tests/test_evolving_spaces.py`: masks follow doors exactly, a masked
move becomes stay, the event fires exactly on switches (period 4 → ticks 3, 7,
11), observations are identical across door configurations at the same
position, the space signature refuses shape/cardinality changes, and a compiled
chunk crosses four switches with **zero recompilations**.

## Limits

- Shapes are fixed by design, not by a mechanism: an environment that genuinely
  needs different shapes must start a new lifetime with a new runner.
- The event reports *that* the valid-action set changed, not a diff; policies
  that need which actions appeared or disappeared compare masks themselves.
- Agents never appear or disappear: `active_mask` marks slots the learner does
  not control, and shapes stay constant. JaxMARL-style agent removal is
  expressed the same way.
