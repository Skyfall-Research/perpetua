# Tutorial: drift and evolving actions

In the world so far, change comes from moving the berry. This step adds a
second kind: the set of available actions changes. A new parameter, `fences`,
holds one value per move direction (1 open, 0 closed). A closed fence:

- masks its move in `info["action_mask"]` (stay is always allowed), and
- blocks it in `step`, where a fenced move becomes stay.

Listing `fences` in `mask_params` tells the framework that this parameter
changes the action set. It then compares each agent's mask before and after a
change and raises `info["space_change"]` on exactly those ticks. This is
Perpetua's version of PettingZoo's action masking, with the difference that the
mask evolves over the lifetime while the action space keeps its shape.

```{literalinclude} ../../../examples/custom_environment/tutorial3_drift_and_masks.py
:language: python
```

Each drifting parameter gets privileged Oracle fields
(`<name>_parameters`, `<name>_phase`, `<name>_changed`), hidden from the agent:

```text
space_change at ticks [16, 32, 48]
mask now [True, True, True, False, False]
hidden from the agent: ['berry_changed', 'berry_parameters', 'berry_phase', 'boundary', 'drift_changed', 'drift_parameters', 'fences_changed', 'fences_parameters', 'fences_phase', 'phase', 'task_id']
```
