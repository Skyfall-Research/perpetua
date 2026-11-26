# Tutorial: environment logic

A forager walks a `size × size` grid. Reaching the berry earns +1, and the
forager respawns on a random cell **on the same tick**: the respawn is reported
in `info["respawn"]`, and the lifetime goes on. The berry's position is the
parameter `berry`, so any drift process can move it.

Points to notice in the code:

- `derive` rounds and clips the raw `berry` value to a cell, so drift smaller
  than half a cell does not move the berry. A `Cyclic` step that small still
  fires `boundary` and counts for the recovery metrics, because a declared
  switch follows the drift phase, not the derived cell: a switch can fire with
  no visible change in the world. The framework calls `derive` before `init`,
  `step` and `observe`; do not call it yourself.
- `step` returns only the new state and the reward. The framework steps drift
  and then calls `observe` with the parameters that follow.
- `observe` returns the three `info` keys an environment provides:
  `action_mask`, `active_mask` and `respawn`. The framework adds the clock `t`
  and `space_change`.
- Shapes and dtypes never change, so a whole lifetime compiles once.

```{literalinclude} ../../../examples/custom_environment/tutorial2_logic.py
:language: python
```

Running the file steps the world by hand, then runs a compiled 256-tick chunk:

```text
clock 256 mean reward 0.01171875
```
