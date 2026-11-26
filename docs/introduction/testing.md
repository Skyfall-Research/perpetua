# Testing environments

`pp.testing.check_env(factory)` checks that an environment honours the
lifetime contract. `factory` is a zero-argument function that builds a fresh,
identically configured environment, because several checks compare two
instances. It raises `pp.testing.ContractError` with a message naming the first
failure, and returns `None` when everything passes.

```python
import perpetua as pp

pp.testing.check_env(lambda: pp.make("ContinualGrid-v0", size=4))
pp.testing.check_env(
    lambda: pp.make("DoorGrid-v0", drift=pp.drift.Cyclic([(1, 1, 1, 1), (1, 0, 1, 0)], 2))
)
```

| Check | What it asserts |
|---|---|
| Shapes | `info` has `action_mask`, `active_mask`, `respawn`, `space_change` and `t`; observations, masks, events and rewards keep their shapes; `info` and the state keep one structure; rewards are finite |
| Public types | observations have the dtype their space declares; `action_mask`, `active_mask`, `respawn` and `space_change` are boolean; rewards are floating point; for discrete actions the mask is `(agents, actions)` and every active agent has a valid action |
| Privilege | none of the environment's declared Oracle fields (`boundary`, `task_id`, `phase`, `drift_parameters` and the per-field aliases such as `goal_changed`) in public `info`; `boundary` and `task_id` are declared (`drift_changed` is only the deprecated alias of `boundary`); every declared Oracle field is produced; `pp.Oracle` exposes exactly what is requested and records it in `metadata["oracle_exposure"]` |
| Reproducibility | two instances with the same seeds and actions give identical states and rewards |
| Continuity | one chunk of `2n` ticks equals two chunks of `n`, the lifetime clock counts every tick, and no reward of the random lifetime is non-finite |
| State types | a JAX world's state keeps the same types from `init` through `step`. A weakly typed leaf (often built from a Python scalar) would make every compiled runner compile twice. Worlds wrapped in `DriftEnv` are normalised automatically, so this check matters for environments that implement the outer contract themselves |
| Compilation | a second chunk compiles nothing, and no reward of the random lifetime is non-finite |
| Checkpoints | a saved lifetime resumes exactly, when `metadata["checkpointable"]` is true, and no reward of the resumed lifetime is non-finite |

The shapes, public types and reproducibility checks call `init` and `step`
directly and look at every tick: four ticks with the all-zero action, then
`ticks` ticks (default 4) with actions sampled from a fixed key. Discrete slots
draw among the actions that tick's `action_mask` allows, and an inactive slot
is sent action 0; a `Box` draws uniformly within its bounds, so it must be
bounded. The other checks run uniformly random lifetimes of `ticks` or
`2 * ticks` ticks through a runner; they compare whole lifetimes and
additionally fail when `metrics.healthy` is false, which a non-finite reward
causes.

A short random walk does not reach every state. A reward that is non-finite
only at a distant goal, or a mask that closes only late, is found when `ticks`
is large enough for random play to get there, so pass a larger `ticks` for
your own worlds:

```python
import perpetua as pp

pp.testing.check_env(lambda: pp.make("ContinualGrid-v0", size=4), ticks=64)
```

Host simulators (`metadata["execution"] == "host"`) run through `HostRunner`;
the state-type and compilation checks apply to JAX worlds only. The project's
own suite runs `check_env` on every registered environment
(`tests/test_environment_conformance.py`).

For long runs, `scripts/soak.py env --spec <id>` adds what short checks cannot
see: finite state and observations over 10⁸ ticks, integer counters that would
overflow within 10¹⁰ ticks, recompiles and memory growth.
