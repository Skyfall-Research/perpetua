# COOM sequences

COOM sequences (`CO8`, `CD8`, ...) in COOM's own task order, each task a worker
process in COOM's pinned runtime.

```bash
uv run python -m perpetua.adapters.coom_setup
```

```python
import perpetua as pp

env = pp.presets.coom_sequence("CO8", period=200_000)
```

- **Learner:** PPO with the CNN torso (12 discrete actions, `(1, 84, 84, 12)`
  observations).
- **Measured:** a pipeline check with tiny budgets (32 training ticks per task), and one real-budget CO8 lifetime (PPO, 200,000 ticks per task, three seeds); see [claim 6](../claims/comparability.md).
- **Limits:** ViZDoom state is opaque, so no checkpoints; COOM's TensorFlow
  continual-learning code is not used.

See [continual benchmarks](../benchmarks.md) for the runtime.
