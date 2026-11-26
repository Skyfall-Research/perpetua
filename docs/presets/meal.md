# MEAL

A reproducible sequence of procedurally generated MEAL Overcooked layouts in one
native JAX lifetime.

```bash
uv sync --locked --extra meal --extra training
```

```python
import perpetua as pp

env = pp.presets.meal_sequence(sequence_length=8, difficulty="medium", seed=42)
```

- **Learner:** PPO or IPPO with the CNN torso; `on_switch="restart"` is
  available because the layouts are functional JAX environments. The torso
  reads the 26 layers as they are (pixel scale 1) on every tick: three of them
  are counts (onions in a pot, onions in a soup, cooking time left, up to 19),
  not pixels.
- **Measured:** measured as a pipeline check with tiny budgets (32 training ticks per task); see [claim 6](../claims/comparability.md); at that budget no soup was delivered, so recovery is
  trivially immediate.
- **Limits:** the `meal` extra pins `tfp-nightly` until a stable
  TensorFlow Probability works with JAX 0.11 (checked nightly).
