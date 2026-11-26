# MORPHEUS regimes

The six chaos regimes of the [MORPHEUS](https://openreview.net/forum?id=31P1VAfLkJ) evaluation suite in one
persistent world:
baseline, EDI spike, infrastructure degradation, financial pressure, workforce
constraint, then a baseline revisit.

```python
import perpetua as pp

env = pp.presets.morpheus_regimes(period=500)  # mock by default
```

After the sixth regime (the baseline revisit) the world stays in it: the
schedule does not wrap into another baseline block.

- **Metrics:** `pp.protocol` reproduces the suite's six metrics (all 140
  reference answers); the revisit is labelled as the baseline it repeats, so
  forgetting compares the two encounters.
- **Measured:** PPO on the mock (H200, 2026-11-03), as a pipeline check with a
  tiny budget: 32 training ticks per regime (2 updates of 16 steps), not the 500
  of `period=500`. Mean −0.002 per tick, IQM 0.006 [−0.024, 0.032], forgetting
  0.000, 3 of 5 switches recovered. This is not a learning result; see
  [claim 6](../claims/comparability.md).
- **Limits:** the live client has not been run against a real deployment by
  Perpetua; it needs `MORPHEUS_URL`. Not checkpointable.

See [MORPHEUS](../morpheus.md).
