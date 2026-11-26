# Task-switching games

Switching-ALE-style sequences: Atari games through EnvPool, all with the full
18-action space so shapes never change, or MinAtar games through gymnax in one
compiled lifetime.

```python
import perpetua as pp

atari = pp.presets.switching_atari(["Breakout-v5", "Asterix-v5"], period=100_000)  # Linux
minatar = pp.presets.switching_minatar(period=100_000)
```

- **Learner:** PPO with the CNN torso, built with
  `observation_space=env.observation_space`: Atari frames are then divided by
  255, MinAtar's binary channels by 1.
- **Measured:** measured as a pipeline check with tiny budgets (32 training ticks per task); see [claim 6](../claims/comparability.md) (Breakout and Asterix).
- **Limits:** ALE is GPL-2.0 and installed only with EnvPool; EnvPool is Linux
  only.
