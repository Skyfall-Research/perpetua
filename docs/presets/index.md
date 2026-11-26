# Benchmarks and presets

A preset is a convenience constructor, not a new environment: it composes
existing adapters with a task switch and a drift schedule. Every preset records
`metadata["configuration"]["preset"]`: its name, the evaluation horizon
(`horizon_ticks`) and a declared k-percent tuning protocol, read back with
`pp.tuning.declared(env)`.

The last column of the table is a pipeline check, not a result: every row
trained for 32 ticks per task, which shows that the benchmark runs and is
scored with the shared metrics, and nothing about how well an agent learns.
Real-budget results exist for Continual World (nine seeds, on the
[Continual World page](continual-world.md)) and for the COOM CO8 lifetime
(three seeds, 200,000 ticks per task, in
[claim 6](../claims/comparability.md#a-real-budget-lifetime-on-co8)).

| Benchmark | Preset | Learner used | Pipeline check (32 ticks per task) | Page |
|---|---|---|---|---|
| Continual World (Meta-World) | `pp.presets.cw10()`, `cw20()` | SAC | IQM 0.217 [0.193, 0.250], forgetting 0.0014 | [Continual World](continual-world.md) |
| COOM | `pp.presets.coom_sequence("CO8")` | PPO (CNN) | IQM 0.083 [0.040, 0.138] | [COOM](coom.md) |
| MEAL | `pp.presets.meal_sequence(...)` | PPO (CNN) | zero reward at this budget | [MEAL](meal.md) |
| Switching-ALE / MinAtar | `pp.presets.switching_atari(...)`, `switching_minatar()` | PPO (CNN) | IQM 0.781 [0.781, 1.563] | [Task switching games](switching.md) |
| MORPHEUS | `pp.presets.morpheus_regimes()` | PPO | mock service: IQM 0.006 [−0.024, 0.032], 3/5 switches recovered | [MORPHEUS regimes](morpheus-regimes.md) |
| AgarCL screen (`agario-screen-v0`) | `pp.make("agarcl/agario-screen-v0")` | SAC | zero reward at this budget | [AgarCL](../agarcl.md) |

All rows come from one script, `examples/compare_benchmarks.py`, scored with the
same lifetime metrics. Rewards are not comparable across benchmarks. Published
agents have not been re-scored.

```{toctree}
:hidden:

continual-world
coom
meal
switching
morpheus-regimes
```
