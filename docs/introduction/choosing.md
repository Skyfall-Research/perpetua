# Which benchmark should I use?

Start from the question you are asking. Every environment below runs under
the same lifetime API and is scored with the same metrics, so you can begin on
a small native world and move to a benchmark later without changing the
experiment code.

| Research question | Start with | Then scale to | Measure with |
|---|---|---|---|
| Loss of plasticity | [`ContinualGrid-v0`](../environments/continualgrid-v0.md) with `Cyclic` drift | [`BeanWorld-v0`](../environments/beanworld-v0.md), [task-switching games](../presets/switching.md) | [health probes](../evaluation.md#learner-health-probes-loss-of-plasticity), [plasticity options](../training-options.md#plasticity-options) |
| Forgetting and forward transfer | the [quickstart](../tutorials/quickstart.md) (two goals in turn) | [Continual World](../presets/continual-world.md), [COOM](../presets/coom.md), [MEAL](../presets/meal.md) | [`revisit`](../evaluation.md#scoring-saved-policies-revisit), [recovery and forgetting](../evaluation.md#recovery-and-forgetting) |
| Multi-agent non-stationarity | [`PartnerGrid-v0`](../environments/partnergrid-v0.md), [`OpponentGrid-v0`](../environments/opponentgrid-v0.md) | [JaxMARL MPE tasks](../environments/families/multi-agent.md), [MEAL Overcooked](../presets/meal.md) | [multi-agent guide](../marl.md) |
| Gradually changing dynamics | [`gymnax/CartPole-v1`](../environments/gymnax-cartpole-v1.md) with `RandomWalk` | [`mjx/Pendulum-v0`](../environments/mjx-pendulum-v0.md), the other [control tasks](../environments/families/control.md) | windowed reward and [recovery](../evaluation.md#recovery-and-forgetting); on the MJX tasks also [same-state regret](../evaluation.md#same-state-regret-against-a-reference-policy), which is [always zero on CartPole](../evaluation.md#where-same-state-regret-reads-zero) |
| Changing action availability | [`DoorGrid-v0`](../environments/doorgrid-v0.md) | | [evolving spaces](../evolving-spaces.md) |
| Drift caused by the agent | [`ForagerGrid-v0`](../environments/foragergrid-v0.md) with `Accumulate` | | [custom drift](../tutorials/custom-drift.md) |
| Open-ended worlds | [`BeanWorld-v0`](../environments/beanworld-v0.md) | [AgarCL](../agarcl.md), [MORPHEUS](../morpheus.md) | lifetime metrics on one clock |
| Compute-aware time | any native world with the [compute-aware runner](../compute-time.md) (experimental) | | the same metrics, with thinking time charged in ticks |

## How to choose within a row

- **Native worlds** (the grids and Bean World) need no extras, compile, and
  run in seconds on a laptop CPU. Use them to isolate one question and to
  check an idea before spending GPU time.
- **Wrapped benchmarks** (Continual World, COOM, MEAL, AgarCL, MORPHEUS) are
  what other papers report on. They need an extra or a separate runtime, and
  most step a CPU simulator, so they run slower. Use them when you need a
  number that others can compare against.
- **Task sequences** come as [presets](../presets/index.md). A preset fixes
  the tasks and their order and records the switch period and a tuning
  budget, so a result can state exactly which protocol it ran.

## What is and is not established

The preset pages report a pipeline check with tiny budgets: each benchmark
runs and is scored, and nothing is claimed about how well an agent learns.
Real-budget results exist for Continual World
([page](../presets/continual-world.md); it is not a reproduction of the
paper's forward transfer) and for a COOM CO8 lifetime
([claim 6](../claims/comparability.md#a-real-budget-lifetime-on-co8)). Treat the bundled learners as
reference implementations, not tuned baselines.
