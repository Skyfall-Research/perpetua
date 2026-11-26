# Continual benchmarks

Existing continual-RL benchmarks under the lifetime API: MEAL, Continual World (Meta-World), COOM, AgarCL and MORPHEUS. None is vendored; each is an optional dependency or a separately built runtime.

See each page: MEAL and Meta-World are extras; COOM and AgarCL build pinned runtimes with `python -m perpetua.adapters.coom_setup` and `python -m perpetua.adapters.agarcl_setup`.

Benchmark sequences are presets (see the benchmarks page); host simulators accept `compiled=True` to run inside the compiled runners.

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [MEAL Overcooked](../meal-overcooked-v0.md) | `meal/overcooked-v0` | Discrete | `2` | `(2, 9, 9, 26)` | [MEAL](https://github.com/TTomilin/MEAL) |
| [Meta-World MT1](../metaworld-mt1-v0.md) | `metaworld/MT1-v0` | Continuous | `1` | `(1, 39)` | [Meta-World](https://metaworld.farama.org/benchmark/benchmark_descriptions/) |
| [COOM](../coom-coom-v0.md) | `coom/COOM-v0` | Discrete | `1` | `(1, 84, 84, 12)` | [COOM](https://github.com/TTomilin/COOM) |
| [MORPHEUS](../morpheus-morpheus-v0.md) | `morpheus/Morpheus-v0` | Discrete | `1` | `(1, 13)` | [MORPHEUS (paper)](https://openreview.net/forum?id=31P1VAfLkJ) |
| [AgarCL (screen)](../agarcl-agario-screen-v0.md) | `agarcl/agario-screen-v0` | Continuous | `1` | `(1, 84, 84, 3)` | [AgarCL](https://github.com/AgarCL/AgarCL) |
| [AgarCL (grid)](../agarcl-agario-grid-v0.md) | `agarcl/agario-grid-v0` | Continuous | `1` | `(1, 128, 128, 8)` | [AgarCL](https://github.com/AgarCL/AgarCL) |
| [AgarCL Grand Arena](../agarcl-grand-arena-v0.md) | `agarcl/grand-arena-v0` | Continuous | `1` | `(1, 128, 128, 3)` | [AgarCL](https://github.com/AgarCL/AgarCL) |

```{toctree}
:hidden:

../meal-overcooked-v0
../metaworld-mt1-v0
../coom-coom-v0
../morpheus-morpheus-v0
../agarcl-agario-screen-v0
../agarcl-agario-grid-v0
../agarcl-grand-arena-v0
```
