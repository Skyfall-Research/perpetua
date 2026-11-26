# Native continual worlds

Worlds written for Perpetua in JAX. They compile, vmap and checkpoint, and they are the fastest way to study one drift question at a time: a moving goal, changing affordances, agent-driven drift, or an open-ended world.

```bash
uv sync --locked
```

Every native world takes `drift=` (one process or a mapping of parameter names to processes). Grid sizes and agent counts are fixed per lifetime.

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [Continual Grid](../continualgrid-v0.md) | `ContinualGrid-v0` | Discrete | `1` | `(1, 4)` | — |
| [Door Grid](../doorgrid-v0.md) | `DoorGrid-v0` | Discrete | `1` | `(1, 4)` | — |
| [Forager Grid](../foragergrid-v0.md) | `ForagerGrid-v0` | Discrete | `1` | `(1, 4)` | — |
| [Bean World](../beanworld-v0.md) | `BeanWorld-v0` | Discrete | `1` | `(1, 150)` | — |

```{toctree}
:hidden:

../continualgrid-v0
../doorgrid-v0
../foragergrid-v0
../beanworld-v0
```
