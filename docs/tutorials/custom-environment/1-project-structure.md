# Tutorial: project structure

An environment package needs three things: the environment, a factory that
attaches drift, and a test. A typical layout:

```text
berry-field/
├── pyproject.toml           # depends on perpetua
├── berry_field/
│   ├── __init__.py          # pp.register("tutorial/BerryField-v0", make)
│   └── env.py               # BerryParams, BerryState, BerryField, make
└── tests/
    └── test_env.py          # pp.testing.check_env(make)
```

An environment is a class with the five required members below and an optional `derive`, plus four attributes:
`num_agents`, `observation_space`, `action_space` and `metadata`. It is a pure function of state and
parameters: all randomness comes from the `key` argument, and everything that
may change over the lifetime is a **parameter**, so a drift process can change
it without rebuilding anything. There is no `reset` and no `done`.

| Member | Purpose |
|---|---|
| `default_params` | Starting values of every drifting parameter, as float arrays |
| `derive(params)` | Optional: raw drift values to effective ones (rounding, clipping) |
| `init(key, params)` | Start the lifetime, once: `(state, obs, info)` |
| `step(state, action, key, params)` | One tick: `(state, reward)` |
| `observe(state, params)` | What the agent sees: `(obs, info)` |
| `privileged(state)` | World-side diagnostics only an Oracle may expose |

The framework's `DriftEnv` wraps the class and adds what every environment
needs but none should implement: the drift processes and their random stream,
declared switches, the lifetime clock, Oracle fields and checkpoint metadata.

The skeleton, with every method filled in by the next step:

```{literalinclude} ../../../examples/custom_environment/tutorial1_skeleton.py
:language: python
```
