# Sweeps and tuning budgets

Two tools for hyperparameters: running a whole sweep as one compiled job, and
declaring how much of a lifetime tuning is allowed to spend.
[Training learners](training.md) covers the learners themselves.

(sweeps-as-data)=
## Hyperparameter sweeps as data

Continuous learner hyperparameters are **traced state**, not compile-time
constants. `PPOState`, `DQNState` and `SACState` each carry a
`hyperparameters` pytree (`PPOHyperparameters`, `DQNHyperparameters`,
`SACHyperparameters`) holding the knobs that sweeps vary: learning rate,
gamma, GAE lambda, clip epsilon, value/entropy coefficients, epsilon,
SAC tau, maximum gradient norm and the L2 coefficient. Structural choices
(rollout length, epochs, minibatches, hidden size, buffer capacity, target
sync interval, reset schedule, alpha autotuning) stay in the config.

```python
import jax
import jax.numpy as jnp

import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner
from perpetua.agents.ppo import PPOHyperparameters

env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
config = PPOConfig(rollout_steps=128)
trainer = PPORunner(env, PPO(action_size=5, config=config))
base = PPOHyperparameters.from_config(config)
sweep = jax.tree.map(  # a batch of two learning rates, everything else fixed
    lambda a, b: jnp.stack([a, b]),
    base._replace(learning_rate=jnp.asarray(1e-4)),
    base._replace(learning_rate=jnp.asarray(1e-3)),
)


def initialize(hp):  # one lifetime per hyperparameter set
    return trainer.init(seed=0, window_size=128, hyperparameters=hp)


states = jax.jit(jax.vmap(initialize))(sweep)  # one compiled call
states = jax.vmap(trainer)(states)  # batched updates
```

`init` may be wrapped in `jax.jit`; the runner itself may not. A learner
runner is already compiled and must be the outermost compiled call, so batch
it with a bare `jax.vmap(trainer)`: `jax.jit(jax.vmap(trainer))` is refused
with an error that names this.

Because the values are state, changing them reuses the same compiled program:
the test suite asserts **zero compilation events** when the same shapes run
with different learning rates, and that the resulting parameters differ.
Checkpoints carry the resolved values, so a resumed sweep keeps its
per-lifetime settings; the checkpoint metadata still records the config
defaults, so read `train_state.agent.hyperparameters.as_dict()` (or the example
reports, which include it) when a run's actual settings matter. Concrete
overrides are validated at `init` (finiteness, learning rate, gamma, clip,
tau, epsilon and coefficient ranges), per slot for `IPPO` and `IQL`; values
inside a `vmap`/`jit` trace skip those checks because they are not concrete.
SAC's fixed `alpha` and autotune `target_entropy` are traced values too. With
`autotune_alpha=True` the traced `alpha` is ignored: `log_alpha` starts from
the configured `SACConfig.alpha`, and `task_boundary_reset` resets it to that
value at each switch. The L2 penalty term is gated
statically by the configured coefficient: sweeping L2 therefore requires a
nonzero configured default. `IPPO` and `IQL` accept a scalar hyperparameter
pytree (broadcast to every slot) or one with a leading agent axis for
per-slot settings.

## Declared tuning protocol

A lifetime has no fixed horizon to tune for, so `pp.tuning` makes the tuning
protocol part of the experiment. `tune` implements k-percent tuning (Mesbahi et
al., 2024): every candidate runs on the first `fraction` of the lifetime for a
few tuning seeds, the best by mean or IQM is kept, and only it is evaluated on
the full lifetime with disjoint seeds. Tuning ranks by the mean of the middle
half of the sorted seed scores, with fractional weight at the cuts, so IQM
selection differs from the mean even with three seeds.

Here `learning_rates` is the list of candidates and `score(candidate, seeds,
ticks)` runs one lifetime per seed and returns one score per seed. The
lifetime is kept short so the example runs in seconds:

```python
from perpetua.agents import PPO, PPOConfig, PPORunner

grid = pp.make("ContinualGrid-v0", size=4)
learning_rates = [1e-4, 1e-3, 1e-2]


def score(learning_rate, seeds, ticks):
    config = PPOConfig(rollout_steps=32, learning_rate=learning_rate)
    trainer = PPORunner(grid, PPO(action_size=5, config=config))
    scores = []
    for seed in seeds:
        state = trainer.init(seed=seed)
        for _ in range(ticks // 32):
            state = trainer(state)
        scores.append(float(state.lifetime.metrics.mean.mean()))
    return scores


protocol = pp.tuning.TuningProtocol(0.1, tuning_seeds=3, evaluation_seeds=10)
result = pp.tuning.tune(
    protocol, learning_rates, score, lifetime_ticks=1_280, key=jax.random.key(0)
)
result.best, result.intervals["iqm"], result.ticks_spent, result.protocol
```

Every benchmark preset declares its protocol in
`metadata["configuration"]["preset"]["tuning"]` together with its evaluation
horizon (`horizon_ticks`). The default, `pp.presets.DEFAULT_TUNING`, tunes on
the first 10% with 3 seeds, selects by IQM and evaluates on 10 fresh seeds (a
`TuningProtocol(...)` you build yourself selects by `"mean"` unless you pass
`selection="iqm"`);
`pp.tuning.declared(env)` reads it back:

```python
env = pp.presets.cw10()
protocol = pp.tuning.declared(env)
horizon = env.metadata["configuration"]["preset"]["horizon_ticks"]  # 10_000_000
```

Pass both to `tune` as above, `tune(protocol, candidates, score,
lifetime_ticks=horizon, key=key)`, with a `score` that trains on this
benchmark.

`score(candidate, seeds, ticks)` returns one score per seed and may vmap the
seeds through traced hyperparameters (above). The result records the protocol,
every tuning score and the environment ticks spent on tuning and evaluation,
so the cost of tuning is reported with the score. The test suite tunes a PPO
learning rate this way.
