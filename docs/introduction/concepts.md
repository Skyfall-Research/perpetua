# Concepts

Perpetua is for **continual reinforcement learning**: an agent lives in one
world that keeps changing and never ends, and it must keep learning. There are
no episodes, no resets, no separate test phase and, by default, nobody tells
the agent when the world changed. Everything below follows from that.

Why it matters: most RL training assumes a fixed task and stops learning once
deployed. Real worlds keep changing, and networks trained for a long time on a
changing stream are known to lose the ability to learn new things, and to
forget what they learned before. Continual RL studies agents that keep
learning anyway, and Perpetua provides the changing worlds and the measurements.

An environment is three parts composed together:

```text
environment = base dynamics  ∘  drift on parameters  ∘  (possibly evolving) action spaces
              step(state, a)     θ changes over time    masks change, shapes do not
```

## Six terms to start with

| Term | Meaning |
|---|---|
| **Lifetime** | One never-ending interaction. `env.init(key)` starts it once; `env.step` advances it forever. It replaces Gymnasium's sequence of episodes. |
| **Tick** | One environment step. The lifetime clock `info["t"]` counts ticks; it is 64-bit, so it never wraps in practice. Read it as a Python integer with `info["t"].to_int()`. |
| **Respawn** | A death, a reached goal or a kept legacy time limit. The world puts the agent back on the same tick and reports it in `info["respawn"]`; the lifetime, learner and clock continue. It replaces `terminated`/`truncated`. |
| **Drift** | How parameters change over time, given as `drift=` to `pp.make`: one process for the default parameter, or a mapping `{"gravity": process}`. Processes: `Stationary` (no change, the default), `Cyclic` (fixed values in turn), `RandomWalk` (gradual noise), `Procedural` (fresh random values every period), `Accumulate` (driven by the agents' own behaviour), `Schedule` (like `Cyclic`, but each lifetime in a batch can have its own table), and the combinators `Sequence` (processes one after another) and `Sum` (a base process plus offsets). |
| **Oracle** | `pp.Oracle(env, expose=[...])` deliberately shows chosen privileged fields to the agent in `info["oracle"]`, and records the choice in `env.metadata["oracle_exposure"]` so results say what the agent knew. |
| **Runner** | Runs many ticks as one compiled program. `pp.runtime.Runner(env, policy, ticks=256)` advances a lifetime by a **chunk** of 256 ticks per call; call it again to continue. |

## The six terms in code

```python
import jax
import perpetua as pp

# Goal alternates between two corners every 500 ticks: two phases, a switch every 500.
env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], 500))
runner = pp.runtime.Runner(env, pp.policies.random, ticks=1000)  # a 1,000-tick chunk
carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=100))
carry = runner(carry)  # continue the same lifetime; never initialise again
print("tick", carry.state.t.to_int(), "mean reward", float(carry.metrics.mean[0]))
assert "boundary" not in carry.info  # privileged: the agent was never told

seen = pp.Oracle(env, expose=["boundary"])
_, _, info = seen.init(jax.random.key(0))
print(sorted(info["oracle"]), seen.metadata["oracle_exposure"])
```

The [quickstart](../tutorials/quickstart.md) runs a full experiment (drift,
learning, recovery and forgetting) in a few seconds. The rest of this page is
reference: come back to it when another page uses a term.

## What the agent sees

| Public: `obs` and `info` | Privileged: shown to the agent only through an Oracle |
|---|---|
| observation, reward | drift parameters, phase |
| `action_mask`, `active_mask` | whether a switch just happened (`boundary`) |
| `respawn`, `space_change` | task id |
| the lifetime clock `t` | world-side diagnostics (for example ForagerGrid's mass) |

The switch flag has one name, `boundary`, for drift and for task sequences
alike. `drift_changed` is a deprecated alias with the same value.

## Measuring continual learning

| Term | Meaning |
|---|---|
| **Non-stationarity** | The world's rules change during the lifetime (here: drift and task switches), so yesterday's best policy may be wrong today. |
| **Recovery** | How many ticks after a switch the agent needs to get its reward back to the level it had before the switch. |
| **Forgetting** | How much worse the agent became at an earlier task or goal than it once was. 0 means nothing was forgotten; negative means the final policy improved on an earlier goal. |
| **Forward transfer** | Whether experience with earlier tasks helps learn a new one faster than learning it from scratch. Positive is good; negative means the past got in the way. |
| **Loss of plasticity** | A network that has trained for a long time gradually loses the ability to learn new things, even though nothing is forgotten on purpose. Health probes such as dormant units track it. |
| **IQM** | Interquartile mean: the mean of the middle 50% of runs, a robust summary across seeds. |

[Evaluation](../evaluation.md) shows how recovery, forgetting, the health
probes and IQM are computed; forward transfer needs single-task reference
runs, as in the [Continual World preset](../presets/continual-world.md).

## Reference vocabulary

The remaining terms, for looking up when a page uses them.

| Term | Meaning |
|---|---|
| **Parameter** | A quantity that may change over the lifetime, such as a goal position or gravity. Each environment page lists its *driftable parameters*. |
| **Period** | For `Cyclic`, the number of ticks each value is held. `Cyclic([a, b], 1000)` uses `a` for ticks 0–999, `b` for 1000–1999, then `a` again. |
| **Phase / switch** | Each `Cyclic` value is a phase. Moving to the next phase is a *declared switch*; recovery is measured from switches. Gradual drift (`RandomWalk`) never switches. The Oracle field for it is `boundary`; the metrics count ticks since it in `metrics.since_switch`. |
| **Privileged data** | What the world knows but the agent should not: the current parameters, the phase, whether a switch just happened, a task id. It never appears in `obs` or `info`. Evaluation code (never the policy) can read it with `env.privileged(state)`. |
| **Slot** | One agent position in a world. Arrays have a leading slot axis: observations `(agents, ...)`, actions `(agents,)`, rewards `(agents,)`. Single-agent worlds have one slot. |
| **Action mask** | `info["action_mask"]`: which actions are valid now. It may change during a lifetime (a door closes) while the action space keeps its shape; `info["space_change"]` flags the ticks where it does. |
| **Active mask** | `info["active_mask"]`: which slots learners control. Scripted partners and opponents are world participants with `active_mask=False`, so learners never train on them. |
| **Carry** | Everything a runner passes from one chunk to the next, as one pytree: world state, latest observation and `info`, random keys and metrics. `pp.runtime.init_lifetime(env, seed=0)` builds the first one, and the lifetime clock is `carry.state.t.to_int()`. Learner runners return a training state whose `.lifetime` is this carry and whose `.agent` is the learner. |
| **Metrics** | Computed inside the lifetime on the same clock as the world: lifetime mean reward, a sliding window (`window_size` ticks), an exponential average, variance, regret against a reference policy, and recovery after the latest switch. |
| **Learner** | `PPO`/`IPPO`, `DQN`/`IQL` and `SAC` in `perpetua.agents`. `act` picks actions every tick; `update` learns from a rollout or replay batch between chunks. A *rollout* is the batch of consecutive ticks (`rollout_steps`) a learner collects before one update. Learner runners such as `PPORunner` do both: one call is one rollout and one update. |
| **Execution** | `jax` worlds run on the accelerator inside the compiled runner. `host` worlds (Meta-World, COOM, AgarCL, EnvPool, MORPHEUS) step a CPU simulator; use `HostRunner`, or `compiled=True` where available. |
| **Checkpoint** | `pp.checkpoint.save` stores a whole carry (world, drift, keys, metrics, learner) so a lifetime resumes exactly; `replay` re-runs a segment. |
| **Preset** | A ready-made benchmark sequence, such as `pp.presets.cw10()` for Continual World. It is a composition of an environment, a task switch and drift, not a new environment. |
| **Tuning protocol** | How much of a lifetime may be spent choosing hyperparameters. Each preset declares one (`pp.tuning.declared`). |

## JAX in five ideas

Perpetua is written in [JAX](https://docs.jax.dev/en/latest/). You need five
ideas to read its code; JAX's [tutorials](https://docs.jax.dev/en/latest/jax-101.html)
explain each in depth.

- **Arrays like NumPy:** `jax.numpy` (usually `jnp`) mirrors NumPy, and arrays
  are immutable.
- **Pure functions:** functions return new state instead of changing it. That
  is why `env.step(state, action, key)` takes and returns `state`.
- **Explicit random keys:** there is no global random seed. A key goes into
  every random call, and `jax.random.split(key)` makes new keys from it.
- **Pytrees:** any nesting of tuples, lists, dicts and named tuples whose leaves
  are arrays. A carry, a learner's parameters and a checkpoint are pytrees, so
  JAX functions apply to them as a whole.
- **Transformations:** `jax.jit` compiles a function the first time it runs (so
  the first call is slower), and `jax.vmap` runs it on a whole batch at once,
  for example many seeds or lifetimes.
