# Compute-aware time

```{admonition} Experimental
:class: warning

This feature is kept working and tested, but it is frozen: no new work is
planned on it, and its interface may change. Build on it with that in mind.
```

Normally the world waits while the agent thinks. In compute-aware time it does
not: a decision that takes `c_ns` nanoseconds of compute costs `k` world ticks,
where one tick stands for `quantum_ns` nanoseconds (the *quantum*), and the
previous action is held meanwhile. A *decision* is one joint action for all
agent slots.

This mode is **opt-in**: `Runner`, `HostRunner`, `PPORunner` and
`HostPPORunner` are synchronous.

It needs a **discrete action space** (`MultiDiscrete`), because the held command
is an int32 vector: `ComputeRunner` and `ComputePPORunner` raise `ValueError`
when they are built on a continuous-action environment (gymnax `Pendulum`, MJX,
JaxMARL continuous tasks). Use the synchronous runners for those.

## Timing contract

For one joint agent decision with synchronized compute cost `c_ns` and world
quantum `quantum_ns`, charge:

`k = max(1, ceil(c_ns / quantum_ns))`

The integer nanosecond calculation avoids float boundary rounding. The previous
command is held for ticks `1..k-1`; the new command arrives on tick `k`. No action
is retroactively applied to time before it was ready. Each held/arriving command
is repaired against the current public mask: the first valid entry, which is
action 0 whenever that is valid. The requested command remains the controller's
held command, even if its current physical execution was repaired.

The first decision has no previous command, so the initial held command is
**action 0 for every slot** (`init_timing`), and a first interval of `k > 1`
ticks runs `k - 1` world ticks under it before the policy's own action arrives.
Action 0 is only "stay" where the environment makes it so: in the native grids
it is stay, in `CartPole` it pushes left.

Every tick advances physics, environment/drift RNG and streaming metrics. Agent
sampling RNG advances only at decisions. Drift is not revealed by the timing
wrapper. Time belongs to the shared world; a batched multi-agent controller's
joint compute cost is charged once, not separately as asynchronous per-agent time.

This is a **discrete compute-latency model**, not hard OS real time. Catch-up
simulation runs after the completed work unit; it reconstructs the ticks that
would have elapsed while the agent was busy. Dynamics, transfers and logging
outside agent work are not added to `c_ns`, and this mode does not promise a
concurrent simulator, wall pacing or hard deadlines.

## Recorded schedules and measured execution

```python
import perpetua as pp

env = pp.make("ContinualGrid-v0", size=3)
runner = pp.compute.ComputeRunner(env, pp.policies.random, decisions=3, quantum_ns=1_000_000)
state = runner.init(seed=0)
trace = pp.compute.TimingTrace.from_ticks([1, 3, 2], quantum_ns=1_000_000)
state = runner(state, trace)
state, measured_trace = runner.measure(state)
```

- A recorded trace must have been built with the runner's `quantum_ns`; a trace
  whose tick counts disagree with it is refused.
- Recorded native replay is a fixed-decision scan with dynamic tick loops, compiled
  once and compatible with `vmap`. Host simulators step explicitly on the host.
- Measured execution returns `MeasuredTimingTrace`, with the same numeric fields
  but a distinct checkpoint schema. Its replay preserves separately compiled
  policy/work and world-step kernels instead of fusing a new whole-chunk program.
  Measured replay is host-scheduled, not an outer-jitted/vmapped chunk.
- `measure()` uses `perf_counter_ns` and waits for JAX output pytrees to finish.
  Asynchronous GPU dispatch is therefore not mistaken for completed computation.
- Measured kernels use per-function `xla_gpu_exclude_nondeterministic_ops`
  and `xla_gpu_autotune_level=0`, and so do the compiled chunks of the learner
  runners and `ComputePPORunner`'s native recorded replay (see
  [training](training.md#provenance-and-limits)). These recorded settings prevent
  fresh compilation from selecting numerically different GPU algorithms. They may
  cost performance; no process-global JAX/XLA flags or drivers are changed.
  `ComputeRunner`'s recorded replay compiles normally, so its reproducibility
  depends on the declared hardware/software/compiler setup.
- Initial policy/learner priming is excluded and uses throwaway keys; it does not
  advance native state or consume lifetime RNG. Shape-changing/recompiling custom
  work is not silently exempted from subsequent measured calls.
- Measured wall costs are nondeterministic external inputs. **Save timing traces**
  to reproduce a measured segment; seed/checkpoint alone cannot reproduce future
  scheduler load. Resume still restores the complete world/controller/model state.

Trace fields are uint32 `(decisions,)` tick counts plus two uint32 `(decisions,)`
limbs holding exact uint64 compute nanoseconds. Controllers retain int32
`held_action (agents,)`, scalar exact decision/ns clocks, last uint32 interval ticks
and a sticky boolean health flag. World clocks/metrics remain authoritative for
physical tick counts. No global JAX 64-bit mode is enabled.

Host schedule construction refuses zero/negative/noninteger/oversized intervals.
Malformed zero intervals in compiled input mark the controller unhealthy without
advancing the world or sampling keys. A single interval must fit uint32; total
world/decision/compute counters retain uint64 limbs. Choose a sensible quantum:
huge accepted intervals genuinely require many simulator ticks; they are not
silently clipped or treated as throughput gains.

## Timed PPO and IPPO

```python
from perpetua.agents import IPPO, ComputePPORunner, PPOConfig

env = pp.make("CooperativeGrid-v0", size=3)
agent = IPPO(action_size=5, config=PPOConfig(rollout_steps=8))
runner = ComputePPORunner(env, agent, decisions=4, quantum_ns=1_000_000)
state = runner.init(seed=0)
state, trace = runner.measure(state)
```

The fixed on-policy buffer is part of `TimedTrainingState`, including observations,
actions, masks, activity, sampled values/log probabilities, discounted interval
rewards and uint32 elapsed ticks. An int32 `filled` cursor permits partial-buffer
checkpoints. A full buffer is optimized at the **next** decision, then that updated
learner selects an action from the current observation. Synchronizing/timing this
whole work unit charges both optimization and inference to the new interval.
This avoids inventing an unrecorded training-time gap or circularly re-running
optimization to learn its own unknown duration.

For an interval of `k` world ticks, learner reward is `sum(gamma**i * reward_i)`;
bootstrap discount is `gamma**k` and continuing GAE trace discount is
`(gamma * gae_lambda)**k`. Raw lifetime metrics still count **every undiscounted
world reward**, including waiting. Requested action/log probability belongs to
the decision-start observation; later mask repair is part of environment dynamics.
Respawns, drift and compute/rollout boundaries never artificially zero discounts.

PPO remains shared-policy; IPPO remains independent per slot. Learner work sees
public observations/transitions, not privileged environment state. This separation
also lets mutable host worlds keep opaque ownership outside the compiled learner.
The held command is controller state, not a new observation feature; memoryless
baselines are not promised fully Markov observations in delayed-control worlds.

## Run, resume and replay

```bash
uv run --extra training python examples/compute_ppo.py --chunks 2 \
  --checkpoint checkpoints/compute-001
uv run --extra training python examples/compute_ppo.py --chunks 1 \
  --resume checkpoints/compute-001
uv run --extra training python examples/compute_ppo.py --replay checkpoints/compute-001

uv run --extra training python examples/compute_ppo.py --mode measured \
  --quantum-ns 10000000 --chunks 3 --checkpoint checkpoints/measured-001
uv run --extra training python examples/compute_ppo.py --mode measured \
  --quantum-ns 10000000 --replay checkpoints/measured-001
```

The CLI stores current state, the last chunk's starting state and its bounded
timing trace, enabling exact full-leaf replay—not only reward-summary equality.
For long runs, persist any earlier traces you need outside compiled loops; do not
retain the entire timing history in the device carry.

Native and supported Gymnasium/EnvPool snapshots can resume their full timing and
training state. The [MPE2 PettingZoo codec](pettingzoo-snapshots.md) also checkpoints
full `simple_spread` state, including timed pending buffers. Opaque/unsupported
PettingZoo backends still refuse checkpoints rather than claiming incomplete resume.

No throughput, guaranteed wall-clock realism, cross-hardware bitwise continuation
or long-duration reliability claim follows from this mode's smoke/conformance tests.
