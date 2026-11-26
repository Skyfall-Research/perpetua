# Evaluating continual learners

How do you tell whether a learner keeps learning in a changing world? This
page covers the measures Perpetua provides, how each is computed, and how to
report them. The [comparing runs](tutorials/evaluation.md) tutorial walks
through a complete comparison.

| Question | Measure | Where |
|---|---|---|
| How well is it doing now? | windowed and EMA reward, lifetime mean | the metrics carry (`carry.metrics`) |
| How long does it take to adapt after a switch? | recovery | `recovery_ticks`, or live `recovered_at` |
| Did it lose what it learned earlier? | forgetting | `revisit` scores, then `forgetting` |
| How far is it from a reference policy, state by state? | same-state regret | `reference=` on any runner |
| Is the network losing the ability to learn? | dormant units, effective rank | `agent.probe`, `ProbedRunner` |
| Is one setting better than another across seeds? | IQM with confidence intervals | `pp.reporting.report` |

The code blocks on this page run in order as one program. They share this
setup, a small PPO learner on a grid whose goal moves every 128 ticks:

```python
import jax
import jax.numpy as jnp
import numpy as np
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

env = pp.make("ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], 128))
agent = PPO(action_size=5, config=PPOConfig(rollout_steps=32))
trainer = PPORunner(env, agent)
state = trainer.init(seed=0, window_size=32)
```

## Recovery and forgetting

Two pure functions turn recorded lifetimes into continual-RL metrics:

- `recovery_ticks(rewards, switches, *, window=32, fraction=0.9)` — given a
  per-tick reward trace and the trace indices of the switches (see
  [which index names a switch](#which-index-names-a-switch)), returns, for each switch, the number
  of ticks after it until the first tick whose following `window` mean is at
  least 90% of the pre-switch `window` mean (`fraction=0.9`; the tolerance is `1 - fraction`). The
  tolerance works for negative rewards too, and values are capped at the
  trace end. For a switch fewer than `window` ticks into the trace the
  reference window starts at tick 0 and includes post-switch ticks.
- `forgetting(scores)` — given a phase-by-phase score matrix `scores[k, j]`
  (the policy saved after phase `k` evaluated under phase-`j` conditions,
  typically from `revisit` over phase-conditioned lifetimes, at least two
  phases), returns how much the **final policy** lost on each earlier phase:
  for every phase `j` before the last, the best score any policy saved from
  `j` up to the second-to-last phase achieved on `j`, minus the final
  policy's score on `j`, averaged over those phases (Chaudhry et al., 2018).
  Scores from before phase `j` was learned are ignored; negative values mean
  backward transfer. The tests include a discriminating pair — a final policy
  that retains every phase scores 0, one that forgets them all scores 1.

Both are pure functions with hand-computed tests. They define the measures;
how a benchmark uses them (which phases, how many seeds) is up to its
protocol.

`recent_rewards(metrics_state, *, steps)` reconstructs the most recent
`steps` per-tick rewards in chronological order from the bounded metrics
window ring, so a live training carry can feed `recovery_ticks` without
keeping a separate trace:

```python
trace, saved, observations = [], [], []
for _ in range(8):
    state = trainer(state)
    trace.append(np.asarray(pp.evaluation.recent_rewards(state.lifetime.metrics, steps=32)))
    saved.append(state.agent)  # the policy after each update, scored further down
    observations.append(state.lifetime.obs)  # used by the health probes at the end
rewards = np.concatenate(trace)[:, 0]  # (256,): the first agent's reward at every tick
```

`steps` must not exceed the populated window entries; a populated check is
raised outside traced code.

### In-carry recovery tracking

Recovery can also be measured on the same clock, inside the compiled lifetime,
with no separate trace. `metrics.update` takes an optional `switch` flag, and `runtime.tick`, the
learner runners and `compute` feed it automatically through
`runtime.drift_switch`:

```python
metrics = pp.metrics.init(num_agents=1, window_size=8)
reward, switch = jnp.asarray([1.0]), jnp.asarray(False)  # switch: bool scalar
metrics = pp.metrics.update(metrics, reward, switch=switch)
metrics.recovered_at  # ticks after the last switch when the window recovered, else -1
metrics.since_switch  # ticks since the last switch
metrics.switch_baseline  # pre-switch window mean used as the reference
```

The reference is the window mean over entries populated *before* the switch
tick, and recovery is only judged once the trailing window holds no pre-switch
rewards — the carry never claims a recovery from stale data, and a window as
long as the drift period reports `-1` rather than guessing. For switches at
least one window into a trace, when `recovery_ticks(...)` is at least 1, the
reported value equals `recovery_ticks(...) + window_size - 1` when
`recovery_ticks` is given the trace index of the step that raised the switch
flag (the post-hoc measure takes the first forward window, the tracker the
first trailing one); the test suite asserts that identity on a hand-computed
trace. The tracker uses the default tolerance `fraction=0.9` of
`metrics.update`; the runners do not change it. The tracker's
window never contains the switch-tick reward, so its earliest window starts
one tick after the switch: when `recovery_ticks` is 0 it reports the first
recovering window starting at least one tick after the switch, plus
`window_size - 1`, and its smallest value is `window_size`. Tracking stays inert when a backend
has no flag (`has_switch` false) and every switch resets it, so only the most
recent switch is reported. The four fields live in the metrics carry and are
therefore checkpointed with it.

Two edges. A switch reported on the very first tick, before the window holds
any reward, is not tracked: there is nothing to take a baseline from, so
`has_switch` stays false and `recovered_at` stays `-1` until the next switch.
And `since_switch` is an int32 that stops at `2**31 - 1` instead of wrapping,
so more than about 2.1 billion ticks after a switch it reads "at least that
many", and a recovery that late reports that value.

**The tracker pools every agent slot.** The baseline and the recovery test use
the mean over all slots, scripted and opponent slots included, because the
carry holds one scalar for each. Where those slots earn rewards that cancel or
that do not belong to the learner, the pooled mean says nothing about the
learner: on `OpponentGrid-v0` the rewards sum to zero across slots, so the
baseline is 0 and `recovered_at` is `window_size` on the first judged tick
even when the learners' own reward fell. Partly scripted worlds
(`PartnerGrid-v0`, `JaxMARLMPE` with scripted slots) mix the scripted slots'
rewards into the baseline. For those, record the per-tick reward of the
learner slots (for example `reward[:num_learners].mean()`) and use
`recovery_ticks` on that trace with the switch ticks from the `boundary`
Oracle field. The tracker is meaningful for a single slot and for symmetric
teams of learners.

**What counts as a switch.** `boundary` is a *declared switch*: the drift
process's `phase` changed on this tick (`pp.drift.switched`). `Cyclic`
and `Procedural` switch on their period; a `Cyclic` switch counts even when
two phases happen to share a value. A task switch also declares a switch on
every tick its active task changes, whatever process drives it. Otherwise,
gradual processes such as `RandomWalk`
never switch, because time-to-recover is only defined relative to a discrete
switch: under gradual drift `has_switch` stays false, which means **not
applicable**, not "never recovered". Track gradual drift with the windowed
and EMA reward or with same-state regret instead; raw parameter movement stays
available to evaluators through the Oracle's `drift_parameters`.

Cost: four scalar fields and one extra comparison per tick.

### Which index names a switch

The step that raises `boundary` returns the last reward earned under the old
parameters: for `Cyclic(values, period)` that is trace index `period - 1`, with
`info['t']` equal to `period`. The first reward earned under the new parameters
is index `period`. Both are valid arguments to `recovery_ticks`, and they give
different results. The index of the flagged step is the one the in-carry tracker
sees, and the one for which the identity above holds. A multiple of `period`
measures from the first reward under the new parameters, and its reference
window also covers the flagged step's reward, so the two results can differ by a
tick or more with no fixed relation. Measured on a hand-built trace (16 rewards
of 1, six of 0, then 1; `window=4`, flag on index 15): `recovery_ticks` gave 7
for index 15 and 6 for index 16, and the carry reported `recovered_at` 10, which
is 7 + 4 - 1. Pick one convention per report and state it.

### Practical notes

- **Finding switch ticks without the Oracle.** The experimenter designed the
  schedule, so the switches are known: `Cyclic(values, period)` switches at
  every multiple of `period`. Pass those ticks to `recovery_ticks`, as the
  [quickstart](tutorials/quickstart.md) does. The Oracle's `boundary` field
  gives the switches too, for evaluation code that should not repeat the
  schedule, but it is raised one step earlier (index `period - 1`); see
  [which index names a switch](#which-index-names-a-switch).
- **`recovered_at` covers only the latest switch.** For one value per switch,
  record rewards with `recent_rewards` and call `recovery_ticks` afterwards.
- **Recovery needs a reward that can drop.** If a random policy already earns
  nearly the maximum (CartPole scores +1 on every tick except a failure),
  every switch looks recovered at once. Use a stricter `fraction`, a task
  whose reward depends on what changed, or the windowed reward.
- **Two forgetting functions.** `pp.evaluation.forgetting(scores)` takes a
  phase-by-phase score matrix from re-evaluating saved policies (the
  quickstart builds one). `pp.protocol.within_trace_forgetting` is the MORPHEUS
  suite's "forgetting": the first encounter of a configuration minus the
  second, within one training trace. The names differ so the two are not
  mixed up in reports.
- **Phase-conditioned lifetimes** hold one phase fixed: build the same
  environment with that phase's value as a one-phase drift, for example
  `pp.drift.Cyclic([goal], 1)`.

## Scoring saved policies: revisit

`act` and `stacked_states` are built as in step 4 of the
[quickstart](tutorials/quickstart.md), here from the policies saved above:

```python
from perpetua.evaluation import revisit

stacked_states = jax.tree.map(lambda *leaves: jnp.stack(leaves), *saved)


def act(agent_state, obs, info, key):
    return agent.act(agent_state, obs, info, key).action


means = revisit(env, act, stacked_states, seeds=jnp.asarray([0, 1, 2]), ticks=256)
# means has shape (policies, seeds); every policy sees the same seeded lifetimes
```

`revisit` scores K stacked past policies (for example PPO states saved at
phase boundaries) against S seeded lifetimes of the same environment
configuration in **one vmapped pass**, instead of a sequential loop over
policies whose cost grows with the number of saved policies. Semantics:

- `states` is a policy stack with leading axis K; `act(state, obs, info, key)`
  applies one unstacked policy and returns the action array (for PPO and SAC
  that is `agent.act(...).action`; `DQN.act` already returns the array).
- Each (policy, seed) pair starts a fresh seeded lifetime whose environment
  and action keys all derive from the seed, so every policy faces identical
  lifetimes for a given seed (common random numbers). `revisit` takes no key;
  the `key` that `act` receives is derived from the seed, so results depend
  only on `seeds` (vary them to draw different lifetimes). Results
  are exact mean lifetime rewards of shape (K, S) and are bitwise
  reproducible for the same inputs.
- The score is the mean reward per tick, averaged over the **learner slots**:
  the slots whose `active_mask` was true at the start or after any tick of the
  evaluated lifetime. Scripted partner and opponent slots report `active_mask`
  false throughout and are left out. On the zero-sum `OpponentGrid-v0` the
  opponents receive the negated learner reward, so an average over every slot
  is zero for any policy; the score is the learners' reward. A learner slot's
  inactive ticks still count as zero-reward ticks. When every slot acted (any
  single-agent or all-active world), or none did, the score is the mean over
  all slots.
- `revisit` needs a JAX environment: it scans the environment under `vmap`, so
  a host environment such as `GymnasiumLifetime` fails inside the adapter.
- Evaluation consumes no training RNG and never mutates the learners. The
  compiled pass is the same `lax.scan` runner used by training.
- Forgetting and forward transfer are built from it: re-evaluate each saved
  policy under each phase's conditions and compare the scores, as the
  [quickstart](tutorials/quickstart.md) does for forgetting. `forgetting` on a
  `revisit` score matrix is therefore the learner slots' forgetting.

## Same-state regret against a reference policy

```python
runner = pp.runtime.Runner(env, pp.policies.random, ticks=256, reference=pp.policies.greedy)
regret_trainer = PPORunner(env, agent, reference=pp.policies.greedy)  # learners too
carry = pp.runtime.init_lifetime(env, seed=0, window_size=128)
carry = runner(carry)  # each tick: one extra environment step for the reference reward
carry.metrics.regret_total  # cumulative reference - reward, per agent
carry.metrics.regret_mean  # cumulative regret per tick
carry.metrics.regret_window_mean  # mean over the bounded window ring
```

`runtime.shadow_reward(env, state, obs, info, reference, key)` computes the
reward the reference policy would have earned from the learner's *pre-action*
state: one extra environment step, reusing the learner's transition key
(common random numbers) with the reference's action draw on an independently
folded key. The shadow state is discarded and the reference reward never enters
the public `info`. The accumulator is the per-tick regret
`reference_reward - reward`, the usual sign: positive when the learner earns
less than the reference, negative when it does better. It is kept as
compensated-float32 `regret_total` plus a
bounded `regret_window` ring (reusing the metrics cursor), so the carry stays
`O(window_size)` and checkpoints with the rest of the metrics.

This is a **same-state regret**, not the literature's dynamic regret against
a best-in-hindsight policy: there is no future knowledge and no re-planning,
and the reference is whatever policy the caller configures. It compares the
reward of one tick only, so it is exactly zero wherever that reward does not
depend on the tick's action: see
[where it reads zero](#where-same-state-regret-reads-zero). Cost: one extra
environment step per tick when a reference is set — measured on a 5×5 grid over
40 × 256-tick chunks, 628k → 397k ticks/s on macOS CPU (1.58× wall time) and
34.1k → 29.5k ticks/s on H200 (1.16×; the tiny grid's step is dispatch-bound
there, a compute-bound environment would approach 2×). Without a reference the
carry, RNG streams and cost are unchanged, and so is the learner's
trajectory (tested).

Every runner takes the same `reference=` argument because they all share one
per-tick body, `runtime.interact` (act) over `runtime.transition` (step,
metrics, declared-switch recovery, regret, clock): `runtime.Runner`,
`HostRunner`, `PPORunner`/`HostPPORunner`, `DQNRunner`/`HostDQNRunner` (DQN
and IQL) and `SACRunner`/`HostSACRunner`. Tests check that a reference
leaves the learner's parameters, optimizer state, replay and world state
exactly as in a run without one, and that against a reference earning
exactly zero, regret is minus the learner's own total. The reference must be a stateless
`(obs, info, key) -> action` policy. The learner runners (`PPORunner`,
`DQNRunner`, `SACRunner` and their host variants) record its qualified name as
`regret_reference` in runner metadata, so a strict checkpoint restore refuses to
continue a regret accumulator against a reference with a different name.
`runtime.Runner` and `HostRunner` copy the environment's metadata and record no
reference, and the check compares names only: two lambdas, or two
`functools.partial` objects, carry the same name (`<lambda>`,
`functools.partial`) and are not told apart. Use named module-level functions
as references, and report which one you used.

Host backends: a shadow step re-steps the pre-action state. Snapshot-capable
host backends (for example Gymnasium CartPole) restore their snapshot on every
step, so the shadow step is exact; opaque simulators cannot fork a state and a
reference is refused when the runner is constructed, before any simulator step.

### Where same-state regret reads zero

Both rewards of a tick come from the learner's state, and the reference's
next state is thrown away. An action whose cost arrives on a later tick
therefore never shows: on CartPole the pole's position after a step is
computed from the velocities before it, so no action changes the reward of
its own tick, and the regret of a policy that drops the pole against one that
balances it is exactly `0.0`. A regret of zero means "the learner matches the
reference" only where the reward of a tick can depend on that tick's action.
Elsewhere use windowed reward and [recovery](#recovery-and-forgetting).

`pp.testing.reward_action_sensitivity` measures this for the environment in
front of you. It follows a uniformly random lifetime, steps several valid
actions from the same state at every tick, and returns the fraction of ticks
on which the reward differed:

```python
pp.testing.reward_action_sensitivity(pp.make("gymnax/CartPole-v1"), ticks=200)  # 0.0
pp.testing.reward_action_sensitivity(env, ticks=200)  # above zero on the grid
```

`0.0` means regret carries no information on the states a random policy
visits. A small value means it is informative on that share of ticks only,
and it says nothing about states random play does not reach.

`scripts/survey_regret.py` runs the diagnostic over every registered
environment that runs as a JAX world. Measured on 2026-11-25 with the default
arguments of each environment: seeds 0, 1 and 2, 1,000 ticks each, 24 actions
compared per tick, on macOS CPU (Apple Silicon, JAX 0.11.2, gymnax 0.0.9,
JaxMARL 0.2.0, MEAL 0.3.1). Of the 45 environments, 20 returned exactly zero
on all 3,000 ticks.

**The reward cannot depend on the tick's action** (16, each confirmed by
reading the reward code). Same-state regret is zero for every pair of
policies:

- `gymnax/CartPole-v1`: the reward is 1 unless the pole or cart is out of
  bounds after the step, and positions advance with the previous velocities.
- `gymnax/MountainCar-v0`: the reward is the constant -1.
- `gymnax/Breakout-MinAtar`: the reward is paid when the ball reaches a brick,
  which the paddle's move on that tick does not change.
- `gymnax/UmbrellaChain-bsuite`: the choice is made on the first step of a
  chain and paid on its last (the default chain has 10 steps); the rewards in
  between are random.
- The JaxMARL MPE tasks whose reward reads positions only, which advance with
  the previous velocities: `MPE_simple_v3`, `MPE_simple_spread-v3`,
  `MPE_simple_tag_v3`, `MPE_simple_world_comm_v3`,
  `MPE_simple_speaker_listener_v4`, `MPE_simple_push_v3`,
  `MPE_simple_adversary_v3`, `MPE_simple_reference_v3` and the four
  `MPE_simple_facmac*_v1` tasks. (`MPE_simple_crypto_v3` pays for the message
  sent on the tick and read 1.0.)

**Zero under random play only** (4). The action can decide the reward of its
own tick, but in states a random policy rarely or never reached, so regret
against a competent learner may still be informative:

- `gymnax/Acrobot-v1` (at the goal height) and `gymnax/Pong-misc` (when the
  paddle's move decides whether the ball is returned): 0.0001 and 0.0008 of
  ticks in a longer run of 20,000 ticks with seed 0.
- `gymnax/SpaceInvaders-MinAtar` (firing with an alien in the row above the
  cannon) and `meal/overcooked-v0` (delivering a soup): still zero in that
  20,000-tick run.

**Above zero** (25), as the fraction of the 3,000 ticks:

| Fraction | Environments |
|---|---|
| 1.0 | `gymnax/Pendulum-v1`, `gymnax/MountainCarContinuous-v0`, `gymnax/DeepSea-bsuite`, `gymnax/GaussianBandit-misc`, `gymnax/Reacher-misc`, `mjx/Pendulum-v0`, `mjx/CartPole-v0`, `jaxmarl/MPE_simple_crypto_v3` |
| 0.5 to 1 | `gymnax/MNISTBandit-bsuite` (0.926), `OpponentGrid-v0` (0.885), `gymnax/BernoulliBandit-misc` (0.805), `gymnax/Swimmer-misc` (0.639) |
| 0.01 to 0.2 | `BeanWorld-v0` (0.180), `gymnax/MemoryChain-bsuite` (0.166), `ForagerGrid-v0` (0.081), `gymnax/Catch-bsuite` (0.055), `PartnerGrid-v0` (0.041), `ContinualGrid-v0` (0.038), `DoorGrid-v0` (0.038), `gymnax/Asterix-MinAtar` (0.026), `gymnax/MetaMaze-misc` (0.014), `gymnax/DiscountingChain-bsuite` (0.010) |
| under 0.01 | `CooperativeGrid-v0` (0.0093), `gymnax/FourRooms-misc` (0.0020), `gymnax/Freeway-MinAtar` (0.0007) |

Host simulators and environments whose runtime was not installed (EnvPool,
COOM, AgarCL, Meta-World, MORPHEUS) were not surveyed. The diagnostic itself
also runs on host backends with snapshot support.

## Reporting with IQM and confidence intervals

```python
scores = means  # (policies, seeds), from revisit above
intervals = pp.reporting.report(scores.T, key=jax.random.key(5))  # (runs, tasks)
intervals["iqm"]  # BootstrapInterval(point, low, high)
intervals["mean"]  # plain-mean interval
intervals["median"]  # median interval
```

`perpetua.reporting` follows the reporting recommendations of Agarwal et al.
(2021): robust aggregates with bootstrap confidence intervals over runs.

- `interquartile_mean` — mean of the middle 50% after sorting, discarding
  `n // 4` values from each end, matching Agarwal et al.'s aggregate-metrics
  definition (rliable/`scipy.stats.trim_mean(..., 0.25)`). Any non-finite
  score makes it NaN, so a diverged run cannot be trimmed away (this differs
  from SciPy for infinities).
- `stratified_bootstrap_ci(scores, statistic, key=...)` — percentile
  confidence bounds from `num_bootstrap` resamples that draw run indices with
  replacement independently per task (scores shape `(runs, tasks)`), with a
  caller-owned explicit RNG key: identical inputs and keys give identical
  intervals, and the whole report compiles under `jit`. The bounds always
  include the point (`low <= point <= high`): where the percentile interval
  would leave it outside, by a float32 rounding step for constant scores or
  further for a skewed statistic on few runs, the bound is moved to the point.
- `report` — IQM, mean and median intervals in one call. All three pool every
  (run, task) entry, so the `"median"` is the median of all entries, not
  rliable's aggregate median (the median over tasks of each task's mean over
  runs); the two differ most when success is bimodal across seeds. Compare a
  Perpetua median with a published rliable median only after recomputing one
  of them.

Non-finite entries propagate as NaN statistics instead of raising, because
value-dependent checks cannot run inside the traced bootstrap; callers
validate their own score matrices. These are reporting utilities for final
analysis, not per-tick telemetry: streaming per-tick metrics remain in
`perpetua.metrics`, and no significance or protocol claim is made by the
module itself.

## Learner health probes (loss of plasticity)

```python
batch_obs = jnp.concatenate(observations)  # (batch, features)
# PPO, DQN, IQL, SAC; on-demand, consumes no RNG
report = agent.probe(state.agent, batch_obs)
report["dormant_actor"]  # fraction of actor hidden units below the variance threshold
report["rank_actor"]  # entropy-based effective rank of the actor activations
```

`perpetua.probes` provides two pure functions over `(batch, units)` activations:

- `dormant_unit_ratio(activations, threshold=1e-3)` — fraction of units whose
  activation variance across the batch is below the absolute threshold. It
  is a variance test, not the normalised mean absolute activation score of
  Sokar et al. (2023).
- `effective_rank(activations)` — entropy-based effective rank (Roy &
  Vetterli, 2007) of the centered activations: ~1 for rank-one representations and approaching
  `min(batch, units)` for isotropic ones.

Both take optional `weights=`, one nonnegative weight per batch row. A row of
weight 0 is left out of the statistic while the batch keeps its shape, which
is how a compiled caller drops rows chosen by data.

`PPO.probe` reports both for the final actor and critic hidden layers of one
unstacked learner state. `DQN.probe` reports the shared Q-network's final
hidden layer (`dormant_q`, `rank_q`); `IQL.probe` probes every independent slot
on the same batch, so its report carries a leading agent axis; `SAC.probe`
reports the actor and the first of the twin critics, probed on the policy's
mean action so no action input is needed. IPPO's stacked per-agent parameters
are not probed by this helper. Probes consume no RNG and never mutate the
learner. Cost: one network forward pass plus the two statistics per call,
measured at 1.4–3.5 ms per call (macOS CPU, eager, batch 32, hidden 16) across
the four learners.

### Probes on the lifetime clock

`pp.probes.ProbedRunner` wraps a PPO, DQN or SAC runner and probes it every
`every` updates, on the agent-slot observations from the last `memory` update
boundaries. It keeps the latest values and a ring of the last `window`
evaluations (probe evaluations, not ticks) next to the training state:

```python
tracked = pp.probes.ProbedRunner(trainer, every=2, window=16, memory=4)
carry = tracked.init(seed=0)
for _ in range(8):
    carry = tracked(carry)  # carry.state is the learner's TrainingState
carry.trace.latest["rank_actor"], carry.trace.history["dormant_actor"]
```

Until `memory` updates have passed, the probe batch is padded with copies of
the observation at tracking start, so the first readings overstate dormant
units and understate rank: discard evaluations before then. `latest` holds
zeros until the first evaluation. A probe batch needs at least two rows (a
single row has zero variance, so every unit would read as dormant). The agent's `probe` must accept the dormant-unit `threshold=` by
keyword.

Only rows whose slot was active count. An inactive slot's observation is a
placeholder (zeros for the scripted slots of `PartnerGrid-v0` and
`OpponentGrid-v0`), and such rows would otherwise dilute both readings: with
one learner and three partners, 24 of 32 rows are zeros. `carry.trace.active`
holds the `active_mask` stored with each observation; inactive rows stay in
the batch, and the runner passes `weights=` (0 for an inactive row) to the
learner's `probe`, which hands them to the two statistics above. A custom
`probe` gets them too if it has a `weights` parameter; one without it is called
as before and counts every row. With fewer than two active
rows in the batch, dormancy reads 1 and rank 1. Runs whose slots are all
active read the same values as without the weights.

 Tracking does not change training (tested for PPO, DQN and SAC), the
carry checkpoints as one pytree, and memory is O(memory + window).
Probing runs per update, not per tick, because one forward pass every tick
would dominate the cost of small environments.
