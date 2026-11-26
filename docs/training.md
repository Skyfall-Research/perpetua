# Training learners

This page covers the bundled learners: how to choose one, run it, and save and
resume it exactly. Their options for continual learning (plasticity, pixels,
task sequences, reward centering) are in [learner options](training-options.md),
and hyperparameter sweeps and tuning budgets in [tuning](tuning.md). Every learner
treats the lifetime as one continuing task: there are no episode resets, and
value estimates bootstrap across respawns and drift.

| Learner | Actions | Agent slots | Runner | Use it for |
|---|---|---|---|---|
| `PPO` | discrete | one shared policy | `PPORunner`, `HostPPORunner` | the default on-policy learner |
| `IPPO` | discrete | one model per slot | the same runners | interacting agents; see [multi-agent](marl.md) |
| `DQN` | discrete | one shared Q-network | `DQNRunner`, `HostDQNRunner` | off-policy learning with replay |
| `IQL` | discrete | one Q-network per slot | the same runners | independent off-policy agents |
| `SAC` | continuous `Box` | one shared policy | `SACRunner`, `HostSACRunner` | continuous control, Continual World |
| `TorchPPO` | discrete | one shared policy | `TorchRunner` | PyTorch agents (host loop, slower) |

`Host...` runners step CPU simulators (Meta-World, EnvPool, Gymnasium lifts)
from Python; the others run as one compiled program. The optional
[compute-aware runner](compute-time.md) also charges the time a learner spends
thinking against the world clock.

## In brief

- Pick a learner from the table above and pair it with its runner. One runner
  call is one rollout and one update; call it again to continue the lifetime.
- Learners need the `training` extra.
- `pp.checkpoint.save` stores the whole training state at an update boundary,
  and a restored run continues exactly on the same software and backend.
- Nothing in a learner resets at a respawn or a switch unless you opt in
  ([plasticity options](training-options.md#plasticity-options),
  [SAC options](training-options.md#sac-options-for-task-sequence-protocols)).
- DQN on a task with a constant reward per tick may need
  [reward centering](training-options.md#average-reward-reward-centering).
- Problems and error messages are in the [FAQ](introduction/faq.md).

## Install and run

PPO uses the optional Flax/Optax training extra. Development installs include it
so the full test suite can run; a core wheel does not require those packages.

```bash
uv sync --locked --extra training
uv run --extra training python examples/train_ppo.py --updates 256 --seed 0
```

For CUDA, sync both extras and run the virtual environment's Python directly:

```bash
uv sync --locked --extra cuda13 --extra training
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python examples/train_ppo.py --require-gpu --updates 256
```

The example is one continuing 3-by-3 world with four noninteracting agent slots,
one shared policy, a 1,024-tick reward window and 128 ticks per rollout (512
transitions across the four slots).
These slots are not four independent training seeds. `--drift cyclic` switches
between two goals every 4,096 ticks; `--drift random-walk` uses bounded noise.

## Functional learner and runner

```python
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
agent = PPO(action_size=5, config=PPOConfig())
runner = PPORunner(env, agent)
train_state = runner.init(seed=0, window_size=1024)
train_state = runner(train_state)
```

`action_size` is the width of the environment's action mask, which is
`env.action_space.nvec[0]` (`PPO(action_size=env.action_space.nvec[0])` works;
NumPy integers are accepted). A learner on a task switch needs the padded size,
the largest task's action count. The default MLP torso reads
`(agents, features)` observations, so pixel environments need
`PPOConfig(torso="cnn")` and a learner built with
`observation_space=env.observation_space`, which fixes
[what the frames are divided by](training-options.md#pixel-observations-cnn-torso);
the learners refuse a continuous action space
and name `pp.agents.SAC`. The rollout's samples (`rollout_steps * agents`
pooled for PPO, `rollout_steps` per agent for IPPO) must divide evenly into
`minibatches` and leave at least two samples in each: the advantages are
normalised per minibatch, so a single sample would normalise to exactly zero
and the policy would never be updated. A runner refuses such a split when it
is built. Float fields of a config (`learning_rate`, `gamma`, ...) accept
Python, NumPy and JAX real scalars and are stored as Python floats; booleans
are refused.

`PPO.act` consumes only public observations/masks and a fresh policy key. It
returns actions, their log probabilities and value predictions. Actor and
critic are separate two-layer tanh MLPs. Adam uses global gradient clipping and
a constant learning rate; there is no finite-budget annealing schedule.

Each runner call collects one bounded rollout and performs the configured
epochs/minibatches. GAE bootstraps from the final observation with no done mask:
respawn, drift and rollout boundaries are not terminal states. Active-slot
weighting excludes inactive samples; an all-inactive minibatch leaves parameters
and optimizer state unchanged. No learner state is reset at a task switch.

PPO optimizes a discounted surrogate (`gamma=0.99` by default), not an
average-reward policy-gradient formulation. Evaluation remains online lifetime
reward in the same world; no separate frozen-policy evaluation is performed.

`jax.vmap(runner)` supports independent learners with the same static configuration
and shapes. Different network architectures in one batch are not supported;
batched hyperparameters are, through [sweeps as data](tuning.md#hyperparameter-sweeps-as-data).
A learner runner (`PPORunner`, `DQNRunner`, `SACRunner`) is already compiled,
with compiler options JAX accepts only on a top-level `jax.jit`, so it must be
the outermost compiled call: use `runner(state)` or a bare `jax.vmap(runner)`.
`jax.jit(runner)` and `jax.jit(jax.vmap(runner))` are refused with an error
that says so.

## Complete state at update boundaries

`TrainingState` contains:

- Environment/drift state, lifetime clock, observations and masks.
- Separate world, drift, policy and optimizer-shuffle RNG streams.
- Lifetime/window/EMA metrics, ring buffer and health state.
- Network parameters and every Adam optimizer state leaf.
- Exact rollout and gradient-update counters, plus last-update diagnostics.

The rollout buffer is consumed within a runner call and is not retained at a
returned boundary. Saving mid-rollout or mid-optimizer update is unsupported.
The off-policy learners (DQN, IQL, SAC) keep their replay buffers in their
state, so a checkpoint includes them.

Update counters use `Clock`; Optax's internal bias-correction counter retains
its library-defined int32 saturation behavior and is not a lifetime/LR clock.
Health checks report non-finite parameters, optimizer leaves, losses and rewards;
the example stops at the next update boundary rather than silently reinitializing.

## Save and resume

```python
pp.checkpoint.save("checkpoints/001", train_state, metadata=runner.metadata)
template = runner.init(seed=999, window_size=1024)
train_state = pp.checkpoint.restore("checkpoints/001", template, expected_metadata=runner.metadata)
train_state = runner(train_state)
```

The template supplies structure and target devices, not the restored values.
All saved RNG/optimizer state takes precedence over the template seed.
`save` stages the checkpoint beside the target and renames it into place only
after every file commits, so killing a save cannot leave a restorable-looking
directory or block a retry. Saving over an existing checkpoint is refused.
`restore` checks the checkpoint's structure and metadata, not the integrity of
the stored array bytes (see below).

```bash
uv run --extra training python examples/train_ppo.py \
  --updates 128 --seed 1 --checkpoint checkpoints/001
uv run --extra training python examples/train_ppo.py \
  --updates 128 --resume checkpoints/001 --checkpoint checkpoints/002
```

Use the same drift/learner configuration on resume. A resumed report does not
claim the template seed was the original initialization seed.

Checkpoints are synchronous, atomic **local Orbax directories**, containing
state plus JSON metadata. Existing destinations are never overwritten. The
versioned format records environment/drift and learner configuration, pytree
structure, shapes, dtypes and software/backend information. Restore rejects
missing or unfinished checkpoints, an unsupported format or version,
configuration changes, and structure, shape or dtype mismatches with the
template, all before loading the state. It does not deserialize Python code or
pickle arbitrary objects.

Restore does not checksum the stored arrays. A flipped byte inside a
structurally valid state file is not detected: in a PPO checkpoint, flipping one
byte in the middle of the largest state file restored without error and changed
one leaf. The test suite flips a byte in a small lifetime state and expects a
rejection, which shows that damage to structural bytes is caught, not that data
bytes are. Keep your own hash of a checkpoint directory if you need to detect
corruption in storage or transfer.

Bitwise continuation is tested on matching software/backends, including a
fresh-process CPU resume. It is not guaranteed across devices or software changes.
A checkpoint records the Perpetua version and source revision, the JAX and
Orbax versions, the backend, Python and platform that wrote it
(`pp.checkpoint.provenance("checkpoints/001")` reads them back). `restore`
compares the Perpetua version and revision, JAX, Orbax and backend with the
running ones and emits one warning listing any difference; it does not refuse,
because a changed revision need not change the state layout. The warning also
fires when the revision is unknown on one side only, or when either side was
written from a tree with uncommitted changes, since such a revision does not
identify the code. The record is not
part of `runner.metadata`, so a checkpoint written at another commit is never
refused for that reason alone. Use the dependency lock, and put
`pp.reporting.provenance()` in experiment reports, as the example scripts do.
A checkpoint restores only into the state layout it was written with; see the
[FAQ](introduction/faq.md#checkpoints).

To audit or debug a stretch of a lifetime, replay it from a checkpoint:

```python
segment = pp.checkpoint.replay("checkpoints/001", template, runner, chunks=4)
```

Every RNG stream lives in the saved carry, so the replayed chunks repeat the
original ones exactly on the same software and backend (tested against the
original run). It is not a cross-device guarantee. A `TorchRunner` is the
exception: its agent's weights and generator live outside the carry, so the
generic functions refuse it; use `TorchRunner.save` and `TorchRunner.restore`.

## Continuing DQN and IQL

```bash
uv run --extra training python examples/train_dqn.py \
  --updates 128 --seed 1 --checkpoint checkpoints/dqn-001
uv run --extra training python examples/train_dqn.py \
  --updates 128 --resume checkpoints/dqn-001
uv run --extra training python examples/train_dqn.py \
  --algorithm iql --updates 128 --checkpoint checkpoints/iql-001
```

`DQNConfig` sizes count ticks, not transitions: one replay row is one tick and
holds every slot, so `batch_size` and `buffer_capacity` are in ticks and shared
DQN trains on `batch_size * slots` transitions per gradient step (IQL on
`batch_size` per slot). `rollout_steps` may not exceed `buffer_capacity`, for
`SACConfig` as well. Every field is described in the `DQNConfig` reference.

`DQN`/`DQNRunner`/`HostDQNRunner` share the lifetime contract: no episodic
resets or done masking (bootstrapping always continues), constant epsilon
exploration with no finite-lifetime annealing, masked epsilon-greedy sampling
and masked next-state maxima so invalid actions are never selected or backed
up, active-masked Huber losses over fixed agent slots, and a shared Q-network
across slots. `IQL` instead keeps independent per-slot models, target
networks, optimizer moments and sampling subkeys (mirroring IPPO): a slot's
rewards/gradients cannot update another slot, wholly inactive slots skip
updates, and statistics/gradient clocks carry a leading agent axis. The replay
buffer is a fixed-shape ring shared by all slots: `cursor` equals the
insertion count until `2 * capacity` and then stays in
`[capacity, 2 * capacity)`, so it never overflows; `filled` saturates at
capacity, sampling uses explicit saved
RNG keys with replacement from the filled prefix, and the whole buffer,
online/target parameters, optimizer moments, step counters and target-sync
age checkpoint and replay exactly. The target network synchronizes every
`target_update_interval` gradient steps through an explicit counter.
Each replay entry stores, per slot, the observation and next observation
(float32), the action mask and next action mask (`action_size` booleans each),
the action (int32), the reward and the bridge discount (float32 each) and the
active flag (boolean), so the buffer takes
`capacity * slots * (8 * obs_size + 2 * action_size + 13)` bytes,
independent of lifetime length (`obs_size` is the number of elements in one
slot's observation). Both learners train through cyclic drift
schedules (`--drift cyclic`): seeded sanity runs on the drifting grids keep a
reward rate well above the random baseline, do not collapse after switches,
and `recovery_ticks` on the reconstructed training trace stays bounded. Those
are sanity checks, not adaptation-speed benchmarks.

## Continuing SAC

```bash
uv run --extra training python examples/train_sac.py \
  --updates 64 --seed 1 --checkpoint checkpoints/sac-001
uv run --extra training python examples/train_sac.py \
  --updates 64 --resume checkpoints/sac-001
```

`SAC`/`SACRunner` run the standard soft actor-critic loop under the lifetime
contract on continuous-action environments (for example
`gymnax/Pendulum-v1`): tanh-squashed Gaussian policy mapped onto the
environment's declared bounds, twin critics with soft-updated target copies
(explicit `tau`), bounded fixed-shape replay with explicit sampling keys, and
no episodic resets or done masking. The entropy temperature is fixed by
default (`alpha`); `autotune_alpha=True` learns `log_alpha` toward
`target_entropy` with a detached
`-(log_alpha * (log_prob + target_entropy))` objective. The policy's
log-probability is in the environment's action units, so `target_entropy` is
too. Unset, it is `-action_dim + sum(log((high - low) / 2))`, the usual
`-action_dim` heuristic in actions normalised to [-1, 1] (exactly `-action_dim`
for bounds of +-1). A target of `-action_dim` in raw units cannot be reached
once an action range (`high - low`) is narrower than about 0.37 (a box of
+-0.18 or less), and the temperature then grows without limit; pass `target_entropy` yourself to override the default. The default 128
gradient steps per 128-tick rollout follow SAC's conventional
one-gradient-step-per-environment-step ratio; checkpoints retain
actor/critics/targets, optimizer moments (including log-alpha), replay
contents and counters exactly. `HostSACRunner` reuses the same learner on the
host lift's continuous Box-action path (for example a CPU `Pendulum-v1`);
without a snapshot codec those backends stay opaque and non-checkpointable
under the documented single-use ownership rules. `HostDQNRunner` and
`HostSACRunner` copy the replay buffer once per call, which keeps the state
passed in readable, and then write each tick's transition into that copy in
place, so the cost of a tick does not grow with `buffer_capacity`. SAC also trains through
cyclic gravity drift: a seeded sanity run on `gymnax/Pendulum-v1` keeps its
windowed return far above the uniform-torque reference across switches with
bounded measured recovery. This is a sanity check only.

`SACConfig(kernel_init=...)` selects the weight initialiser for every Dense
layer of the actor and the critics: `"orthogonal"` (the default) or
`"glorot_uniform"`. On Continual World the two were not distinguishable
with three seeds each (see the [preset page](presets/continual-world.md)).

### Slots that go inactive

DQN, IQL and SAC store a transition only when its slot has a real next
observation. When a slot is active on one tick and inactive on the next (an
EnvPool slot on the tick its game ends, a PettingZoo agent that drops out), its
transition waits, adds the discounted reward of the ticks it sits out, and is
stored when the slot acts again with the discount it accumulated. With
`reward_centering` the reward rate is subtracted once per stored transition,
not once per tick it spans, and it is moved by the undivided TD error of that
transition. With inactive stretches their `reward_rate` is therefore not an
average reward per tick: a stored transition that spans several ticks carries
the discounted reward of all of them and has the rate removed once.

PPO, IPPO and `TorchPPO` have no stored transitions. Inside a rollout their advantage
estimate skips the ticks a slot sits out in the same way: the tick before the
stretch bootstraps from the next tick the slot acts, discounted across the
stretch and including its rewards, and the value computed from a placeholder
observation is never used. With `reward_centering` PPO subtracts the rate from
every tick's reward, the ticks a slot sits out included, so a stretch of `k`
ticks has the rate removed `k` times (discounted) where DQN, IQL and SAC remove
it once from the accumulated reward. The rate itself is moved by the same
bridged TD errors the advantages use, each divided by the discounted number of
ticks it spans (`1 + gamma + ... + gamma**(k-1)`), so `reward_rate` stays an
average reward per tick and its step does not grow with the length of the
stretch. A rollout does not carry an open stretch
across its end: a slot that is inactive when the rollout ends bootstraps the
advantage of its last action from the critic's value of its placeholder
observation, a value the critic is never trained on.

## PyTorch agents

`perpetua.torch_bridge` runs PyTorch agents on any lifetime (optional `torch`
extra). A PyTorch policy cannot run inside a compiled scan, so `TorchRunner` is
a host loop: each tick, observations and masks cross to PyTorch through DLPack
(no copy when both use the same device), the agent acts, and the action goes
back through `runtime.transition`, the per-tick body every JAX runner uses.
Metrics, recovery tracking and the Oracle behave as for the JAX learners.
`TorchRunner` takes no reference policy, so it reports no regret.

```python
from perpetua.torch_bridge import TorchPPO, TorchRunner

env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
runner = TorchRunner(env, TorchPPO(env.observation_space, env.action_space, seed=0))
train_state = runner.init(seed=0)
for _ in range(64):
    train_state = runner(train_state)
runner.save("checkpoints/torch", train_state)  # lifetime + agent state dict, atomically
```

Any object with `act(obs, info) -> (action, extras)`, `update(rollout)`,
`state_dict()`, `load_state_dict()`, `device` and `configuration` is an agent.
`info` holds `action_mask`, `active_mask`, `respawn` and `space_change` as
tensors, the tick `t` as an int64 scalar and, under an `Oracle`, the exposed
fields as tensors in `info["oracle"]`.
`TorchRunner` refuses a `TorchPPO` whose rollout (`rollout_steps * slots`) does
not split into `minibatches` of at least two samples when it is built. Its
checkpoint metadata records the installed `torch` version, so a checkpoint
restores only under the same version: the error names the `framework` entry
with both versions.
`TorchRunner.healthy` checks every floating tensor anywhere in the agent's
`state_dict()` (nested dicts, lists and tuples are walked), so optimizer
moments and any other saved tensors count as well as the network.
`TorchPPO` is a reference continuing PPO for discrete per-slot actions: it masks
invalid actions, skips the ticks a slot sits out in its advantage estimate
exactly as the JAX PPO does ([slots that go inactive](#slots-that-go-inactive);
`TorchPPO.advantages(rollout)` returns the estimate), weights inactive slots
out of its losses (its per-minibatch
advantage normalisation still includes them), seeds its initial weights with
`torch.manual_seed(seed)` (which reseeds PyTorch's global RNG) and keeps its
sampling generator in the state dict, so a restored run repeats an uninterrupted one exactly on CPU
(tested). On the 3×3 grid with four slots it reaches a 0.43 reward window after
64 updates against 0.05 for random, at about 2,300 ticks/s on a Mac CPU: the
per-tick host round trip is the price of leaving the compiled path.

## Provenance and limits

The PPO architecture, clipped losses, GAE and scan-based update structure adapt
PureJaxRL revision `31756b197773a52db763fdbe6d635e4b46522a73`. Its Apache-2.0
license, copyright and modification notice are distributed under
`perpetua/agents/LICENSE-PureJaxRL` and `perpetua/agents/NOTICE`.

The learners are reference implementations for continual-RL experiments, not
tuned benchmark baselines, and no statistical cross-benchmark comparison is
established here. The [CPU bridge](adapters.md) reuses the same learners
through the `Host...` runners. [Supported Gymnasium CartPole snapshots](cpu-snapshots.md)
can resume full CPU/learning state; opaque backends remain non-checkpointable.
The optional `native/envpool_snapshot/README.md`
also supports complete synchronous CartPole/PPO continuation.

Host PPO/IPPO kernels use per-function
deterministic GPU compilation, recorded as `host_compiler_options` in checkpoint
metadata, to prevent fresh-process autotuning differences. The same options apply
to native compiled `PPORunner`, `DQNRunner` and `SACRunner` chunks (recorded as
`compiler_options`), the `HostDQNRunner`/`HostSACRunner` kernels, and
`ComputePPORunner`'s native replay and measured paths, so fresh-process GPU
resume is exact for those runners. It is not for `TorchRunner`, the stateless
`ComputeRunner`'s replay, or `ComputePPORunner` replaying a plain
`TimingTrace` on a host world: those compile without the options.
No global XLA flags are changed, and deterministic kernels may be slower. See
[snapshot timing/restore](pettingzoo-snapshots.md).
