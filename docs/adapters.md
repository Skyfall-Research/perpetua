# Environment adapters and the CPU bridge

## Supported integrations

| Integration | Execution | Drift | Full checkpoint |
|---|---|---|---|
| `ContinualGrid-v0` | Native JAX | Goal coordinates | Yes |
| `DoorGrid-v0` | Native JAX (evolving action space) | Door affordances `(4,)` | Yes |
| `ForagerGrid-v0` | Native JAX (agent-driven drift) | Per-agent mass, fed by the agent's own collections | Yes |
| `BeanWorld-v0` | Native JAX, never-ending toroidal world in the style of Jelly Bean World | Item rewards and regrowth delays | Yes |
| `CooperativeGrid-v0` | Interacting native JAX | Target anchor | Yes |
| `PartnerGrid-v0` | Native JAX + scripted partners | Target anchor + hidden partner policy | Yes |
| `OpponentGrid-v0` | Native JAX + scripted opponents | Shared goal + hidden opponent policy | Yes |
| `gymnax/CartPole-v1` | Functional JAX | Gravity | Yes |
| `gymnax/Pendulum-v1` | Functional JAX (continuous actions) | Gravity | Yes |
| `mjx/Pendulum-v0`, `mjx/CartPole-v0` | MuJoCo on MJX, functional JAX (optional `mjx` extra) | Gravity, body mass/inertia scale | Yes |
| `gymnax/<20 others>` (Acrobot, MountainCar, MinAtar, bsuite, misc) | Functional JAX, generic adapter | Any non-precomputed float parameter, by name (11 of the 20 have one; see below) | Yes |
| `jaxmarl/MPE_simple_spread-v3` | Functional JAX | Reward multiplier, audited dynamics fields (`local_ratio`, `dt`, `mass`, `damping`, `contact_margin`) and the hidden scripted-policy `beta` | Yes |
| `jaxmarl/<12 other MPE tasks>` (see [MARL](marl.md)) | Functional JAX, generic `JaxMARLMPE` | Reward multiplier + audited dynamics fields | Yes |
| `meal/overcooked-v0` | Native JAX (optional MEAL 0.3.1, MIT) | Reward multiplier; layout switching via `pp.presets.meal_sequence` | Yes |
| `metaworld/MT1-v0` | Host MuJoCo (optional Meta-World 3.1.1, MIT) | Reward multiplier; CW10/CW20 via `pp.presets.continual_world` | No (opaque state) |
| `coom/COOM-v0` | Host ViZDoom worker in COOM's pinned runtime (`scripts/setup_coom.py`) | Reward multiplier; sequences via `pp.presets.coom_sequence` | No (opaque state) |
| `morpheus/Morpheus-v0` | Hosted service over HTTP (`morpheus` extra), or seeded mock | Paper chaos regimes (pushed to the world); reward multiplier | No |
| `agarcl/agario-screen-v0`, `agarcl/agario-grid-v0`, `agarcl/grand-arena-v0` (the paper's full game) | Native worker in AgarCL's pinned runtime (`scripts/setup_agarcl.py`) | Reward multiplier; hybrid `Box(3)` actions | With `replay_capacity`: seeded action-log replay |
| `GymnasiumLifetime(env)` | Host-controlled CPU | Reward multiplier | Supported CartPole codec only |
| `AECLifetime(env)` | Host-controlled CPU (sequential cycles) | Reward multiplier | Opaque single-use |
| `PettingZooLifetime(env, snapshot=True)` | Host-controlled CPU | Reward multiplier | Headless MPE2 simple_spread codec only |
| `envpool/CartPole-v1` | Batched C++ CPU + JAX policy/learner | Reward multiplier | No |
| `envpool/Atari-v0` (any Atari game) | Batched C++ CPU + JAX policy/learner (Linux, optional GPL-2.0) | Reward multiplier; task switching via `pp.presets.switching_atari` | No (opaque state) |
| `envpool/CartPole-v1`, `snapshot=True` | Optional headless C++ binding + JAX learner | Reward multiplier | Yes, consumed synchronous batches |

Optional libraries load only when requested. The compatible gymnax adapter pins
0.0.9, tested with JAX 0.11.2 and Gymnasium 1.4.0; gymnax 1.0.0 declares older,
incompatible version bounds. This integration does not downgrade our stack.
See [continual benchmarks](benchmarks.md) for MEAL, Meta-World and COOM.

## Functional control environment

```bash
uv sync --locked --extra interop --extra training
uv run --extra interop --extra training python examples/control_ppo.py --backend gymnax
```

```python
import perpetua as pp
from perpetua.agents import PPO, PPORunner

env = pp.make("gymnax/CartPole-v1", drift=pp.drift.Cyclic([(9.8,), (5.0,)], period=128))
trainer = PPORunner(env, PPO(action_size=2))
state = trainer(trainer.init(seed=0))
```

The adapter uses low-level `step_env`/`reset_env`, not gymnax's autoresetting
public step. It discards backend episode-age limits and preserves the Perpetua
clock. Physical failures respawn inside the same transition. Failure reward is
explicitly zero, so instant respawns do not turn CartPole's +1 reward into a
constant independent of behavior. This reward/time policy is recorded in metadata;
it is an adapted continuing world, not the original episodic benchmark.

Observations/actions/rewards retain the agent-slot axis: `(1, 4)`, `(1,)`, `(1,)`.
The same `PPORunner`, online metrics, Oracle wrapper and checkpoint code work on
the grid and this control environment. Checkpoints include all physics and RNG.

## Every other gymnax environment

```python
import perpetua as pp

env = pp.make("gymnax/Acrobot-v1", drift={"link_mass_1": pp.drift.Cyclic([1.0, 1.2], 128)})
env = pp.make("gymnax/Asterix-MinAtar")  # pixel observations (1, 10, 10, 4)
env = pp.make("gymnax/DeepSea-bsuite", drift={"unscaled_move_cost": pp.drift.Stationary()})
```

The other 20 gymnax environments run through one generic adapter,
`perpetua.adapters.gymnax_params.GymnaxParameters`, wrapped in the same
`DriftEnv` as the native grids, so drift, the Oracle, the clock and metadata
behave identically everywhere. `scripts/probe_gymnax.py` prints the measured
coverage for the installed gymnax version.

- **Any float parameter field can drift, by name**, except fields that another
  field is computed from. Nine of the 20 generic environments have no
  driftable field at all (Asterix, Breakout, Freeway, SpaceInvaders, Catch,
  MemoryChain, UmbrellaChain, MNISTBandit and Pong: none has a float
  parameter), so `drift=` is refused on them; their
  environment pages say "Driftable parameters: none". The other 11 are
  Acrobot, MountainCar, MountainCarContinuous, DeepSea, DiscountingChain,
  FourRooms, MetaMaze, BernoulliBandit, GaussianBandit, Reacher and Swimmer. gymnax precomputes derived fields at construction and
  its parameter structs have no `__post_init__`, so drifting CartPole's
  `masspole` would silently leave `total_mass` stale. The adapter reads the
  parameter class to find those inputs and refuses them with an explanation;
  the fields the physics reads directly stay driftable. Measured across all 24
  environments with `scripts/probe_gymnax.py`: only CartPole has such inputs
  (`masscart`, `masspole`, `length`), so in practice the rule removes three
  fields from one environment.
- **Drift tables follow the field's shape.** Each phase must have the
  parameter's shape. The generic adapter's fields are mostly scalars, so a
  two-phase table is one-dimensional: `Cyclic([1.0, 1.2], 128)`. The
  hand-written CartPole and Pendulum wrappers use shape-`(1,)` fields
  (`gravity`, `g`), so their phases are tuples: `Cyclic([(9.8,), (12.0,)], 128)`.
  A mismatch raises an error that states both shapes, and each environment
  page shows a working drift example.
- **Episodes become respawns.** The time limit is removed, so an episode ends
  only on genuine termination; that termination is reported as `respawn`, and
  gymnax's autoreset supplies the fresh state inside the same lifetime.
- **A bare process drives the default drift field.** The field is the one in
  the environment page's "Default drift field" row (`dt` on Acrobot,
  `min_position` on MountainCar, `mean_mu` on GaussianBandit): the first
  driftable field in the order gymnax declares its parameters, which is not
  always the first entry of the alphabetical "Driftable parameters" row. There
  is no warning when the environment has several; to drift another field pass
  a mapping,
  `drift={"field": process}`. Environments with no driftable field
  (Pong, the MinAtar games, Catch, MemoryChain, UmbrellaChain and
  MNISTBandit) raise `ValueError` saying so for any `drift=`.
- **Actions are checked for shape and dtype, not value.** A discrete environment
  takes an integer array of shape `(1,)`; a Box one (`MountainCarContinuous`,
  `Reacher`, `Swimmer`) takes a float array of the action space's shape, so
  `pp.policies.random` does not apply to it and raises an error pointing to
  `pp.policies.uniform(env.action_space)`. Values are not checked inside `jit`:
  an index outside `[0, n - 1]` reaches the backend as it is. Measured on gymnax
  0.0.9, most backends look the index up in a table, so an index past the end
  acts as the last action and `-1` as the last action too; MountainCar scales
  its push with the index (action 6 applies five times the force of action 2).
  Clip in the policy if your agent can produce such values.
- **Rewards are the backend's own**, including on a terminating step. The two
  hand-written wrappers above exist because they deliberately change that:
  measured over 400 ticks with the same seed, generic CartPole collects 400
  and the hand-written one 358 with the same 42 respawns, because the
  hand-written wrapper zeroes the reward on each failure tick. The Pendulum
  wrapper does not change the backend's semantics: measured over 100 ticks with
  the same keys, its observations and rewards are identical to the generic
  adapter's. It differs in exposing gravity as a shape-`(1,)` field (`g`) and in
  rejecting an action that is not a float array of shape `(1, 1)`. So
  `gymnax/CartPole-v1` and `gymnax/Pendulum-v1` keep their wrappers, and an
  environment that needs different reward or respawn semantics needs one too.

Limits: `SimpleBandit-bsuite` and `PointRobot-misc` are unregistered because
their gymnax 0.0.9 implementations are broken (positional state-field order
and mismatched `select` shapes respectively). The other 20 generic
environments pass the full conformance suite (`tests/test_environment_conformance.py`).
The adapter carries the backend's own observation in state rather than calling
`get_obs`, whose signature varies across environments; its state is a JAX
pytree and checkpoints with the lifetime. It has no host or snapshot path.

## Functional multi-agent environment

```bash
uv sync --locked --extra jaxmarl --extra training
uv run --extra jaxmarl --extra training python examples/train_marl.py \
  --backend jaxmarl --updates 2 --checkpoint checkpoints/jaxmarl-001
uv run --extra jaxmarl --extra training python examples/train_marl.py \
  --backend jaxmarl --updates 1 --resume checkpoints/jaxmarl-001
```

```python
import perpetua as pp

env = pp.make(
    "jaxmarl/MPE_simple_spread-v3",
    num_agents=3,
    num_landmarks=3,
    local_ratio=0.5,
    max_steps=25,
    drift=pp.drift.Cyclic([(1.0,), (0.5,)], period=128),
)
```

The optional `jaxmarl` extra (JaxMARL 0.2.0, Apache-2.0, JAX >= 0.4.25) lifts
JaxMARL's jit-able MPE `simple_spread` into a native continuing lifetime.
Dictionary observations/actions/rewards become fixed `(3, 18)`/`(3,)`/`(3,)`
agent slots. The adapter uses `step_env`, not JaxMARL's public `step` whose
both-branch auto-reset computes every transition twice; the legacy
`max_steps` limit respawns the world on the same tick through a single
conditional reset, retaining the transition reward and advancing the
Perpetua clock. JaxMARL MPE exposes no action masks, so every action stays
valid, and no legacy info or task identity leaks into public data.

The reward-multiplier drift scales rewards without touching world dynamics or
RNG streams; phases/parameters stay Oracle-only. `local_ratio`, `dt`, `mass`,
`damping` and `contact_margin` can drift the physics instead, and `scripted`
drifts the partner policy below; [multi-agent](marl.md) lists the measured
fields. With ``num_scripted > 0`` the
last slots become scripted partner/opponent agents with hidden drifting
policy parameters (beta-greedy toward or away from their nearest landmark,
else uniform, on an independent RNG stream): they keep world rewards and
appear in learner observations as ordinary peers, but their masks allow only
the placeholder stay action, their observations are zero and their
``active_mask`` is False, so IPPO never trains on them. The world ``State``
pytree, both drifts and RNG checkpoint and replay exactly through the shared
native contracts, including vmapped worlds and compiled chunks. JaxMARL's own
``done`` uses a pre-increment check, so ``max_steps=N`` truncates after N+1
transitions; the adapter mirrors that timing exactly. ``local_ratio`` must be
a Python float here (JaxMARL itself accepts ints).

## Gymnasium API view

```python
import perpetua as pp
from perpetua.adapters import GymnasiumAdapter

view = GymnasiumAdapter(pp.make("ContinualGrid-v0"), horizon=256)
obs, info = view.reset(seed=0)
obs, reward, terminated, truncated, info = view.step(0)
```

The current view supports single-agent Box observations and discrete action slots.
`reset` initializes once; later unseeded resets return the current world without
advancing or restarting it. A later seed is rejected. The first `reset()` without
a seed starts the lifetime from seed 0, so two adapters built the same way give
identical lifetimes; pass `seed=` for another. `info["t"]` is the lifetime `Clock`
(call `.to_int()`), and any non-empty `options` raises `ValueError`, whereas the
PettingZoo view ignores reset options. The optional horizon emits
synthetic truncations while the lifetime clock continues. These choices are
recorded in adapter metadata.

This is familiar Gymnasium notation, **not identical episodic reset semantics**.
Gym's deterministic-reset checks or learners that reset recurrent state at every
synthetic boundary need deliberate handling. Multi-agent reward aggregation and
continuous-action views are not silently inferred.

## Lift a CPU Gymnasium environment

```python
import gymnasium as gym
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import PPO, HostPPORunner

env = GymnasiumLifetime(gym.make("CartPole-v1"))
try:
    trainer = HostPPORunner(env, PPO(action_size=2))
    state = trainer(trainer.init(seed=0))
finally:
    env.close()
```

The CPU lift supports Box observations with Discrete or Box slot actions. An
observation keeps the environment's own shape with a leading slot axis, cast to
float32: a `Box(0, 255, (H, W, C), uint8)` environment gives `(1, H, W, C)`, and
needs `PPOConfig(torso="cnn")` and a learner built with
`observation_space=env.observation_space`, which then divides the frames by
255. For
continuous actions (for example a CPU `Pendulum-v1`), the lifetime's action
space is a per-slot Box, actions are validated as finite float vectors and
clipped into the declared bounds, and the compatibility `action_mask` exposes
one validity flag per slot while `active_mask` governs participation. A
`SAC` learner runs through `HostSACRunner` on the same lift. Legacy termination
and truncation cause internal respawns, not public done flags. Backend reward
and time-limit semantics are retained and recorded. Retaining dense +1 rewards
with instant CartPole respawns can give a degenerate reward signal; benchmark
authors must define suitable reward/respawn dynamics, rather than claim
equivalence to the functional control adapter above. Host lifts own one mutable
simulator, so they cannot be `vmap`ped or traced with a lifetime seed: that
raises a `TypeError` that says so, and batching comes from the backend's own
slots. Backends without a snapshot codec remain opaque and non-checkpointable,
with single-use ownership rules; only the supported CartPole codec resumes.

## Batched C++ simulator bridge

On Linux:

```bash
uv sync --locked --extra cpu-bridge --extra training
uv run --extra cpu-bridge --extra training python examples/control_ppo.py \
  --backend envpool --num-envs 8
```

For a GPU policy/learner, add `cuda13`, then run the same example with
`JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, no overriding
`LD_LIBRARY_PATH`, and `--require-gpu`. Simulators remain on CPU.

EnvPool supplies batched C++ stepping, without Python process pools. The adapter
sorts returned batches by simulator ID. Its synchronous product-world has one
clock and multiple independent simulator/agent slots, not interacting MARL.
Legacy time limits and next-tick autoresets are recorded; reset-only slots are
inactive for PPO credit. Reward scaling is composable drift and is not exposed
to the agent unless Oracle enables it.

`HostRunner` and `HostPPORunner` orchestrate CPU stepping. The policy and the
**same PPO loss, GAE and optimizer** are compiled JAX code. Mutable CPU states
without a snapshot codec have owner/generation checks: stale, forked or failed
states cannot be reused as if functional. Supported Gymnasium CartPole carries
a numeric snapshot and restores it before stepping. Always close backends.

The optional `native/envpool_snapshot/README.md`
provides the same complete-state/replay contract for synchronous CartPole batches.
Build it locally and request `snapshot=True` explicitly. It preserves upstream
physics, MT19937 RNGs, episode counters and pending reset-credit masks; it does
not replace the installed EnvPool wheel or expose an async checkpoint API.

## Runtime and checkpoint limits

EnvPool 1.2.7's installed API exposes no full simulator/RNG snapshot operation.
The ordinary EnvPool bridge and unsupported Gymnasium environments declare
`checkpointable=False`; saving a full lifetime fails before creating files.
Supported Gymnasium CartPole and the optional headless EnvPool binding can resume
physics, RNG, reset state and learning state;
see [CPU snapshots](cpu-snapshots.md). Saving only neural weights is never
represented as resuming an opaque simulator lifetime.

The host loop uses explicit batched host/device transfers. The compiled bridge
below runs the simulator inside the compiled tick instead; there is no
pinned-memory path, because transfers were not the bottleneck in measurement.

The Linux wheel imports rendering modules even for CartPole. On the tested
Ubuntu container it required user-space `libegl1`, `libgles2`, `libopengl0`
(plus dependency `libegl-mesa0`); no NVIDIA driver was changed. The macOS wheel
requires external Qt paths, so this project does not currently enable EnvPool
there. Core/native/gymnax paths remain usable on macOS (checked by hand; CI
runs on Linux only).

## Compiled CPU bridge

`compiled=True` on `GymnasiumLifetime`, `EnvPoolCartPole`, `EnvPoolAtari`,
`MetaWorldLifetime`, `COOMLifetime` and `AgarCLLifetime` turns the simulator step into a `jax.experimental.io_callback`, so the whole
tick (drift, metrics, the policy and a learner's update) compiles into one
program and the ordinary `Runner`, `PPORunner`, `DQNRunner` and `SACRunner`
drive the host simulator:

```python
import perpetua as pp
from perpetua.agents import PPO, PPORunner

env = pp.make("envpool/CartPole-v1", num_envs=256, num_threads=16, compiled=True)
trainer = PPORunner(env, PPO(action_size=2))
```

Results are bit-identical to the eager host loop (`HostRunner`,
`HostPPORunner`), and it is 5.8–14× faster on the H200 host (see
[claim 1](claims/throughput.md)). The simulator is still one mutable object, so
compiled lifetimes are single-use (a stale or forked state is refused), cannot
be vmapped or checkpointed (so AgarCL refuses `compiled=True` with
`replay_capacity`), and need JAX's CPU platform next to the GPU
(`JAX_PLATFORMS=cuda,cpu`).

`EnvPoolCartPole(overlap=True)` uses EnvPool's asynchronous mode: each tick
half the slots act while the other half steps, so a slot advances about every
other tick (with `num_threads > 1` EnvPool returns whichever slots finish
first, so the halves can mix) and the slots that did not return are masked
out. The reward for an action arrives on the slot's next active tick. It is
opt-in and not bit-reproducible (which half returns first depends on thread timing). On the
H200 it was slower than the synchronous compiled bridge, because the simulator
stopped being the bottleneck once the tick was compiled.

Only EnvPool steps many simulators in one batched C++ call (ALE and CartPole,
`num_envs` slots per lifetime). COOM (ViZDoom), AgarCL and Meta-World (MuJoCo)
run one simulator per lifetime: COOM and AgarCL in a worker process each,
Meta-World in-process. More lifetimes mean more workers, not a larger batch.
MuJoCo control also runs natively on the accelerator through MJX
(`mjx/Pendulum-v0`, `mjx/CartPole-v0`).

## Worker process deadlines

COOM and AgarCL share a JSON-lines worker process rather than importing their
pinned runtimes. Every request has a deadline (600 s by default) covering both
the write and the complete reply. A native library that hangs is killed,
reaped and reported with the tail of its stderr, so a lifetime cannot block
forever. AgarCL replay gets `max(600 s, ticks / 200)` because replay is slower
than stepping; a million-tick replay is allowed about 83 minutes.

Saving a checkpoint writes into a private staging directory beside the target
and renames it into place only after Orbax has committed every file. A process
killed mid-save therefore leaves no directory at the requested path, cannot
damage an earlier checkpoint, and does not prevent a later save to that path.
Restore rejects a missing or unfinished checkpoint, an unsupported format or
version, a configuration mismatch, and a structure, shape or dtype mismatch with
the template. It does not checksum the state files: a flipped byte inside a
structurally valid file is not detected (a one-byte flip in the largest state
file of a PPO checkpoint restored without error and changed one leaf). The tests
cover a killed save, a corrupted byte in a small lifetime state and a ten-save
resume chain against an uninterrupted run.

This is initial interoperability: not every Gym environment, simulator, space
type or legacy benchmark is supported yet.

## Parallel PettingZoo adapters

The optional `marl` extra provides `PettingZooAdapter(native_env)` and
`PettingZooLifetime(parallel_env)`. Both retain per-agent reward vectors/masks;
no single-agent reward aggregation is inferred. The view uses synthetic
truncations and lifetime-preserving unseeded resets. The lift keeps a fixed slot
superset, hides legacy done/task-info fields and refuses opaque simulator
checkpoints. See [MARL interoperability](marl.md) for signatures, shapes, real MPE2
execution, supported spaces and lifecycle/RNG ownership. The opt-in
[MPE2 snapshot codec](pettingzoo-snapshots.md) adds complete `simple_spread`
checkpoint support without changing default opaque backend behavior.
