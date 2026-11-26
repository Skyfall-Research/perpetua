# Multi-agent RL

Every Perpetua environment is multi-agent shaped: agents are slots on a leading
array axis, each with its own reward and masks, inside one never-ending
lifetime. This page covers worlds where agents interact, scripted partners and
opponents whose behaviour drifts, and multi-agent environments from JaxMARL and
PettingZoo.

| You want | Use |
|---|---|
| A small native world where agents must cooperate | `CooperativeGrid-v0` (below) |
| A partner or opponent whose behaviour changes over time | `PartnerGrid-v0`, `OpponentGrid-v0`, or `num_scripted=` on any multi-agent JaxMARL MPE task |
| Standard multi-agent particle tasks | `jaxmarl/MPE_*` (the `jaxmarl` extra) |
| Your own PettingZoo environment | `PettingZooLifetime` or `AECLifetime` (the `marl` extra) |
| Continual cooperative benchmarks | the MEAL presets in [continual benchmarks](benchmarks.md) |

**Learners.** `IPPO` gives every slot its own actor, critic and optimizer;
`PPO` shares one policy across slots; `IQL` is the independent off-policy
counterpart of `DQN`; `SAC` works on continuous MPE tasks with identical
action bounds. Scripted partners and opponents are never trained on. See
[training](training.md).

## Native interacting world

`CooperativeGrid-v0` is a small, fully observed cooperative environment—not merely
independent agents sharing a goal. Agents can block one another's movement,
contest free cells and determine whether the whole team receives reward.

```python
import perpetua as pp
from perpetua.agents import IPPO, PPORunner

env = pp.make("CooperativeGrid-v0", size=3, num_agents=2)
trainer = PPORunner(env, IPPO(action_size=5))
train_state = trainer.init(seed=0)
train_state = trainer(train_state)
```

The grid supports sizes 2–64 and 2–`min(32, size**2 // 2)` fixed slots. Initial
and respawn positions are unique and exclude all current targets. Target `i`
is the drift-anchor cell plus `i` in cyclic row-major order. With the default
size 5 and two agents, the initial targets are `(4, 4)` and `(0, 0)`; with size 3
they are `(2, 2)` and `(0, 0)`.

### Public arrays and interactions

| Field | Shape | dtype / meaning |
|---|---|---|
| `obs` | `(agents, 4 + 3*(agents-1))` | float32 in `[0, 1]` |
| `action` | `(agents,)` | Integer: stay/up/down/left/right = 0/1/2/3/4 |
| `reward` | `(agents,)` | float32, +1 for each active agent on joint collection |
| `action_mask` | `(agents, 5)` | bool, wall/occupancy/activity availability |
| `active_mask`, `respawn` | `(agents,)` | bool |
| `t` | `Clock` | the lifetime tick counter, 64-bit; `.to_int()` reads it |

Each observation contains own normalized position/assigned target, then peers'
normalized positions and activity in slot order, excluding self. Inactive
observations and inactive peer features are zero; only placeholder stay is valid.
Physical target locations are visible; drift phase/parameters/RNG are not public
unless explicitly requested through `Oracle` and recorded in metadata.

Movement is simultaneous. Cells occupied at tick start block movement even if
their occupant proposes leaving; swaps/chains through occupied cells are not
allowed. Multiple proposals to the same free cell all stay. Out-of-range or
masked integer native actions become stay; shape/dtype errors are refused.

At least two active agents must all reach their assigned targets for joint
collection. Collision/reward/respawn use the **old** drift parameters. Respawning
agents draw unique positions from the old target complement; drift then advances
and the returned observation describes new targets. There is no episode/reset API.

Environment/agent/drift RNG streams remain separate. Storage and compiled shapes
are bounded in lifetime length; `Runner`, `PPORunner`, `vmap`, streaming metrics
and complete native checkpoints use the existing contracts unchanged.

## Independent PPO

`IPPO` composes the existing PPO loss, continuing GAE and optimizer. Every fixed
slot has a separate actor/critic parameter tree and optimizer moments. No task IDs,
task-specific heads, policy sharing or privileged central critic are introduced.
Each model uses only its own public observation and action mask.

All parameter and optimizer leaves and the gradient-step counters have a
leading agent axis. `statistics` fields are float32 `(agents,)`; training-update/world clocks
remain scalar per lifetime. A wholly inactive slot does not update its parameters,
moments or gradient counter. Minibatch divisibility applies to **time samples per
agent**, unlike shared PPO's pooled `time * agents` samples.

Saved sampling/shuffle roots derive independent slot subkeys. Shuffle subkeys
are fully consumed within one update; the next saved root determines future
updates. Native checkpoints therefore include every model, moment, key, drift,
world state and metric; no segment history or episode buffer is needed.

```bash
uv run --extra training python examples/train_marl.py --updates 2 \
  --checkpoint checkpoints/marl-001
uv run --extra training python examples/train_marl.py --updates 1 \
  --resume checkpoints/marl-001
```

## Partner/opponent policy drift

`PartnerGrid-v0` shares the cooperative grid world with scripted partners whose
policy parameters drift over the lifetime:

```python
import perpetua as pp

env = pp.make(
    "PartnerGrid-v0",
    size=3,
    num_learners=1,
    num_partners=1,
    partner_drift=pp.drift.Cyclic([(1.0,), (0.0,)], period=128),
)
```

The first `num_learners` slots are policy-controlled; the last `num_partners`
slots are scripted partners. Each tick a partner moves beta-greedily toward its
assigned target among currently valid moves, or samples uniformly among them,
where per-partner `beta` in `[0, 1]` is hidden drift state clipped from the
`partner_drift` family. `beta = 1` is a fully cooperating partner; `beta = 0`
wanders. The target-anchor `drift` argument keeps its `CooperativeGrid-v0`
meaning and defaults to stationary targets.

Partners remain **world participants**: they block occupied cells, contend for
free cells, must reach their assigned targets for the joint +1 and respawn with
everyone. They are **not learner-controlled**: partner public action masks
allow only the placeholder stay action, partner observations are zero and
partner `active_mask` is False, so PPO/IPPO never record or train on them while
their peer positions stay visible in learner observations. Learner actions
passed for partner slots are ignored, whichever value they carry.

Partner actions draw from an independent RNG stream carried in state; world
(respawn) randomness, goal-drift randomness, partner-policy randomness and
learner sampling all advance separately. Per-partner parameters/phases are
privileged `partner_parameters`/`partner_phase`/`partner_changed` Oracle
fields, never ordinary info. Full lifetime/IPPO checkpoints retain the partner
drift state, its RNG stream and every learner leaf.

```bash
uv run --extra training python examples/train_marl.py \
  --backend partner --updates 3 --checkpoint checkpoints/partner-001
uv run --extra training python examples/train_marl.py \
  --backend partner --updates 2 --resume checkpoints/partner-001
```

Adversarial **opponents** exist in the same pattern through `OpponentGrid-v0`:
scripted opponents with hidden drifting aggression hunt their nearest learner.
A learner within one step of an opponent (including co-location) is caught:
collection is interrupted, it scores -1 and respawns. Rewards are strictly
zero-sum—opponent slots share the negated learner reward sum each tick:

```python
env = pp.make(
    "OpponentGrid-v0",
    size=5,
    num_learners=1,
    num_opponents=1,
    opponent_drift=pp.drift.Cyclic([(1.0,), (0.0,)], period=128),
)
```

Learners collect the shared drifting goal for +1 and respawn outside it.
Opponents block cells and contend for free cells like partners, except among
themselves: when only opponents propose the same free cell, the lowest slot
moves and the others stay, so two fully aggressive opponents cannot hold each
other in place. They are equally uncontrollable, never trained on (their slots take no gradient steps)
and visible as peers in learner observations. Their parameters/phases are
privileged `opponent_parameters`/`opponent_phase`/`opponent_changed` Oracle
fields, and full lifetime/IPPO checkpoints retain opponent drift state, its
independent RNG stream and every learner leaf. CLI: `--backend opponent`.
Opponents spawn uniformly on distinct cells (possibly on the goal or next to
a learner) and caught learners respawn outside the goal only, so an immediate re-catch is
possible; both are how the world is defined, not bugs.

## JaxMARL functional interoperability

The optional `jaxmarl` extra (JaxMARL 0.2.0, Apache-2.0) lifts JaxMARL's
jit-able MPE `simple_spread` into a native continuing lifetime with
reward-multiplier drift, three fixed agent slots with 18 observation features each, same-tick legacy
respawns and exact native checkpoints; see
[environment adapters](adapters.md) for the full contract. CLI:
`--backend jaxmarl`. With `num_scripted > 0` the last slots are scripted
partner/opponent agents whose hidden drifting parameters control whether they
move toward or away from their nearest landmark, while learners see them as
ordinary peers and never train on them (`--backend jaxmarl-scripted`). The
MEAL continual-MARL presets are documented in [continual benchmarks](benchmarks.md).

### More MPE tasks through the generic adapter

`JaxMARLMPE` covers all 12 other registered MPE tasks. Their ids keep
JaxMARL's spelling with an underscore before the version
(`jaxmarl/MPE_simple_tag_v3`). `jaxmarl/MPE_simple_spread-v3` is the one id
spelled with a hyphen; it runs through its own adapter with scripted partners,
described above. `JaxMARLMPE` pads each agent's
observation to the widest actual feature count (some FACMAC variants report
stale observation-space widths) and masks per-agent action counts or continuous
action dimensions. This supports `MPE_simple_speaker_listener_v4` (3 and 5
discrete actions), the continuous `facmac` family, and `world_comm` (one
9-dimensional and five 5-dimensional continuous action vectors). Shared
multi-agent SAC requires identical per-slot bounds and is tested on
`MPE_simple_facmac_3a_v1`. Reward drift works everywhere; dynamics drift is
offered only for the fields measured to work (next section).

Every multi-agent task also takes scripted partner or opponent slots, so
partner/opponent drift is not limited to `simple_spread`:

```python
env = pp.make(
    "jaxmarl/MPE_simple_push_v3",
    num_scripted=1,  # the last slot; or scripted_slots=(0,)
    scripted_behavior="opponent",  # or "partner"
    scripted_drift=pp.drift.Cyclic([(1.0,), (0.0,)], period=4096),
)
```

The policy uses only the reward, so it works on any task: each tick a scripted
slot scores each candidate action (every discrete action; no-op and each
full-strength dimension for continuous slots) by the learners' summed reward
over a two-tick lookahead with the learners' actual actions. Two ticks because
MPE moves bodies by their previous velocity, so an action first changes reward
one tick later. With probability `beta` a partner takes the best candidate for
the learners and an opponent the worst; otherwise the slot acts uniformly at
random. Per-slot `beta` is the hidden drift parameter `scripted`
(`scripted_parameters` through an Oracle). Scripted slots are inactive, observe
zeros, allow only the placeholder action and ignore actions passed for them.

Measured with random learners over 1,024 ticks (seed 0, learners' summed mean
reward; partner at beta 1, random at beta 0, opponent at beta 1):

| Task | Partner | Random | Opponent |
|---|---:|---:|---:|
| `simple_speaker_listener_v4` | −0.304 | −1.812 | −10.727 |
| `simple_push_v3` | 1.915 | 0.108 | −0.779 |
| `simple_adversary_v3` | −0.370 | −0.803 | −1.079 |
| `simple_reference_v3` | −0.525 | −1.137 | −2.514 |
| `simple_tag_v3` | 0.703 | 0.410 | 0.029 |
| `simple_world_comm_v3` (zero learner actions) | 3.615 | −0.365 | −0.476 |

The first four are checked in `tests/test_jaxmarl_mpe.py`. On the FACMAC tasks
the collision reward is sparse, most candidates tie and partners are not
reliably better than random. A speaker's message affects reward only through
the listener's later reaction, which a held-action lookahead cannot see. FACMAC
variants move their prey with a built-in policy and discard actions for them,
so those slots are always inactive (`backend_scripted_slots` in the metadata)
and cannot be scripted. `simple_v3` has one agent and no room for a scripted
slot.

### Which JaxMARL dynamics can drift

Each environment page lists its driftable fields; this section explains how
that list was measured. JaxMARL keeps dynamics on the environment object rather than in a parameter
argument, so Perpetua drifts them by binding traced values onto a shallow copy
of the environment inside a jitted step. `scripts/audit_jaxmarl.py` checks every
float attribute of every registered environment on a 12-tick jitted rollout and
classifies it:

- **driftable** — injecting the default reproduces the trajectory exactly and a
  perturbed value changes it;
- **inert** — injection is faithful but no effect was seen in this short
  constant-action rollout (it may still matter elsewhere; contact forces, for
  example, act only on collisions);
- **broken** — injection fails to trace or changes even the default trajectory,
  so it must never be offered as a drift field.

Measured on JaxMARL 0.2.0 (38 registered environments):

| Group | Result |
|---|---|
| MPE (13 environments) | 12 have driftable dynamics, typically `dt`, `mass`, `damping`, `rad`, `contact_margin`; `simple_crypto` has none. `simple_spread` drifts `local_ratio`, `dt`, `mass`, `damping`, `contact_margin`. Some MPE variants have broken attributes (`world_comm`, `tag`, the `facmac` family). |
| SMAX | 5 driftable (`time_per_step` and unit-type velocity, sight range, radius, health) |
| jaxnav | 5 driftable, 1 broken |
| `switch_riddle` | 1 driftable (`reward_all_die`) |
| storm 2p/np, hanabi, overcooked, overcooked_v2, coin_game | no float dynamics attributes to drift |
| `HeuristicEnemySMAX`, `storm` | the audit's rollout fails (RecursionError, KeyError) |
| MaBrax (5), JaxRobotarium (8), `LearnedPolicyEnemySMAX` | do not build here (missing `brax`, construction errors) |

The `simple_spread` adapter uses this list: `pp.make("jaxmarl/MPE_simple_spread-v3",
drift={"damping": pp.drift.Cyclic([0.25, 0.9], 512)})` drifts the physics, alongside
the existing `reward_multiplier` (`drift=`) and `scripted` (`scripted_drift=`)
fields. So the audit found injectable dynamics in 15 JaxMARL environments, all of them
measured rather than assumed; Perpetua registers the 13 MPE tasks, of which 12
have driftable dynamics. Every environment can still use reward and
observation drift. The audit is a script rather than a test because its table
tracks the installed JaxMARL version.

## PettingZoo parallel interoperability

Install the optional `marl` extra (PettingZoo 1.27 and MPE2 1.1.1) for either
compatibility direction; IPPO still uses the separate `training` extra.

```python
import perpetua as pp
from perpetua.adapters import PettingZooAdapter

env = pp.make("CooperativeGrid-v0", size=3, num_agents=2)
view = PettingZooAdapter(env, horizon=128)
obs, info = view.reset(seed=0)
actions = {
    name: view.action_space(name).sample(mask=info[name]["action_mask"]) for name in view.agents
}
obs, reward, terminated, truncated, info = view.step(actions)
view.close()
```

The view returns dictionaries for stable `agent_0`, `agent_1`, ... names, cached
per-agent spaces and int8 compatibility masks. Inactive slots remain named with
their public activity flag/placeholder action, not falsely terminated. An optional
horizon ends only a reporting segment and clears `agents` until `reset()`.
Unseeded reset observes the same world without consuming RNG; reseeding a running
lifetime is refused. Reset options are explicitly ignored and cannot reconfigure
native dynamics. The privileged PettingZoo `state()` API is not implemented.
The official parallel API test is exercised; episodic reseed semantics are not
claimed. The view's host-side segment bookkeeping is not a native checkpoint API.

The upward `PettingZooLifetime` lift accepts parallel environments with homogeneous
Box/zero-based Discrete spaces, including masked Box dictionaries. Agent names
define a fixed superset of float32 observation/action/reward slots. Removed slots
are inactive/zero padded; activity and action masks are preserved. Legacy info is
discarded except masks, activity and respawn signals—no task IDs/done flags leak.
When every legacy agent has finished, the lift resets the legacy world on the
same tick and reports `respawn` for every active slot. A lifetime-preserving
view (a `PettingZooAdapter` with a `horizon`, lifted again) is the exception:
its reset only reopens a reporting segment, so `respawn` is then what the
wrapped world reported on that tick, not a flag for every slot.

`AECLifetime` lifts **sequential** AEC environments with the same homogeneous
Box/Discrete restriction. One Perpetua tick drives one full AEC cycle in
selector order, mirroring PettingZoo's official parallel conversion, and
accumulates each agent's reward across the cycle. A terminated or truncated
agent's slot deactivates on that tick (zero-padded observations,
placeholder-only masks) until the legacy world respawns, and the agent is sent
`None` whenever the selector reaches it. An agent that dies before its turn in
a cycle does not act in it, and the action submitted for its slot is dropped.
Actions of inactive slots are ignored, as in the parallel lift. Agent
reordering is refused rather than guessed. Turn-based games are not supported:
an environment whose `metadata["is_parallelizable"]` is `False` (PettingZoo
classic's `connect_four_v3` or `tictactoe_v3`, for example) raises a
`ValueError` when the lift is built. An environment that does not declare the
key is accepted, so a turn-based environment of your own must declare it.
Tests check the lift against the parallel version of the same environment:
the same seed and joint actions give identical observations, rewards and
respawns.
Legacy state/RNG is opaque and single-use; checkpoint attempts are refused.

When all legacy agents finish, the backend internally resets on the same tick,
retaining the transition reward and advancing the Perpetua clock. Ordinary resets
use the supplied step key's derived seed. A known lifetime-preserving view instead
receives unseeded segment resets. Surviving agents are not reset when one leaves.

```bash
uv run --extra training --extra marl python examples/train_marl.py \
  --backend pettingzoo --updates 2
```

This example uses real MPE2 `simple_spread` physics through PettingZoo's `EnvSpec`
creation API and the **same** IPPO implementation in `HostPPORunner`. Host stepping
and transfers are explicit. One host controller owns each external simulator;
stale or failed mutable states are not replayable. MPE2 simulator/RNG state is
opaque by default, so complete host-lifetime checkpoint attempts are refused.
The opt-in [MPE2 snapshot codec](pettingzoo-snapshots.md) supports full `simple_spread`
state/RNG/IPPO continuation through `snapshot=True` or CLI `pettingzoo-snapshot`.

## Limits

- PettingZoo lifts need homogeneous `Box` or zero-based `Discrete` spaces
  across agents; heterogeneous spaces are refused.
- Inactive native slots are placeholders for availability: the grids have no
  automatic schedule of agents being born or dying.
- No multi-agent throughput or benchmark-level learning result is claimed on
  this page, and bitwise resume is guaranteed only on the same software and
  hardware.
