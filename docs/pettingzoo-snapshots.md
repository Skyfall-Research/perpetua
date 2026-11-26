# PettingZoo lifetime checkpoints

```{admonition} Experimental
:class: warning

This feature is kept working and tested, but it is frozen: no new work is
planned on it, and its interface may change. Build on it with that in mind.
```

## Supported codec

The opt-in codec supports **headless MPE2 1.1.1 `simple_spread`**, discrete actions,
PettingZoo 1.27.0 and owned PCG64 RNG. The parallel wrapper and any underlying
`OrderEnforcingWrapper` / `AssertOutOfBoundsWrapper` must be exact known types.
Unknown wrappers/subclasses, rendering, benchmark-info payloads, scripted agents,
nonzero action/communication noise and other environments are explicitly refused.

```python
from pettingzoo import EnvSpec, make
from perpetua.adapters import PettingZooLifetime

spec = EnvSpec(
    id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
)
env = PettingZooLifetime(make("parallel", spec, N=3, max_cycles=128), snapshot=True)
```

Default `snapshot=False` behavior is unchanged: mutable single-use state,
explicit host ownership and no incomplete lifetime checkpoints. No new dependency
or mandatory rendering library is added; the existing `marl` extra supplies MPE2.

```bash
uv run --extra training --extra marl python examples/train_marl.py \
  --backend pettingzoo-snapshot --updates 3 --checkpoint checkpoints/mpe-001
uv run --extra training --extra marl python examples/train_marl.py \
  --backend pettingzoo-snapshot --updates 2 --resume checkpoints/mpe-001

uv run --extra training --extra marl python examples/compute_ppo.py \
  --backend pettingzoo-snapshot --chunks 2 --checkpoint checkpoints/mpe-timed-001
uv run --extra training --extra marl python examples/compute_ppo.py \
  --backend pettingzoo-snapshot --replay checkpoints/mpe-timed-001
```

## Snapshot contents

`MPE2Snapshot` contains owned NumPy arrays, not simulator objects or Python pickles.
For N agents/N landmarks:

| Field | Shape / dtype | Contents |
|---|---|---|
| `version` | scalar uint32 | Codec schema version |
| `configuration` | uint8 JSON bytes | Versions, wrapper layout, agent names and static dynamics/settings |
| `physics` | `(2*N, 4)` float64 | Entity positions and velocities |
| `agent_state` | `(N, 6)` float64 | Communication, physical and communication action vectors |
| `action_present` | `(N, 2)` bool | Optional cached action-vector presence |
| `current_actions` | `(N,)` int64 | Queued discrete actions, -1 for None |
| `rng` | `(10,)` uint32 | PCG64 state/increment limbs and cached uint32 flag/value |
| `agents`, `selector` | `(2, N)` / `(N,)` int32 | Parallel/raw agent lists and AEC selector ordering |
| `counters` | `(6,)` int64 | Steps, selector cursor/selection, agent selection, dead-step skip and stage |
| `rewards` | `(N, 2)` float64 | Rewards and cumulative rewards |
| `done` | `(N, 2)` bool | Termination/truncation flags |
| `present` | `(N, 5)` bool | Per-agent rewards/cumulative/done/info dictionary membership |
| `wrappers` | `(wrappers, 2)` bool | Reset/update order-enforcement flags |
| `camera` | scalar float64 | Numeric original camera range; graphics resources are not serialized |
| `active`, `mask` | `(N,)` / `(N, actions)` bool | Perpetua adapter activity/action-mask state |

Float64 physical state remains NumPy/Orbax data; it is never reconstructed from
float32 agent observations. No process-global JAX x64 policy is enabled. Global
NumPy noise is not captured because it belongs to the whole process; configurations
using it are refused rather than perturbing unrelated environments' RNG streams.

## Restore and timing

Restore validates version, configuration, every array shape/dtype, finite numeric
state, memberships, selector state, cached RNG flags and masks **before** mutation.
The entire RNG and all known AEC bookkeeping are restored into a fresh initialized
simulator. Partial AEC action state can be captured/restored directly, but lifted
parallel runners checkpoint only at returned joint-tick/update boundaries. Do not
mutate/step the raw simulator independently while a Perpetua host runner owns it.

Snapshot-backed CPU states contain no opaque owner token. Each host transition
rehydrates its snapshot before stepping; sequential replay/retry and fresh-instance
continuation therefore use saved state rather than whatever remains in a mutable
object. Observation/mask queries read portable state without mutating the simulator.
One host controller owns each backend; concurrent Python restore/step pairs are
not promised atomic.

The same checkpoint APIs retain IPPO parameters/moments, drift, sampling/shuffle
keys, streaming metrics and—when timed—held commands and pending on-policy buffers.
Legacy respawns and success termination remain world events inside a continuing
lifetime, not learning reset/done boundaries. Static physics or wrapper settings
changed after construction are refused instead of mislabeled as the same experiment.

`HostPPORunner` kernels record `host_compiler_options` in learner metadata and
use per-function `xla_gpu_exclude_nondeterministic_ops=True` /
`xla_gpu_autotune_level=0`. GPU autotuning otherwise changes optimizer rounding
across fresh processes despite identical physics/RNG. These settings may cost
throughput; native compiled `PPORunner` chunks and recorded timed replay apply
the same per-function options, and the library sets no process-global flags.
Strict expected-metadata validation refuses a checkpoint whose compilation
profile differs from the runner's, rather than relabeling it under other
compilation settings.

## Version limits

MPE2 1.1.1 inherits a `BaseScenario.curriculum_stage` property whose setter is a
no-op and getter returns zero. Its `set_curriculum_stage()` therefore cannot advance
the real stage. The codec preserves actual stage-zero behavior and refuses injected
nonzero stages; it does not repair or claim functioning curriculum control upstream.

This is one complete simulator-specific codec, not arbitrary PettingZoo snapshot
support. Other tasks/wrappers need their own validated schemas. Cross-hardware
bitwise continuation, rendering restoration, throughput gains and sanitizer-level
assurance are not claimed. Keep the lockfile with every checkpoint; the
checkpoint itself records the source revision.

