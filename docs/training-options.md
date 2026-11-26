# Learner options for continual learning

Options of the bundled learners that matter when the world keeps changing:
keeping a network able to learn, learning from pixels, following a
task-sequence protocol, and learning average reward. All are off by default.
[Training learners](training.md) covers choosing and running a learner.

## Plasticity options

`PPOConfig` exposes the reference plasticity mechanisms on PPO and IPPO:

```python
import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
config = PPOConfig(activation="crelu", l2_coefficient=1e-5, reset_interval=16)
trainer = PPORunner(env, PPO(action_size=5, config=config))
```

The same three options exist on the off-policy learners through `DQNConfig`
(`DQN` and `IQL`) and `SACConfig` (`SAC`):

```python
from perpetua.agents import DQN, DQNConfig, DQNRunner

config = DQNConfig(activation="crelu", l2_coefficient=1e-5, reset_interval=16)
trainer = DQNRunner(env, DQN(action_size=5, config=config))
```

- `activation="crelu"` replaces tanh with concatenated ReLU
  (`[relu(x), relu(-x)]`), widening each hidden output to `2 * hidden_size`.
  Parameter shapes differ from tanh, so checkpoints do not interchange
  across activations.
- `l2_coefficient` adds `coefficient * sum(params**2)` to the loss: PPO's
  single loss, DQN's Huber loss, and both SAC's critic and actor losses
  (a plain L2 regulariser toward zero, not toward the initial weights).
- `reset_interval` schedules a periodic reset of the final layers, as in
  Nikishin et al. (2022, primacy bias). It resets the deepest
  `reset_last_layers` (default 2) Dense layers, counting actor and critic
  heads, plus a LayerNorm on a reset layer: fresh parameters **and** a fresh
  Adam state (moments and step count) are grafted from a re-initialization
  while the other layers (and any CNN torso) and the gradient clock are
  preserved exactly. The reset layers keep their own Adam step count, so after
  a reset their steps are bias-corrected as in a new network. PPO creates its
  actor and critic layers alternately (`Dense_0`/`Dense_2` actor hidden,
  `Dense_1`/`Dense_3` critic hidden, `Dense_4` policy head, `Dense_5` value
  head), so the default resets both heads, 4 also resets both second hidden
  layers, and an odd count resets the critic's layer at that depth but not the
  actor's. In SAC the count is a depth that means the same in every network:
  the actor's mean and log-std heads read the same hidden layer and count as
  one layer, so 1 resets the heads (both actor heads and each critic's value
  head) and the default 2 resets each network's last hidden layer and its
  heads. DQN's default resets its last hidden layer and Q head.
- `reset_interval` units differ: PPO and IPPO count runner updates (one per
  rollout), while DQN, IQL and SAC count `update` calls, that is gradient
  steps (`gradient_steps` per rollout). The age also advances on steps
  skipped because every sampled slot was inactive. The reset age is explicit
  checkpointed state, and when resets are disabled no extra RNG is consumed.
  DQN also grafts the same fresh top layers into the target network so
  bootstrapping does not immediately fight the reset; SAC grafts the actor,
  both online critics and their target copies (this reset never touches the
  temperature). IQL carries a per-slot reset age, like IPPO.

Mechanism tests cover the CReLU transform, exact layer grafting (parameters
and Adam moments), the schedule boundary, IPPO/IQL per-slot reset ages and
exact checkpoint continuation with all options enabled. Whether they reduce
loss of plasticity over very long lifetimes has not been measured here; the
dormant-unit and effective-rank probes in [evaluation](evaluation.md) are the
tools to measure it.

## Pixel observations: CNN torso

`PPOConfig(torso="cnn")` and `DQNConfig(torso="cnn")` (also used by IPPO and
IQL) replace the MLP torso with a small Nature-CNN-style encoder
(`perpetua.agents.encoders.CNNTorso`): three 3x3 convolutions, two of them
stride-2, then a dense feature layer. Observations are channel-last,
`(agents, H, W, C)`; the torso preserves every leading axis (batch and agent
slots) and flattens only spatial/channel axes, so the existing heads, rollouts
and replay buffers work unchanged. EnvPool Atari emits `(84, 84, 4)`
per slot, with the four stacked frames as channels. The default is `torso="mlp"`. Replay buffers accept `obs_dim`
as an integer or a shape tuple, so DQN/IQL replay stores full pixel frames.
`SAC` takes flat observations only.

The torso divides every observation by one number, the pixel scale, which is
fixed when the learner is built and never depends on what a frame holds:

```python
minatar = pp.make("gymnax/Breakout-MinAtar")
pixel_agent = PPO(
    action_size=3,
    config=PPOConfig(torso="cnn"),
    observation_space=minatar.observation_space,
)
assert pixel_agent.pixel_scale == 1.0  # 255.0 on EnvPool Atari
```

- With `observation_space`, the scale is 255 when the space is a `Box` whose
  bounds are 0 and 255 in every entry (raw 8-bit pixels: EnvPool Atari, a
  Gymnasium `uint8` image environment, AgarCL's screen), and 1 otherwise
  (MinAtar and gymnax Pong, MEAL Overcooked, COOM's already normalised frames,
  AgarCL's grid).
- `PPOConfig(pixel_scale=...)` or `DQNConfig(pixel_scale=...)` sets it
  yourself, to any finite positive number, whatever the space says.
- With neither, the scale is 1.

A runner refuses, when it is built and again at `init`, a learner whose
derived scale is not the one its environment's space gives (a scale of 1 on a
0-255 space, or 255 on any other), and names both remedies, so raw pixels
never reach the network unscaled by accident. A configured `pixel_scale` is
your choice and is not checked. The derivation reads the bounds only: a
0-255 space that holds symbolic indices rather than pixel intensities also
derives 255, unless you set `pixel_scale=1`.

The scale in use is `agent.pixel_scale`, and a runner records it as
`runner.metadata["learner"]["pixel_scale"]`, which a checkpoint stores.

Tests cover torso shapes, PPO and DQN updates on `gymnax/Breakout-MinAtar`
(finite losses, correct replay shapes), and a checkpoint round-trip. Learning
from pixels has not yet been measured against a random baseline.

## SAC options for task-sequence protocols

`SACConfig` has opt-in options that reproduce protocols such as Continual
World's; all are off by default.

- `hidden_layers`, `layer_norm` (Dense, LayerNorm and tanh on the first layer)
  and `activation="leaky_relu"` set the network.
- `random_steps` acts uniformly within the bounds for the first ticks, and
  `update_after` delays learning. Both count ticks from the start of the
  lifetime and, only with `task_boundary_reset=True`, from each task switch.
  `random_steps` is checked every tick; `update_after` is checked once per
  runner call after the rollout, so learning starts at rollout granularity.
- `task_boundary_reset=True` empties the replay buffer and restarts the
  optimizer moments and the temperature when the Oracle's `boundary` fires.
  A runner call whose last tick empties the buffer does not learn.
  It refuses to run unless the environment is wrapped in
  `pp.Oracle(env, expose=["boundary", ...])`, so the privilege is recorded.
- `mask_respawns=True` marks transitions that cross a respawn inactive, so a
  fresh episode's first observation never enters a TD target. Inactive
  transitions are also left out of the actor and temperature losses. A
  transition that waits while its slot is inactive (see
  [slots that go inactive](training.md#slots-that-go-inactive)) crosses a
  respawn when the slot is flagged on any tick from the one it acted on to
  the one that completes it, not only on the completing tick.

`pp.OneHotObservation(env, field="task_id", size=n)` appends an exposed Oracle
field to the observation as a one-hot code and records it in the metadata.

SAC's squashed log-probability uses the exact tanh correction,
`2 * (log 2 - u - softplus(-2u))`, rather than `log(1 - tanh(u)**2 + 1e-6)`.
The floored form has no gradient once tanh saturates, so a policy whose mean
ran away could not recover and the automatic temperature grew until values
became non-finite. With the exact
form the entropy term pulls the mean back. The Gaussian part of the
log-probability is computed from the sampled noise directly, so it stays exact
when the policy is nearly deterministic (recomputing the noise from the action
loses it to float32 rounding once the log standard deviation is below about
-14).

## Average reward: reward centering

`PPOConfig`, `DQNConfig` and `SACConfig` accept `reward_centering=True`
(default off). Each learner then keeps a running average-reward estimate
`reward_rate` in its state (per agent for IPPO/IQL) and uses
`r - reward_rate` in GAE and TD targets. After every update it moves the
estimate by `centering_step_size * mean(TD error)` (Naik et al., 2024,
value-based centering), which is robust off-policy. `centering_step_size` is a
traced hyperparameter (default 0.01), so it can be swept like learning rate
without recompiling.

It is off by default, and turning it off leaves training exactly as without
the feature. Tests cover offset invariance (adding a constant to every reward
while starting the estimate at that constant leaves updates unchanged),
convergence of the estimate to a known average reward, and bounded advantages
at gamma near 1.

For SAC the TD error that moves the estimate is taken against the entropy
augmented target, so `reward_rate` is the average of the reward plus the
discounted entropy bonus: it settles at the reward plus `gamma * alpha` times
the policy's entropy (about 1.12 rather than 1.0 on a replay whose rewards are
all 1.0, at `alpha=0.2` and `gamma=0.9`, where the per-batch entropy statistic
is about 0.6 and varies from batch to batch). It equals the average
reward only when `alpha` is zero, so do not compare it with `metrics.mean`.
For PPO and IPPO it is the average reward per tick, including on slots that
sit out ticks (see [slots that go inactive](training.md#slots-that-go-inactive)).

### When DQN needs it

Turn it on for DQN when the task pays about the same reward on every tick, as
survival tasks do. A CartPole lifetime pays +1 on each tick the pole is up, so
with `gamma=0.99` every state is worth roughly 80 to 90 whatever the action,
and the difference between actions is a small fraction of that. Without
centering the Q-network has to fit that common offset (its mean Q-value was
about 80 to 90 in the runs below, against 2 to 10 with centering), and
training can end below where it started. PPO did not show this in the same
setting; it normalises advantages.

Measured as the mean reward per tick over the last 1,024 ticks (1.0 would mean
the pole never falls; the uncentered runs started near 0.95):

| Environment | Seeds | `reward_centering=False` | `reward_centering=True` |
|---|---:|---|---|
| `gymnax/CartPole-v1` | 3 | 0.897, 0.994, 0.896 | 0.998, 0.996, 0.993 |
| `envpool/CartPole-v1`, 64 slots | 5 | 0.906 on every seed | 0.975, 0.978, 0.913, 0.976, 0.969 |

These are single untuned configurations (about 123,000 ticks on gymnax and
9,600 ticks of 64 slots on EnvPool), recorded on 2026-11-17. Centering does
not make DQN match PPO on EnvPool CartPole, which reached 0.995 in the same
setting, and one EnvPool seed rose to 0.984 and fell back. SAC and other
survival tasks were not measured.

**Under compute-aware time** (`ComputePPORunner`) one decision spans `k` world
ticks and its reward is the discounted sum of those ticks' rewards. Centering
then subtracts `reward_rate * (1 - gamma**k) / (1 - gamma)` (`reward_rate * k`
when `gamma` is 1), the average reward once per tick, and moves the estimate
by the per-tick TD error (the TD error divided by that count), so
`reward_rate` stays an average reward per tick. With one tick per decision
both factors are 1 and nothing changes. Tests cover offset invariance with
decisions of 1 to 5 ticks and convergence of the estimate to the per-tick
reward.

**When a slot sits out ticks** PPO's TD error for the tick before the stretch
is bridged to the slot's next action, so it contains the centered rewards of
every tick in between and moves by `-(1 + gamma + ... + gamma**(k-1))` per
unit of `reward_rate` for a span of `k` ticks. PPO divides each bridged error
by that discounted span count before averaging, for the same reason as above:
undivided, the effective step is `centering_step_size` times the count, and
the estimate diverges once that exceeds 2 (step 0.1, `gamma` 0.99 and a slot
acting every 32nd tick gives 2.75). The two cases compose: when decisions also
span several world ticks, the divisor is the sum over the steps of the span of
each step's tick count, discounted by the ticks before it, which is
`(1 - gamma**K) / (1 - gamma)` for the `K` world ticks of the whole span.
Without inactive ticks the divisor is the per-decision count and the update is
unchanged. DQN, IQL and SAC do not do this: see
[slots that go inactive](training.md#slots-that-go-inactive). The [reward centering tutorial](tutorials/reward-centering.md) shows
what it changes in practice.
