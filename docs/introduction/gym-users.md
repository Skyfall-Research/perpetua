# Perpetua for Gymnasium users

Perpetua keeps Gymnasium's names (`make`, `step`, `obs`, `action`, `reward`,
`info`, spaces) and changes what an interaction *is*: one lifetime that never
resets, instead of a sequence of episodes. This page maps your Gymnasium and
PettingZoo habits onto Perpetua and shows how to port a training loop.

## The mapping

| Gymnasium | Perpetua | Why |
|---|---|---|
| `env = gym.make(id)` | `env = pp.make(id)` | Same registry idea; drift is an argument (`drift=`). |
| `obs, info = env.reset(seed=s)` | `state, obs, info = env.init(key)` | Called **once per lifetime**. State and randomness are explicit. |
| `obs, r, terminated, truncated, info = env.step(a)` | `state, obs, reward, info = env.step(state, action, key)` | A pure function: no hidden state, no episode flags. |
| `terminated` / `truncated` | `info["respawn"]` | Death or a goal is an event inside the lifetime, not a boundary. |
| `TimeLimit` | none | Lifetimes have no length. Native worlds have no time limit. Most gymnax worlds drop theirs, but the bandits (`BernoulliBandit-misc`, `GaussianBandit-misc`), `DiscountingChain-bsuite` and the JaxMARL MPE tasks keep a fixed-length limit, which appears as a respawn (every 100 ticks for the bandits, measured). A CPU-lifted Gymnasium or EnvPool env keeps its own limit, and reaching it is a respawn (the limit is recorded as `legacy_time_limit` in `metadata["configuration"]["backend"]`). |
| Autoreset modes in vector envs | none | Nothing resets, so there is no reset branch to compute. |
| `reset(seed=s)` for reproducibility | `jax.random.key(s)`, split for every use | One seed reproduces the whole lifetime (see [seeding](#seeding)). |
| `Discrete(n)` action, scalar reward | `MultiDiscrete([n] * agents)`, rewards of shape `(agents,)` | Every array has a leading agent axis, even with one agent, so single- and multi-agent code is the same. |
| `env.action_space.sample()` | `pp.policies.random(obs, info, key)` for discrete actions; `pp.policies.uniform(env.action_space)` builds the equivalent policy for `Box` actions | Spaces are ordinary Gymnasium spaces (NumPy sampling); inside compiled JAX code, sample with a key. |
| `Box` action | A float array of shape `(agents, action_dim)` | `SAC` is the learner for continuous actions. |
| `env.render()`, `render_mode` | none | There is no renderer; inspect `state` directly. `GymnasiumAdapter` gives Gymnasium tooling a standard `reset`/`step` view, but declares no render modes. |
| `env.close()` | `env.close()` on host simulators only | Meta-World, EnvPool, COOM, AgarCL and MORPHEUS hold a CPU simulator or worker process, so they have `close()`. Native and gymnax worlds hold nothing and have no `close` method, so call it only on host simulators (`pp.Oracle` and `pp.OneHotObservation` accept `close()` and forward it only if the inner environment has one). |
| Wrappers (`ObservationWrapper`, ...) | An object with the same `init`/`step`/`observe`/`privileged` methods and `num_agents`, `observation_space`, `action_space`, `metadata` and `oracle_fields` attributes, which calls the inner env | `pp.Oracle` and `pp.OneHotObservation` are wrappers written this way. To change dynamics over time, use drift instead. |
| Episode return, `RecordEpisodeStatistics` | `pp.metrics`: lifetime mean, window, EMA, regret, recovery | Performance *is* online reward. A per-life return is easy to compute if you want it ([below](#hand-written-loop)). |
| Separate evaluation env | none by default | The agent is always deployed; `pp.evaluation.revisit` exists for protocols that define a revisit. |
| Task ID in `info` | Hidden; `pp.Oracle(env, expose=["task_id", "boundary"])` | Privilege is opt-in and recorded in `metadata["oracle_exposure"]`. |
| `gym.vector.AsyncVectorEnv` | `jax.vmap(runner)`; batched slots for CPU simulators | Lifetimes batch as data, including different drift schedules (see [seeding](#seeding) for the two lines). |

(hand-written-loop)=
## A hand-written loop

This is the closest equivalent of a Gymnasium `while True: env.step(...)` loop.
It runs one step at a time from Python, which is easy to read and debug but
slow; the runners below compile the whole loop.

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], 127))
step = jax.jit(env.step)  # compile one tick once

key = jax.random.key(0)
key, init_key = jax.random.split(key)
state, obs, info = env.init(init_key)  # once per lifetime; there is no reset

life_return = jnp.zeros(env.num_agents)  # reward collected since the last respawn
returns = []
for tick in range(500):
    key, act_key, step_key = jax.random.split(key, 3)
    action = pp.policies.random(obs, info, act_key)  # shape (agents,)
    state, obs, reward, info = step(state, action, step_key)
    life_return = life_return + reward
    if bool(info["respawn"].any()):  # the agent reached the goal and respawned
        returns.append(float(life_return[0]))
        life_return = jnp.where(info["respawn"], 0.0, life_return)

print(obs.shape, reward.shape, len(returns))  # (1, 4) (1,) and the number of lives
```

`obs` has shape `(agents, features)` and `reward` has shape `(agents,)`. A
respawn does not end anything: the next `step` continues from the respawned
state, and the drift schedule keeps running.

## Your own policy in a runner

A policy is any function `policy(obs, info, key) -> action` that JAX can trace,
returning one action per agent. Pass it to `Runner`, which compiles the policy,
the world and the metrics into one program:

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], 127))


def mostly_greedy(obs, info, key):
    """Walk to the visible goal, but take a random action 10% of the time."""
    explore_key, random_key = jax.random.split(key)
    explore = jax.random.uniform(explore_key, (env.num_agents,)) < 0.1
    return jnp.where(
        explore,
        pp.policies.random(obs, info, random_key),
        pp.policies.greedy(obs, info, key),  # greedy ignores its key
    )


runner = pp.runtime.Runner(env, mostly_greedy, ticks=256)
carry = pp.runtime.init_lifetime(env, seed=0)
for _ in range(4):
    carry = runner(carry)  # continue the same lifetime; never initialize again
print(carry.state.t.to_int(), carry.metrics.mean)  # 1024 ticks, mean reward per agent
```

The `carry` holds everything the loop threads from call to call: the world
state, the last observation and info, the random keys and the metrics. Calling
the runner again continues the lifetime exactly where it stopped. A neural
network policy works the same way: close over its parameters, or use one of the
bundled learners below.

(seeding)=
## Seeding

Randomness in JAX is explicit: a *key* goes into every random operation, and
you split a key to get new independent ones (`key, sub = jax.random.split(key)`)
instead of reusing it. Reusing a key repeats the same random numbers.

`pp.runtime.init_lifetime(env, seed=s)` splits one seed into the world's
initial key and separate streams for the environment and the agent, so `seed`
reproduces the whole lifetime: the same seed and code give the same result.
`seed` is an integer in `[0, 2**32)`; anything else (a float, a negative or
larger integer) raises `ValueError`, and a
PRNG key is refused (start an environment from a key with `env.init(key)`).
Seeds are also data, so `jax.vmap` over an array of seeds runs independent
lifetimes in one compiled call. This is what replaces a vector environment:

```python
import jax
import jax.numpy as jnp

import perpetua as pp

env = pp.make("ContinualGrid-v0", size=4)
runner = pp.runtime.Runner(env, pp.policies.random, ticks=256)
carries = jax.vmap(lambda seed: pp.runtime.init_lifetime(env, seed=seed))(jnp.arange(8))
carries = jax.vmap(runner)(carries)  # eight lifetimes; carries.metrics.mean is (8, agents)
```

## Learners and respawns

The bundled JAX learners (PPO, IPPO, DQN, IQL, SAC) follow the same split:
`agent.act(agent_state, obs, info, key)` returns an action (with log-probabilities
and values for PPO), and runners such as `PPORunner` collect a rollout and call
`agent.update` between chunks. Learner state, optimizer state, replay buffers and
the lifetime form one pytree, so a checkpoint resumes everything exactly.

A lifetime is one continuing task, so the learners **keep bootstrapping across
a respawn** (bootstrapping is using the predicted value of the next state in
the learning target): there is no `done` mask that zeroes that value.
PPO's advantages (GAE) and DQN's targets continue through respawns, drift
changes and rollout boundaries. `SACConfig(mask_respawns=True)` instead drops the
transitions that cross a respawn, for protocols that treat a respawn as a
fresh start. See [training](../training.md) for each learner.

## PettingZoo users

Every Perpetua environment is multi-agent shaped: agents are slots on the
leading axis, `info["active_mask"]` says which slots are taking part, and
`info["action_mask"]` says which actions are legal. How PettingZoo's APIs map:

| PettingZoo | Perpetua | Notes |
|---|---|---|
| Parallel API, `step(actions_dict)` | one tick = one `step(state, actions, key)` with an `(agents,)` action array | `PettingZooLifetime` lifts a parallel env in. |
| AEC API, agents act in turn | one tick = one full cycle of agent turns | `AECLifetime` lifts an AEC env in; each agent's rewards over the cycle are summed. |
| `agents` that come and go | a fixed set of slots, with `info["active_mask"]` | The lifts zero-pad inactive slots, and the learners do not train on them. |
| `env.state()` (global state) | not provided; world-side fields go through `pp.Oracle(env, expose=[...])` | Privileged information is opt-in and recorded. |

## Using existing code

- **A Gymnasium env inside Perpetua:** `GymnasiumLifetime(gym.make(...))` from
  `perpetua.adapters`, with `compiled=True` to run it inside the compiled
  runners. Gymnasium is part of the core install.
- **A Perpetua env for Gymnasium tools:** `GymnasiumAdapter(env, horizon=...)`
  adds synthetic truncations for legacy tooling without resetting the world.
- **PettingZoo:** `PettingZooLifetime` and `AECLifetime` (from
  `perpetua.adapters`, needs the `marl` extra) lift environments in;
  `PettingZooAdapter` exposes a Perpetua environment as a parallel API.
- **PyTorch agents:** `perpetua.torch_bridge.TorchRunner` (the `torch` extra).

See [adapters](../adapters.md) for the details and limits, and
[multi-agent](../marl.md) for interacting worlds.
