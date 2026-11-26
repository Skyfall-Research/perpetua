# Perpetua

Perpetua is a JAX library of **never-ending, changing worlds** for
reinforcement learning research, and the tools to study agents that keep
learning in them. It is for researchers working on continual RL,
non-stationarity, plasticity and continual multi-agent RL, and for developers
who want to add such worlds.

Most RL training assumes a fixed task split into episodes. Perpetua changes
three things:

- **One lifetime, no resets.** `init` once, `step` forever. A death or a reached
  goal is a *respawn* reported in `info["respawn"]`, not the end of an episode.
- **Drift.** The world changes through parameters that move over time (a goal,
  gravity, a partner's behaviour), set with `drift=` when the environment is
  made. [What drifts](https://github.com/Skyfall-Research/perpetua/blob/main/docs/benchmarks.md#what-drifts-in-wrapped-benchmarks) depends on the
  environment family.
- **Privilege is opt-in.** The agent is not told when the world changed. An
  `Oracle` can expose that, and the choice is recorded with the results.

**Status:** pre-release. Not on PyPI yet (the name `perpetua-rl` is
provisional). Perpetua is released under the [MIT licence](https://github.com/Skyfall-Research/perpetua/blob/main/LICENSE).

## Install

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Skyfall-Research/perpetua.git
cd perpetua
uv sync --locked --extra training   # core + the PPO/DQN/SAC learners
uv run python -c "import perpetua as pp; pp.testing.check_env(lambda: pp.make('ContinualGrid-v0')); print('ok')"
```

On macOS this needs Apple Silicon and an arm64 Python, because JAX publishes
no Intel macOS wheels: if `uv sync` reports that `jaxlib` has no wheel for the
platform, run `uv python install cpython-3.12-macos-aarch64-none` and add
`--python cpython-3.12-macos-aarch64-none` to the `uv sync` line. macOS is
checked by hand; CI runs on Linux only.

Other environments and backends are optional extras (`interop` for gymnax,
`marl` for PettingZoo, `cuda13` for NVIDIA GPUs on Linux, ...); see
[installation](https://github.com/Skyfall-Research/perpetua/blob/main/docs/introduction/installation.md).

## A first lifetime

```python
import perpetua as pp

# The goal alternates between two corners every 127 ticks (environment steps).
env = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], 127))
runner = pp.runtime.Runner(env, pp.policies.random, ticks=1024)  # 1,024 ticks, compiled
carry = runner(pp.runtime.init_lifetime(env, seed=0))
# [0]: agent slot 0. Metrics have a leading agent axis, even with one agent.
print(carry.state.t.to_int(), float(carry.metrics.mean[0]))  # ticks so far, mean reward
```

This prints `1024` and the random policy's mean reward per tick, about 0.005:
random moves rarely find the goal. The [quickstart](https://github.com/Skyfall-Research/perpetua/blob/main/docs/tutorials/quickstart.md)
trains PPO through goal switches and measures recovery and forgetting, in about
five seconds on a laptop CPU.

## Coming from Gymnasium or PettingZoo

| Gymnasium | Perpetua |
|---|---|
| `obs, info = env.reset(seed=s)` | `state, obs, info = env.init(key)`, once per lifetime |
| `obs, r, terminated, truncated, info = env.step(a)` | `state, obs, reward, info = env.step(state, action, key)`, a pure function |
| `terminated` / `truncated` | `info["respawn"]`; the lifetime continues |

[Perpetua for Gymnasium users](https://github.com/Skyfall-Research/perpetua/blob/main/docs/introduction/gym-users.md) covers porting a
training loop, seeding, wrappers and PettingZoo's parallel and AEC APIs.

## What's included

- **53 environments in six families:** native continual worlds, multi-agent
  worlds, gymnax and MuJoCo MJX control, MinAtar and bsuite, the continual
  benchmarks MEAL, Continual World, COOM, AgarCL and MORPHEUS, and an EnvPool
  CPU bridge. See the [environments](https://github.com/Skyfall-Research/perpetua/blob/main/docs/environments/index.md).
- **Drift and task switching:** cyclic, random-walk, procedural and
  agent-driven drift, per-lifetime schedules, and several environments in one
  lifetime. See [lifetimes and drift](https://github.com/Skyfall-Research/perpetua/blob/main/docs/native-api.md) and
  [task switching](https://github.com/Skyfall-Research/perpetua/blob/main/docs/task-switching.md).
- **Learners:** continuing PPO/IPPO, DQN/IQL and SAC in JAX, plus a PyTorch
  bridge. See [training](https://github.com/Skyfall-Research/perpetua/blob/main/docs/training.md).
- **Measurement:** streaming lifetime metrics, recovery, forgetting, regret,
  plasticity probes and IQM reporting. See [evaluation](https://github.com/Skyfall-Research/perpetua/blob/main/docs/evaluation.md).
- **Checkpoints:** atomic full-state checkpoints (world, drift, keys, metrics,
  learner) and exact segment replay.
- **Adapters:** Gymnasium, PettingZoo and gymnax environments in, and Perpetua
  environments out to Gymnasium and PettingZoo tools. See
  [adapters](https://github.com/Skyfall-Research/perpetua/blob/main/docs/adapters.md) and [multi-agent](https://github.com/Skyfall-Research/perpetua/blob/main/docs/marl.md).
- **Writing your own environment:** a four-step
  [tutorial](https://github.com/Skyfall-Research/perpetua/blob/main/docs/tutorials/custom-environment/index.md) and the
  `pp.testing.check_env` contract checker.

## Performance

On one NVIDIA H200 GPU, PPO on the JAX-native CartPole lifetime processes 21×
as many environment steps per second as PyTorch PPO on one Gymnasium
`AsyncVectorEnv` worker. With 128 lifetimes trained side by side under
`jax.vmap` it processes 41–49× as many as that baseline with 128 workers (the
range spans two measurements). This compares throughput: the 128 lifetimes are 128 independent learners, and the
baseline is one learner with 128 environments. See
[measured claims](https://github.com/Skyfall-Research/perpetua/blob/main/docs/measurement.md) for the setup, the other claims and
their limits.

## Documentation

- [Installation](https://github.com/Skyfall-Research/perpetua/blob/main/docs/introduction/installation.md), [concepts](https://github.com/Skyfall-Research/perpetua/blob/main/docs/introduction/concepts.md)
  and the [quickstart](https://github.com/Skyfall-Research/perpetua/blob/main/docs/tutorials/quickstart.md)
- [Tutorials](https://github.com/Skyfall-Research/perpetua/blob/main/docs/tutorials/index.md)
- [Environments](https://github.com/Skyfall-Research/perpetua/blob/main/docs/environments/index.md) and [benchmarks](https://github.com/Skyfall-Research/perpetua/blob/main/docs/presets/index.md)
- [API reference](https://github.com/Skyfall-Research/perpetua/blob/main/docs/api/index.md)

The docs are a Sphinx site built from `docs/`; see [contributing](https://github.com/Skyfall-Research/perpetua/blob/main/CONTRIBUTING.md)
to build them.

## Slides

[Perpetua: A Standard Interface for Continual Reinforcement Learning](https://github.com/Skyfall-Research/perpetua/blob/main/docs/slides/Perpetua_Framework.pdf)
(PDF, 24 pages) is the proposal talk from September 2026: why continual RL needs
its own interface, a survey of existing frameworks and benchmarks, the design
principles, and the planned measurements. It was written before the library
existed in its current form, so its API sketch, roadmap and "claims we will
measure" are intentions, not results. The [documentation](https://github.com/Skyfall-Research/perpetua/blob/main/docs/introduction/concepts.md)
and [measured claims](https://github.com/Skyfall-Research/perpetua/blob/main/docs/measurement.md) describe what Perpetua does today.

## Contributing

See [CONTRIBUTING.md](https://github.com/Skyfall-Research/perpetua/blob/main/CONTRIBUTING.md) for the engineering rules and the checks
every change runs.

## Acknowledgements and citation

Perpetua builds on other people's work, and we are grateful to the authors. The
PPO learner is adapted from [PureJaxRL](https://github.com/luchris429/purejaxrl)
(Lu et al.). The environments, interfaces and benchmarks Perpetua wraps or
reproduces come from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
[PettingZoo](https://github.com/Farama-Foundation/PettingZoo),
[MPE2](https://github.com/Farama-Foundation/MPE2),
[gymnax](https://github.com/RobertTLange/gymnax) (with its JAX versions of
[MinAtar](https://github.com/kenjyoung/MinAtar) and
[bsuite](https://github.com/google-deepmind/bsuite) tasks),
[JaxMARL](https://github.com/bold-lab-ai/JaxMARL),
[Meta-World](https://github.com/Farama-Foundation/Metaworld) and the Continual
World benchmark, [COOM](https://github.com/TTomilin/COOM) and
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom),
[MEAL](https://github.com/TTomilin/MEAL),
[AgarCL](https://github.com/AgarCL/AgarCL),
[EnvPool](https://github.com/sail-sg/envpool),
[MuJoCo and MJX](https://github.com/google-deepmind/mujoco) and the
[MORPHEUS evaluation suite](https://github.com/Skyfall-Research/morpheus-evals).
`BeanWorld-v0` is an original implementation in the style of
[Jelly Bean World](https://github.com/eaplatanios/jelly-bean-world). These are
optional dependencies and are not vendored; each keeps its own licence, listed in
[THIRD_PARTY_NOTICES.md](https://github.com/Skyfall-Research/perpetua/blob/main/THIRD_PARTY_NOTICES.md). If you use Perpetua, please cite
it and the benchmarks you run; [how to cite](https://github.com/Skyfall-Research/perpetua/blob/main/docs/introduction/citing.md) has the
BibTeX and the list of papers.

## Licence

Perpetua is released under the [MIT licence](https://github.com/Skyfall-Research/perpetua/blob/main/LICENSE).
