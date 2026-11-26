# Benchmark setup: MEAL, Continual World and COOM

These integrations are optional pip dependencies, never vendored. Each records
its backend version in environment metadata and has a generated page in the
[environment index](environments/index.md). See also the dedicated
[MORPHEUS](morpheus.md) and [AgarCL](agarcl.md) pages.

(what-drifts)=
## What drifts in wrapped benchmarks

The native, gymnax and MJX environments drift their dynamics: a goal, gravity,
a partner's behaviour. Most wrapped benchmarks and CPU simulators drift a
reward multiplier and change through [task switching](task-switching.md). Each
[environment page](environments/index.md) lists its driftable parameters.

## MEAL (MIT)

`meal-bench==0.3.1` is verified with Python 3.12, JAX 0.11.2, Flax 0.12.10,
Optax 0.2.8 and JaxMARL 0.2.0. `distrax==0.1.9` uses the locked
`tfp-nightly` build; an attempted downgrade to Distrax 0.1.5/stable TFP was
reverted because that combination does not import under JAX 0.11. The pinned
nightly is pinned exactly (`tfp-nightly==0.26.0.dev20261030`) so relocking
cannot drift to an untested build; its import and distribution sampling are
tested. Stable `tensorflow-probability` 0.25.0 (November 2024, still the newest
on 2026-11-03) calls `jax.interpreters.xla.pytype_aval_mappings`, which JAX
0.11 removed. A nightly CI job (`stable-tfp` in
`.github/workflows/integrations.yml`) tries the newest stable release with the
pinned JAX and fails deliberately once it works, so the nightly can be dropped.
The Overcooked adapter is native JAX, carries
MEAL's observations in state, and reports MEAL auto-resets as `respawn` events.
Procedural layouts come from the seeded sequence API, so equal seeds produce
the same layout as well as the same reset state.

```bash
uv sync --locked --extra meal --extra training
```

```python
import perpetua as pp

env = pp.make("meal/overcooked-v0", difficulty="medium", num_agents=2, seed=0)
sequence = pp.presets.meal_sequence(
    sequence_length=8, difficulty="medium", num_agents=2, seed=42, period=100_000
)
```

Observations have shape `(agents, H, W, C)` and work with the CNN torso, which
reads them unscaled. Task
switching resumes previous layouts by default; `on_switch="restart"` is
available because these are functional JAX environments.

## Continual World / Meta-World (MIT; MuJoCo Apache-2.0)

Farama Meta-World 3.1.1 and MuJoCo 3.3.0 are verified on macOS (by hand; CI
runs on Linux only) and Linux with Gymnasium 1.4.0. The original Continual World repository has no visible licence,
so no code from it is used. The published CW10 order is mapped to current v3
task names; CW20 is that order repeated twice. This preserves the published
sequence, not numerical equivalence to the old v1 implementations.

```bash
uv sync --locked --extra metaworld --extra training
```

```python
import perpetua as pp

env = pp.make("metaworld/MT1-v0", env_name="hammer-v3", task_index=0)
cw10 = pp.presets.cw10(period=1_000_000)
cw20 = pp.presets.cw20(period=1_000_000)  # MetaWorld ∘ Switch(CW10, 2, 1M)
```

`cw20()` is the CW10 cycle with a 20-task horizon recorded in
`metadata["configuration"]["preset"]`: a lifetime never ends, so the
repetition count is an evaluation horizon, and the second pass revisits the
same ten simulators. `continual_world(CW20, ...)` still builds twenty separate
lifetimes when you want the second pass on fresh simulators.

MT1 goals are pinned by `task_index` because otherwise Meta-World samples goals
from an independent RNG on reset. Tasks run for 200 simulator steps per backend
episode by default, with termination/truncation reported as respawns. Host SAC
is verified on `hammer-v3`. The preset uses resume semantics; host lifts cannot
restart on task entry.

Continual World scores task success, not reward. Meta-World's
`info["success"]` is exposed only through the Oracle (`expose=["success"]`) of
a single-task `MetaWorldLifetime`; the switched presets (`cw10()`, `cw20()`,
`continual_world()`) do not offer a `success` field, with or without
`compiled=True`. `examples/compare_benchmarks.py --benchmark continual_world`
and `scripts/reproduce_continual_world.py` build separate one-task evaluation
environments for it. The example reports average performance and forgetting on
success as the fraction of evaluation ticks with success. That is a per-tick definition, not the paper's per-episode one;
forward transfer needs single-task reference runs and is not reported by
that example. `scripts/reproduce_continual_world.py` follows the paper's
protocol, with per-episode success and forward transfer; see the
[preset page](presets/continual-world.md).

## COOM (MIT; ViZDoom MIT)

COOM cannot share Perpetua's environment, so it runs as a worker process in its
own pinned runtime. Nothing from COOM is vendored and its dependencies never
enter Perpetua's lockfile.

```bash
uv run python -m perpetua.adapters.coom_setup   # clones COOM 7929801 and builds its runtime
```

```python
import perpetua as pp

env = pp.make("coom/COOM-v0", scenario="pitfall", task="default")
co8 = pp.presets.coom_sequence("CO8", period=200_000)
```

The setup script clones a pinned COOM commit and creates a separate Python 3.12
environment with the dependency set measured to work: Gymnasium 0.28.1, Gym
0.26.2, ViZDoom 1.3.2, SciPy 1.11.4, NumPy 1.26.4 and OpenCV (headless).
`PERPETUA_COOM_PYTHON` and `PERPETUA_COOM_SOURCE` point the adapter elsewhere.
Each lifetime starts `coom_worker.py` in that runtime and exchanges JSON lines
plus raw float32 observations; the worker redirects its own stdout so ViZDoom
prints cannot corrupt the protocol, and worker errors are raised in the parent
instead of hanging it. If the worker dies, the error carries its exit code and
the tail of its stderr (shared helper `perpetua.adapters.worker`).

How each upstream problem is handled:

| Upstream problem (measured) | Handling |
|---|---|
| PyPI `COOM==1.0.0` wheel has no importable package; source wheel omits `COOM.env.scenarios` | No wheel is installed: the worker imports the pinned source checkout from `PYTHONPATH` |
| Pins `gymnasium==0.28.1` and imports `FrameStack`, removed in Gymnasium 1.4; needs legacy `gym` | Separate runtime; Perpetua's Gymnasium 1.4 is untouched |
| `make_sequence` defaults leave `action_space_fn` unset | The worker builds every task with `make_env` and `build_multi_discrete_actions`; sequence order comes from COOM's own tables |
| COOM wrappers reject `reset(seed=...)` | The seed goes to ViZDoom through `doom_kwargs`; resets take no seed |
| Continual-learning extra needs TensorFlow 2.11 (no Python 3.12 wheels) | Not installed; only the environments are used |

Observations keep COOM's default wrappers (resize to 84x84, rescale, running
observation normalisation, four-frame stack); the four RGB frames are merged
into channels, `(1, 84, 84, 12)`, so the CNN torso applies directly. All
scenarios use COOM's 12-action space. Episode ends are respawns. Seeded
lifetimes are bit-reproducible across respawns; ViZDoom state is opaque, so
COOM lifetimes cannot be checkpointed. The main CI job does not build the
runtime, so COOM tests skip there. A separate workflow
(`.github/workflows/integrations.yml`) builds it on Ubuntu 22.04 and runs them,
failing if any skip. It runs on changes to the COOM files, nightly and on
demand.

## Considered, not integrated

**DiscoveryWorld** (AI2, Apache-2.0) is a benchmark of scientific-discovery
tasks. Its actions are JSON verbs addressed to object identifiers, and its
observations are scene descriptions: an interface built for language-model
agents. Perpetua's numeric learners would need it squeezed into a fixed discrete
menu, and results there would measure the fit of the menu, not continual
learning, so it is not integrated. A text-action interface would be the
prerequisite for revisiting this.
