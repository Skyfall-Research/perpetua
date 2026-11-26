# Installation

Perpetua needs Python 3.12 or 3.13.

## Install with pip

```bash
git clone https://github.com/Skyfall-Research/perpetua.git
cd perpetua
pip install ".[training]"        # the core plus the learners
```

The installed copy records the commit of the clone it was built from, so
`pp.reporting.provenance()` can name it in results; it does not follow the
clone afterwards, so install again after `git pull`
(see [reproducibility](reproducibility.md#versions)).

This is enough for the quickstart and the tutorials that use only native
worlds. Three tutorials need more: the gymnax and task-switching tutorials need
the `interop` extra, and the Continual World tutorial needs `training` (already
installed above) and `metaworld`.
`pip install ".[training,interop,metaworld]"` covers all of them. Leave out
`[training]` for the core alone (JAX, Gymnasium, Orbax), or name other extras
from the table below, for example `pip install ".[training,interop]"`.

```{note}
Perpetua is not on PyPI yet. Once it is published the install line becomes
`pip install "perpetua-rl[training]"` (the name is provisional).
```

## Install for development with uv

Contributors use [uv](https://docs.astral.sh/uv/), a Python package and
project manager: `uv sync` creates the project's virtual environment from the
lock file, so everyone runs the same dependency versions, and `uv run` runs a
command inside it. The commands in the rest of the documentation are written
this way; with a pip install, drop the `uv run` prefix and its `--extra`
flags.

```bash
uv sync --locked                       # the core plus the dev and meal groups (tests, learners, adapters)
uv sync --locked --no-default-groups   # the core alone
```

## macOS

On a Mac, Perpetua needs an Apple Silicon machine and an arm64 Python: JAX
publishes no wheels for Intel macOS. If the Python that uv picks is an Intel
(x86-64) build, which happens on Apple Silicon when an Intel Python is first
on the `PATH`, `uv sync` stops and reports that `jaxlib` has no wheel for the
current platform. Install an arm64 interpreter with uv and point the project
at it:

```bash
uv python install cpython-3.12-macos-aarch64-none
uv sync --locked --python cpython-3.12-macos-aarch64-none
```

`python -c "import platform; print(platform.machine())"` prints `arm64` for
a suitable interpreter. macOS is checked by hand, not in CI: every automated
job runs on Linux. JAX runs on the CPU there, and EnvPool (`cpu-bridge`) and
`cuda13` are Linux-only.

## Extras

The core runs every native world, the drift library, the runners, metrics and
checkpoints. Everything else is an optional extra, and no third-party benchmark
code is vendored.

| Extra | Adds | For |
|---|---|---|
| `cuda13` | `jax[cuda13]` (Linux) | NVIDIA GPUs |
| `training` | Flax, Optax | the PPO, IPPO, DQN, IQL and SAC learners |
| `interop` | gymnax 0.0.9 | `gymnax/*` environments |
| `mjx` | MuJoCo 3.3.0 and MJX | `mjx/Pendulum-v0`, `mjx/CartPole-v0` |
| `jaxmarl` | JaxMARL 0.2 | `jaxmarl/MPE_*` tasks |
| `marl` | PettingZoo, MPE2 | PettingZoo lifts and adapters |
| `cpu-bridge` | EnvPool (Linux) | `envpool/*`, Atari switching |
| `meal` | MEAL 0.3.1 | `meal/overcooked-v0`, `pp.presets.meal_sequence` |
| `metaworld` | Meta-World 3.1.1, MuJoCo | `metaworld/MT1-v0`, Continual World presets |
| `morpheus` | httpx | a live MORPHEUS deployment (the mock needs nothing) |
| `torch` | PyTorch | `perpetua.torch_bridge` agents |

**Which extras do I need?** For the quickstart, the Bean World, custom-drift,
evaluation, reward-centering and custom-environment tutorials, `training`
(the custom-drift tutorial runs on the core alone). Add `interop` for the
`gymnax/*` environments, which the gymnax and task-switching tutorials use,
`metaworld` for the Continual World tutorial, and `marl` for PettingZoo.
`cuda13` and `cpu-bridge` are Linux-only; on macOS JAX runs on the CPU, which
is enough for every tutorial.

Combine extras as needed. With uv, pass the same `--extra` flags to `uv run`:

```bash
uv sync --locked --extra training --extra interop
uv run --extra training --extra interop python examples/train_ppo.py
```

On a GPU host, include `cuda13`; CUDA 13 needs a supported NVIDIA GPU and
driver 580 or newer (this project does not install drivers). Check the GPU
with `uv run --extra cuda13 python scripts/check_runtime.py --require-gpu`.
The compiled CPU bridge (`compiled=True`) also needs JAX's CPU platform, so set
`JAX_PLATFORMS=cuda,cpu` there.

## Separate runtimes

COOM and AgarCL pin dependencies that conflict with this project, so each runs in
its own environment, built once:

```bash
uv run python -m perpetua.adapters.coom_setup      # COOM on ViZDoom
uv run python -m perpetua.adapters.agarcl_setup    # AgarCL's native simulator
```

Both modules ship in the installed package, so the same commands work without
a checkout (drop `uv run`).

The optional EnvPool snapshot binding (`EnvPoolCartPole(snapshot=True)`) is built
from `native/envpool_snapshot`; see [CPU snapshots](../cpu-snapshots.md).

## Check the install

```bash
python -c "import perpetua as pp; pp.testing.check_env(lambda: pp.make('ContinualGrid-v0')); print('ok')"
```

(With uv, prefix it with `uv run`.)

This builds a world, runs the contract checks on it and prints `ok` (about
ten seconds, mostly compilation). Then run the
[quickstart](../tutorials/quickstart.md) (needs the `training` extra) for a
complete experiment. The full test suite (`uv run --all-extras pytest -n auto`,
from a development install) takes roughly 10 to 20 minutes on a multi-core
laptop and skips what is not installed; see
[CONTRIBUTING](https://github.com/Skyfall-Research/perpetua/blob/main/CONTRIBUTING.md)
for what it covers and what CI does not run.
