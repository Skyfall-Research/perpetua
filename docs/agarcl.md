# AgarCL setup, observations and replay

AgarCL is an MIT-licensed, non-episodic agar.io environment with continuous
movement and a split/feed choice. Perpetua exposes single-agent **screen** and
**grid** observations through a worker in AgarCL's own pinned native runtime.
Nothing from AgarCL is vendored or patched.

## Setup

```bash
uv run python -m perpetua.adapters.agarcl_setup
```

The module clones AgarCL at `d256e7b` and GLFW 3.4, builds a separate Python
3.12 environment with CMake 3.31, and installs AgarCL with `--no-deps` so its
Gymnasium 0.29 metadata cannot alter Perpetua's Gymnasium 1.4. No sudo or
system package installation is used.

On Linux, AgarCL needs Clang (GCC 11 rejects its sources) and OpenGL, GLU and
EGL development files. The script uses them if they are installed. Otherwise it
downloads the missing ones (`apt-get download`) and extracts them into a
private prefix, choosing the distribution's default `clang-N` (the one the
`clang` package depends on), or the newest if that cannot be determined. GLFW is built
without X11/Wayland (rendering goes through EGL), and GLU is linked explicitly.

Prerequisites: `git`, network access, a base C/C++ toolchain (libc and
libstdc++ headers, a linker, e.g. `build-essential`) and, if anything must be
fetched, apt package lists (`apt-get update`). Distributions without apt must
provide Clang (or `CXX`) and the OpenGL/GLU/EGL files themselves.
Tested platforms (2026-11-03):
- **macOS:** Apple M-series, hidden OpenGL window.
- **Ubuntu 22.04 x86_64 (H200):** NVIDIA EGL.
- **GitHub Actions, CPU-only with Mesa EGL:** Ubuntu 22.04 x86_64, Ubuntu
  24.04 x86_64 and aarch64, and Debian 12 x86_64 (container).

All run the same AgarCL test set (see `.github/workflows/integrations.yml`).
CPU-only Linux hosts need Mesa's EGL driver (`libegl-mesa0`,
`libgl1-mesa-dri`), and the worker selects Mesa's `EGL_PLATFORM=surfaceless`
unless you set `EGL_PLATFORM` yourself.

Override the runtime with `PERPETUA_AGARCL_PYTHON` and
`PERPETUA_AGARCL_SOURCE`.

## Use

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("agarcl/agario-screen-v0")  # or "agarcl/agario-grid-v0"
state, obs, info = env.init(jax.random.key(0))
# x/y movement in [-1, 1], then 0=noop, 1=split, 2=feed.
state, obs, reward, info = env.step(
    state, jnp.asarray([[0.25, -0.5, 0.0]], dtype=jnp.float32), jax.random.key(1)
)
```

`agarcl/grand-arena-v0` is the paper's full game, the Grand Arena, with the
world from its Appendix G.1: 500 pellets, 10 viruses and 8 heuristic bots, and
128×128 screen observations. The benchmark repository's committed
`env_config.json` differs (arena 300, 600 pellets, 4 bots), so every value is
a keyword override (`arena_size`, `num_pellets`, `num_viruses`, `num_bots`,
`screen_len`). Observations are upstream's RGB render rather than the paper's
four-channel tensor, and action noise stays off for reproducibility. In
20,000 seeded random ticks it produced no natural death either.

- **Screen** observations are `(1, 84, 84, 3)` RGB by default (`screen_len`,
  `agent_view`).
- **Grid** observations are `(1, grid_size, grid_size, 8)` (default 128). The
  eight channels are out-of-bounds, pellets (presence and count), viruses
  (presence and count), the agent's own cells, and other players' minimum and
  maximum mass. Upstream's `agario-grid-v0` wrapper is broken: it references
  an undefined `args` and passes 11 arguments to a 10-argument constructor. So
  the worker calls the low-level binding with upstream's own default settings.
  Upstream also builds grid frames once per step with `tick_index=0`, so frames
  are only written when `num_frames == ticks_per_step`. The worker configures
  it that way and returns frame 0, which holds the post-step state.
- The upstream hybrid action `(Box(2), Discrete(3))` is encoded as `Box(3)`:
  x/y plus a rounded/clipped noop/split/feed value. `add_noise=False` is
  required because AgarCL's sanitizer forwards the original actions when noise
  is enabled.
- SAC accepts flat observations only, so use `flatten=True` for it.
- The default reward is AgarCL's per-tick mass change, so totals telescope to
  net mass gained; an agent that stays at minimum mass scores exactly 0.

Seeded lifetimes reproduce exactly on the tested builds. `reset(seed=...)` is
ignored by AgarCL, so the worker seeds before reset and lets later respawns
continue the RNG stream. Upstream defaults are used (`mode=0`, no step limit,
no bots), so a respawn happens only when the agent dies. `num_bots` adds
upstream scripted bots, and `mode` is passed through unchanged; both default to
off/0. No natural death occurred in 30,000 seeded random grid ticks with 128
bots, so `respawn_after=N` is also available as an explicit periodic respawn
for benchmark time limits and stress tests (it is off by default).

## Checkpoints by replay

AgarCL's own JSON snapshots omit RNG, tick and timer state, so they are not
used. Instead, pass `replay_capacity` to make a lifetime checkpointable:

```python
env = pp.make("agarcl/agario-grid-v0", replay_capacity=1_000_000)
```

The snapshot is the seed plus the action log, kept in a preallocated
`(replay_capacity, 3)` float32 buffer (12 bytes per tick; 12 MB per million
ticks) so checkpoint shapes are fixed. Restoring replays the log in a fresh
worker, which is exact because seeded lifetimes reproduce. The conformance
suite checks this for screen and grid with save, restore into a differently
seeded lifetime, and continue. A test also branches from an older state and
returns to the original line, and another replays through two deterministic
respawns with eight bots.

Costs and limits:
- **Stepping:** the live state is never replayed, so each tick costs only one
  log write.
- **Restoring** a loaded checkpoint or an older state replays every logged tick
  at simulator speed. On an Apple M-series Mac, replay measured about 800
  ticks/s for screen and 3,100 ticks/s for grid. A million screen ticks
  therefore take about 20 minutes to restore.
- **Overflow:** a lifetime refuses to step past its capacity rather than lose
  exactness.
- **Large capacity:** constructing with `replay_capacity=1_000_000` allocates
  the full 12 MB buffer, and a save/restore of a 20-tick log at that capacity
  passed (0.02 s save, 0.01 s restore on the test Mac). CI does not replay a
  full million-tick log. On the H200 host, a soak restored a full 10⁶-tick
  16×16 grid log (eight bots) in 497 s and matched the next step exactly.
  Screen replay is slower; the ~20-minute
  estimate above still applies to it.
- **Respawns:** replay across a respawn follows the same code path and is
  tested with `respawn_after`. Natural death with bots is not yet observed:
  128 bots and 30,000 seeded random grid ticks produced no death, so natural
  death plus replay remains unmeasured.

## Limits

- **RAM observations** have no upstream binding (`agario-ram-v0` is not
  registered and `obs_type="ram"` asserts), so they are not available.
- **One simulator per worker process:** a second construction in one process
  renders an all-black screen, and a third segfaults (measured on the pinned
  build). The adapter therefore starts a fresh worker for every lifetime and
  every restore, so the all-black frame from a second construction in one
  process does not occur.
- **Rendering paths:** the macOS build uses a hidden OpenGL window and Linux
  uses EGL. These are environment-step paths, not training-throughput
  measurements.
- **Metadata conflict:** the native dependency declares `gymnasium~=0.29`,
  which conflicts with Perpetua's metadata even though the screen and grid
  paths work with Gymnasium 1.4 in their own runtime. This is why the adapter
  uses a separate environment.
- **Worker crashes** report the worker's exit code and the tail of its stderr.
