"""AgarCL screen and grid observations through its pinned native runtime.

AgarCL is MIT-licensed but has a C++/GLFW build and declares Gymnasium 0.29,
so it runs in a separate Python environment built by ``agarcl_setup.py``.
Single-agent screen and grid observations are exposed. The grid bypasses
upstream's broken ``agario-grid-v0`` wrapper (see ``agarcl_worker.py``). RAM
observations have no upstream binding and are not available.

AgarCL's own JSON snapshots omit RNG and timer state, so they are not used.
With ``replay_capacity`` set, a lifetime is checkpointable instead by replaying
its seeded action log, which is exact because seeded lifetimes reproduce.

The upstream ``(Box(2), Discrete(3))`` action is encoded as ``Box(3)``:
continuous x/y in [-1, 1], then 0=noop, 1=split or 2=feed (rounded/clipped).
"""

import os
import platform
import sys
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from perpetua.adapters.host import BackendResult, CPULifetime
from perpetua.adapters.snapshots import BackendSnapshot, ReplaySnapshot
from perpetua.adapters.worker import WorkerProcess
from perpetua.core import JSONValue
from perpetua.drift import Drift

__all__ = [
    "AGARCL_COMMIT",
    "GLFW_COMMIT",
    "DEFAULT_ROOT",
    "available",
    "GRAND_ARENA",
    "AgarCLLifetime",
]

AGARCL_COMMIT = "d256e7b6a61bebfd3a23014f0343c2f9195ac9d4"
GLFW_COMMIT = "7b6aead9fb88b3623e3b3725ebb42670cbe4c579"
DEFAULT_ROOT = Path.home() / ".cache" / "perpetua" / "agarcl" / AGARCL_COMMIT
WORKER = Path(__file__).with_name("agarcl_worker.py")


def runtime() -> tuple[Path, Path]:
    """The AgarCL Python and source paths, from the environment or the default cache."""
    python = Path(
        os.environ.get("PERPETUA_AGARCL_PYTHON", DEFAULT_ROOT / "venv" / "bin" / "python")
    )
    source = Path(os.environ.get("PERPETUA_AGARCL_SOURCE", DEFAULT_ROOT / "src"))
    return python, source


def available() -> bool:
    """Whether the AgarCL runtime built by ``agarcl_setup.py`` is present."""
    python, source = runtime()
    return python.exists() and (source / "gym_agario").is_dir()


class _Worker(WorkerProcess):
    def __init__(self) -> None:
        python, source = runtime()
        if not available():
            raise RuntimeError(
                "AgarCL runtime not found; run `python -m perpetua.adapters.agarcl_setup` "
                "or set PERPETUA_AGARCL_PYTHON and PERPETUA_AGARCL_SOURCE."
            )
        root = python.parent.parent.parent
        library_path = os.pathsep.join(
            (
                str(root / "prefix" / "lib"),
                str(root / "prefix" / "usr" / "lib" / f"{platform.machine()}-linux-gnu"),
            )
        )
        super().__init__(
            "AgarCL",
            [str(python), "-P", str(WORKER)],
            env={
                # AgarCL calls eglGetDisplay(EGL_DEFAULT_DISPLAY) and renders to a
                # pbuffer. Without a display server, Mesa needs the surfaceless
                # platform; NVIDIA's EGL works with it too (checked on the H200).
                **({"EGL_PLATFORM": "surfaceless"} if sys.platform == "linux" else {}),
                **os.environ,
                "PYTHONPATH": str(source),
                **{
                    name: os.pathsep.join(filter(None, (library_path, os.environ.get(name))))
                    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
                },
            },
            dtype=np.uint8,
        )


class _AgarCLBackend:
    num_slots = 1
    uses_reset_seed = False

    def __init__(
        self,
        *,
        obs_type: str,
        screen_len: int,
        agent_view: bool,
        grid_size: int,
        num_bots: int,
        mode: int,
        respawn_after: int | None,
        arena_size: int | None,
        num_pellets: int | None,
        num_viruses: int | None,
        flatten: bool,
        replay_capacity: int | None,
    ) -> None:
        self.request: dict[str, JSONValue] = {
            "obs_type": obs_type,
            "screen_len": screen_len,
            "agent_view": agent_view,
            "grid_size": grid_size,
            "num_bots": num_bots,
            "mode": mode,
            "respawn_after": respawn_after,
            "arena_size": arena_size,
            "num_pellets": num_pellets,
            "num_viruses": num_viruses,
        }
        self.obs_type = obs_type
        self.flatten = flatten
        self.capacity = replay_capacity
        self.checkpointable = replay_capacity is not None
        self.seed: int = 0
        self.length: int = 0
        self.log: NDArray[np.float32] = np.zeros((replay_capacity or 0, 3), dtype=np.float32)
        self.worker = _Worker()
        try:
            _, obs = self.worker.request_array(
                command="init", seed=0, **cast(dict[str, Any], self.request)
            )
        except BaseException:
            self.worker.close()
            raise
        frame_shape = tuple(int(dimension) for dimension in self._frame(obs).shape)
        shape = (int(np.prod(frame_shape)),) if flatten else frame_shape
        if obs_type == "grid":
            self.observation_space = gym.spaces.Box(-1, np.inf, shape, dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(0, 255, shape, dtype=np.float32)
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Box(
            np.asarray([-1.0, -1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.configuration: dict[str, JSONValue] = {
            "backend": f"agarcl-{AGARCL_COMMIT[:12]}",
            "observations": (
                f"single-agent {obs_type}"
                + (f" {grid_size}x{grid_size}, post-step frame" if obs_type == "grid" else "")
            ),
            "screen_len": screen_len if obs_type == "screen" else None,
            "agent_view": agent_view if obs_type == "screen" else None,
            "num_bots": num_bots,
            "mode": mode,
            "respawn_after": respawn_after,
            "arena_size": arena_size,
            "num_pellets": num_pellets,
            "num_viruses": num_viruses,
            "actions": "Box(3): x/y movement plus noop/split/feed",
            "add_noise": False,
            "flattened_for_sac": flatten,
            "checkpoint": (
                f"replay of the seeded action log, capacity {replay_capacity} ticks"
                if replay_capacity is not None
                else "none (AgarCL's own snapshots are incomplete)"
            ),
        }

    def _frame(self, obs: NDArray[Any]) -> NDArray[Any]:
        # The screen keeps AgarCL's leading frame axis (num_frames=1).
        return obs[0] if self.obs_type == "screen" else obs

    def _values(self, obs: NDArray[Any]) -> NDArray[np.float32]:
        values = self._frame(obs).astype(np.float32)
        return values.reshape(1, -1) if self.flatten else values[None]

    def _start(self, seed: int) -> NDArray[Any]:
        """Construct the simulator as the first one in a fresh worker process.

        Measured on the pinned build: a second construction in one process
        renders an all-black screen, and a third segfaults. So every lifetime
        start and every restore gets its own worker; the construction probe's
        worker is discarded.
        """
        # Until the caller records the new state, no snapshot matches this
        # backend: a start or replay that fails must not leave the old
        # bookkeeping pointing at a different simulator.
        self.length = -1
        self.worker.close()
        self.worker = _Worker()
        _, obs = self.worker.request_array(
            command="init", seed=seed, **cast(dict[str, Any], self.request)
        )
        return obs

    def init(self, seed: int) -> NDArray[np.float32]:
        obs = self._start(seed)
        self.seed, self.length = seed, 0
        self.log = np.zeros_like(self.log)
        return self._values(obs)

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        del reset_seed
        vector = np.asarray(action[0], dtype=np.float32)
        row = np.asarray([vector[0], vector[1], np.clip(np.rint(vector[2]), 0, 2)], np.float32)
        if self.length < 0:
            raise RuntimeError(
                "AgarCL has no valid simulator state after a failed start or restore; "
                "call init or restore again."
            )
        if self.capacity is not None:
            if self.length >= self.capacity:
                raise RuntimeError(
                    f"AgarCL replay log is full ({self.capacity} ticks); "
                    "construct the lifetime with a larger replay_capacity."
                )
            # Earlier states share this buffer but only read rows below their length.
            self.log[self.length] = row
            self.length += 1
        header, obs = self.worker.request_array(
            command="step", move=[float(row[0]), float(row[1])], discrete=int(row[2])
        )
        return BackendResult(
            self._values(obs),
            np.asarray([header["reward"]], dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            np.asarray([header["respawn"]], dtype=np.bool_),
        )

    def snapshot(self) -> BackendSnapshot:
        if self.capacity is None:
            raise ValueError("Construct AgarCL with replay_capacity to enable checkpoints.")
        return ReplaySnapshot(
            np.asarray(1, dtype=np.uint32),
            np.asarray(self.seed, dtype=np.int64),
            np.asarray(self.length, dtype=np.int64),
            self.log,
        )

    def restore(self, snapshot: BackendSnapshot) -> None:
        if self.capacity is None:
            raise ValueError("Construct AgarCL with replay_capacity to enable checkpoints.")
        if not isinstance(snapshot, ReplaySnapshot):
            raise ValueError("AgarCL restores only its own replay snapshots.")
        expected = (
            ((), np.uint32),
            ((), np.int64),
            ((), np.int64),
            ((self.capacity, 3), np.float32),
        )
        for value, (shape, dtype) in zip(cast(tuple[object, ...], snapshot), expected, strict=True):
            array = cast(NDArray[Any], value) if isinstance(value, np.ndarray) else None
            if array is None or array.shape != shape or array.dtype != dtype:
                raise ValueError("AgarCL replay snapshot shape/dtype does not match the backend.")
        length, seed = int(snapshot.length), int(snapshot.seed)
        if int(snapshot.version) != 1 or not 0 <= length <= self.capacity:
            raise ValueError("Invalid AgarCL replay snapshot version or length.")
        if snapshot.actions is self.log and length == self.length and seed == self.seed:
            return  # the backend is already at this state (the per-tick case)
        rows = snapshot.actions[:length]
        if not np.all(np.isfinite(rows)) or not np.all(np.isin(rows[:, 2], (0.0, 1.0, 2.0))):
            raise ValueError("Invalid AgarCL replay actions.")
        # A branch or a loaded checkpoint: copy, so states sharing the old buffer stay valid.
        log = np.array(snapshot.actions, copy=True)
        self._start(seed)
        if length:
            # Replay is much slower than a normal step (the measured floor is
            # ~800 screen ticks/s); grant it time without weakening the guard
            # on ordinary calls. 200 ticks/s is still 4x slower than measured.
            header, _ = self.worker.request_array(
                np.ascontiguousarray(rows).tobytes(),
                command="replay",
                timeout=max(self.worker.timeout, length / 200.0),
            )
            if header.get("ticks") != length:
                raise RuntimeError(
                    f"AgarCL replay ran {header.get('ticks')} of {length} logged ticks; "
                    "the restored state would be wrong."
                )
        self.seed, self.length, self.log = seed, length, log

    def close(self) -> None:
        self.worker.close()


# The full game ("Grand Arena") as specified in the AgarCL paper, Appendix G.1
# (arXiv:2505.18347): 500 pellets, 10 viruses and 8 heuristic bots, with
# 128x128 pixel observations. Its benchmark repository's committed
# env_config.json differs (arena 300, 600 pellets, 4 bots); any value can be
# overridden by keyword. Frame skip, regeneration and mass decay are the pinned
# engine's own. Screen observations are upstream's RGB render, not the paper's
# four-channel tensor, and action noise stays off for reproducibility.
GRAND_ARENA: dict[str, Any] = {
    "obs_type": "screen",
    "screen_len": 128,
    "arena_size": 128,
    "num_pellets": 500,
    "num_viruses": 10,
    "num_bots": 8,
}


class AgarCLLifetime(CPULifetime):
    """One non-episodic AgarCL world (screen or grid) as a host-controlled lifetime.

    ``obs_type`` selects ``"screen"`` (an RGB render ``screen_len`` pixels on
    a side, at least 16; ``agent_view`` is passed to upstream's screen
    environment) or ``"grid"`` (a ``grid_size`` by ``grid_size`` grid, at
    least 8). ``flatten=True`` presents each observation as one flat vector,
    which SAC needs. ``env_id`` overrides the recorded id (default
    ``agarcl/agario-<obs_type>-v0``).

    ``replay_capacity`` enables exact checkpoints by replaying the seeded action
    log (12 bytes per tick, preallocated); restoring a non-current state costs
    one replay of its ticks. It also caps the lifetime: once the log holds
    ``replay_capacity`` ticks, ``step`` raises ``RuntimeError``. Without it,
    lifetimes are not checkpointable and have no length cap.
    ``num_bots`` adds upstream scripted bots (0 by default; useful for death,
    respawn and replay stress tests). ``mode`` is passed to upstream unchanged;
    the default 0 is the non-episodic mode. ``respawn_after`` optionally resets
    the simulator after that many ticks, reporting a respawn like a death; it is
    off by default and exists to make non-episodic respawn boundaries testable.
    ``arena_size``, ``num_pellets`` and ``num_viruses`` override upstream's
    "normal" world (``None`` keeps it); ``agarcl/grand-arena-v0`` sets them to
    the paper's full-game values.
    """

    def __init__(
        self,
        *,
        obs_type: str = "screen",
        screen_len: int = 84,
        agent_view: bool = False,
        grid_size: int = 128,
        num_bots: int = 0,
        mode: int = 0,
        respawn_after: int | None = None,
        arena_size: int | None = None,
        num_pellets: int | None = None,
        num_viruses: int | None = None,
        flatten: bool = False,
        replay_capacity: int | None = None,
        drift: Drift | None = None,
        env_id: str | None = None,
        compiled: bool = False,
    ) -> None:
        if compiled and replay_capacity is not None:
            raise ValueError("compiled=True states are single-use; drop replay_capacity.")
        if obs_type not in ("screen", "grid"):
            raise ValueError("obs_type must be 'screen' or 'grid'.")
        if type(screen_len) is not int or screen_len < 16:
            raise ValueError("screen_len must be an integer >= 16.")
        if type(grid_size) is not int or grid_size < 8:
            raise ValueError("grid_size must be an integer >= 8.")
        if type(num_bots) is not int or num_bots < 0:
            raise ValueError("num_bots must be a non-negative integer.")
        if type(mode) is not int or mode < 0:
            raise ValueError("mode must be a non-negative integer.")
        if respawn_after is not None and (type(respawn_after) is not int or respawn_after < 1):
            raise ValueError("respawn_after must be a positive integer or None.")
        for name, value, minimum in (
            ("arena_size", arena_size, 16),
            ("num_pellets", num_pellets, 0),
            ("num_viruses", num_viruses, 0),
        ):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum} or None.")
        if replay_capacity is not None and (
            type(replay_capacity) is not int or replay_capacity < 1
        ):
            raise ValueError("replay_capacity must be a positive integer or None.")
        super().__init__(
            _AgarCLBackend(
                obs_type=obs_type,
                screen_len=screen_len,
                agent_view=agent_view,
                grid_size=grid_size,
                num_bots=num_bots,
                mode=mode,
                respawn_after=respawn_after,
                arena_size=arena_size,
                num_pellets=num_pellets,
                num_viruses=num_viruses,
                flatten=flatten,
                replay_capacity=replay_capacity,
            ),
            env_id=env_id or f"agarcl/agario-{obs_type}-v0",
            drift=drift,
            compiled=compiled,
        )
