"""COOM (ViZDoom) through a worker process in COOM's own pinned environment.

COOM pins Gymnasium 0.28.1 and needs legacy Gym, which conflict with
Perpetua's Gymnasium 1.4, and its published wheels are incomplete. So COOM is
never imported here: ``coom_setup.py`` builds a separate environment
from a pinned COOM commit, and each lifetime talks to a worker process
(``coom_worker.py``) over a small JSON-plus-raw-bytes protocol. Nothing from
COOM is vendored.

The runtime is located through ``PERPETUA_COOM_PYTHON`` and
``PERPETUA_COOM_SOURCE``, defaulting to the directory the setup module writes.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from perpetua.adapters.host import BackendResult, CPULifetime
from perpetua.adapters.snapshots import BackendSnapshot
from perpetua.adapters.worker import WorkerProcess
from perpetua.core import JSONValue
from perpetua.drift import Drift

__all__ = [
    "COOM_COMMIT",
    "DEFAULT_ROOT",
    "available",
    "sequence_tasks",
    "COOMLifetime",
]

COOM_COMMIT = "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e"
DEFAULT_ROOT = Path.home() / ".cache" / "perpetua" / "coom" / COOM_COMMIT
WORKER = Path(__file__).with_name("coom_worker.py")


def runtime() -> tuple[Path, Path]:
    """Return the legacy interpreter and COOM source checkout."""
    python = Path(os.environ.get("PERPETUA_COOM_PYTHON", DEFAULT_ROOT / "venv" / "bin" / "python"))
    source = Path(os.environ.get("PERPETUA_COOM_SOURCE", DEFAULT_ROOT / "src"))
    return python, source


def available() -> bool:
    """Whether a COOM runtime is installed where the adapter will look."""
    python, source = runtime()
    return python.exists() and (source / "COOM").is_dir()


class _Worker(WorkerProcess):
    """One COOM worker process; ViZDoom runs in its own temporary directory."""

    def __init__(self) -> None:
        python, source = runtime()
        if not available():
            raise RuntimeError(
                "COOM runtime not found; run `python -m perpetua.adapters.coom_setup` "
                "or set PERPETUA_COOM_PYTHON and PERPETUA_COOM_SOURCE."
            )
        # ViZDoom writes _vizdoom.ini into its working directory.
        self.workdir = tempfile.mkdtemp(prefix="perpetua-coom-")
        try:
            super().__init__(
                "COOM",
                # -P: the worker's own directory holds Perpetua's ``gymnasium.py``
                # adapter, which must not shadow the real package.
                [str(python), "-P", str(WORKER)],
                env={**os.environ, "PYTHONPATH": str(source)},
                dtype=np.float32,
                cwd=self.workdir,
            )
        except BaseException:
            shutil.rmtree(self.workdir, ignore_errors=True)
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)


def sequence_tasks(sequence: str) -> list[tuple[str, str]]:
    """COOM's own (scenario, task) order for a named sequence such as ``CO8``."""
    worker = _Worker()
    try:
        header, _ = worker.request(command="sequence", sequence=sequence)
        return [(str(scenario), str(task)) for scenario, task in header["tasks"]]
    finally:
        worker.close()


class _COOMBackend:
    """One COOM task as a single host slot; ViZDoom state is opaque."""

    num_slots = 1
    uses_reset_seed = False
    checkpointable = False
    configuration: dict[str, JSONValue]

    def __init__(self, scenario: str, task: str) -> None:
        self.scenario = scenario
        self.task = task
        self.worker = _Worker()
        # Spaces are needed before the seeded lifetime exists; probe once.
        try:
            header, obs = self.worker.request_array(
                command="init", scenario=scenario, task=task, seed=0
            )
        except BaseException:
            self.worker.close()  # a failed probe must not leak the worker or its directory
            raise
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(int(header["actions"]))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, obs.shape, dtype=np.float32)
        self.configuration = {
            "backend": "coom-worker",
            "coom_commit": COOM_COMMIT,
            "scenario": scenario,
            "task": task,
            "wrappers": "COOM defaults; 4 RGB frames merged into 12 channels",
            "respawn": "same-tick-on-termination-or-truncation",
        }

    def init(self, seed: int) -> NDArray[np.float32]:
        _, obs = self.worker.request_array(
            command="init", scenario=self.scenario, task=self.task, seed=seed
        )
        return obs[None]

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        del reset_seed  # ViZDoom's own seeded stream continues across respawns.
        header, obs = self.worker.request_array(command="step", action=int(action[0]))
        return BackendResult(
            obs[None],
            np.asarray([header["reward"]], dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            np.asarray([header["respawn"]], dtype=np.bool_),
        )

    def snapshot(self) -> BackendSnapshot:
        raise ValueError("COOM (ViZDoom) state is opaque; snapshots are not supported.")

    def restore(self, snapshot: BackendSnapshot) -> None:
        del snapshot
        raise ValueError("COOM (ViZDoom) state is opaque; snapshots are not supported.")

    def close(self) -> None:
        self.worker.close()


class COOMLifetime(CPULifetime):
    """One COOM scenario/task as a host-controlled continual lifetime."""

    def __init__(
        self,
        *,
        scenario: str = "pitfall",
        task: str = "default",
        drift: Drift | None = None,
        compiled: bool = False,
    ) -> None:
        super().__init__(
            _COOMBackend(scenario, task), env_id="coom/COOM-v0", drift=drift, compiled=compiled
        )
