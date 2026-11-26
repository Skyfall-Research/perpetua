"""The timed training CLI must resume pending buffers and replay measured traces."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest


def run(script: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=check,
        timeout=90,
    )


def test_recorded_time_training_resumes_and_replays_exactly(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/compute_ppo.py"
    path = tmp_path / "timed"
    run(script, ["--chunks", "2", "--checkpoint", str(path)])
    continued = json.loads(run(script, ["--chunks", "1", "--resume", str(path)]).stdout)
    full = json.loads(run(script, ["--chunks", "3"]).stdout)
    assert continued["world_ticks"] == full["world_ticks"] == 24
    assert continued["decisions"] == full["decisions"] == 12
    assert continued["updates"] == full["updates"] == 1
    assert continued["mean_reward"] == full["mean_reward"]
    replay = json.loads(run(script, ["--replay", str(path)]).stdout)
    assert replay["exact_replay"] is True
    assert replay["world_ticks"] == 16


def test_measured_time_trace_can_be_replayed_without_remeasuring(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/compute_ppo.py"
    path = tmp_path / "measured"
    measured = json.loads(
        run(
            script,
            [
                "--mode",
                "measured",
                "--quantum-ns",
                "1000000000",
                "--chunks",
                "1",
                "--checkpoint",
                str(path),
            ],
        ).stdout
    )
    replay = json.loads(
        run(
            script, ["--mode", "measured", "--quantum-ns", "1000000000", "--replay", str(path)]
        ).stdout
    )
    assert replay["exact_replay"] is True
    assert replay["world_ticks"] == measured["world_ticks"]
    assert replay["compute_ns"] == measured["compute_ns"] > 0


def test_timed_pettingzoo_checkpoint_is_explicitly_refused(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/compute_ppo.py"
    result = run(
        script,
        ["--backend", "pettingzoo", "--checkpoint", str(tmp_path / "unsupported")],
        check=False,
    )
    assert result.returncode != 0
    assert "checkpoint" in result.stderr
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="Exact measured GPU replay requires CUDA"
)
@pytest.mark.parametrize("backend", ["native", "pettingzoo-snapshot"])
def test_measured_gpu_replay_matches_optimizer_state_in_a_fresh_process(
    tmp_path: Path,
    backend: str,
) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/compute_ppo.py"
    path = tmp_path / "gpu-measured"
    common = [
        "--backend",
        backend,
        "--mode",
        "measured",
        "--quantum-ns",
        "10000000",
        "--require-gpu",
    ]
    env = {**os.environ, "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    subprocess.run(
        [sys.executable, str(script), *common, "--chunks", "3", "--checkpoint", str(path)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, str(script), *common, "--replay", str(path)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        assert json.loads(result.stdout)["exact_replay"] is True
