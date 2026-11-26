"""Exercise the real runtime check, including its GPU requirement guard."""

import json
import os
import subprocess
import sys
from pathlib import Path

import perpetua


def test_package_import() -> None:
    assert perpetua.__doc__ is not None


def test_cpu_runtime_check() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_runtime.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)
    assert report["backend"] == "cpu"
    assert report["chunk_equivalence"] is True
    assert report["batch_size"] == 64


def test_gpu_requirement_rejects_cpu() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_runtime.py"
    result = subprocess.run(
        [sys.executable, str(script), "--require-gpu"],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "JAX has no GPU device" in result.stderr
