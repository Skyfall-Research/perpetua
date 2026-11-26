"""The optional control example must use the same continuing learner contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["gymnax", "envpool"])
def test_control_example(backend: str) -> None:
    if backend == "envpool" and sys.platform != "linux":
        pytest.skip("EnvPool CPU bridge currently requires Linux runtime libraries.")
    script = Path(__file__).resolve().parents[1] / "examples" / "control_ppo.py"
    result = subprocess.run(
        [sys.executable, str(script), "--backend", backend, "--updates", "2", "--num-envs", "4"],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    report = json.loads(result.stdout)
    assert report["lifetime_ticks"] == 256
    assert report["updates"] == 2
    assert report["healthy"] is True
    assert report["metadata"]["environment"]["checkpointable"] is (backend == "gymnax")
