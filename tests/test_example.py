"""The documented example must execute against every initial drift family."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("drift", ["stationary", "cyclic", "random-walk"])
def test_continual_grid_example(drift: str) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "continual_grid.py"
    result = subprocess.run(
        [sys.executable, str(script), "--chunks", "1", "--drift", drift],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)
    assert report["ticks_per_lifetime"] == 256
    assert report["lifetimes"] == 4
    assert report["agents"] == 2
    assert report["healthy"] is True
    assert report["metadata"]["oracle_exposure"] == []
    assert len(report["mean_reward"]) == 4


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["switching", "--ticks", "8", "--chunks", "1", "--period", "4"], {"period": 4}),
        (
            ["pixels", "--updates", "1", "--rollout-steps", "8", "--seeds", "0,1"],
            {"training_ticks_per_seed": 8, "eval_ticks_per_seed": 8, "seeds": [0, 1]},
        ),
    ],
    ids=["switching", "pixels"],
)
def test_switching_measurement_example(arguments: list[str], expected: dict[str, object]) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "measure_switching.py"
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    report = json.loads(result.stdout)
    assert expected.items() <= report.items()
    assert set(report["provenance"]) >= {"perpetua", "revision", "dirty", "jax"}
    if arguments[0] == "switching":
        assert report["backend"] == "minatar"
        assert report["ticks_per_second"] > 0
    else:
        assert set(report["scores"]) == set(report["reports"]) == {"ppo", "random"}
        assert all(len(scores) == 2 for scores in report["scores"].values())
