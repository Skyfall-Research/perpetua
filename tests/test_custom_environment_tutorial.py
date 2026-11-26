"""Every step of the custom-environment tutorial runs as shown in the docs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEPS = sorted((ROOT / "examples" / "custom_environment").glob("tutorial*.py"))


def test_the_tutorial_has_four_steps_and_pages() -> None:
    assert [step.name for step in STEPS] == [
        "tutorial1_skeleton.py",
        "tutorial2_logic.py",
        "tutorial3_drift_and_masks.py",
        "tutorial4_testing.py",
    ]
    pages = ROOT / "docs" / "tutorials" / "custom-environment"
    for step in STEPS:
        included = [page for page in pages.glob("*.md") if step.name in page.read_text()]
        assert included, f"{step.name} is not included in any tutorial page"


@pytest.mark.parametrize("step", STEPS, ids=lambda path: path.stem)
def test_step_runs(step: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(step)],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
