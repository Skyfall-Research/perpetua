"""Every Python block in the tutorials and introduction must run, isolated and unchanged."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
TUTORIALS = DOCS / "tutorials"
PAGES = sorted([*TUTORIALS.glob("*.md"), *(DOCS / "introduction").glob("*.md"), DOCS / "index.md"])
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_tutorial_code_blocks_execute(path: Path) -> None:
    blocks = BLOCK.findall(path.read_text())
    if not blocks:
        pytest.skip(f"{path.name} is an index page without code")
    for index, code in enumerate(blocks):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, f"{path.name} block {index} failed:\n{result.stderr[-2000:]}"


def test_environment_creation_guide_runs_as_one_script() -> None:
    # The reference guide builds one example across several blocks; after the
    # contract sketch, pasting them in order must give a working program.
    blocks = re.findall(r"```python\n(.*?)```", (DOCS / "custom-environments.md").read_text(), re.S)
    assert blocks[0].startswith("class MyEnv:")
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(blocks[1:])],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


# The guides are reference pages: their blocks build on each other and leave
# out the common imports, so each page runs as one program after PREAMBLE.
GUIDES = sorted(
    page
    for page in [
        *DOCS.glob("*.md"),
        *(DOCS / "usage").glob("*.md"),
        *(DOCS / "presets").glob("*.md"),
        DOCS.parent / "README.md",
    ]
    if page.relative_to(DOCS.parent).as_posix()
    not in {"docs/index.md", "docs/custom-environments.md"}
)
PREAMBLE = "import jax\nimport jax.numpy as jnp\nimport numpy as np\nimport perpetua as pp\n"
# Blocks that are not run, by a line they contain. Each must match one block.
EXEMPT: dict[str, dict[str, str]] = {
    "docs/adapters.md": {'pp.make("envpool/CartPole-v1", num_envs=256': "needs EnvPool (Linux)"},
    "docs/agarcl.md": {
        'env = pp.make("agarcl/agario-screen-v0")': "needs the AgarCL runtime",
        "replay_capacity=1_000_000": "needs the AgarCL runtime",
    },
    "docs/benchmarks.md": {'pp.make("coom/COOM-v0"': "needs the COOM runtime"},
    "docs/presets/coom.md": {'pp.presets.coom_sequence("CO8"': "needs the COOM runtime"},
    "docs/presets/switching.md": {"pp.presets.switching_atari(": "needs EnvPool Atari (Linux)"},
    "docs/task-switching.md": {"# Atari through EnvPool": "needs EnvPool Atari (Linux)"},
}
RUN_BLOCKS = """
import json, sys, traceback
namespace = {}
for index, code in json.load(sys.stdin):
    try:
        exec(code, namespace)
    except BaseException:
        traceback.print_exc()
        sys.exit(f"block {index} failed")
"""


def _name(path: Path) -> str:
    return path.relative_to(DOCS.parent).as_posix()


# Blocks that run only where an optional package is installed (the eight CI
# groups have no torch; the torch jobs of ci.yml and learning.yml select the
# guides that hold a torch block with `-m torch`).
OPTIONAL: dict[str, dict[str, str]] = {
    "docs/benchmarks.md": {'pp.make("meal/overcooked-v0"': "meal"},
    "docs/presets/meal.md": {"pp.presets.meal_sequence(": "meal"},
    "docs/training.md": {"from perpetua.torch_bridge import TorchPPO, TorchRunner": "torch"},
}


def _exempt(path: Path, code: str) -> bool:
    if any(marker in code for marker in EXEMPT.get(_name(path), {})):
        return True
    return any(
        marker in code and importlib.util.find_spec(package) is None
        for marker, package in OPTIONAL.get(_name(path), {}).items()
    )


def _guides() -> list[Any]:
    partial = [pytest.mark.torch(partial=True)]
    return [
        pytest.param(
            path,
            id=_name(path),
            marks=partial * ("torch" in OPTIONAL.get(_name(path), {}).values()),
        )
        for path in GUIDES
    ]


@pytest.mark.parametrize("path", _guides())
def test_guide_code_blocks_execute_in_order(path: Path, tmp_path: Path) -> None:
    blocks = BLOCK.findall(path.read_text())
    if not blocks:
        pytest.skip(f"{path.name} has no Python blocks")
    program = [(-1, PREAMBLE)]
    program += [(index, code) for index, code in enumerate(blocks) if not _exempt(path, code)]
    result = subprocess.run(
        [sys.executable, "-c", RUN_BLOCKS],
        input=json.dumps(program),
        cwd=tmp_path,  # some blocks write checkpoints
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, f"{_name(path)}:\n{result.stderr[-2000:]}"


def test_every_guide_exemption_matches_exactly_one_block() -> None:
    pages = {_name(page): page for page in GUIDES}
    assert set(EXEMPT) | set(OPTIONAL) <= set(pages)
    for name, markers in (*EXEMPT.items(), *OPTIONAL.items()):
        blocks = BLOCK.findall(pages[name].read_text())
        for marker in markers:
            assert sum(marker in code for code in blocks) == 1, (name, marker)
