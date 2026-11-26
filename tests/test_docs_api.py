"""The API reference documents every public module and declared name, and its code blocks run."""

import ast
import importlib
import json
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_tutorials import BLOCK, PREAMBLE, RUN_BLOCKS

import perpetua

API = Path(__file__).resolve().parents[1] / "docs" / "api"
DOCUMENTED = sorted(
    set(re.findall(r"automodule:: (\S+)", "\n".join(p.read_text() for p in API.glob("*.md"))))
)
# Workers run in other runtimes, so they neither import here nor declare a surface.
WORKERS = {"perpetua.adapters.agarcl_worker", "perpetua.adapters.coom_worker"}
# Packages are documented through their modules; the setup modules are commands
# (`python -m`), described in the guides.
SETUP = {"perpetua.adapters.agarcl_setup", "perpetua.adapters.coom_setup"}
UNDOCUMENTED = {"perpetua.adapters", "perpetua.agents", "perpetua.envs", *WORKERS, *SETUP}
MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(perpetua.__path__, "perpetua.")
    if not module.name.rsplit(".", 1)[-1].startswith("_")
)

_ALIAS = "a type alias: it has no docstring and appears in the signatures that use it"
_RUNTIME = "the pin or location of a separately built runtime, read by its setup script"
_TABLE = "a lookup table the tests index; the page of the class that uses it describes it"
_PRESET = "a task-name tuple the benchmarks guide describes in prose"
# Names a module declares public that autodoc does not render, each with the reason.
UNDOCUMENTED_NAMES = {
    "perpetua.adapters.agarcl.AGARCL_COMMIT": _RUNTIME,
    "perpetua.adapters.agarcl.GLFW_COMMIT": _RUNTIME,
    "perpetua.adapters.agarcl.DEFAULT_ROOT": _RUNTIME,
    "perpetua.adapters.agarcl.GRAND_ARENA": "the preset behind agarcl/grand-arena-v0, whose "
    "environment page lists its values",
    "perpetua.adapters.coom.COOM_COMMIT": _RUNTIME,
    "perpetua.adapters.coom.DEFAULT_ROOT": _RUNTIME,
    "perpetua.adapters.mjx.TASKS": _TABLE,
    "perpetua.adapters.morpheus.FEATURES": _TABLE,
    "perpetua.adapters.morpheus.State": _ALIAS,
    "perpetua.adapters.morpheus.RewardFunction": _ALIAS,
    "perpetua.adapters.pettingzoo.ParallelAction": _ALIAS,
    "perpetua.adapters.snapshots.BackendSnapshot": _ALIAS,
    "perpetua.adapters.worker.STDERR_TAIL": "the stderr bound WorkerProcess documents",
    "perpetua.agents.dqn.Transition": _ALIAS,
    "perpetua.agents.encoders.Activation": _ALIAS,
    "perpetua.agents.encoders.Torso": _ALIAS,
    "perpetua.agents.plasticity.Params": _ALIAS,
    "perpetua.agents.sac.SACTransition": _ALIAS,
    "perpetua.agents.sac.SACActivation": _ALIAS,
    "perpetua.agents.sac.SACKernelInit": _ALIAS,
    "perpetua.core.JSONValue": _ALIAS,
    "perpetua.envs.bean_world.ITEMS": _TABLE,
    "perpetua.envs.continual_grid.MOVES": _TABLE,
    "perpetua.evaluation.Act": _ALIAS,
    "perpetua.policies.Policy": _ALIAS,
    "perpetua.presets.CW10": _PRESET,
    "perpetua.presets.CW20": _PRESET,
    "perpetua.reporting.Statistic": _ALIAS,
    "perpetua.tasks.OnSwitch": _ALIAS,
    "perpetua.tuning.Score": _ALIAS,
}


def load(name: str) -> ModuleType:
    return importlib.import_module(name)


def modules(names: list[str]) -> list[Any]:
    """One parameter per module; the PyTorch bridge needs the torch extra."""
    torch = {"perpetua.torch_bridge"}
    return [pytest.param(name, marks=[pytest.mark.torch] * (name in torch)) for name in names]


def rendered(module: ModuleType) -> set[str]:
    """The module-level names autodoc renders: a docstring, or a ``#:`` comment above."""
    assert module.__file__ is not None
    source = Path(module.__file__).read_text()
    lines = source.splitlines()
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            if ast.get_docstring(node):
                names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign) and lines[node.lineno - 2].startswith(
            "#:"
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def test_every_public_module_has_an_api_page() -> None:
    assert set(MODULES) - UNDOCUMENTED - set(DOCUMENTED) == set()
    assert "perpetua.envs.continual_grid" in MODULES


@pytest.mark.parametrize("name", modules(DOCUMENTED))
def test_documented_modules_import(name: str) -> None:
    # Sphinx is configured to ignore autodoc import errors from optional
    # extras, so an import failure would otherwise leave an empty page.
    load(name)


@pytest.mark.parametrize("name", modules(["perpetua", *sorted(set(MODULES) - WORKERS)]))
def test_every_module_declares_names_that_exist(name: str) -> None:
    module = load(name)
    declared = getattr(module, "__all__", None)
    assert declared is not None, f"{name} declares no __all__"
    assert len(set(declared)) == len(declared)
    for public in declared:
        assert not public.startswith("_"), f"{name}.{public}"
        getattr(module, public)


@pytest.mark.parametrize("name", modules(DOCUMENTED))
def test_every_declared_name_is_documented_or_listed_with_a_reason(name: str) -> None:
    module = load(name)
    shown = rendered(module)
    listed = {key.rsplit(".", 1)[1] for key in UNDOCUMENTED_NAMES if key.rsplit(".", 1)[0] == name}
    declared = set(module.__all__)
    assert declared - shown - listed == set()
    # A stale entry: the name is no longer declared, or it has gained documentation.
    assert listed - (declared - shown) == set()


def test_undocumented_names_belong_to_documented_modules() -> None:
    assert {key.rsplit(".", 1)[0] for key in UNDOCUMENTED_NAMES} <= set(DOCUMENTED)
    assert all(UNDOCUMENTED_NAMES.values())


PAGES_WITH_CODE = sorted(page for page in API.glob("*.md") if BLOCK.search(page.read_text()))


@pytest.mark.parametrize("path", PAGES_WITH_CODE, ids=lambda page: page.name)
def test_api_page_code_blocks_execute_in_order(path: Path, tmp_path: Path) -> None:
    # Like the guides, a reference page runs as one program after the common
    # imports. No block is exempt: none needs a runtime CI lacks.
    program = [(-1, PREAMBLE), *enumerate(BLOCK.findall(path.read_text()))]
    result = subprocess.run(
        [sys.executable, "-c", RUN_BLOCKS],
        input=json.dumps(program),
        cwd=tmp_path,  # the checkpoint page writes one
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, f"{path.name}:\n{result.stderr[-2000:]}"
