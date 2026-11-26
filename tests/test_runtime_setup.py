"""The COOM and AgarCL setup modules repair a runtime environment that no longer runs."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from perpetua.adapters import agarcl_setup, coom_setup

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH.")
SETUPS = [coom_setup, agarcl_setup]


def dangling(python: Path) -> None:
    python.parent.mkdir(parents=True)
    python.symlink_to(python.parent / "removed" / "python3.12")


def failing(python: Path) -> None:
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)


@pytest.mark.parametrize("setup", SETUPS, ids=["coom", "agarcl"])
@pytest.mark.parametrize("damage", [dangling, failing])
def test_an_environment_whose_interpreter_does_not_run_is_recreated(
    setup: ModuleType, damage: object, tmp_path: Path
) -> None:
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    assert callable(damage)
    damage(python)
    (venv / "stale").write_text("")
    assert not setup.usable(python)
    setup.ensure_venv(venv)
    assert setup.usable(python)
    assert not (venv / "stale").exists()


@pytest.mark.parametrize("setup", SETUPS, ids=["coom", "agarcl"])
def test_a_working_environment_is_kept(setup: ModuleType, tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    assert not setup.usable(venv / "bin" / "python")
    setup.ensure_venv(venv)
    (venv / "installed").write_text("")
    setup.ensure_venv(venv)
    assert (venv / "installed").exists()
    assert setup.usable(venv / "bin" / "python")


def test_coom_setup_reruns_over_a_dangling_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "coom"
    (root / "src" / ".git").mkdir(parents=True)
    python = root / "venv" / "bin" / "python"
    dangling(python)
    calls: list[tuple[str, ...]] = []

    def run(*command: str) -> None:
        calls.append(command)
        if command[:2] == ("uv", "venv"):
            subprocess.run(command, check=True)

    monkeypatch.setattr(coom_setup, "run", run)
    monkeypatch.setattr(sys, "argv", ["coom_setup", "--root", str(root)])
    coom_setup.main()
    assert [command[:3] for command in calls] == [
        ("git", "-C", str(root / "src")),
        ("uv", "venv", "--quiet"),
        ("uv", "pip", "install"),
    ]
    assert "--clear" in calls[1]
    assert coom_setup.usable(python)
    assert f"PERPETUA_COOM_PYTHON={python}" in capsys.readouterr().out
