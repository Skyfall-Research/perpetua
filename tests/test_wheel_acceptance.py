"""A new user can install the wheel, follow a tutorial and reproduce a result."""

import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    # The clean wheel installs CPU-only JAX; an inherited JAX_PLATFORMS=cuda
    # (as on the GPU suite) would make it fail to initialise a backend.
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, f"{command[:3]} failed:\n{result.stderr[-3000:]}"
    return result


def test_clean_wheel_tutorial_and_measured_example(tmp_path: Path) -> None:
    run("uv", "build", "--wheel", "--out-dir", str(tmp_path / "dist"), cwd=ROOT)
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    run("uv", "venv", "--quiet", str(tmp_path / "venv"))
    python = tmp_path / "venv" / "bin" / "python"
    run("uv", "pip", "install", "--quiet", "--python", str(python), str(wheel))

    # What the "runtime not found" errors tell a user to run is in the wheel.
    packaged = set(zipfile.ZipFile(wheel).namelist())
    for name in ("coom", "agarcl"):
        assert f"perpetua/adapters/{name}_setup.py" in packaged
        assert f"perpetua/adapters/{name}_worker.py" in packaged
        usage = run(
            str(python), "-I", "-m", f"perpetua.adapters.{name}_setup", "--help", cwd=tmp_path
        )
        assert "--root" in usage.stdout

    # A wheel built from a checkout names its commit, and the install reports it.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "pyproject.toml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode == 0 and tracked.returncode == 0:
        commit = head.stdout.strip()
        generated = zipfile.ZipFile(wheel).read("perpetua/_revision.py").decode()
        assert f"revision = {commit!r}" in generated
        assert str(ROOT) not in generated
        assert str(Path.home()) not in generated
        shown = run(
            str(python),
            "-I",
            "-c",
            "import json, perpetua; print(json.dumps(perpetua.reporting.provenance()))",
            cwd=tmp_path,
        )
        record = json.loads(shown.stdout)
        assert record["revision"] == commit
        assert isinstance(record["dirty"], bool)
    assert not (ROOT / "src" / "perpetua" / "_revision.py").exists()

    tutorial = (ROOT / "docs" / "tutorials" / "custom-environment.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", tutorial, re.DOTALL)
    assert blocks
    for code in blocks:
        run(str(python), "-I", "-c", code, cwd=tmp_path)

    result = run(
        str(python),
        "-I",
        str(ROOT / "examples" / "continual_grid.py"),
        "--chunks",
        "1",
        "--drift",
        "cyclic",
        cwd=tmp_path,
    )
    report = json.loads(result.stdout)
    assert report["ticks_per_lifetime"] == 256
    np.testing.assert_allclose(
        np.asarray(report["mean_reward"]),
        np.asarray(
            [
                [0.140625, 0.1484375],
                [0.109375, 0.140625],
                [0.140625, 0.17578125],
                [0.13671875, 0.14453125],
            ]
        ),
        atol=1e-6,
    )
