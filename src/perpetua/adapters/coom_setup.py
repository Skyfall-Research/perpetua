"""Build COOM's pinned legacy runtime for ``perpetua.adapters.coom``.

COOM cannot share Perpetua's environment (it pins Gymnasium 0.28.1 and needs
legacy Gym), and its PyPI wheel is empty, so this module clones a pinned COOM
commit and creates a separate Python 3.12 environment with the dependency set
measured to work. Nothing is installed into Perpetua's own environment.

    python -m perpetua.adapters.coom_setup [--root DIR]

Requires ``git`` and ``uv``. Point the adapter elsewhere with
``PERPETUA_COOM_PYTHON`` and ``PERPETUA_COOM_SOURCE``.
"""

import argparse
import subprocess
from pathlib import Path

from perpetua.adapters.coom import COOM_COMMIT, DEFAULT_ROOT

__all__ = ["main"]

REPOSITORY = "https://github.com/TTomilin/COOM.git"
REQUIREMENTS = (
    "gymnasium==0.28.1",
    "gym==0.26.2",
    "vizdoom==1.3.2",
    "scipy==1.11.4",
    "numpy==1.26.4",
    "opencv-python-headless==4.11.0.86",
)


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def usable(python: Path) -> bool:
    """Whether an environment's interpreter is there and starts."""
    try:
        return subprocess.run([str(python), "-c", ""], capture_output=True).returncode == 0
    except OSError:
        return False


def ensure_venv(venv: Path) -> None:
    """Create the runtime's environment, replacing one whose interpreter no longer runs.

    An environment outlives the Python it was made from (a removed or moved
    uv-managed interpreter leaves ``bin/python`` dangling), and ``uv venv``
    refuses an existing directory without ``--clear``.
    """
    if not usable(venv / "bin" / "python"):
        run("uv", "venv", "--quiet", "--clear", "--python", "3.12", str(venv))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    root = parser.parse_args().root
    source, venv = root / "src", root / "venv"
    if not (source / ".git").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", REPOSITORY, str(source))
    run("git", "-C", str(source), "checkout", "--quiet", COOM_COMMIT)
    ensure_venv(venv)
    run("uv", "pip", "install", "--quiet", "--python", str(venv / "bin" / "python"), *REQUIREMENTS)
    print(f"PERPETUA_COOM_PYTHON={venv / 'bin' / 'python'}")
    print(f"PERPETUA_COOM_SOURCE={source}")


if __name__ == "__main__":
    main()
