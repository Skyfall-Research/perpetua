"""Build hook: record the source commit in the wheel and the sdist.

Building from a git checkout writes ``perpetua/_revision.py`` (the commit hash
and whether tracked files had uncommitted changes) into the distribution, so a
non-editable install can still name its code. The file is generated in a
temporary directory and never written into the source tree. An sdist carries
it, and a wheel built from that sdist keeps it. Outside a checkout, with no
such file, nothing is recorded. No path, host name or user name is stored.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_MODULE = "src/perpetua/_revision.py"


def _git(root: Path, *arguments: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _checkout_revision(root: Path) -> tuple[str, bool | None] | None:
    # An unpacked sdist can sit beneath an unrelated repository, so the
    # checkout must track this project.
    if _git(root, "ls-files", "--error-unmatch", "pyproject.toml") is None:
        return None
    commit = _git(root, "rev-parse", "HEAD")
    if not commit or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return commit, None if status is None else bool(status)


class RevisionBuildHook(BuildHookInterface[Any]):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        # An editable install is a checkout and reads its live git state.
        if version == "editable":
            return
        root = Path(self.root)
        found = _checkout_revision(root)
        if found is not None:
            self._scratch = tempfile.mkdtemp(prefix="perpetua-revision-")
            source = Path(self._scratch) / "_revision.py"
            source.write_text(
                '"""Generated at build time; not part of the source tree."""\n\n'
                f"revision = {found[0]!r}\ndirty = {found[1]!r}\n"
            )
        elif (root / _MODULE).is_file():
            source = root / _MODULE
        else:
            return
        destination = _MODULE if self.target_name == "sdist" else _MODULE.removeprefix("src/")
        build_data["force_include"][str(source)] = destination

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        scratch = getattr(self, "_scratch", None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
