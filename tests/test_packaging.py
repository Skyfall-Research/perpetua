"""What the package index shows: the README as its long description, and the metadata."""

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Skyfall-Research/perpetua"
LINK = re.compile(r"""\]\(([^)\s]+)\)|(?:src|href)=["']([^"']+)["']""")


def test_readme_links_are_absolute_and_point_into_the_repository() -> None:
    targets = [a or b for a, b in LINK.findall((ROOT / "README.md").read_text())]
    assert len(targets) > 20
    relative = [target for target in targets if not re.match(r"https?://", target)]
    assert relative == [], "the package index renders the README outside the repository"
    for prefix in (f"{REPOSITORY}/blob/main/", "https://raw.githubusercontent.com/"):
        for target in targets:
            if target.startswith(prefix):
                path = target.removeprefix(prefix).partition("#")[0]
                path = path.removeprefix("Skyfall-Research/perpetua/main/")
                assert (ROOT / path).is_file(), target


def test_metadata_names_the_project_without_a_personal_address() -> None:
    project: dict[str, Any] = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["authors"] == [{"name": "Skyfall Research"}]
    assert project["keywords"]
    urls: dict[str, str] = project["urls"]
    assert {"Repository", "Issues", "Documentation"} <= urls.keys()
    assert "Homepage" not in urls  # no site is deployed yet
    assert all(url.startswith(REPOSITORY) for url in urls.values())
    assert (ROOT / "docs" / "index.md").is_file()
