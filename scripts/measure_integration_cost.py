"""Measure user-facing integration glue against pinned upstream examples.

For each benchmark, clone its pinned commit to a temporary directory, select a
README/example code block that constructs and steps an environment, and count
its code lines. Compare that with an equivalent Perpetua snippet (construct and
step with uniformly random actions), per benchmark and, as the claim is worded,
for one agent across all benchmarks (the upstream total is the sum of the
separate examples, since each needs its own API).
Adapter implementation sizes are reported separately; they are not compared
with user examples as if they were the same quantity. Missing upstream examples
are reported as unavailable, not replaced with an invented baseline.
"""

import argparse
import ast
import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from perpetua.reporting import provenance

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "meal": ("https://github.com/TTomilin/MEAL.git", "8e7857c76eedc44623a7bfe987235f2a9aadb626"),
    "continual_world": (
        "https://github.com/Farama-Foundation/Metaworld.git",
        "c0dafaf1557f31cad3b96ba2b62462f45638f5a5",
    ),
    "coom": (
        "https://github.com/TTomilin/COOM.git",
        "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e",
    ),
    "agarcl": (
        "https://github.com/AgarCL/AgarCL.git",
        "d256e7b6a61bebfd3a23014f0343c2f9195ac9d4",
    ),
    "morpheus": (
        "https://github.com/Skyfall-Research/morpheus_sdk.git",
        "3a7d1d2fd89f67f452ea14719d925de179f30c0a",
    ),
}
ADAPTER_FRAGMENTS = {"continual_world": [("src/perpetua/adapters/host.py", "MetaWorldLifetime")]}
ADAPTERS = {
    "meal": ["src/perpetua/adapters/meal.py"],
    "continual_world": [],
    "coom": [
        "src/perpetua/adapters/coom.py",
        "src/perpetua/adapters/coom_worker.py",
        "src/perpetua/adapters/worker.py",  # shared worker protocol
        "src/perpetua/adapters/coom_setup.py",
    ],
    "agarcl": [
        "src/perpetua/adapters/agarcl.py",
        "src/perpetua/adapters/agarcl_worker.py",
        "src/perpetua/adapters/worker.py",  # shared worker protocol
        "src/perpetua/adapters/agarcl_setup.py",
    ],
    "morpheus": ["src/perpetua/adapters/morpheus.py"],
}
ENV_IDS = {
    "meal": "meal/overcooked-v0",
    "continual_world": "metaworld/MT1-v0",
    "coom": "coom/COOM-v0",
    "agarcl": "agarcl/agario-screen-v0",
    "morpheus": "morpheus/Morpheus-v0",
}
# Like-for-like with the upstream examples: construct, then step with uniformly
# random actions (their ``env.action_space.sample()``) for 1,000 ticks.
SINGLE_SNIPPET = """import perpetua as pp
env = pp.make("benchmark/id-v0")
runner = pp.runtime.HostRunner(env, pp.policies.uniform(env.action_space), ticks=1000)
runner(pp.runtime.init_lifetime(env, seed=0))
"""
# The claim as worded: the glue to run one agent on all of the benchmarks.
ALL_SNIPPET = f"""import perpetua as pp
for env_id in {list(ENV_IDS.values())!r}:
    env = pp.make(env_id)
    runner = pp.runtime.HostRunner(env, pp.policies.uniform(env.action_space), ticks=1000)
    runner(pp.runtime.init_lifetime(env, seed=0))
"""


def code_blocks(text: str) -> list[str]:
    """Extract fenced Markdown code blocks that contain a ``.step(`` call.

    A fence with a language tag always opens a block, so one unbalanced bare
    fence earlier in a README cannot invert the pairing of every later block.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            tagged = bool(stripped[3:].strip())
            if current is not None and not tagged:
                blocks.append("\n".join(current))
                current = None
            else:
                current = []
        elif current is not None:
            current.append(line)
    return [block for block in blocks if ".step(" in block]


def code_lines(source: str) -> int:
    """Count executable source lines, excluding comments and docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sum(
            bool(line.strip()) and not line.lstrip().startswith("#") for line in source.splitlines()
        )
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # docstrings and other literal-only lines are not glue
        if hasattr(node, "lineno"):
            start = getattr(node, "lineno", 0)
            lines.update(range(start, getattr(node, "end_lineno", start) + 1))
    return len(lines)


def fragment_lines(path: Path, class_name: str) -> int:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return code_lines(ast.get_source_segment(path.read_text(), node) or "")
    raise ValueError(f"{class_name} not found in {path}")


def upstream_example(repository: Path) -> tuple[int | None, str | None]:
    candidates = [
        repository / "README.md",
        repository / "README.rst",
        *sorted((repository / "examples").glob("**/*.py")),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        blocks = code_blocks(text) if path.suffix == ".md" else [text]
        for block in blocks:
            if ".step(" in block and ("make(" in block or "make_env(" in block):
                return code_lines(block), str(path.relative_to(repository))
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="also run the all-benchmarks snippet (8 ticks each); needs every runtime",
    )
    if parser.parse_args().execute:
        # Benchmarks such as MEAL print to stdout; keep stdout pure JSON.
        with contextlib.redirect_stdout(sys.stderr):
            exec(ALL_SNIPPET.replace("ticks=1000", "ticks=8"), {})  # noqa: S102 - our own snippet
        print("executed the all-benchmarks snippet", file=sys.stderr)
    results: dict[str, Any] = {
        "perpetua_single_benchmark_loc": code_lines(SINGLE_SNIPPET),
        "perpetua_all_benchmarks_loc": code_lines(ALL_SNIPPET),
        "provenance": provenance(),
        "benchmarks": {},
    }
    with tempfile.TemporaryDirectory(prefix="perpetua-integration-cost-") as directory:
        for name, (url, commit) in BENCHMARKS.items():
            destination = Path(directory) / name
            subprocess.run(["git", "clone", "--quiet", url, str(destination)], check=True)
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--quiet", commit], check=True
            )
            reference_loc, source = upstream_example(destination)
            adapter_loc = sum(code_lines((ROOT / path).read_text()) for path in ADAPTERS[name])
            adapter_loc += sum(
                fragment_lines(ROOT / path, class_name)
                for path, class_name in ADAPTER_FRAGMENTS.get(name, [])
            )
            results["benchmarks"][name] = {
                "commit": commit,
                "upstream_example_loc": reference_loc,
                "upstream_example": source,
                "adapter_loc": adapter_loc,
                "adapter_files": ADAPTERS[name],
            }
    counts = [entry["upstream_example_loc"] for entry in results["benchmarks"].values()]
    results["upstream_all_benchmarks_loc"] = None if None in counts else sum(counts)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
