"""Comparability runner and integration-cost measurement helpers."""

# pyright: reportUnknownMemberType=false

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_measure() -> Any:
    path = ROOT / "scripts" / "measure_integration_cost.py"
    spec = importlib.util.spec_from_file_location("measure_integration_cost", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = load_measure()
code_blocks = measure.code_blocks
code_lines = measure.code_lines


def load_comparability() -> Any:
    path = ROOT / "examples" / "compare_benchmarks.py"
    spec = importlib.util.spec_from_file_location("compare_benchmarks", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_markdown_step_blocks_and_code_line_counts() -> None:
    text = """Before.
```python
# comment
env = gym.make("x")
for _ in range(2):
    observation, reward, terminated, truncated, info = env.step(action)
```
After."""
    blocks = code_blocks(text)
    assert len(blocks) == 1
    assert "env.step" in blocks[0]
    assert code_lines(blocks[0]) == 3
    assert code_lines('"""docstring"""\nvalue = 1\n') == 1


def test_morpheus_comparability_smoke() -> None:
    module = load_comparability()
    args = argparse.Namespace(tasks=1, updates_per_task=1, rollout_steps=4, eval_ticks=4, seeds=[0])
    result = module.report("morpheus", args)
    assert result["benchmark"] == "morpheus"
    assert result["forgetting"] is None
    assert result["recovery_ticks"] is None
    assert result["scores"][0][0][0] == result["mean_reward"]


def test_switch_recovery_excludes_wraparound_and_judges_inside_new_task() -> None:
    module = load_comparability()
    period = 8
    # Task 0 earns 1; task 1 earns 0 for 3 ticks then 1 (recovers);
    # task 2 earns 0 throughout (never recovers). Index k*period-1 is the switch.
    trace = np.ones(3 * period, dtype=np.float32)
    trace[period - 1 : period + 2] = 0.0
    trace[2 * period - 1 :] = 0.0
    assert module.switch_recovery(trace, 3, period) == [3, None]


def test_integration_snippets_cover_every_benchmark() -> None:
    assert code_lines(measure.SINGLE_SNIPPET) == 4
    assert code_lines(measure.ALL_SNIPPET) == 5
    assert all(env_id in measure.ALL_SNIPPET for env_id in measure.ENV_IDS.values())
    compile(measure.ALL_SNIPPET, "<all-benchmarks snippet>", "exec")


def test_uniform_policy_samples_valid_discrete_and_continuous_actions() -> None:
    import gymnasium as gym
    import jax
    import jax.numpy as jnp

    import perpetua as pp

    box = gym.spaces.Box(np.full((2, 3), -1.0), np.full((2, 3), 2.0), dtype=np.float32)
    action = pp.policies.uniform(box)(jnp.zeros((2, 1)), cast(Any, {}), jax.random.key(0))
    assert action.shape == (2, 3)
    assert box.contains(np.asarray(action))
    discrete = gym.spaces.MultiDiscrete([3, 3])
    assert pp.policies.uniform(discrete) is pp.policies.random


def test_continual_world_reports_success_metrics() -> None:
    module = load_comparability()
    args = argparse.Namespace(tasks=2, updates_per_task=1, rollout_steps=4, eval_ticks=4, seeds=[0])
    success = module.report("continual_world", args)["success"]
    assert success["definition"].startswith("fraction of evaluation ticks")
    assert 0.0 <= success["average_performance"] <= 1.0
    assert success["forgetting"] is not None
    assert success["forward_transfer"] is None
    assert np.asarray(success["rates"]).shape == (1, 2, 2)  # (seeds, policies, tasks)


def test_metaworld_success_is_oracle_only() -> None:
    import jax
    import jax.numpy as jnp

    import perpetua as pp

    inner: Any = pp.make("metaworld/MT1-v0", env_name="reach-v3")
    try:
        assert "success" in inner.oracle_fields
        env = pp.Oracle(inner, expose=["success"])
        state, _, info = env.init(jax.random.key(0))
        assert "success" not in info
        state, _, _, info = env.step(state, jnp.zeros((1, 4), jnp.float32), jax.random.key(1))
        oracle = cast(dict[str, Any], info.get("oracle"))
        assert float(oracle["success"][0]) == float(inner.env.backend.last_info["success"])
    finally:
        inner.close()


def test_metaworld_success_is_required_after_a_step() -> None:
    import jax
    import jax.numpy as jnp
    import pytest

    import perpetua as pp

    inner: Any = pp.make("metaworld/MT1-v0", env_name="reach-v3")
    try:
        state, _, _ = inner.init(jax.random.key(0))
        assert float(inner.privileged(state)["success"][0]) == 0.0  # no step yet
        state, *_ = inner.step(state, jnp.zeros((1, 4), jnp.float32), jax.random.key(1))
        del inner.env.backend.last_info["success"]  # e.g. a Meta-World release renames it
        with pytest.raises(RuntimeError, match="no 'success' entry"):
            inner.privileged(state)
    finally:
        inner.close()


def test_continual_world_reproduction_script_runs_end_to_end(tmp_path: Path) -> None:
    import json
    import os
    import subprocess

    script = ROOT / "scripts" / "reproduce_continual_world.py"
    small = [
        "--tasks", "2", "--steps-per-task", "100", "--eval-every", "50",
        "--eval-episodes", "1", "--episode-steps", "25",
        "--random-steps", "25", "--update-after", "25", "--buffer", "500",
    ]  # fmt: skip
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(*args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(script), *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        return result.stdout

    run("cl", "--seed", "0", "--out", str(tmp_path / "cl-0.jsonl"), *small)
    for task in ("0", "1"):
        run("reference", "--task", task, "--out", str(tmp_path / f"ref-{task}.jsonl"), *small)
    records = [json.loads(line) for line in (tmp_path / "cl-0.jsonl").read_text().splitlines()]
    assert records[0]["oracle_exposure"] == ["task_id", "boundary"]
    assert set(records[0]["provenance"]) >= {"perpetua", "revision", "dirty", "jax"}
    assert [r["tick"] for r in records if r["kind"] == "eval"] == [50, 100, 150, 200]
    assert len([r for r in records if r["kind"] == "boundary_eval"]) == 4
    summary = json.loads(
        run(
            "metrics",
            "--cl",
            str(tmp_path / "cl-*.jsonl"),
            "--reference",
            str(tmp_path / "ref-*.jsonl"),
        )
    )
    assert summary["seeds"] == 1
    assert set(summary["provenance"]) >= {"perpetua", "revision", "dirty", "jax"}
    for name in ("performance", "forgetting"):
        assert len(summary[name]) == 3


def test_continual_world_metrics_stay_valid_json_when_forward_transfer_is_undefined(
    tmp_path: Path,
) -> None:
    import json
    import subprocess

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run(index: int | None) -> dict[str, Any]:
        return {
            "kind": "run",
            "seed": 0,
            "tasks": ["a", "b"],
            "steps_per_task": 200,
            "reference_task_index": index,
        }

    def evaluation(kind: str, tick: int, training: int, task: int, success: float) -> Any:
        return {
            "kind": kind,
            "tick": tick,
            "training_task": training,
            "task": task,
            "success": success,
        }

    write(
        "cl.jsonl",
        [
            run(None),
            evaluation("eval", 100, 0, 0, 0.2),
            evaluation("eval", 200, 0, 0, 0.6),
            evaluation("eval", 300, 1, 1, 0.5),
            evaluation("eval", 400, 1, 1, 0.9),
            evaluation("boundary_eval", 200, 0, 0, 0.6),
            evaluation("boundary_eval", 200, 0, 1, 0.1),
            evaluation("boundary_eval", 400, 1, 0, 0.3),
            evaluation("boundary_eval", 400, 1, 1, 0.9),
            {"kind": "done", "complete": True},
        ],
    )
    # Task 0's reference is partly learned; task 1's is solved from the start
    # (AUC 1), so its forward transfer is undefined.
    write(
        "ref0.jsonl",
        [
            run(0),
            evaluation("eval", 100, 0, 0, 0.2),
            evaluation("eval", 200, 0, 0, 0.4),
            {"kind": "done", "complete": True},
        ],
    )
    write(
        "ref1.jsonl",
        [
            run(1),
            evaluation("eval", 100, 0, 0, 1.0),
            evaluation("eval", 200, 0, 0, 1.0),
            {"kind": "done", "complete": True},
        ],
    )
    script = ROOT / "scripts" / "reproduce_continual_world.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "metrics",
            "--cl",
            str(tmp_path / "cl.jsonl"),
            "--reference",
            str(tmp_path / "ref*.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert "NaN" not in result.stdout
    report = json.loads(result.stdout)
    seed = report["per_seed"][0]
    assert seed["forward_transfer_undefined_tasks"] == 1
    assert seed["per_task"][1]["ft"] is None
    # (AUC 0.4 - reference 0.3) / (1 - 0.3), from the one defined task.
    assert seed["forward_transfer"] == pytest.approx(0.1 / 0.7)
    assert report["forward_transfer"] is not None

    # A reference run with no completed done record would bias the baseline.
    write("ref1.jsonl", [run(1), evaluation("eval", 100, 0, 0, 1.0)])
    refused = subprocess.run(
        [
            sys.executable,
            str(script),
            "metrics",
            "--cl",
            str(tmp_path / "cl.jsonl"),
            "--reference",
            str(tmp_path / "ref*.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert refused.returncode != 0
    assert "did not finish healthy" in refused.stderr
    assert "ref1.jsonl" in refused.stderr
