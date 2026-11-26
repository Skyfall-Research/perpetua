"""Scripts describe themselves with --help without running their job, and fail loudly."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_help(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=ROOT,
    )


@pytest.mark.parametrize(
    "script",
    [
        "scripts/generate_env_docs.py",
        "scripts/probe_gymnax.py",
        "scripts/record_baselines.py",
        "scripts/reproduce_continual_world.py",
        "scripts/survey_regret.py",
        "examples/custom_env.py",
    ],
)
def test_help_prints_usage_and_does_not_run_the_job(script: str) -> None:
    before = {
        page: page.stat().st_mtime_ns for page in (ROOT / "docs" / "environments").glob("*.md")
    }
    result = run_help(ROOT / script)
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.startswith("usage:")
    assert "wrote " not in result.stdout
    after = {
        page: page.stat().st_mtime_ns for page in (ROOT / "docs" / "environments").glob("*.md")
    }
    assert before == after


# Written for other tools rather than as a result: the regression fixture, and
# the summaries the plots print of reports that already carry their provenance.
NOT_REPORTS = {"scripts/record_baselines.py", "scripts/plot_claim_figures.py"}


def test_every_json_report_carries_its_provenance() -> None:
    sources = sorted([*(ROOT / "examples").glob("*.py"), *(ROOT / "scripts").glob("*.py")])
    reports = {
        str(path.relative_to(ROOT)) for path in sources if "json.dumps(" in path.read_text()
    } - NOT_REPORTS
    assert len(reports) >= 22
    missing = sorted(
        name for name in reports if not re.search(r'"provenance": ', (ROOT / name).read_text())
    )
    assert not missing, f"JSON reports without a provenance key: {missing}"


def load_reproduction() -> ModuleType:
    path = ROOT / "scripts" / "reproduce_continual_world.py"
    spec = importlib.util.spec_from_file_location("reproduce_continual_world", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_run(path: Path, *, kind: str, evals: int, complete: bool = True, **header: Any) -> None:
    rows: list[dict[str, Any]] = [{"kind": "header", **header}]
    rows += [{"kind": "eval", "tick": i, "task": 0, "success": 0.1 * i} for i in range(evals)]
    rows.append({"kind": "done", "tick": evals, "complete": complete})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def write_cl(path: Path, *, complete: bool = True, seed: int = 0) -> None:
    rows: list[dict[str, Any]] = [
        {"kind": "header", "tasks": ["a"], "steps_per_task": 10, "seed": seed}
    ]
    rows += [{"kind": "eval", "tick": i, "task": 0, "success": 0.2 * i} for i in range(1, 4)]
    rows.append(
        {"kind": "boundary_eval", "tick": 10, "task": 0, "training_task": 0, "success": 0.5}
    )
    rows.append({"kind": "done", "tick": 10, "complete": complete})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def args_for(module: ModuleType, cl: list[str], reference: list[str]) -> Any:
    import argparse

    return argparse.Namespace(cl=cl, reference=reference)


def test_metrics_fail_on_an_empty_glob(tmp_path: Path) -> None:
    module = load_reproduction()
    with pytest.raises(SystemExit, match="--cl matched no files"):
        module.metrics(args_for(module, [str(tmp_path / "none-*.jsonl")], []))
    write_cl(tmp_path / "cl.jsonl")
    with pytest.raises(SystemExit, match="--reference matched no files"):
        module.metrics(
            args_for(module, [str(tmp_path / "cl.jsonl")], [str(tmp_path / "ref-*.jsonl")])
        )


def test_metrics_refuse_aborted_and_mismatched_reference_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_reproduction()
    write_cl(tmp_path / "cl.jsonl")
    cl = [str(tmp_path / "cl.jsonl")]
    write_run(tmp_path / "ref-a.jsonl", kind="reference", evals=3, reference_task_index=0)
    module.metrics(args_for(module, cl, [str(tmp_path / "ref-a.jsonl")]))
    assert '"seeds": 1' in capsys.readouterr().out
    write_run(
        tmp_path / "ref-b.jsonl", kind="reference", evals=3, complete=False, reference_task_index=0
    )
    with pytest.raises(SystemExit, match=r"did not finish healthy.*ref-b\.jsonl"):
        module.metrics(args_for(module, cl, [str(tmp_path / "ref-*.jsonl")]))
    (tmp_path / "ref-b.jsonl").unlink()
    write_run(tmp_path / "ref-c.jsonl", kind="reference", evals=2, reference_task_index=0)
    with pytest.raises(SystemExit, match=r"3 evaluations.*\[2, 3\].*same window"):
        module.metrics(args_for(module, cl, [str(tmp_path / "ref-*.jsonl")]))


def test_metrics_skip_a_cl_run_that_ended_unhealthy_after_its_final_evaluations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_reproduction()
    write_cl(tmp_path / "cl-0.jsonl", seed=0)
    write_cl(tmp_path / "cl-1.jsonl", seed=1, complete=False)
    module.metrics(args_for(module, [str(tmp_path / "cl-*.jsonl")], []))
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert [seed["seed"] for seed in report["per_seed"]] == [0]
    assert "cl-1.jsonl: aborted" in captured.err
    (tmp_path / "cl-0.jsonl").unlink()
    with pytest.raises(SystemExit, match="no complete CL run"):
        module.metrics(args_for(module, [str(tmp_path / "cl-*.jsonl")], []))


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reset_figure_refuses_more_batch_sizes_than_it_has_styles(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib", reason="the figures need matplotlib")
    matplotlib.use("Agg")
    module = load_script("plot_claim_figures")

    def report(batches: list[int]) -> str:
        rows = [
            {
                "env": "gymnax/CartPole-v1",
                "batch": batch,
                "lifetime_steps_per_second": 2.0,
                "autoreset_steps_per_second": 1.0,
                "autoreset_flops": 3.0,
                "step_flops": 2.0,
            }
            for batch in batches
        ]
        path = tmp_path / f"reset-{len(batches)}.json"
        path.write_text(json.dumps({"backend": "cpu", "rows": rows}))
        return str(path)

    module.plot_reset(report([1, 64]), tmp_path / "two.png")
    assert (tmp_path / "two.png").stat().st_size > 0
    with pytest.raises(ValueError, match=r"at most 2 batch sizes.*\[1, 64, 1024\]"):
        module.plot_reset(report([1, 64, 1024]), tmp_path / "three.png")
    assert not (tmp_path / "three.png").exists()


def test_soak_growth_ignores_samples_of_an_exited_worker() -> None:
    module = load_script("soak")
    assert module._settled_growth([100, None, 130, None]) == 30
    assert module._settled_growth([None, None]) is None
    assert module._settled_growth([]) is None
    # The first tenth of the 20 readable samples is warm-up: 118 - 101.
    assert module._settled_growth([1, *range(100, 119), None]) == 17


def test_gymnax_probe_counts_the_fields_the_adapter_can_drift() -> None:
    gymnax = pytest.importorskip("gymnax", reason="the probe needs gymnax")
    from perpetua.adapters.gymnax_params import GymnaxParameters

    module = load_script("probe_gymnax")
    for name in ("CartPole-v1", "MountainCar-v0", "Catch-bsuite", "Freeway-MinAtar"):
        _, params = gymnax.make(name)
        floats = module.float_fields(params)
        assert "max_steps_in_episode" not in floats, name
        assert all(type(getattr(params, field)) is float for field in floats), name
        # The adapter lists its own float fields when asked to drift an unknown one.
        with pytest.raises(ValueError, match=re.escape(f"float fields are {floats}")):
            GymnaxParameters(name, drift_fields=("not_a_field",))
    _, params = gymnax.make("Catch-bsuite")
    assert module.float_fields(params) == []  # its parameters are integers only


def run_baseline(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "measure_baseline.py"), "--device", "cpu", *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=ROOT,
    )


def test_regret_survey_reports_a_blind_and_an_informative_environment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "survey_regret.py"),
            *("--ticks", "64", "--actions", "4", "--seeds", "0"),
            *("--env", "gymnax/MountainCar-v0", "--env", "ContinualGrid-v0"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-500:]
    rows = dict(re.findall(r"^\| `(.+?)` \| (\S+) \|$", result.stdout, re.M))
    assert set(rows) == {"gymnax/MountainCar-v0", "ContinualGrid-v0"}
    assert float(rows["gymnax/MountainCar-v0"]) == 0.0
    assert float(rows["ContinualGrid-v0"]) > 0.0


@pytest.mark.torch
def test_baseline_rejects_minibatches_that_do_not_divide_the_batch() -> None:
    for minibatches in ("5", "32"):  # a remainder; minibatches of one sample
        result = run_baseline(
            "--num-envs", "2", "--num-steps", "16", "--num-minibatches", minibatches
        )
        assert result.returncode != 0
        assert "must divide the rollout batch size" in result.stderr, result.stderr[-500:]


@pytest.mark.torch
def test_baseline_progress_is_valid_json_before_the_first_episode_ends() -> None:
    result = run_baseline(
        "--num-envs", "2", "--num-steps", "2", "--num-minibatches", "2", "--warmup-updates", "1",
        "--total-steps", "4", "--progress-every", "1",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-500:]

    def reject(constant: str) -> Any:
        raise ValueError(f"non-finite JSON constant {constant}")

    progress = [
        json.loads(line, parse_constant=reject)
        for line in result.stderr.splitlines()
        if line.startswith("{")
    ]
    assert progress
    assert progress[0]["mean_episode_return"] is None
    assert "Mean of empty slice" not in result.stderr
