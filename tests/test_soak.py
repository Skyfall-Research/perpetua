"""Smoke runs of the soak and learning-guard scripts at tiny sizes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import perpetua as pp

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script: str, *args: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def test_native_soak_checks_clocks_memory_and_recompiles() -> None:
    records = run(
        "soak.py",
        "native",
        "--ticks",
        "262144",
        "--batch",
        "8",
        "--chunk",
        "1024",
        "--sample-every",
        "8",
    )
    summary = records[-1]
    assert summary["summary"] is True
    assert summary["ticks_per_lifetime"] == 33 * 1024  # 32 measured calls + warm-up
    assert summary["checks"] == {
        "healthy": True,
        "exact_clocks": True,
        "exact_metric_ticks": True,
        "no_recompiles_after_warmup": True,
        # None where the platform cannot measure it: RSS needs /proc (Linux),
        # device memory a GPU backend. Unmeasured is reported, not passed.
        "rss_growth_under_limit": True if sys.platform == "linux" else None,
        "device_memory_flat": None,
    }


def test_learner_soak_checks_updates_and_recompiles() -> None:
    summary = run("soak.py", "ppo", "--ticks", "4096", "--rollout", "256", "--sample-every", "4")[
        -1
    ]
    assert summary["ticks"] == 17 * 256
    assert summary["updates"] == 17
    assert all(value is not False for value in summary["checks"].values())


def test_env_soak_lists_specs_and_checks_any_environment() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "soak.py"), "env", "--list"],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    specs = result.stdout.split()
    assert "BeanWorld-v0" in specs
    assert "jaxmarl/MPE_simple_push_v3+scripted" in specs
    assert not any(spec.startswith(("envpool/", "coom/", "metaworld/")) for spec in specs)
    # The list is the registry's: every id that does not step a host simulator.
    listed = {spec for spec in specs if "+" not in spec}
    assert listed == {
        env_id for env_id in pp.environments(builtin=True) if not pp.spec(env_id).host
    }
    summaries = {}
    for spec in ("BeanWorld-v0", "jaxmarl/MPE_simple_push_v3+scripted", "gymnax/Freeway-MinAtar"):
        summaries[spec] = summary = run(
            "soak.py",
            "env",
            "--spec",
            spec,
            "--ticks",
            "16384",
            "--chunk",
            "2048",
            "--sample-every",
            "2",
        )[-1]
        assert summary["ticks_per_lifetime"] == 9 * 2048, spec  # 8 measured calls + warm-up
        assert all(value is not False for value in summary["checks"].values()), (
            spec,
            summary["checks"],
        )
    # BeanWorld's uint32 regrow clock wraps by design: listed for review, not failed.
    assert "[0].env.tick" in summaries["BeanWorld-v0"]["unsigned_projected_to_wrap"]
    # gymnax's episode counter grows but the adapter clamps it, so it is not flagged.
    assert summaries["gymnax/Freeway-MinAtar"]["integers_projected_to_overflow"] == {}


def test_dqn_and_sac_soaks_check_updates_and_recompiles() -> None:
    for mode, rollout in (("dqn", 32), ("sac", 128)):
        summary = run(
            "soak.py",
            mode,
            "--ticks",
            str(8 * rollout),
            "--rollout",
            str(rollout),
            "--sample-every",
            "2",
        )[-1]
        assert summary["mode"] == mode
        assert summary["updates"] == 9
        assert all(value is not False for value in summary["checks"].values()), (
            mode,
            summary["checks"],
        )


def test_learning_guard_reports_intervals_and_passes() -> None:
    summary = run("learning_guard.py", "--algorithms", "ppo,dqn", "--seeds", "2")[-1]
    results = summary["results"]
    for name in ("ppo", "dqn"):
        low, high = results[name]["difference_iqm_ci95"]
        assert low <= results[name]["difference_iqm"] <= high
        assert results[name]["passed"] is True
    assert summary["passed"] is True
