"""Seeded results of this revision against the ones recorded from an earlier one."""

import copy
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import pytest
from helpers import load_script

import perpetua as pp

CHANGED = (
    "A seeded result changed. If the change is intended, add a 'Changes results' entry to "
    "CHANGELOG.md and re-record the fixture of every platform: "
    "`uv run --all-extras python scripts/record_baselines.py` on an Apple Silicon Mac for "
    "baselines-darwin-arm64.json, and the 'Record baselines' workflow "
    "(.github/workflows/baselines.yml) for baselines-linux-x86_64.json."
)
# Two GitHub runners are not the same machine, and how a float rounds depends
# on the CPU's vector instructions. Two recordings on different runner CPUs
# (EPYC 7763 and EPYC 9V45, 622 floats) differed by at most 6.9e-6 relatively
# above 0.01 in magnitude and by at most 8.9e-8 absolutely below it; the two
# bounds allow about five times that between a Linux run and the Linux
# fixture. Booleans and integers compare exactly on every platform, and every
# number does on macOS arm64.
LINUX_RTOL = 5e-5
ABSOLUTE = 5e-7
RTOL = {"linux": LINUX_RTOL}.get(sys.platform, 0.0)

baselines = load_script("record_baselines")
FIXTURE: Path = baselines.fixture_path()
FIXTURES: list[Path] = sorted(baselines.FIXTURES.glob("baselines-*.json"))


def differences(actual: Any, recorded: Any, *, rtol: float) -> list[str]:
    """The digest entries that differ, as ``path: recorded -> actual`` lines.

    ``rtol=0`` compares every number exactly. Otherwise a float may differ by
    ``rtol * |recorded| + ABSOLUTE``; booleans and integers stay exact.
    """
    now: dict[str, bool | int | float] = baselines.flatten(actual)
    then: dict[str, bool | int | float] = baselines.flatten(recorded)
    if now.keys() != then.keys():
        return [f"entries differ: {sorted(now.keys() ^ then.keys())}"]
    changed: list[str] = []
    for path, old in then.items():
        new = now[path]
        close = rtol > 0 and isinstance(old, float) and abs(new - old) <= rtol * abs(old) + ABSOLUTE
        if type(new) is not type(old) or not (new == old or close):
            changed.append(f"{path}: {old!r} -> {new!r}")
    return changed


def fixture_is_required(environ: Mapping[str, str], sys_platform: str, machine: str) -> bool:
    """Whether a missing fixture is a failure: in CI on Linux x86_64, where one is committed."""
    return (sys_platform, machine) == ("linux", "x86_64") and bool(environ.get("CI"))


def recorded_for_this_platform() -> dict[str, Any]:
    """This platform's fixture; skips (fails in CI on Linux x86_64) where none applies."""
    here: dict[str, str | bool] = baselines.current_platform()
    if here["x64"]:
        pytest.skip("jax_enable_x64 draws different random numbers; the baselines are 32-bit.")
    if here["backend"] != "cpu":
        pytest.skip(f"The baselines are recorded on the CPU backend, not {here['backend']}.")
    if not FIXTURE.exists():
        reason = (
            f"No baselines are recorded for this platform (tests/fixtures/{FIXTURE.name} is "
            "missing), so seeded results are not compared here; see scripts/record_baselines.py."
        )
        if fixture_is_required(os.environ, str(here["sys_platform"]), str(here["machine"])):
            pytest.fail(reason)
        pytest.skip(reason)
    recorded: dict[str, Any] = json.loads(FIXTURE.read_text())
    assert recorded["platform"] == here, f"{FIXTURE.name} was recorded on {recorded['platform']}."
    return recorded


def runs(names: Mapping[str, object]) -> list[Any]:
    """One parameter per run; a run whose optional dependency group is absent skips."""
    cases: list[Any] = []
    for name in sorted(names):
        group: str | None = baselines.NEEDS.get(name)
        missing = group is not None and importlib.util.find_spec(group) is None
        reason = f"{name}: the optional {group} group is not installed."
        cases.append(pytest.param(name, marks=pytest.mark.skipif(missing, reason=reason)))
    return cases


def test_the_fixture_of_the_recording_mac_is_committed() -> None:
    assert baselines.fixture_path("darwin", "arm64") in FIXTURES
    assert FIXTURE.name == f"baselines-{sys.platform}-{baselines.platform.machine()}.json"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.name)
def test_the_recorded_runs_are_the_defined_runs(fixture: Path) -> None:
    recorded = json.loads(fixture.read_text())
    assert sorted(recorded["runs"]) == sorted(baselines.RUNS), CHANGED
    where = recorded["platform"]
    assert fixture == baselines.fixture_path(where["sys_platform"], where["machine"])
    assert (where["backend"], where["x64"]) == ("cpu", False)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.name)
def test_every_recorded_run_holds_numbers_that_follow_the_trajectory(fixture: Path) -> None:
    for name, digest in json.loads(fixture.read_text())["runs"].items():
        if name == "ppo/cnn-meal-pot-fill":
            assert {"logits", "value"} <= digest["after"].keys()
            continue
        assert set(digest["observation"]) == {"sum", "abs_sum", "weighted_sum"}, name
        if name in baselines.RANDOM:
            assert set(digest["trajectory"]) == {"actions", "observations"}, name
        else:
            assert digest["params"], name


@pytest.mark.parametrize("name", runs(baselines.RUNS))
def test_seeded_result_matches_the_recorded_baseline(name: str) -> None:
    recorded = recorded_for_this_platform()
    changed = differences(baselines.RUNS[name](), recorded["runs"][name], rtol=RTOL)
    how = "exactly" if RTOL == 0 else f"within a relative {RTOL:g}"
    assert not changed, (
        f"{CHANGED}\n{name} (recorded on {recorded['platform']} in {FIXTURE.name}, "
        f"compared {how}):\n" + "\n".join(changed)
    )


@pytest.mark.parametrize("name", runs(baselines.RANDOM))
def test_other_actions_change_a_random_digest(name: str) -> None:
    make, ticks = baselines.RANDOM[name]
    env = make()
    policy = pp.policies.uniform(env.action_space)
    runner = pp.runtime.Runner(env, policy, ticks=ticks)
    lifetime = pp.runtime.init_lifetime(env, seed=0, window_size=16)
    other = lifetime._replace(agent_key=jax.random.key(1))
    first = baselines.trajectory_digest(env, policy, lifetime, ticks)
    second = baselines.trajectory_digest(env, policy, other, ticks)
    assert first["actions"] != second["actions"]
    assert first["observations"] != second["observations"]
    final, other_final = (
        baselines.lifetime_digest(runner(lifetime)),
        baselines.lifetime_digest(runner(other)),
    )
    assert final["observation"] != other_final["observation"]


def test_a_changed_number_is_reported_with_its_path() -> None:
    recorded = {"mean_reward": [0.5, 0.25], "tick": 8, "params": {"Dense_0": {"sum": 1.0}}}
    assert differences(json.loads(json.dumps(recorded)), recorded, rtol=0.0) == []
    nudged = {"mean_reward": [0.5, 0.2500001], "tick": 8, "params": {"Dense_0": {"sum": 1.0}}}
    assert differences(nudged, recorded, rtol=0.0) == ["mean_reward/1: 0.25 -> 0.2500001"]
    assert differences(nudged, recorded, rtol=LINUX_RTOL) == []
    late = {**recorded, "tick": 9}
    assert differences(late, recorded, rtol=LINUX_RTOL) == ["tick: 8 -> 9"]
    far = {**recorded, "params": {"Dense_0": {"sum": 1.2}}}
    assert differences(far, recorded, rtol=LINUX_RTOL) == ["params/Dense_0/sum: 1.0 -> 1.2"]
    assert differences({"tick": 8}, recorded, rtol=LINUX_RTOL)[0].startswith("entries differ")
    assert differences({**recorded, "tick": 8.0}, recorded, rtol=LINUX_RTOL) == ["tick: 8 -> 8.0"]


def scaled(digest: Any, target: str, factor: float, prefix: str = "") -> Any:
    """A copy of a digest with the number at one ``flatten`` path multiplied."""

    def below(key: object) -> str:
        return f"{prefix}/{key}" if prefix else str(key)

    if isinstance(digest, dict):
        entries = cast(dict[str, Any], digest)
        return {key: scaled(value, target, factor, below(key)) for key, value in entries.items()}
    if isinstance(digest, list):
        values = cast(list[Any], digest)
        return [scaled(value, target, factor, below(i)) for i, value in enumerate(values)]
    return digest * factor if prefix == target else digest


def test_a_tenth_of_a_percent_fails_both_comparisons() -> None:
    assert LINUX_RTOL <= 1e-4
    assert ABSOLUTE <= 1e-6
    recorded = json.loads(baselines.fixture_path("darwin", "arm64").read_text())["runs"]
    floats = [
        (name, path)
        for name, digest in recorded.items()
        for path, value in baselines.flatten(digest).items()
        if isinstance(value, float) and abs(value) > 1e-3
    ]
    assert len(floats) > 400
    for name, path in floats:
        mutated = scaled(recorded[name], path, 1.001)
        for rtol in (0.0, LINUX_RTOL):
            changed = differences(mutated, recorded[name], rtol=rtol)
            assert len(changed) == 1, (name, path, rtol)
            assert changed[0].startswith(f"{path}: ")
    small = {"loss": 1e-6}
    assert differences({"loss": -1e-6}, small, rtol=LINUX_RTOL) == ["loss: 1e-06 -> -1e-06"]
    assert differences({"loss": 1.001e-6}, small, rtol=0.0) == ["loss: 1e-06 -> 1.001e-06"]


def test_only_linux_ci_fails_without_a_fixture() -> None:
    assert fixture_is_required({"CI": "true"}, "linux", "x86_64")
    assert not fixture_is_required({}, "linux", "x86_64")
    assert not fixture_is_required({"CI": ""}, "linux", "x86_64")
    assert not fixture_is_required({"CI": "true"}, "linux", "aarch64")
    assert not fixture_is_required({"CI": "true"}, "darwin", "arm64")


def test_compare_reports_the_largest_relative_difference() -> None:
    fixture = {"runs": {"a": {"mean_reward": [0.5, 2.0], "tick": 8, "zero": 0.0}}}
    assert baselines.compare(copy.deepcopy(fixture), fixture) == ([], 0.0, "")
    moved = {"runs": {"a": {"mean_reward": [0.5005, 2.0002], "tick": 8, "zero": 0.0}}}
    broken, largest, where = baselines.compare(moved, fixture)
    assert broken == []
    assert where == "a/mean_reward/0"
    assert largest == pytest.approx(1e-3)
    late = {"runs": {"a": {"mean_reward": [0.5, 2.0], "tick": 9}}}
    broken, largest, where = baselines.compare(late, fixture)
    assert broken == ["only in one file: a/zero", "a/tick: 8 -> 9"]
    assert (largest, where) == (0.0, "")
    nonzero = {"runs": {"a": {"mean_reward": [0.5, 2.0], "tick": 8, "zero": 1e-12}}}
    assert baselines.compare(nonzero, fixture) == ([], float("inf"), "a/zero")
