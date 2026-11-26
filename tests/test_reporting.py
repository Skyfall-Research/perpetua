"""IQM and stratified bootstrap reporting must be exact and reproducible."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua import reporting


def test_interquartile_mean_trims_the_middle_half_exactly() -> None:
    scores = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert float(pp.reporting.interquartile_mean(scores)) == 4.5
    constant = jnp.full((4, 3), 2.5)
    assert float(pp.reporting.interquartile_mean(constant)) == 2.5
    assert float(pp.reporting.interquartile_mean(jnp.asarray([7.0]))) == 7.0
    assert float(pp.reporting.mean_statistic(scores)) == 4.5
    assert float(pp.reporting.median_statistic(scores)) == 4.5
    with pytest.raises(ValueError, match=r"Scores must contain at least one value\."):
        pp.reporting.interquartile_mean(jnp.asarray([]))
    assert bool(jnp.isnan(pp.reporting.interquartile_mean(jnp.asarray([1.0, float("nan")]))))


def test_interquartile_mean_matches_scipy_across_sample_sizes() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    for size in range(1, 21):
        values = rng.normal(size=size).astype(np.float32)
        expected = scipy_stats.trim_mean(values, 0.25)
        np.testing.assert_allclose(pp.reporting.interquartile_mean(values), expected, rtol=1e-6)
    ties = np.asarray([1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 9.0, 10.0], dtype=np.float32)
    np.testing.assert_allclose(
        pp.reporting.interquartile_mean(ties), scipy_stats.trim_mean(ties, 0.25), rtol=1e-6
    )
    # Small samples are defined, not NaN: no values are trimmed below n = 4.
    assert float(pp.reporting.interquartile_mean(jnp.asarray([1.0, 4.0]))) == 2.5


def test_non_finite_scores_are_never_trimmed_away() -> None:
    nan, inf = float("nan"), float("inf")
    # Sorting puts these at the ends, exactly where trimming would drop them.
    for values in (
        [1.0, 2.0, 3.0, nan],
        [1.0, 2.0, 3.0, 4.0, nan],
        [1.0, 2.0, 3.0, inf],
        [-inf, 1.0, 2.0, 3.0],
    ):
        assert bool(jnp.isnan(pp.reporting.interquartile_mean(jnp.asarray(values)))), values
    scores = jnp.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, nan], [4.0, 5.0]])
    interval = pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(0), num_bootstrap=16)
    assert bool(jnp.isnan(interval.point))


def test_bootstrap_intervals_are_reproducible_and_contain_the_point() -> None:
    scores = jnp.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])
    first = pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(7))
    second = pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(7))
    np.testing.assert_array_equal(np.asarray(first.point), np.asarray(second.point))
    np.testing.assert_array_equal(np.asarray(first.low), np.asarray(second.low))
    np.testing.assert_array_equal(np.asarray(first.high), np.asarray(second.high))
    assert float(first.low) <= float(first.point) <= float(first.high)
    # With many distinct scores the resamples, and so the bounds, depend on the key.
    varied = jnp.asarray(np.random.default_rng(3).normal(size=(32, 3)), dtype=jnp.float32)
    one = pp.reporting.stratified_bootstrap_ci(varied, key=jax.random.key(7), num_bootstrap=64)
    two = pp.reporting.stratified_bootstrap_ci(varied, key=jax.random.key(8), num_bootstrap=64)
    assert (float(one.low), float(one.high)) != (float(two.low), float(two.high))
    assert float(one.point) == float(two.point)
    tight = pp.reporting.stratified_bootstrap_ci(
        scores, key=jax.random.key(7), num_bootstrap=50, confidence=0.5
    )
    assert float(tight.high - tight.low) <= float(first.high - first.low)


def test_bootstrap_bounds_include_the_point_for_constant_scores() -> None:
    # Quantile interpolation between equal float32 samples can land one ulp
    # away from them: 0.1 gave high = 0.10000000149 below the point 0.10000000894.
    for value in (0.1, 0.3, 1.0 / 3.0, 0.7, 12345.678):
        for statistic in (
            pp.reporting.interquartile_mean,
            pp.reporting.mean_statistic,
            pp.reporting.median_statistic,
        ):
            interval = pp.reporting.stratified_bootstrap_ci(
                np.full((5, 3), value, np.float32), statistic, key=jax.random.key(3)
            )
            assert float(interval.low) <= float(interval.point) <= float(interval.high)
    # A statistic the percentile interval need not cover is still included.
    skewed = pp.reporting.stratified_bootstrap_ci(
        jnp.asarray([[0.0], [0.0], [0.0], [100.0]]),
        lambda values: jnp.max(values),
        key=jax.random.key(0),
        confidence=0.2,
        num_bootstrap=200,
    )
    assert float(skewed.low) <= float(skewed.point) == float(skewed.high) == 100.0


def test_report_returns_jittable_statistics() -> None:
    scores = jnp.asarray([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    key = jax.random.key(3)
    eager = pp.reporting.report(scores, key=key, num_bootstrap=64)

    def run(values: jax.Array, k: jax.Array) -> dict[str, pp.reporting.BootstrapInterval]:
        return pp.reporting.report(values, key=k, num_bootstrap=64)

    compiled = cast(dict[str, pp.reporting.BootstrapInterval], jax.jit(run)(scores, key))
    for name in ("iqm", "mean", "median"):
        for field in ("point", "low", "high"):
            np.testing.assert_array_equal(
                np.asarray(getattr(eager[name], field)),
                np.asarray(getattr(compiled[name], field)),
            )
    assert float(eager["iqm"].point) == 3.5


def test_reporting_validates_inputs() -> None:
    scores = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match=r"num_bootstrap must be a positive integer\."):
        pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(0), num_bootstrap=0)
    with pytest.raises(ValueError, match=r"confidence must be in"):
        pp.reporting.stratified_bootstrap_ci(scores, key=jax.random.key(0), confidence=1.5)
    with pytest.raises(ValueError, match=r"Bootstrap scores must have shape \(runs, tasks\)\."):
        pp.reporting.stratified_bootstrap_ci(jnp.asarray([1.0, 2.0]), key=jax.random.key(0))
    assert not bool(
        jnp.isfinite(
            pp.reporting.stratified_bootstrap_ci(
                jnp.asarray([[1.0, float("inf")]]), key=jax.random.key(0), num_bootstrap=8
            ).point
        )
    )


def test_reporting_composes_with_revisit_evaluation() -> None:
    env = pp.make("ContinualGrid-v0", size=3)

    def greedy_act(state: jax.Array, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        del state
        return pp.policies.greedy(obs, info, key)

    def random_act(state: jax.Array, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        del state
        return pp.policies.random(obs, info, key)

    seeds = jnp.asarray([0, 1, 2, 3])
    states = jnp.zeros((2,))
    greedy = pp.evaluation.revisit(env, greedy_act, states, seeds=seeds, ticks=32)
    random = pp.evaluation.revisit(env, random_act, states, seeds=seeds, ticks=32)
    # revisit returns (policies, seeds); reporting consumes (runs, tasks).
    greedy_report = pp.reporting.report(greedy.T, key=jax.random.key(5), num_bootstrap=256)
    random_report = pp.reporting.report(random.T, key=jax.random.key(5), num_bootstrap=256)
    assert float(greedy_report["iqm"].point) > float(random_report["iqm"].point)
    assert float(greedy_report["iqm"].low) > float(random_report["iqm"].high)


@pytest.mark.parametrize(
    "statistic",
    [pp.reporting.mean_statistic, pp.reporting.median_statistic, pp.reporting.interquartile_mean],
)
def test_non_finite_scores_give_nan_for_every_statistic(statistic: object) -> None:
    """A diverged run must not be averaged or medianed away."""
    for bad in (jnp.inf, -jnp.inf, jnp.nan):
        scores = jnp.asarray([[1.0], [bad], [2.0]])
        assert np.isnan(float(cast(Any, statistic)(scores)))


def test_bootstrap_bounds_are_nan_whenever_the_point_is() -> None:
    scores = jnp.arange(8.0).reshape(8, 1).at[3, 0].set(jnp.nan)
    for seed in range(10):
        interval = pp.reporting.stratified_bootstrap_ci(
            scores, key=jax.random.key(seed), num_bootstrap=1
        )
        assert np.isnan(np.asarray(interval)).all(), seed


def test_provenance_names_the_code_and_software_without_local_details() -> None:
    record = pp.reporting.provenance()
    assert set(record) == {
        "perpetua",
        "revision",
        "dirty",
        "jax",
        "backend",
        "python",
        "platform",
    }
    assert record["perpetua"] == importlib.metadata.version("perpetua-rl")
    assert record["jax"] == jax.__version__
    assert record["backend"] == jax.default_backend()
    assert json.loads(json.dumps(record)) == record
    assert record["python"] == platform.python_version()
    assert record["platform"] == f"{platform.system()}-{platform.machine()}"
    assert str(Path.home()) not in json.dumps(record)


def uncached_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = reporting._source_revision  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(reporting, "_source_revision", lookup.__wrapped__)


def test_provenance_reads_the_checkout_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("needs git and a source checkout")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    uncached_revision(monkeypatch)
    record = pp.reporting.provenance()
    assert record["revision"] == head
    assert isinstance(record["dirty"], bool)


def test_provenance_does_not_raise_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    uncached_revision(monkeypatch)
    monkeypatch.setenv("PATH", "")
    record = pp.reporting.provenance()
    # The editable install records a local directory, not a VCS commit.
    assert record["revision"] is None
    assert record["dirty"] is None
    assert record["perpetua"] is not None


def build_record(monkeypatch: pytest.MonkeyPatch, **fields: object) -> None:
    module = types.ModuleType("perpetua._revision")
    for name, value in fields.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "perpetua._revision", module)


def test_provenance_falls_back_to_the_revision_recorded_at_build_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_checkout(*arguments: str) -> None:
        return None

    uncached_revision(monkeypatch)
    monkeypatch.setattr(reporting, "_git", no_checkout)
    # This environment is an editable install: no checkout, no VCS record.
    assert pp.reporting.provenance()["revision"] is None
    build_record(monkeypatch, revision="0123abcd", dirty=True)
    record = pp.reporting.provenance()
    assert (record["revision"], record["dirty"]) == ("0123abcd", True)
    build_record(monkeypatch, revision="0123abcd")
    assert pp.reporting.provenance()["dirty"] is None
    build_record(monkeypatch, revision=7, dirty="yes")
    record = pp.reporting.provenance()
    assert (record["revision"], record["dirty"]) == (None, None)


def test_a_checkout_reports_its_live_state_over_a_stale_build_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("needs git and a source checkout")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    uncached_revision(monkeypatch)
    build_record(monkeypatch, revision="5" * 40, dirty=False)
    assert pp.reporting.provenance()["revision"] == head


def test_a_vcs_install_record_wins_over_the_build_record(monkeypatch: pytest.MonkeyPatch) -> None:
    build_record(monkeypatch, revision="5" * 40, dirty=False)
    test_provenance_takes_the_commit_of_a_vcs_install(monkeypatch)


def test_provenance_takes_the_commit_of_a_vcs_install(monkeypatch: pytest.MonkeyPatch) -> None:
    class Installed:
        version = "0.0.1a0"

        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://example.org/perpetua.git",
                    "vcs_info": {"vcs": "git", "commit_id": "0123abcd"},
                }
            )

    uncached_revision(monkeypatch)

    def no_checkout(*arguments: str) -> None:
        return None

    def installed(name: str) -> Installed:
        return Installed()

    monkeypatch.setattr(reporting, "_git", no_checkout)
    monkeypatch.setattr(importlib.metadata, "distribution", installed)
    record = pp.reporting.provenance()
    assert (record["revision"], record["dirty"]) == ("0123abcd", None)
