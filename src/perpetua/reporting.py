"""Seed-aware reporting: interquartile means and stratified bootstrap intervals.

Statistics are pure functions of their scores; non-finite entries propagate as
NaN results rather than raising, because value-dependent checks cannot run
inside the bootstrap's traced resampling. Callers validate their own inputs.
:func:`provenance` names the code and software a result was produced with.
"""

# JAX device members are partially annotated.
# pyright: reportUnknownMemberType=false

import functools
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from perpetua.core import as_int

__all__ = [
    "Statistic",
    "BootstrapInterval",
    "interquartile_mean",
    "mean_statistic",
    "median_statistic",
    "stratified_bootstrap_ci",
    "provenance",
    "report",
]

type Statistic = Callable[[jax.Array], jax.Array]

_DISTRIBUTION = "perpetua-rl"


def _git(*arguments: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


@functools.cache
def _source_revision() -> tuple[str | None, bool | None]:
    # An installed package can sit beneath an unrelated repository (a virtual
    # environment inside a project), so the checkout must track this package.
    if _git("ls-files", "--error-unmatch", "__init__.py") is not None:
        commit = _git("rev-parse", "HEAD")
        if commit:
            status = _git("status", "--porcelain", "--untracked-files=no")
            return commit, None if status is None else bool(status)
    try:
        text = importlib.metadata.distribution(_DISTRIBUTION).read_text("direct_url.json")
        commit = json.loads(text or "{}")["vcs_info"]["commit_id"]
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError, LookupError, TypeError):
        commit = None
    if isinstance(commit, str):
        return commit, None
    # Written by the build hook (hatch_build.py) when the distribution was
    # built from a checkout. It comes last, so it never overrides live git state.
    try:
        built = importlib.import_module("perpetua._revision")
    except ImportError:
        return None, None
    commit, dirty = getattr(built, "revision", None), getattr(built, "dirty", None)
    if not isinstance(commit, str):
        return None, None
    return commit, dirty if isinstance(dirty, bool) else None


def provenance() -> dict[str, str | bool | None]:
    """The code and software producing a result, as a JSON-ready dict.

    Put it in every report so a result can be placed against the changelog.
    :func:`perpetua.checkpoint.save` records the same entries. Keys:

    - ``perpetua``: the installed package version, None when the package is
      imported without being installed.
    - ``revision``: the source commit. From a git checkout (an editable
      install included) it is the checked-out commit, read now; for an
      install from a VCS URL it is the commit the installer recorded (PEP
      610); for any other install it is the commit the wheel or sdist was
      built from, which the build records when it runs in a git checkout
      (``pip install .`` from a clone, for example). None when none of the
      three is known: a distribution built outside a checkout, or a checkout
      without ``git``.
    - ``dirty``: whether tracked files of the checkout had uncommitted changes
      (untracked files are not counted), now for a checkout and at build time
      for a built distribution; None when unknown, as for a VCS-URL install.
      A dirty revision does not identify the code.
    - ``jax``, ``backend``: the JAX version and ``jax.default_backend()``.
    - ``python``, ``platform``: the interpreter version and
      ``<system>-<machine>``.

    The revision is looked up once per process, with a short timeout, and no
    failure there raises. No path, host name or user name is included.
    """
    try:
        version = importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    revision, dirty = _source_revision()
    return {
        "perpetua": version,
        "revision": revision,
        "dirty": dirty,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
    }


class BootstrapInterval(NamedTuple):
    """Point statistic and percentile confidence bounds over bootstrap resamples."""

    point: jax.Array
    low: jax.Array
    high: jax.Array


def _flatten(scores: ArrayLike) -> jax.Array:
    values = jnp.asarray(scores, dtype=jnp.float32)
    if values.size < 1:
        raise ValueError("Scores must contain at least one value.")
    return values.reshape(-1)


def interquartile_mean(scores: ArrayLike) -> jax.Array:
    """Interquartile mean: discard the bottom and top quarter of sorted scores.

    With ``k = n // 4``, the retained slice is ``sorted[k : n - k]``. This is
    the Agarwal et al. definition used by rliable and
    ``scipy.stats.trim_mean(..., 0.25)``; for fewer than four scores nothing is
    trimmed. It is not an average of values between the 25th and 75th
    percentiles, which differs for most sample sizes and can be empty.

    Any non-finite score makes the result NaN. Sorting would otherwise move NaN
    or infinities to the ends where trimming silently drops them, hiding a
    diverged run; this deliberately differs from SciPy for infinities.
    """
    values = jnp.sort(_flatten(scores))
    k = values.size // 4
    trimmed = jnp.mean(values[k : values.size - k])
    return jnp.where(jnp.all(jnp.isfinite(values)), trimmed, jnp.nan)


def mean_statistic(scores: ArrayLike) -> jax.Array:
    """Mean over all entries; NaN if any entry is non-finite, as for the IQM."""
    values = _flatten(scores)
    return jnp.where(jnp.all(jnp.isfinite(values)), jnp.mean(values), jnp.nan)


def median_statistic(scores: ArrayLike) -> jax.Array:
    """Median over all entries; NaN if any entry is non-finite, as for the IQM."""
    values = _flatten(scores)
    return jnp.where(jnp.all(jnp.isfinite(values)), jnp.median(values), jnp.nan)


def stratified_bootstrap_ci(
    scores: ArrayLike,
    statistic: Statistic = interquartile_mean,
    *,
    key: jax.Array,
    confidence: float = 0.95,
    num_bootstrap: int = 2000,
) -> BootstrapInterval:
    """Percentile interval resampling runs with replacement within task strata.

    ``scores`` has shape (runs, tasks). Each bootstrap draw resamples run
    indices independently per task, mirroring rliable's stratified bootstrap,
    then applies ``statistic`` to the resampled (runs, tasks) matrix (the
    built-in statistics flatten it themselves). The caller owns the
    explicit RNG key; identical inputs and keys give identical intervals.
    The bounds always include the point: ``low <= point <= high``, widening
    the percentile interval to the point where it falls outside (by a
    rounding step for constant scores, or further for a skewed statistic on
    few runs). All three are NaN when the point is.
    """
    values = jnp.asarray(scores, dtype=jnp.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("Bootstrap scores must have shape (runs, tasks).")
    num_bootstrap = as_int(num_bootstrap, "num_bootstrap must be a positive integer.", minimum=1)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    point = statistic(values)
    keys = jax.random.split(key, num_bootstrap)

    def resample(draw_key: jax.Array) -> jax.Array:
        indices = jax.random.randint(
            draw_key, (values.shape[0], values.shape[1]), 0, values.shape[0]
        )
        return statistic(jnp.take_along_axis(values, indices, axis=0))

    samples = jax.vmap(resample)(keys)
    tail = (1.0 - confidence) / 2.0
    low = jnp.quantile(samples, tail)
    # Interpolating between equal samples can round the lower bound one ulp above
    # the upper bound on some backends.
    high = jnp.maximum(low, jnp.quantile(samples, 1.0 - tail))
    # The same interpolation can leave the point one ulp outside bounds it
    # equals, so the interval is widened to contain it.
    low, high = jnp.minimum(low, point), jnp.maximum(high, point)
    # A few draws can all miss the non-finite entry that makes the point NaN.
    finite = jnp.all(jnp.isfinite(point))
    return BootstrapInterval(
        point, jnp.where(finite, low, jnp.nan), jnp.where(finite, high, jnp.nan)
    )


def report(
    scores: ArrayLike,
    *,
    key: jax.Array,
    confidence: float = 0.95,
    num_bootstrap: int = 2000,
) -> dict[str, BootstrapInterval]:
    """IQM, mean and median intervals for a (runs, tasks) score matrix.

    All three statistics pool every (run, task) entry. The ``"median"`` is
    therefore the median of all entries, not the aggregate median of Agarwal et
    al. (2021) and rliable, which is the median over tasks of each task's mean
    over runs; the two differ most when success is bimodal across seeds. The
    IQM and mean match the pooled definitions in rliable.
    """
    return {
        "iqm": stratified_bootstrap_ci(
            scores,
            interquartile_mean,
            key=key,
            confidence=confidence,
            num_bootstrap=num_bootstrap,
        ),
        "mean": stratified_bootstrap_ci(
            scores,
            mean_statistic,
            key=key,
            confidence=confidence,
            num_bootstrap=num_bootstrap,
        ),
        "median": stratified_bootstrap_ci(
            scores,
            median_statistic,
            key=key,
            confidence=confidence,
            num_bootstrap=num_bootstrap,
        ),
    }
