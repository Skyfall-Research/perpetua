# Contributing to Perpetua

## Design rules

- Keep Gym/RL vocabulary: `obs`, `action`, `reward`, `info`, `key` and space
  descriptors (`observation_space`, `action_space`).
- Native stepping is functional, with explicit state and RNG. There is no
  reset/done lifecycle: respawns are events inside one lifetime.
- Keep agent-visible data separate from privileged drift and evaluator state.
  Hidden parameters reach a policy only through an explicit `pp.Oracle`.
- Document public array shapes, dtypes, masks, transition timing and RNG
  ownership for every new environment, drift process or learner.
- Prefer simple functions, small state pytrees and composition over deep
  inheritance.
- Keep I/O, host logging and checkpoint serialization outside compiled tick
  loops.

## Tests

- Write behavioural tests for new logic. Do not substitute mocks or coverage
  targets for checking behaviour.
- Check seeded reproducibility, chunk continuity, shape stability and
  privilege isolation for anything that runs inside a lifetime.
- A test should be able to fail on the bug it guards against: compare
  discriminating cases (for example "nothing forgotten" and "everything
  forgotten"), not only values that the implementation happens to produce.
- `tests/test_baselines.py` compares short seeded runs with digests recorded
  from an earlier revision, one fixture per platform
  (`tests/fixtures/baselines-<sys.platform>-<machine>.json`, CPU, 32-bit):
  exactly on macOS arm64, and on Linux x86_64 with booleans and integers exact
  and floats within the relative `LINUX_RTOL` (5e-5) plus an absolute 5e-7, about
  five times the spread measured between two CI runner CPUs. A platform without a fixture skips the comparison,
  except CI on Linux x86_64, where a missing fixture fails. If the test fails
  and the change in results is intended, add an entry under "Changes results"
  in `CHANGELOG.md`, say in it which baselines moved, and re-record both
  fixtures: `uv run --all-extras python scripts/record_baselines.py` on an
  Apple Silicon Mac, and for Linux run the `Record baselines` workflow
  (`.github/workflows/baselines.yml`) on the branch, download its
  `baselines-linux-x86_64` artifact and commit the file as
  `tests/fixtures/baselines-linux-x86_64.json`. Otherwise the failure is a
  bug. `scripts/record_baselines.py --compare` prints the largest relative
  difference between a fresh recording and the committed fixture.

## Checks before every change

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run --all-extras pytest -n auto
```

`-n auto` runs the tests in parallel, one worker per CPU core, and
`--all-extras` adds PyTorch, so the torch-gated tests run instead of
skipping. Expect roughly 10 to 20 minutes on a multi-core laptop, depending
on the core count; the recorded durations in `.test_durations` (recorded on
an Apple Silicon laptop) add up to about 57 minutes run one at a time. Tests whose
platform or runtime is missing skip; add `-rs` to list them with reasons.
Each test has a 30-minute limit, so a hang fails instead of stalling the run.
CI runs the suite as eight parallel groups balanced by `.test_durations`.

If `.test_durations` is missing or stale, a full run rewrites it (the timings
are those of the machine it runs on):

```bash
uv run pytest -n auto --store-durations --clean-durations
```

MEAL needs a pinned nightly of TensorFlow Probability (`tfp-nightly`), kept in
its own dependency group. If it stops installing, `uv sync --no-group meal`
gives the rest of the environment, and the MEAL tests are skipped.

A test that needs PyTorch carries the `torch` marker: put `@pytest.mark.torch`
on it (or on its parameter with `pytest.param(..., marks=pytest.mark.torch)`)
instead of calling `pytest.importorskip("torch")`. `tests/conftest.py` skips
marked tests when torch is missing and marks every test of a module that
imports torch at the top, and `pytest -m torch` runs exactly these tests.

To run the first three automatically on every commit, install the optional
hooks once with `uvx pre-commit install`.

Documentation changes also build the docs site with warnings as errors:

```bash
uv run --group docs sphinx-build -W docs _build/html
```

Environment pages are generated: after editing `scripts/env_content.py` or
`scripts/generate_env_docs.py`, run `uv run python scripts/generate_env_docs.py`.

Library code and tests use strict type checking. JAX-heavy files narrowly
disable unknown-member diagnostics because JAX's device and tracer annotations
are incomplete; argument, return and other strict checks stay on. Examples and
runtime probes use standard checking, with runtime assertions for numerical
results.

Build and verify a wheel before a release. CPU CI runs on pull requests; GPU
runs execute only trusted code.

## What CI does not run

Every workflow runs on Linux CPU runners. The following are checked by hand,
and a passing CI run says nothing about them:

- **GPU-only tests** (five cases, skipped without a CUDA device):
  `tests/test_compute_example.py::test_measured_gpu_replay_matches_optimizer_state_in_a_fresh_process`
  (two backends) and the `cuda` case of
  `tests/test_marl_example.py::test_partner_drift_ippo_resume_matches_uninterrupted_run`,
  `tests/test_marl_example.py::test_opponent_drift_ippo_resume_matches_uninterrupted_run`
  and
  `tests/test_pettingzoo_snapshot_process.py::test_mpe_ippo_fresh_resume_crosses_drift_and_legacy_respawn`.
  On a GPU machine, run the suite with both platforms visible (the compiled
  CPU bridge needs the CPU one):

  ```bash
  JAX_PLATFORMS=cuda,cpu uv run --all-extras pytest -rs
  ```

- **The live MORPHEUS smoke test**
  (`tests/test_morpheus.py::test_live_morpheus_smoke`) needs a running
  service and its address in `MORPHEUS_URL`; no workflow sets one. Everything
  else in that file runs against the bundled mock.
- **Guide code blocks exempted by name** in `EXEMPT` of
  `tests/test_tutorials.py`, each with its reason: the EnvPool block of
  `docs/adapters.md`, the two AgarCL blocks of `docs/agarcl.md`, the COOM
  blocks of `docs/benchmarks.md` and `docs/presets/coom.md`, and the EnvPool
  Atari blocks of `docs/presets/switching.md` and `docs/task-switching.md`.
  They are never executed, on any machine, by the test suite.
- **macOS.** No job runs there; the suite is run by hand on Apple Silicon.
  EnvPool is Linux-only, so its tests and generated pages are skipped on a
  Mac.

Some checks run in CI, but not on every pull request:

- COOM and AgarCL tests, conformance cases and environment-page checks run
  in `integrations.yml`: nightly, and on pushes that touch those adapters,
  their setup scripts, tests or pages. The main jobs skip them (the runtimes
  are not built there) and name each skip.
- The learning regression guard (`learning.yml`) runs nightly and on pushes
  that touch the learners.

The torch-gated tests (every test with the `torch` marker, which includes
the guide `docs/training.md` with its PyTorch block) and the EnvPool snapshot
conformance case run on every pull request, in the `torch` and `wheel` jobs
of `ci.yml`; both jobs fail if one of those tests is skipped. The eight test
groups have no torch and skip the marked tests. CI passes `-rs`, so each
job's log lists what it skipped and why.

## Licence

Perpetua is MIT-licensed. By contributing you agree that your contribution is
released under the same licence.

## Reporting results

Report the checks you actually ran and their limits. Speed or cost claims need
a measurement with stated conditions; smoke tests do not support throughput
claims.
