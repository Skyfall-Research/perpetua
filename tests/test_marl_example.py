"""Fresh-process IPPO resume must preserve every learner and interacting world leaf."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import IPPO, PPOConfig, PPORunner


def test_native_ippo_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), *extra],
            env=process_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        return json.loads(result.stdout)

    saved, resumed, full = (tmp_path / name for name in ["saved", "resumed", "full"])
    run(["--updates", "3", "--checkpoint", str(saved)])
    a = run(
        ["--updates", "2", "--seed", "999", "--resume", str(saved), "--checkpoint", str(resumed)]
    )
    b = run(["--updates", "5", "--checkpoint", str(full)])
    assert a["lifetime_ticks"] == b["lifetime_ticks"] == 160
    assert a["gradient_steps_per_agent"] == b["gradient_steps_per_agent"] == [80, 80]
    assert a["mean_reward"] == b["mean_reward"]
    env = pp.make(
        "CooperativeGrid-v0", size=2, drift=pp.drift.Cyclic([(1.0, 1.0), (0.0, 0.0)], 128)
    )
    trainer = PPORunner(env, IPPO(action_size=5, config=PPOConfig(rollout_steps=32)))
    template = trainer.init(seed=123, window_size=128)
    left = pp.checkpoint.restore(resumed, template, expected_metadata=trainer.metadata)
    right = pp.checkpoint.restore(full, template, expected_metadata=trainer.metadata)
    assert int(left.lifetime.state.drift["goal"].phase) == 1
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(x), jax.random.key_data(y))
        else:
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "platform",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                jax.default_backend() != "gpu",
                reason="Fresh GPU native IPPO continuation requires CUDA",
            ),
        ),
    ],
)
def test_partner_drift_ippo_resume_matches_uninterrupted_run(tmp_path: Path, platform: str) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    process_env = {
        **os.environ,
        "JAX_PLATFORMS": platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), "--backend", "partner", *extra],
            env=process_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        return json.loads(result.stdout)

    saved, resumed, full = (tmp_path / name for name in ["saved", "resumed", "full"])
    run(["--updates", "3", "--checkpoint", str(saved)])
    a = run(
        ["--updates", "2", "--seed", "999", "--resume", str(saved), "--checkpoint", str(resumed)]
    )
    b = run(["--updates", "5", "--checkpoint", str(full)])
    assert a["lifetime_ticks"] == b["lifetime_ticks"] == 160
    assert a["gradient_steps_per_agent"] == b["gradient_steps_per_agent"] == [80, 0]
    assert a["mean_reward"] == b["mean_reward"]
    assert a["healthy"] is True
    env = pp.make("PartnerGrid-v0", size=3, partner_drift=pp.drift.Cyclic([(1.0,), (0.0,)], 128))
    trainer = PPORunner(env, IPPO(action_size=5, config=PPOConfig(rollout_steps=32)))
    template = trainer.init(seed=123, window_size=128)
    left = pp.checkpoint.restore(resumed, template, expected_metadata=trainer.metadata)
    right = pp.checkpoint.restore(full, template, expected_metadata=trainer.metadata)
    assert int(left.lifetime.state.drift["partner"].phase) == 1
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(x), jax.random.key_data(y))
        else:
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "platform",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                jax.default_backend() != "gpu",
                reason="Fresh GPU native IPPO continuation requires CUDA",
            ),
        ),
    ],
)
def test_opponent_drift_ippo_resume_matches_uninterrupted_run(
    tmp_path: Path, platform: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    process_env = {
        **os.environ,
        "JAX_PLATFORMS": platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), "--backend", "opponent", *extra],
            env=process_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        return json.loads(result.stdout)

    saved, resumed, full = (tmp_path / name for name in ["saved", "resumed", "full"])
    run(["--updates", "3", "--checkpoint", str(saved)])
    a = run(
        ["--updates", "2", "--seed", "999", "--resume", str(saved), "--checkpoint", str(resumed)]
    )
    b = run(["--updates", "5", "--checkpoint", str(full)])
    assert a["lifetime_ticks"] == b["lifetime_ticks"] == 160
    assert a["gradient_steps_per_agent"] == b["gradient_steps_per_agent"] == [80, 0]
    assert a["mean_reward"] == b["mean_reward"]
    assert isinstance(a["mean_reward"], list)
    mean = cast(list[float], a["mean_reward"])
    assert mean[1] == pytest.approx(-mean[0])
    assert a["healthy"] is True
    env = pp.make("OpponentGrid-v0", size=3, opponent_drift=pp.drift.Cyclic([(1.0,), (0.0,)], 128))
    trainer = PPORunner(env, IPPO(action_size=5, config=PPOConfig(rollout_steps=32)))
    template = trainer.init(seed=123, window_size=128)
    left = pp.checkpoint.restore(resumed, template, expected_metadata=trainer.metadata)
    right = pp.checkpoint.restore(full, template, expected_metadata=trainer.metadata)
    assert left.lifetime.state.t.to_int() == 160
    assert int(left.lifetime.state.drift["opponent"].phase) == 1
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(x), jax.random.key_data(y))
        else:
            np.testing.assert_array_equal(x, y)


def test_jaxmarl_example_runs_and_resumes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), "--backend", "jaxmarl", *extra],
            env=process_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        return json.loads(result.stdout)

    run(["--updates", "2", "--checkpoint", str(tmp_path / "jaxmarl")])
    summary = run(
        [
            "--updates",
            "1",
            "--resume",
            str(tmp_path / "jaxmarl"),
            "--checkpoint",
            str(tmp_path / "resumed"),
        ]
    )
    assert summary["lifetime_ticks"] == 96
    assert summary["gradient_steps_per_agent"] == [48, 48, 48]
    assert summary["healthy"] is True


def test_jaxmarl_scripted_example_runs_and_resumes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), "--backend", "jaxmarl-scripted", *extra],
            env=process_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return json.loads(result.stdout)

    saved = run(["--updates", "2", "--checkpoint", str(tmp_path / "scripted")])
    assert saved["lifetime_ticks"] == 64
    assert saved["gradient_steps_per_agent"] == [32, 32, 0]
    resumed = run(
        [
            "--updates",
            "1",
            "--resume",
            str(tmp_path / "scripted"),
            "--checkpoint",
            str(tmp_path / "r"),
        ]
    )
    assert resumed["lifetime_ticks"] == 96
    assert resumed["gradient_steps_per_agent"] == [48, 48, 0]
    assert resumed["healthy"] is True


def test_ippo_example_runs_real_pettingzoo_world() -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    result = subprocess.run(
        [sys.executable, str(script), "--backend", "pettingzoo", "--updates", "1"],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    summary = json.loads(result.stdout)
    assert summary["lifetime_ticks"] == 32
    assert summary["gradient_steps_per_agent"] == [16, 16, 16]
    assert summary["healthy"] is True
