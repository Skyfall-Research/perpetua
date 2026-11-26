"""A fresh process must resume CPU simulation and PPO without a seed restart."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import PPO, HostPPORunner


@pytest.mark.parametrize(
    "backend",
    [
        "gymnasium",
        pytest.param(
            "envpool-snapshot",
            marks=pytest.mark.skipif(
                find_spec("perpetua_envpool_snapshot") is None or sys.platform != "linux",
                reason="Requires the optional Linux native/envpool_snapshot binding",
            ),
        ),
    ],
)
def test_cpu_resume_across_processes(tmp_path: Path, backend: str) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "control_ppo.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(extra: list[str]) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(script), "--backend", backend, *extra],
            env=process_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        return json.loads(result.stdout)

    checkpoint = tmp_path / "cpu"
    resumed_checkpoint = tmp_path / "resumed"
    complete_checkpoint = tmp_path / "complete"
    saved = run(["--updates", "2", "--checkpoint", str(checkpoint)])
    restored = run(
        [
            "--updates",
            "1",
            "--resume",
            str(checkpoint),
            "--checkpoint",
            str(resumed_checkpoint),
        ]
    )
    complete = run(["--updates", "3", "--checkpoint", str(complete_checkpoint)])
    assert saved["lifetime_ticks"] == 256
    assert restored["lifetime_ticks"] == 384
    assert restored["gradient_steps"] == 48
    assert restored["mean_reward"] == complete["mean_reward"]
    assert restored["healthy"] is True
    env = (
        GymnasiumLifetime(gym.make("CartPole-v1"))
        if backend == "gymnasium"
        else pp.make(
            "envpool/CartPole-v1",
            num_envs=8,
            num_threads=2,
            snapshot=True,
            drift=pp.drift.Cyclic([(1.0,), (0.5,)], period=128),
        )
    )
    try:
        trainer = HostPPORunner(env, PPO(action_size=2))
        template = trainer.init(seed=999, window_size=128)
        a = pp.checkpoint.restore(resumed_checkpoint, template, expected_metadata=trainer.metadata)
        b = pp.checkpoint.restore(complete_checkpoint, template, expected_metadata=trainer.metadata)
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            if isinstance(left, jax.Array) and jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
            else:
                np.testing.assert_array_equal(left, right)
    finally:
        env.close()
