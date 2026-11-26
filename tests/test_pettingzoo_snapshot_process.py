"""Fresh-process MPE2/IPPO checkpoints must preserve full simulator and learner state."""

# Optional simulator/JAX APIs are partially annotated.
# pyright: reportUnknownMemberType=false

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import jax
import pytest
from helpers import assert_same
from pettingzoo import EnvSpec, ParallelEnv, make  # pyright: ignore[reportMissingTypeStubs]

import perpetua as pp
from perpetua.adapters import PettingZooLifetime
from perpetua.agents import IPPO, ComputePPORunner, HostPPORunner, PPOConfig
from perpetua.compute import TimingTrace


def execute(script: Path, args: list[str], *, platform: str = "cpu") -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(script), "--backend", "pettingzoo-snapshot", *args],
        env={**os.environ, "JAX_PLATFORMS": platform, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "platform",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                jax.default_backend() != "gpu", reason="Fresh GPU IPPO continuation requires CUDA"
            ),
        ),
    ],
)
def test_mpe_ippo_fresh_resume_crosses_drift_and_legacy_respawn(
    tmp_path: Path,
    platform: str,
) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/train_marl.py"
    saved, resumed, full = (tmp_path / name for name in ["saved", "resumed", "full"])
    execute(script, ["--updates", "3", "--checkpoint", str(saved)], platform=platform)
    actual = execute(
        script,
        ["--updates", "2", "--resume", str(saved), "--checkpoint", str(resumed)],
        platform=platform,
    )
    expected = execute(script, ["--updates", "5", "--checkpoint", str(full)], platform=platform)
    assert actual["lifetime_ticks"] == expected["lifetime_ticks"] == 160
    assert actual["gradient_steps_per_agent"] == [80, 80, 80]
    assert actual["mean_reward"] == expected["mean_reward"]
    spec = EnvSpec(
        id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
    )
    env = PettingZooLifetime(
        cast(ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=128)),
        snapshot=True,
        drift=pp.drift.Cyclic([(1.0,), (0.5,)], 128),
    )
    try:
        runner = HostPPORunner(env, IPPO(action_size=5, config=PPOConfig(rollout_steps=32)))
        assert runner.metadata["host_compiler_options"] == pp.compute.measured_compiler_options()
        template = runner.init(seed=999, window_size=128)
        a = pp.checkpoint.restore(resumed, template, expected_metadata=runner.metadata)
        b = pp.checkpoint.restore(full, template, expected_metadata=runner.metadata)
        assert_same(a, b)
    finally:
        env.close()


def test_timed_mpe_snapshot_preserves_partial_buffer_and_replay(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/compute_ppo.py"
    saved, resumed, full = (tmp_path / name for name in ["saved", "resumed", "full"])
    execute(script, ["--chunks", "1", "--checkpoint", str(saved)])
    actual = execute(
        script, ["--chunks", "2", "--resume", str(saved), "--checkpoint", str(resumed)]
    )
    expected = execute(script, ["--chunks", "3", "--checkpoint", str(full)])
    assert actual["world_ticks"] == expected["world_ticks"] == 24
    assert actual["updates"] == expected["updates"] == 1
    assert execute(script, ["--replay", str(resumed)])["exact_replay"] is True
    spec = EnvSpec(
        id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
    )
    env = PettingZooLifetime(
        cast(ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=3)), snapshot=True
    )
    try:
        config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=16)
        runner = ComputePPORunner(env, IPPO(action_size=5, config=config), decisions=4)
        template = runner.init(seed=999, window_size=16)
        payload = {
            "state": template,
            "start": template,
            "trace": TimingTrace.from_ticks([2] * 4, quantum_ns=1_000_000),
        }
        metadata = {**runner.metadata, "cost_source": "recorded", "recorded_profile_ticks": 2}
        a = pp.checkpoint.restore(resumed, payload, expected_metadata=metadata)
        b = pp.checkpoint.restore(full, payload, expected_metadata=metadata)
        assert_same(a["state"], b["state"])
    finally:
        env.close()
