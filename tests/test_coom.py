"""COOM through its pinned worker runtime (skipped where it is not installed)."""

# Third-party environments have partially annotated interfaces.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters import coom

ROOT = Path(__file__).resolve().parents[1]

needs_coom = pytest.mark.skipif(
    not coom.available(),
    reason="COOM runtime not installed (python -m perpetua.adapters.coom_setup).",
)


def test_missing_runtime_explains_the_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("PERPETUA_COOM_PYTHON", str(tmp_path / "missing" / "python"))
    with pytest.raises(RuntimeError, match="python -m perpetua.adapters.coom_setup"):
        pp.make("coom/COOM-v0")


def test_the_setup_command_is_a_module_of_the_installed_package() -> None:
    # The checkout's script is a wrapper around the module the error names.
    for command in (
        ["-m", "perpetua.adapters.coom_setup"],
        [str(ROOT / "scripts" / "setup_coom.py")],
    ):
        result = subprocess.run(
            [sys.executable, *command, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "--root" in result.stdout


def test_the_working_directory_is_removed_when_the_worker_cannot_start_or_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import sys
    import tempfile

    from perpetua.adapters.worker import WorkerProcess

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "source" / "COOM").mkdir(parents=True)
    not_executable = tmp_path / "python"
    not_executable.write_text("")
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setenv("PERPETUA_COOM_SOURCE", str(tmp_path / "source"))

    monkeypatch.setenv("PERPETUA_COOM_PYTHON", str(not_executable))
    with pytest.raises(OSError, match=r"\[Errno 13\] Permission denied:"):
        coom._Worker()  # pyright: ignore[reportPrivateUsage]
    assert list(scratch.iterdir()) == []

    def failing_close(self: WorkerProcess) -> None:
        self.process.kill()
        self.process.wait()
        raise RuntimeError("close failed")

    monkeypatch.setenv("PERPETUA_COOM_PYTHON", sys.executable)
    worker = coom._Worker()  # pyright: ignore[reportPrivateUsage]
    assert len(list(scratch.iterdir())) == 1
    monkeypatch.setattr(WorkerProcess, "close", failing_close)
    with pytest.raises(RuntimeError, match="close failed"):
        worker.close()
    assert list(scratch.iterdir()) == []


@needs_coom
def test_worker_errors_are_reported_not_hung() -> None:
    with pytest.raises(RuntimeError, match="COOM worker: KeyError"):
        pp.make("coom/COOM-v0", scenario="no_such_scenario")


@needs_coom
def test_seeded_lifetimes_reproduce_across_respawns() -> None:
    first, second = pp.make("coom/COOM-v0"), pp.make("coom/COOM-v0")
    try:
        a, _, _ = first.init(jax.random.key(3))
        b, _, _ = second.init(jax.random.key(3))
        respawns = 0
        for tick in range(300):
            action = jnp.asarray([tick % 12], dtype=jnp.int32)
            a, obs_a, reward_a, info_a = first.step(a, action, jax.random.key(tick))
            b, obs_b, reward_b, _ = second.step(b, action, jax.random.key(tick))
            np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
            np.testing.assert_array_equal(np.asarray(reward_a), np.asarray(reward_b))
            respawns += int(info_a["respawn"][0])
        assert respawns > 0  # the check crossed at least one COOM episode end
    finally:
        cast(Any, first).close()
        cast(Any, second).close()


@needs_coom
def test_coom_trains_through_host_ppo_with_the_cnn() -> None:
    from perpetua.agents import PPO, HostPPORunner, PPOConfig

    env = pp.make("coom/COOM-v0", scenario="health_gathering")
    try:
        config = PPOConfig(torso="cnn", rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
        trainer = HostPPORunner(env, PPO(action_size=12, config=config))
        state = trainer(trainer.init(seed=0))
        assert bool(state.healthy)
        assert state.agent.gradient_steps.to_int() == 1
    finally:
        cast(Any, env).close()


@needs_coom
def test_coom_sequence_switches_tasks_in_coom_order() -> None:
    env = pp.presets.coom_sequence("CO4", period=2)
    try:
        scenarios = [task.metadata["configuration"]["backend"]["scenario"] for task in env.env.envs]
        # COOM's CO4 is the last four CO8 scenarios.
        assert scenarios == ["chainsaw", "raise_the_roof", "run_and_gun", "health_gathering"]
        runner = pp.runtime.HostRunner(env, pp.policies.random, ticks=8)
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
        assert carry.state.t.to_int() == 8
        assert int(carry.state.drift["task"].phase) == 0  # four tasks, period 2: back to 0
        assert carry.obs.shape == (1, 84, 84, 12)
    finally:
        env.close()


def test_a_failed_reinit_does_not_close_the_old_environment_twice(tmp_path: Any) -> None:
    import os
    import sys
    from pathlib import Path

    from perpetua.adapters.worker import WorkerProcess

    for package in ("COOM", "COOM/env", "COOM/utils"):
        (tmp_path / package).mkdir()
        (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / "COOM" / "utils" / "config.py").write_text(
        "Scenario = {'PITFALL': 0}\nSequence = sequence_scenarios = sequence_tasks = {}\n"
    )
    (tmp_path / "COOM" / "env" / "builder.py").write_text(
        """
import numpy as np

built = 0
build_multi_discrete_actions = None


class Env:
    closed = False

    class action_space:
        n = 3

    def reset(self):
        return np.zeros((4, 2, 2, 3), np.float32), {}

    def close(self):
        if self.closed:
            raise RuntimeError("closed twice")
        self.closed = True


def make_env(*args):
    global built
    built += 1
    if built == 2:
        raise RuntimeError("build failed")
    return Env()
"""
    )
    script = Path(coom.__file__).with_name("coom_worker.py")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    process = WorkerProcess("COOM", [sys.executable, str(script)], env=env, dtype=np.float32)
    try:
        request: dict[str, Any] = {"scenario": "pitfall", "task": "default", "seed": 0}
        process.request_array(command="init", **request)
        with pytest.raises(RuntimeError, match="build failed"):
            process.request(command="init", **request)
        process.request_array(command="init", **request)
    finally:
        process.close()
    assert process.process.returncode == 0
