"""The public training example must resume a saved lifetime, not restart it."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_dqn_training_example_resumes_across_processes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "train_dqn.py"
    checkpoint = tmp_path / "first"
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    first = subprocess.run(
        [
            sys.executable,
            str(script),
            "--updates",
            "2",
            "--seed",
            "1",
            "--checkpoint",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    saved = json.loads(first.stdout)
    assert saved["lifetime_ticks"] == 256
    assert saved["updates"] == 2
    assert saved["gradient_steps"] == 8
    assert saved["buffer_cursor"] == 256
    assert saved["healthy"] is True
    resumed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--updates",
            "1",
            "--seed",
            "999",
            "--resume",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    restored = json.loads(resumed.stdout)
    assert restored["lifetime_ticks"] == 384
    assert restored["gradient_steps"] == 12
    assert restored["buffer_cursor"] == 384
    assert restored["resumed_from"] == str(checkpoint)
    uninterrupted = subprocess.run(
        [sys.executable, str(script), "--updates", "3", "--seed", "1"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    complete = json.loads(uninterrupted.stdout)
    assert restored["mean_reward"] == complete["mean_reward"]
    assert restored["windowed_reward"] == complete["windowed_reward"]
    assert restored["loss"] == complete["loss"]


def test_sac_training_example_resumes_across_processes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "train_sac.py"
    checkpoint = tmp_path / "first"
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    first = subprocess.run(
        [
            sys.executable,
            str(script),
            "--updates",
            "2",
            "--seed",
            "1",
            "--checkpoint",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    saved = json.loads(first.stdout)
    assert saved["lifetime_ticks"] == 256
    assert saved["updates"] == 2
    assert saved["gradient_steps"] == 256
    assert saved["buffer_cursor"] == 256
    assert saved["healthy"] is True
    resumed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--updates",
            "1",
            "--seed",
            "999",
            "--resume",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    restored = json.loads(resumed.stdout)
    assert restored["lifetime_ticks"] == 384
    assert restored["gradient_steps"] == 384
    assert restored["buffer_cursor"] == 384
    assert restored["resumed_from"] == str(checkpoint)
    uninterrupted = subprocess.run(
        [sys.executable, str(script), "--updates", "3", "--seed", "1"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    complete = json.loads(uninterrupted.stdout)
    assert restored["mean_reward"] == complete["mean_reward"]
    assert restored["windowed_reward"] == complete["windowed_reward"]
    assert restored["critic_loss"] == complete["critic_loss"]


def test_training_example_resumes_across_processes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "train_ppo.py"
    checkpoint = tmp_path / "first"
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    first = subprocess.run(
        [
            sys.executable,
            str(script),
            "--updates",
            "2",
            "--seed",
            "1",
            "--checkpoint",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    saved = json.loads(first.stdout)
    assert saved["lifetime_ticks"] == 256
    assert saved["updates"] == 2
    assert saved["healthy"] is True
    resumed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--updates",
            "1",
            "--seed",
            "999",
            "--resume",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    restored = json.loads(resumed.stdout)
    assert restored["lifetime_ticks"] == 384
    assert restored["updates"] == 3
    assert restored["gradient_steps"] == 48
    assert restored["healthy"] is True
    assert restored["resumed_from"] == str(checkpoint)
    uninterrupted = subprocess.run(
        [sys.executable, str(script), "--updates", "3", "--seed", "1"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    complete = json.loads(uninterrupted.stdout)
    assert restored["mean_reward"] == complete["mean_reward"]
    assert restored["windowed_reward"] == complete["windowed_reward"]
    assert restored["loss"] == complete["loss"]
