"""Measurement instruments must be exact, inert and reproducible."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import logging
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp


def test_compile_counter_counts_only_new_compilations() -> None:
    def scale_impl(values: jax.Array) -> jax.Array:
        return values * 2.0

    scale = jax.jit(scale_impl)
    handlers_before = list(logging.getLogger("jax").handlers)
    flag_before = bool(jax.config.jax_log_compiles)
    with pp.measurement.count_compilations() as cold:
        jax.block_until_ready(cast(jax.Array, scale(jnp.ones(8))))
    assert cold.count >= 1
    with pp.measurement.count_compilations() as warm:
        jax.block_until_ready(cast(jax.Array, scale(jnp.ones(8))))
    assert warm.count == 0
    with pp.measurement.count_compilations() as changed:
        widened = cast(jax.Array, scale(jnp.ones(16)))
        jax.block_until_ready(widened)
    assert changed.count >= 1
    assert bool(jax.config.jax_log_compiles) is flag_before
    assert list(logging.getLogger("jax").handlers) == handlers_before


def test_compile_counter_sees_compilations_when_logging_is_muted() -> None:
    emitter = logging.getLogger("jax._src.interpreters.pxla")
    disable_before, level_before = logging.root.manager.disable, emitter.level
    logging.disable(logging.WARNING)
    emitter.setLevel(logging.ERROR)
    try:

        @jax.jit
        def muted(values: jax.Array) -> jax.Array:
            return values * 5.0 + 3.0

        with pp.measurement.count_compilations() as cold:
            jax.block_until_ready(cast(jax.Array, muted(jnp.ones(3))))
        assert cold.count >= 1
        assert logging.root.manager.disable == logging.WARNING
        assert emitter.level == logging.ERROR
    finally:
        logging.disable(disable_before)
        emitter.setLevel(level_before)


def test_xla_flops_is_static_and_ordered() -> None:
    def tanh_sum(values: jax.Array) -> jax.Array:
        return jnp.tanh(values).sum()

    def plus_one(values: jax.Array) -> jax.Array:
        return values + 1.0

    small = pp.measurement.xla_flops(tanh_sum, jnp.ones(8))
    large = pp.measurement.xla_flops(tanh_sum, jnp.ones(64))
    assert small > 0
    assert large > small
    assert pp.measurement.xla_flops(plus_one, jnp.ones(4)) > 0


def test_autoreset_step_selects_fresh_state_on_limit() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    state = env.init(jax.random.key(0))[0]
    action = jnp.asarray([0])
    key = jax.random.key(7)
    step_key, reset_key = jax.random.split(key)
    stepped, reward, elapsed = pp.measurement.autoreset_step(
        env, state, action, key, limit=2, ticks=jnp.int32(0)
    )
    expected_state, _, expected_reward, _ = env.step(state, action, step_key)
    assert_same(stepped, expected_state)
    np.testing.assert_array_equal(np.asarray(reward), np.asarray(expected_reward))
    assert int(elapsed) == 1
    fresh_state, _, _ = env.init(reset_key)
    selected, _, reset_elapsed = pp.measurement.autoreset_step(
        env, state, action, key, limit=2, ticks=jnp.int32(1)
    )
    assert_same(selected, fresh_state)
    assert int(reset_elapsed) == 0
    again = pp.measurement.autoreset_step(env, state, action, key, limit=2, ticks=jnp.int32(1))
    assert_same(selected, again[0])
    with pytest.raises(ValueError, match=r"limit must be a positive integer\."):
        pp.measurement.autoreset_step(env, state, action, key, limit=0, ticks=jnp.int32(0))


def test_transition_scan_matches_manual_lifetime_and_autoresets() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    state = env.init(jax.random.key(0))[0]
    action = jnp.asarray([0])
    key = jax.random.key(5)
    final, rewards = pp.measurement.transition_scan(env, state, action, key, ticks=4)
    manual_state = state
    manual_key = key
    manual_rewards: list[jax.Array] = []
    for _ in range(4):
        manual_key, step_key = jax.random.split(manual_key)
        manual_state, _, manual_reward, _ = env.step(manual_state, action, step_key)
        manual_rewards.append(manual_reward)
    assert_same(final, manual_state)
    np.testing.assert_array_equal(np.asarray(rewards), np.asarray(manual_rewards))
    assert rewards.shape == (4, 1)
    reset_state, reset_rewards = pp.measurement.transition_scan(
        env, state, action, key, ticks=4, autoreset_limit=2
    )
    assert reset_rewards.shape == (4, 1)
    assert not np.array_equal(
        np.asarray(reset_state.env.positions), np.asarray(final.env.positions)
    )
    with pytest.raises(ValueError, match=r"ticks must be a positive integer\."):
        pp.measurement.transition_scan(env, state, action, key, ticks=0)
    with pytest.raises(ValueError, match=r"autoreset_limit must be None or a positive"):
        pp.measurement.transition_scan(env, state, action, key, ticks=4, autoreset_limit=0)


def test_code_lines_separates_roles(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        '"""Module docstring."""\n'
        "\n"
        "# a comment\n"
        "import jax\n"
        "\n"
        "\n"
        "def f() -> int:\n"
        '    """Function docstring."""\n'
        "    return 1\n"
    )
    counts = pp.measurement.code_lines(source)
    assert counts.total == 9
    assert counts.blank == 3
    assert counts.comment == 1
    assert counts.docstring == 2
    assert counts.code == 3
    quoted = tmp_path / "quoted.py"
    quoted.write_text(
        "def f() -> str:\n"
        '    """Summary.\n'
        "\n"
        "    # a heading inside the docstring\n"
        '    """\n'
        '    text = """\n'
        "# a line of data, not a comment\n"
        '"""\n'
        "    return text  # trailing\n"
        "    # a real comment\n"
    )
    assert pp.measurement.code_lines(quoted) == (10, 1, 1, 3, 5)
    module_counts = pp.measurement.code_lines(Path(pp.measurement.__file__))
    assert module_counts.total > 0
    assert module_counts.code > 0
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    with pytest.raises(SyntaxError):
        pp.measurement.code_lines(bad)


def test_measure_example_runs_and_reports_all_sections() -> None:
    import json
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "examples" / "measure_claims.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phases",
            "8",
            "--ticks-per-phase",
            "4",
            "--chunk",
            "16",
            "--b2-ticks",
            "256",
            "--autoreset-limit",
            "16",
        ],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    report = json.loads(result.stdout)
    b1 = report["b1_compile_events"]
    assert b1["lifetime"]["warm_compile_events"] == 0
    assert b1["lifetime"]["cold_compile_events"] >= 1
    assert b1["naive_new_constant_per_task"]["compile_events"] >= 8
    b2 = report["b2_reset_path"]
    assert b2["autoreset_step_flops"] > b2["step_flops"] > 0
    assert b2["reset_path_flops_overhead"] > 0
    assert b2["lifetime_steps_per_second"] > b2["autoreset_steps_per_second"] > 0
    assert b2["reset_ticks_per_second"] > 0
    pendulum = report["b2_reset_path_pendulum"]
    # The static cost model can report no difference (Pendulum's reset work is
    # below its resolution) while the measured loop is far slower.
    assert pendulum["autoreset_step_flops"] >= pendulum["step_flops"] > 0
    assert pendulum["init_flops"] > 0
    assert pendulum["lifetime_steps_per_second"] > pendulum["autoreset_steps_per_second"] > 0
    glue = report["b3_glue"]
    assert glue["totals"]["adapters"]["files"] >= 5
    assert glue["totals"]["adapters"]["code"] > 0
    assert glue["totals"]["examples"]["code"] > 0


def test_throughput_example_reports_iqm_intervals() -> None:
    import json
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "examples" / "measure_throughput.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--envs",
            "ContinualGrid-v0",
            "--chunks",
            "16",
            "--batches",
            "1,4",
            "--ticks",
            "64",
            "--seeds",
            "4",
            "--repetitions",
            "1",
        ],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    report = json.loads(result.stdout)
    assert report["backend"] == "cpu"
    assert report["policy"] == "auto"
    configs = report["configs"]
    assert all(config["policy"] == "random" for config in configs)
    assert [config["batch"] for config in configs] == [1, 4]
    assert [config["chunk"] for config in configs] == [16, 16]
    assert configs[0]["steps_per_measurement"] == 64
    assert configs[1]["steps_per_measurement"] == 256
    for config in configs:
        assert config["calls_per_measurement"] == 4
        assert len(config["steps_per_second_per_seed"]) == 4
        assert config["best_steps_per_second"] > 0
        assert 0 < config["worst_steps_per_second"] <= config["best_steps_per_second"]
        assert config["iqm_low"] <= config["iqm_steps_per_second"] <= config["iqm_high"]
        assert config["mean_low"] <= config["mean_steps_per_second"] <= config["mean_high"]
        assert config["median_low"] <= config["median_steps_per_second"] <= config["median_high"]


def test_training_throughput_example_reports_native_intervals() -> None:
    import json
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "examples" / "measure_training_throughput.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "native",
            "--env",
            "gymnax/CartPole-v1",
            "--batch",
            "2",
            "--duration",
            "2",
            "--seeds",
            "3",
        ],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    report = json.loads(result.stdout)
    assert report["backend"] == "cpu"
    measurement = report["measurement"]
    assert measurement["stack"] == "lifetime-ppo-native-vmap"
    assert measurement["env"] == "gymnax/CartPole-v1"
    assert measurement["batch"] == 2
    assert measurement["rollout_steps"] == 128
    assert len(measurement["steps_per_second_per_seed"]) == 3
    assert measurement["best_steps_per_second"] > 0
    assert measurement["iqm_low"] <= measurement["iqm"] <= measurement["iqm_high"]
    assert measurement["mean_low"] <= measurement["mean"] <= measurement["mean_high"]


@pytest.mark.torch
def test_baseline_script_reports_throughput_when_torch_is_available() -> None:
    import json
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_baseline.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-envs",
            "2",
            "--num-steps",
            "16",
            "--warmup-updates",
            "1",
            "--duration",
            "2",
            "--device",
            "cpu",
        ],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    report = json.loads(result.stdout)
    assert report["stack"] == "async-vector-env-pytorch-ppo"
    assert report["num_envs"] == 2
    assert report["batch_size"] == 32
    assert report["steps_measured"] > 0
    assert report["steps_per_second"] > 0
    assert report["updates"] >= 1


def test_tuning_sweep_measures_vmap_against_job_array() -> None:
    import json
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_tuning.py"
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "sweep",
            "--sets",
            "2",
            "--seeds",
            "1",
            "--updates",
            "2",
        ],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=1200,
    )
    report = json.loads(result.stdout)
    assert report["backend"] == "cpu"
    assert report["sweep"]["configs"] == 2
    assert report["vmapped"]["healthy"] is True
    assert report["job_array"]["jobs"] == 2
    assert report["job_array"]["healthy"] is True
    assert report["vmapped"]["ticks_per_lifetime"] == 2 * 128
    assert report["gpu_hours"]["job_array"] > report["gpu_hours"]["vmapped"]
    assert report["gpu_hours"]["ratio"] > 1.0
