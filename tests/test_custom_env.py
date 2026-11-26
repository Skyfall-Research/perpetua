"""The custom-environment guide's example must satisfy the documented contract."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom_env.py"


def load_example() -> Any:
    spec = importlib.util.spec_from_file_location("custom_env_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lineworld_satisfies_the_documented_contract() -> None:
    env = load_example().make(drift=pp.drift.Cyclic([3.0, 1.0], period=4))
    state, obs, info = env.init(jax.random.key(0))
    twin, twin_obs, _twin_info = env.init(jax.random.key(0))
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(twin_obs))
    assert int(state.t.lo) == int(twin.t.lo)
    assert env.observation_space.contains(np.asarray(obs))
    assert np.asarray(obs).shape == (1, 2)
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    assert "oracle" not in info
    assert info["action_mask"].shape == (1, 3)
    assert info["active_mask"].shape == (1,)
    next_state, next_obs, reward, next_info = env.step(state, jnp.asarray([2]), jax.random.key(1))
    observed, observed_info = env.observe(next_state)
    np.testing.assert_array_equal(np.asarray(next_obs), np.asarray(observed))
    np.testing.assert_array_equal(
        np.asarray(next_info["action_mask"]), np.asarray(observed_info["action_mask"])
    )
    np.testing.assert_array_equal(
        np.asarray(next_info["active_mask"]), np.asarray(observed_info["active_mask"])
    )
    np.testing.assert_array_equal(
        np.asarray(next_info["respawn"]), np.asarray(observed_info["respawn"])
    )
    assert int(next_info["t"].lo) == int(observed_info["t"].lo)
    assert np.asarray(reward).shape == (1,)
    repeat = env.step(state, jnp.asarray([2]), jax.random.key(1))
    np.testing.assert_array_equal(np.asarray(repeat[2]), np.asarray(reward))
    assert {"phase", "drift_parameters", "drift_changed"} <= set(env.privileged(next_state))
    assert {"goal_parameters", "goal_phase", "goal_changed"} <= set(env.privileged(next_state))


def test_lineworld_compiles_batches_and_composes() -> None:
    env = load_example().make(drift=pp.drift.Cyclic([3.0, 1.0], period=8))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=16)

    def initialize(seed: jax.Array) -> Any:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=16)

    batched = jax.vmap(initialize)(jnp.asarray([0, 1]))
    result = jax.vmap(runner)(batched)
    np.testing.assert_array_equal(np.asarray(result.state.t.lo), 16)
    composed = pp.compose(env, pp.drift.RandomWalk(low=1, high=3, scale=0.1))
    metadata = cast(dict[str, Any], composed.metadata)
    configuration = cast(dict[str, Any], metadata.get("configuration"))
    drift = cast(dict[str, Any], configuration["drift"])
    assert drift["type"] == "random-walk"
    oracle = pp.Oracle(env, expose=["drift_parameters"])
    oracle_state, _, oracle_info = oracle.init(jax.random.key(0))
    assert "oracle" in oracle_info
    assert set(oracle_info["oracle"]) == {"drift_parameters"}
    assert oracle.metadata["oracle_exposure"] == ("drift_parameters",)
    del oracle_state


def test_lineworld_author_code_has_no_drift_bookkeeping() -> None:
    # The point of the parameter contract: the environment itself never
    # touches a drift process, a drift key, the clock or Oracle fields.
    source = EXAMPLE.read_text()
    world = source[source.index("class LineWorld") : source.index("def make(")]
    for framework_concern in ("drift_key", "Clock", "switched", "oracle_fields", "with_drift"):
        assert framework_concern not in world


def test_custom_env_example_runs() -> None:
    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    assert "respawn True" in result.stdout
    assert "composed drift: random-walk" in result.stdout
    json.dumps({})  # keep the import used for parity with other example tests
