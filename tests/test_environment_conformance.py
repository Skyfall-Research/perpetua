"""The contract checks for every registered environment.

``pp.testing.check_env`` runs the generic checks (seeded reproducibility,
chunk continuity and clock, privilege isolation and Oracle, shapes, state
types, compilation and, where supported, a checkpoint round-trip); this file
adds the registry's own: no space change while stationary, metadata and
unknown-field refusal. EnvPool is Linux-only; snapshot
support is checked separately for EnvPool CartPole.
"""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import sys
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from helpers import unavailable

import perpetua as pp

ENV_IDS = (
    *pp.environments(builtin=True),
    "envpool/CartPole-v1@snapshot",
    "agarcl/agario-screen-v0@replay",
    "agarcl/agario-grid-v0@replay",
)


def make(case_id: str) -> Any:
    """Construct one environment, keeping host batches small for the suite."""
    env_id, _, variant = case_id.partition("@")
    kwargs: dict[str, Any] = {}
    if env_id.startswith("envpool/"):
        kwargs = {"num_envs": 2, "num_threads": 1}
    if variant == "snapshot":
        kwargs["snapshot"] = True
    if variant == "replay":
        kwargs["replay_capacity"] = 64
    if env_id == "agarcl/agario-grid-v0":
        kwargs["grid_size"] = 32
    return pp.make(env_id, **kwargs)


def action_for(env: Any) -> jax.Array:
    space = env.action_space
    if hasattr(space, "nvec"):
        return jnp.zeros((env.num_agents,), dtype=jnp.int32)
    return jnp.zeros(space.shape, dtype=jnp.float32)


def configuration_of(env: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", cast("dict[str, Any]", env.metadata)["configuration"])


def check_space_change_and_stable_shapes(case_id: str) -> None:
    env = make(case_id)
    try:
        state, obs, info = env.init(jax.random.key(0))
        obs_shape, mask_shape = obs.shape, info["action_mask"].shape
        space_change = cast("jax.Array", info.get("space_change"))
        assert space_change.shape == (env.num_agents,)
        assert not bool(jnp.any(space_change))
        for tick in range(4):
            state, obs, _, info = env.step(state, action_for(env), jax.random.key(tick))
            assert obs.shape == obs_shape
            assert info["action_mask"].shape == mask_shape
            assert not bool(jnp.any(cast("jax.Array", info.get("space_change"))))
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def check_metadata_and_unknown_drift_fields(case_id: str) -> None:
    env_id = case_id.partition("@")[0]
    env = make(case_id)
    try:
        configuration = configuration_of(env)
        assert configuration["drift"]["type"] == "stationary"
        assert (env.metadata.get("execution", "jax") == "host") == pp.spec(env_id).host
        if env_id == "envpool/Atari-v0":
            assert env.metadata["env_id"] == "envpool/Breakout-v5"
        else:
            assert env.metadata["env_id"] == env_id
        with pytest.raises(ValueError, match="Unknown drift parameters"):
            pp.make(env_id, drift={"hatch": pp.drift.Stationary()})
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


@pytest.fixture(autouse=True)
def release_compiled_programs() -> Any:
    """Free each environment's compiled programs before the next one.

    Every case compiles several runners; keeping all of them alive made this
    file need about 6 GB in one process.
    """
    yield
    jax.clear_caches()


@pytest.mark.parametrize("case_id", ENV_IDS)
def test_all_checks(case_id: str, tmp_path: Any) -> None:
    reason = unavailable(case_id.partition("@")[0])
    if reason is not None:
        pytest.skip(reason)
    if case_id.endswith("@snapshot"):
        from importlib.util import find_spec

        if find_spec("perpetua_envpool_snapshot") is None:
            pytest.skip("Build native/envpool_snapshot for EnvPool snapshots.")
    # The public checker: reproducibility, continuity and clock, privilege and
    # Oracle, shapes, state types, compilation and checkpoints.
    pp.testing.check_env(lambda: make(case_id))
    check_space_change_and_stable_shapes(case_id)
    check_metadata_and_unknown_drift_fields(case_id)


@pytest.mark.skipif(sys.platform != "linux", reason="EnvPool Atari is Linux-only.")
def test_atari_reward_drift_reaches_the_backend() -> None:
    env = pp.make(
        "envpool/Atari-v0",
        num_envs=2,
        num_threads=1,
        drift=pp.drift.Cyclic([(1.0,), (2.0,)], 2),
    )
    try:
        state, _, _ = env.init(jax.random.key(0))
        for tick in range(2):
            state, _, _, _ = env.step(state, action_for(env), jax.random.key(tick))
        assert float(env.privileged(state)["reward_multiplier_parameters"][0]) == 2.0
        assert int(state.drift["reward_multiplier"].phase) == 1
    finally:
        cast(Any, env).close()
