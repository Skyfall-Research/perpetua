"""Third-party registration: ids resolve, refuse collisions and resume exactly."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import importlib.util
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import perpetua as pp
from perpetua.envs.continual_grid import ContinualGrid
from perpetua.params import DriftEnv


def grid_factory(**kwargs: Any) -> pp.Environment[Any]:
    return pp.make("ContinualGrid-v0", size=3, **kwargs)


def test_register_resolves_ids_and_lists_them() -> None:
    try:
        pp.register("test-grid-v0", grid_factory)
        assert "test-grid-v0" in pp.registered()
        assert pp.environments()[-1] == "test-grid-v0"
        assert "ContinualGrid-v0" in pp.environments()
        env = pp.make("test-grid-v0")
        assert env.metadata["env_id"] == "ContinualGrid-v0"
        assert env.num_agents == 1
    finally:
        pp.unregister("test-grid-v0")
    assert "test-grid-v0" not in pp.registered()
    with pytest.raises(ValueError, match="Unknown environment"):
        pp.make("test-grid-v0")


def test_register_refuses_collisions_unless_overwriting() -> None:
    with pytest.raises(ValueError, match="built-in"):
        pp.register("ContinualGrid-v0", grid_factory)
    pp.register("test-grid-v0", grid_factory)
    try:
        with pytest.raises(ValueError, match="already registered"):
            pp.register("test-grid-v0", grid_factory)
        pp.register("test-grid-v0", grid_factory, overwrite=True)
        with pytest.raises(ValueError, match=r"Registered environment ids must be non-empty"):
            pp.register("", grid_factory)
        with pytest.raises(ValueError, match=r"Registered environment factories must be"):
            pp.register("test-grid-v0", cast(Any, "not callable"), overwrite=True)
        with pytest.raises(ValueError, match="not a registered"):
            pp.unregister("never-registered-v0")
    finally:
        pp.unregister("test-grid-v0")


def test_overwriting_a_builtin_restores_it_on_unregister() -> None:
    def tiny(**kwargs: Any) -> pp.Environment[Any]:
        # An override factory must build the built-in directly: calling make
        # for the same id would recurse into this registration.
        drift = cast("pp.drift.Drift | None", kwargs.pop("drift", None))
        return DriftEnv(ContinualGrid(size=2), drift, env_id="ContinualGrid-v0")

    pp.register("ContinualGrid-v0", tiny, overwrite=True)
    try:
        assert pp.make("ContinualGrid-v0").size == 2
        assert pp.environments().count("ContinualGrid-v0") == 1
        assert pp.environments()[0] == "ContinualGrid-v0"
    finally:
        pp.unregister("ContinualGrid-v0")
    assert pp.make("ContinualGrid-v0").size == 5


def test_registered_environment_checkpoints_and_resumes(tmp_path: Path) -> None:
    pp.register("test-grid-v0", grid_factory)
    try:
        env = pp.make("test-grid-v0")
        runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
        carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
        path = tmp_path / "registered"
        pp.checkpoint.save(path, carry, metadata=env.metadata)
        restored = pp.checkpoint.restore(
            path,
            pp.runtime.init_lifetime(env, seed=999, window_size=4),
            expected_metadata=env.metadata,
        )
        np.testing.assert_array_equal(
            np.asarray(restored.state.env.positions), np.asarray(carry.state.env.positions)
        )
        assert int(restored.state.t.to_int()) == 4
        np.testing.assert_array_equal(np.asarray(restored.obs), np.asarray(carry.obs))
        np.testing.assert_array_equal(
            np.asarray(restored.metrics.mean), np.asarray(carry.metrics.mean)
        )
    finally:
        pp.unregister("test-grid-v0")


_MISSING_EXTRAS = """
import importlib, sys
for name in ("gymnax", "flax", "torch", "pettingzoo", "jaxmarl"):
    sys.modules[name] = None
import perpetua as pp

def message(action):
    try:
        action()
    except ImportError as error:
        return str(error)
    raise AssertionError("no ImportError")

print(message(lambda: pp.make("gymnax/Acrobot-v1")))
print(message(lambda: pp.make("jaxmarl/MPE_simple_spread-v3")))
print(message(lambda: pp.agents))
print(message(lambda: importlib.import_module("perpetua.torch_bridge")))
print(message(lambda: importlib.import_module("perpetua.adapters.pettingzoo")))
print(pp.make("ContinualGrid-v0").num_agents)
"""


def test_a_missing_extra_names_the_extra_to_install() -> None:
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _MISSING_EXTRAS],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 6, result.stdout
    expectations = [
        ("gymnax/Acrobot-v1", "gymnax", "interop"),
        ("jaxmarl/MPE_simple_spread-v3", "jaxmarl", "jaxmarl"),
        ("perpetua.agents", "flax", "training"),
        ("perpetua.torch_bridge", "torch", "torch"),
        ("perpetua.adapters.pettingzoo", "pettingzoo", "marl"),
    ]
    for line, (feature, package, extra) in zip(lines, expectations, strict=False):
        assert feature in line, line
        assert f"'{package}'" in line, line
        assert f"Perpetua's {extra} extra" in line, line
        assert f'perpetua-rl[{extra}]"' in line, line
    assert lines[-1] == "1"


def test_builtin_environments_exclude_third_party_ids() -> None:
    builtin = pp.environments(builtin=True)
    pp.register("test-grid-v0", grid_factory)
    try:
        assert "test-grid-v0" in pp.environments()
        assert pp.environments(builtin=True) == builtin
        assert pp.environments()[: len(builtin)] == builtin
    finally:
        pp.unregister("test-grid-v0")
    assert pp.environments() == builtin


def test_spec_records_what_an_id_needs_without_constructing_it() -> None:
    assert pp.spec("ContinualGrid-v0") == pp.EnvSpec("ContinualGrid-v0")
    assert pp.spec("gymnax/Acrobot-v1") == pp.EnvSpec(
        "gymnax/Acrobot-v1", namespace="gymnax", extra="interop"
    )
    envpool = pp.spec("envpool/Atari-v0")
    assert (envpool.extra, envpool.host, envpool.platform) == ("cpu-bridge", True, "linux")
    coom = pp.spec("coom/COOM-v0")
    assert (coom.extra, coom.host, coom.setup) == (None, True, "perpetua.adapters.coom_setup")
    morpheus = pp.spec("morpheus/Morpheus-v0")
    assert (morpheus.extra, morpheus.optional_extra, morpheus.host) == (None, "morpheus", True)
    for env_id in pp.environments(builtin=True):
        assert pp.spec(env_id).id == env_id
    with pytest.raises(ValueError, match="not a built-in environment"):
        pp.spec("never-registered-v0")
    pp.register("test-grid-v0", grid_factory)
    try:
        with pytest.raises(ValueError, match="not a built-in environment"):
            pp.spec("test-grid-v0")
    finally:
        pp.unregister("test-grid-v0")


def test_spec_names_real_extras_and_setup_scripts() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for env_id in pp.environments(builtin=True):
        record = pp.spec(env_id)
        for extra in (record.extra, record.optional_extra):
            assert extra is None or extra in extras, env_id
        assert record.setup is None or importlib.util.find_spec(record.setup) is not None, env_id
        assert record.setup is None or record.host, env_id


def test_adapter_install_hints_agree_with_the_registry() -> None:
    """The adapters that word their own hint name the extra or script the registry records."""
    import inspect

    from perpetua.adapters import agarcl, coom, host

    hints = {
        "metaworld/MT1-v0": host,
        "envpool/CartPole-v1": host,
        "coom/COOM-v0": coom,
        "agarcl/agario-grid-v0": agarcl,
    }
    for env_id, owner in hints.items():
        record = pp.spec(env_id)
        source = " ".join(inspect.getsource(owner).split())
        if record.setup is not None:
            assert f"python -m {record.setup}" in source, env_id
        else:
            assert f"Install Perpetua's {record.extra} extra" in source, env_id
    worded = {pp.spec(env_id).namespace for env_id in hints}
    for env_id in pp.environments(builtin=True):
        record = pp.spec(env_id)
        # Every other id gets its hint from ``requires_extra`` in ``make``.
        assert record.namespace in worded or record.setup is None, env_id


@pytest.mark.parametrize(
    ("env_id", "field", "kwargs"),
    [
        ("PartnerGrid-v0", "partner", {}),
        ("OpponentGrid-v0", "opponent", {}),
        ("jaxmarl/MPE_simple_spread-v3", "scripted", {"num_scripted": 1}),
        ("jaxmarl/MPE_simple_tag_v3", "scripted", {"num_scripted": 1}),
    ],
)
def test_a_field_drift_keyword_and_the_drift_mapping_cannot_name_one_field(
    env_id: str, field: str, kwargs: dict[str, Any]
) -> None:
    if env_id.startswith("jaxmarl/"):
        pytest.importorskip("jaxmarl")
    keyword = f"{field}_drift"
    slow, fast = pp.drift.Cyclic([(1.0,), (0.0,)], 5), pp.drift.Cyclic([(0.0,), (1.0,)], 3)
    with pytest.raises(ValueError, match=rf"{keyword}= and drift\['{field}'\] both set"):
        pp.make(env_id, drift={field: fast}, **{keyword: slow}, **kwargs)
    # Each alone, and the keyword next to a mapping for another field, still work.
    other = "goal" if field != "scripted" else "reward_multiplier"
    values = [(0.0, 0.0), (1.0, 1.0)] if other == "goal" else [(1.0,), (2.0,)]
    both = cast(
        Any, pp.make(env_id, drift={other: pp.drift.Cyclic(values, 7)}, **{keyword: slow}, **kwargs)
    )
    assert both.drifts[field] is slow
    assert both.drifts[other].period == 7
    assert cast(Any, pp.make(env_id, drift={field: fast}, **kwargs)).drifts[field] is fast
    assert cast(Any, pp.make(env_id, **{keyword: slow}, **kwargs)).drifts[field] is slow
