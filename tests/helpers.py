"""Assertions shared by the test modules."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_same(left: object, right: object) -> None:
    """Assert two pytrees have the same structure and exactly equal leaves (keys by key data)."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_array_equal(a, b)


def unavailable(env_id: str) -> str | None:
    """Why this machine cannot construct a built-in id, or ``None`` when it can."""
    record = pp.spec(env_id)
    if record.platform is not None and sys.platform != record.platform:
        return f"{env_id} runs on {record.platform} only."
    if record.extra == "meal" and importlib.util.find_spec("meal") is None:
        return f"{env_id}: the optional meal group is not installed."
    if record.setup is not None:
        adapter = importlib.import_module(f"perpetua.adapters.{record.namespace}")
        if not adapter.available():
            name = {"coom": "COOM", "agarcl": "AgarCL"}[record.namespace or ""]
            return f"{name} runtime not installed (python -m {record.setup})."
    return None
