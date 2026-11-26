"""Verify JAX compilation, batching and chunk continuity on the selected device."""

import argparse
import json
import time
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from perpetua.reporting import provenance


class RuntimeReport(TypedDict):
    """Results of the smoke check, not a representative RL benchmark."""

    jax_version: str
    backend: str
    provenance: dict[str, str | bool | None]
    devices: list[str]
    batch_size: int
    ticks_per_chunk: int
    compile_and_first_run_seconds: float
    warm_run_seconds: float
    chunk_equivalence: bool


def check_runtime(*, require_gpu: bool = False) -> RuntimeReport:
    """Check compiled stepping and exact chunk continuity; optionally require CUDA."""
    devices = jax.devices()
    if require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("GPU validation requested, but JAX has no GPU device.")

    batch_size = 64
    ticks = 256

    def advance(value: jax.Array) -> jax.Array:
        def tick(carry: jax.Array, _: None) -> tuple[jax.Array, None]:
            return carry + jnp.uint32(1), None

        result, _ = jax.lax.scan(tick, value, None, length=ticks)
        return result

    compiled = jax.jit(jax.vmap(advance))
    values = jnp.arange(batch_size, dtype=jnp.uint32)
    started = time.perf_counter()
    result = compiled(values).block_until_ready()
    first_run = time.perf_counter() - started
    np.testing.assert_array_equal(np.asarray(result), np.arange(batch_size) + ticks)

    started = time.perf_counter()
    continued = compiled(result).block_until_ready()
    warm_run = time.perf_counter() - started
    np.testing.assert_array_equal(np.asarray(continued), np.arange(batch_size) + 2 * ticks)

    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": provenance(),
        "devices": [str(device) for device in devices],
        "batch_size": batch_size,
        "ticks_per_chunk": ticks,
        "compile_and_first_run_seconds": first_run,
        "warm_run_seconds": warm_run,
        "chunk_equivalence": True,
    }


def main() -> None:
    """Print a machine-readable runtime report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_runtime(require_gpu=bool(args.require_gpu)), indent=2))


if __name__ == "__main__":
    main()
