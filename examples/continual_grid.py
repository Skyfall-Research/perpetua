"""Run four batched continuing worlds with two scripted agents each."""

import argparse
import json
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState


def run_example(
    *, chunks: int = 8, drift_name: str = "cyclic", require_gpu: bool = False
) -> dict[str, object]:
    """Run bounded chunks and report online reward, not learning performance."""
    if type(chunks) is not int or chunks < 1:
        raise ValueError("chunks must be a positive integer.")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU execution requested, but JAX is not using a GPU.")
    drifts: dict[str, pp.drift.Drift] = {
        "stationary": pp.drift.Stationary(),
        "cyclic": pp.drift.Cyclic([(7, 7), (0, 0), (0, 7)], period=127),
        "random-walk": pp.drift.RandomWalk(low=0, high=7, scale=0.2),
    }
    if drift_name not in drifts:
        raise ValueError(f"Unknown drift: {drift_name}")
    env = pp.make("ContinualGrid-v0", size=8, num_agents=2, drift=drifts[drift_name])

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[GridState]]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=128)

    carry = jax.vmap(initialize)(jnp.arange(4, dtype=jnp.uint32))
    runner = pp.runtime.Runner(env, pp.policies.greedy, ticks=256)
    run_batch = cast(
        Callable[
            [pp.runtime.Lifetime[DriftEnvState[GridState]]],
            pp.runtime.Lifetime[DriftEnvState[GridState]],
        ],
        jax.jit(jax.vmap(runner)),
    )
    for _ in range(chunks):
        carry = run_batch(carry)
        if not bool(jnp.all(carry.metrics.healthy)):
            raise RuntimeError("A non-finite reward was detected at a chunk boundary.")
    hi, lo = np.asarray(carry.state.t.hi), np.asarray(carry.state.t.lo)
    ticks = (int(hi[0]) << 32) | int(lo[0])
    assert ticks == chunks * 256
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
        "drift": drift_name,
        "policy": "greedy (scripted, not learning)",
        "lifetimes": 4,
        "agents": 2,
        "ticks_per_lifetime": ticks,
        "total_environment_ticks": ticks * 4,
        "mean_reward": np.asarray(carry.metrics.mean).tolist(),
        "windowed_reward": np.asarray(carry.metrics.window_mean).tolist(),
        "ema_reward": np.asarray(carry.metrics.ema).tolist(),
        "healthy": True,
        "metadata": runner.metadata,
    }


def main() -> None:
    """Print a machine-readable report for the selected drift family."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument(
        "--drift", choices=["stationary", "cyclic", "random-walk"], default="cyclic"
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = run_example(
        chunks=int(args.chunks), drift_name=str(args.drift), require_gpu=bool(args.require_gpu)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
