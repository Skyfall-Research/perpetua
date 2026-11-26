"""Measure tuning cost: one vmapped sweep versus per-config job arrays.

Claim 4 of the measured claims (docs/measurement.md): *tuning becomes affordable*
(baseline: per-config job arrays; measurement: GPU-hours for a fixed sweep of
seeds × hyperparameters). Because continuous hyperparameters are traced state,
a whole sweep runs as **one vmapped compiled job**. The baseline emulates the
common practice of one job per configuration, each paying its own compilation.

Both modes run the same work: `updates` PPO updates of 128 ticks per lifetime
on the same drifting environment. Timings include compilation, because that is
the cost the baseline pays per configuration.

```bash
# Mac/CPU smoke test
python scripts/measure_tuning.py --mode sweep --sets 2 --seeds 1 --updates 3

# H200 measurement
python scripts/measure_tuning.py --mode sweep --sets 4 --seeds 4 --updates 100 --require-gpu
```
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Any, cast

import jax
import jax.numpy as jnp

import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner
from perpetua.agents.ppo import PPOHyperparameters

LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
ENTROPY_COEFFICIENTS = (0.0, 0.01, 0.05, 0.1)
ROLLOUT_STEPS = 128
WINDOW = 128


def sweep_configs(sets: int, seeds: int) -> list[tuple[float, float, int]]:
    """Return the deterministic sweep as (learning rate, entropy, seed) triples."""
    if min(sets, seeds) < 1:
        raise ValueError("sets and seeds must be positive integers.")
    configs: list[tuple[float, float, int]] = []
    for index in range(sets):
        learning_rate = LEARNING_RATES[index % len(LEARNING_RATES)]
        entropy = ENTROPY_COEFFICIENTS[(index // len(LEARNING_RATES)) % len(ENTROPY_COEFFICIENTS)]
        for seed in range(seeds):
            configs.append((learning_rate, entropy, seed))
    return configs


def build_trainer() -> Any:
    """One runner shape shared by both modes."""
    env = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], period=64))
    return PPORunner(env, PPO(action_size=5, config=PPOConfig(rollout_steps=ROLLOUT_STEPS)))


def _hyperparameters(
    base: PPOHyperparameters, learning_rate: float, entropy: float
) -> PPOHyperparameters:
    return base._replace(
        learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
        entropy_coefficient=jnp.asarray(entropy, dtype=jnp.float32),
    )


def run_single(config: tuple[float, float, int], updates: int) -> dict[str, Any]:
    """One configuration in its own process: compile, then train."""
    learning_rate, entropy, seed = config
    trainer = build_trainer()
    base = PPOHyperparameters.from_config(trainer.agent.config)
    hyperparameters = _hyperparameters(base, learning_rate, entropy)
    started = time.perf_counter()
    state = trainer.init(seed=seed, window_size=WINDOW, hyperparameters=hyperparameters)
    state = jax.block_until_ready(trainer(state))
    for _ in range(updates - 1):
        state = jax.block_until_ready(trainer(state))
    seconds = time.perf_counter() - started
    return {
        "config": [learning_rate, entropy, seed],
        "updates": updates,
        "seconds": seconds,
        "ticks": int(state.lifetime.state.t.to_int()),
        "healthy": bool(state.healthy),
    }


def run_vmap(configs: list[tuple[float, float, int]], updates: int) -> dict[str, Any]:
    """The whole sweep as one vmapped compiled job."""
    trainer = build_trainer()
    base = PPOHyperparameters.from_config(trainer.agent.config)
    hyperparameters = jax.tree.map(
        lambda *values: jnp.stack(values),
        *[_hyperparameters(base, learning_rate, entropy) for learning_rate, entropy, _ in configs],
    )
    seeds = jnp.asarray([seed for _, _, seed in configs], dtype=jnp.int32)

    def initialize(hp: PPOHyperparameters, seed: jax.Array) -> Any:
        return trainer.init(seed=seed, window_size=WINDOW, hyperparameters=hp)

    def advance(state: Any) -> Any:
        return trainer(state)

    started = time.perf_counter()
    states = jax.vmap(advance)(jax.vmap(initialize)(hyperparameters, seeds))
    for _ in range(updates - 1):
        states = jax.block_until_ready(jax.vmap(advance)(states))
    states = jax.block_until_ready(states)
    seconds = time.perf_counter() - started
    return {
        "configs": len(configs),
        "updates": updates,
        "seconds": seconds,
        "ticks_per_lifetime": int(states.lifetime.state.t.lo[0]),
        "healthy": bool(jnp.all(states.healthy)),
    }


def sweep(sets: int, seeds: int, updates: int, *, require_gpu: bool) -> dict[str, Any]:
    """Run the vmapped job and the per-config job array, then compare."""
    configs = sweep_configs(sets, seeds)
    vmap_result = run_vmap(configs, updates)
    job_results: list[dict[str, Any]] = []
    for index in range(len(configs)):
        command = [
            sys.executable,
            __file__,
            "--mode",
            "single",
            "--index",
            str(index),
            "--sets",
            str(sets),
            "--seeds",
            str(seeds),
            "--updates",
            str(updates),
        ]
        if require_gpu:
            command.append("--require-gpu")
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        job_results.append(cast(dict[str, Any], json.loads(completed.stdout)))
    job_seconds = sum(float(result["seconds"]) for result in job_results)
    vmap_seconds = float(vmap_result["seconds"])
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
        "sweep": {
            "configs": len(configs),
            "sets": sets,
            "seeds": seeds,
            "updates": updates,
            "rollout_steps": ROLLOUT_STEPS,
            "config_list": [list(config) for config in configs],
        },
        "vmapped": vmap_result,
        "job_array": {
            "jobs": len(job_results),
            "seconds_per_job": [float(result["seconds"]) for result in job_results],
            "total_seconds": job_seconds,
            "healthy": all(bool(result["healthy"]) for result in job_results),
        },
        "gpu_hours": {
            "vmapped": vmap_seconds / 3600.0,
            "job_array": job_seconds / 3600.0,
            "ratio": job_seconds / vmap_seconds,
        },
        "note": (
            "Both modes run the same per-lifetime work (same environment, PPO "
            "shape and update count) and include compilation: the job array "
            "pays it once per configuration, the vmapped job once in total. "
            "GPU-hours are wall-clock on one device; a cluster parallelises "
            "the job array without reducing its GPU-hour total."
        ),
    }


def main() -> None:
    """Run the requested measurement mode and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sweep", "single"], required=True)
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU measurement requested, but JAX is not using a GPU.")
    if args.mode == "single":
        configs = sweep_configs(int(args.sets), int(args.seeds))
        if not 0 <= args.index < len(configs):
            raise ValueError("index is outside the sweep.")
        print(json.dumps(run_single(configs[args.index], int(args.updates)), indent=2))
        return
    print(
        json.dumps(
            sweep(
                int(args.sets),
                int(args.seeds),
                int(args.updates),
                require_gpu=bool(args.require_gpu),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
