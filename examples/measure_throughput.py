"""Measure end-to-end lifetime throughput on the local backend (claim-1 harness).

Environment steps/s for the native compiled path, swept over chunk sizes
(ticks per compiled call, exposing host-dispatch overhead) and vmapped
lifetime batches (device parallelism), with warm-up, device synchronization,
per-seed timings and IQM + bootstrap confidence intervals from
``perpetua.reporting``. Every lifetime advances the same compiled runner
through its drift schedule, so drift is included in the measured path.

This is the harness for the throughput claim (docs/measurement.md). It does not by
itself establish the >=10x comparison against Gymnasium AsyncVectorEnv +
PyTorch PPO: that needs the external baseline on the same hardware, and no
throughput claim should be made from one machine or one run.
"""

import argparse
import json
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
from gymnasium import spaces

import perpetua as pp


def _make_env(env_id: str) -> pp.Environment[Any]:
    """Build a drifting environment for the throughput sweep."""
    if env_id == "ContinualGrid-v0":
        return pp.make(env_id, size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], period=64))
    if env_id == "gymnax/Pendulum-v1":
        return pp.make(env_id, drift=pp.drift.Cyclic([(10.0,), (5.0,)], period=256))
    return pp.make(env_id)


def _policy_for(name: str, env: pp.Environment[Any]) -> Any:
    """Resolve the policy; ``auto`` picks by action space."""
    if name == "auto":
        name = "uniform" if isinstance(env.action_space, spaces.Box) else "random"
    if name == "random":
        return pp.policies.random
    if name == "greedy":
        return pp.policies.greedy
    if name == "uniform":
        box = cast(spaces.Box, env.action_space)
        low = jnp.asarray(box.low, dtype=jnp.float32)
        high = jnp.asarray(box.high, dtype=jnp.float32)

        def uniform_policy(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
            del obs, info
            return jax.random.uniform(key, low.shape, minval=low, maxval=high)

        return uniform_policy
    raise ValueError(f"Unknown policy: {name}")


def measure_config(
    env_id: str,
    *,
    chunk: int,
    batch: int,
    ticks: int,
    seeds: int,
    repetitions: int,
    window: int,
    policy_name: str,
) -> dict[str, object]:
    """Time one (chunk, batch) configuration across seeds with warm-up and sync.

    ``window`` is held fixed across the sweep: the metrics window is copied
    per step, so tying it to the chunk size would confound the chunk sweep.
    """
    if min(chunk, batch, ticks, seeds, repetitions, window) < 1:
        raise ValueError("chunk, batch, ticks, seeds, repetitions and window must be positive.")
    env = _make_env(env_id)
    policy = _policy_for(policy_name, env)
    policy_name = (
        "uniform"
        if policy_name == "auto" and isinstance(env.action_space, spaces.Box)
        else "random"
        if policy_name == "auto"
        else policy_name
    )
    runner = pp.runtime.Runner(env, policy, ticks=chunk)
    calls = max(ticks // chunk, 1)
    measured_ticks = calls * chunk

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[Any]:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=window)

    init_batch = jax.jit(jax.vmap(initialize))
    run_batch = jax.jit(jax.vmap(runner))
    seed_batch = jnp.arange(batch, dtype=jnp.int32)
    carry = jax.block_until_ready(run_batch(init_batch(seed_batch)))
    per_seed: list[float] = []
    for seed_index in range(seeds):
        carry = init_batch(seed_batch + jnp.int32(seed_index * batch))
        carry = jax.block_until_ready(run_batch(carry))
        timings = []
        for _ in range(repetitions):
            started = time.perf_counter()
            for _ in range(calls):
                carry = run_batch(carry)
            carry = jax.block_until_ready(carry)
            timings.append(time.perf_counter() - started)
        per_seed.append(measured_ticks * batch / min(timings))
    scores = jnp.asarray(per_seed, dtype=jnp.float32)
    intervals = pp.reporting.report(scores[:, None], key=jax.random.key(0))
    return {
        "env": env_id,
        "policy": policy_name,
        "chunk": chunk,
        "batch": batch,
        "calls_per_measurement": calls,
        "ticks_per_lifetime": measured_ticks,
        "window_size": window,
        "steps_per_measurement": measured_ticks * batch,
        "seeds": seeds,
        "repetitions": repetitions,
        "steps_per_second_per_seed": per_seed,
        "best_steps_per_second": max(per_seed),
        "worst_steps_per_second": min(per_seed),
        # IQM discards n // 4 values from each end of the sorted scores, as in
        # rliable. Small samples are defined; mean and median are also reported.
        "iqm_steps_per_second": float(intervals["iqm"].point),
        "iqm_low": float(intervals["iqm"].low),
        "iqm_high": float(intervals["iqm"].high),
        "mean_steps_per_second": float(intervals["mean"].point),
        "mean_low": float(intervals["mean"].low),
        "mean_high": float(intervals["mean"].high),
        "median_steps_per_second": float(intervals["median"].point),
        "median_low": float(intervals["median"].low),
        "median_high": float(intervals["median"].high),
    }


def main() -> None:
    """Run the throughput sweep and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default="ContinualGrid-v0,gymnax/Pendulum-v1")
    parser.add_argument("--chunks", default="16,128,1024")
    parser.add_argument("--batches", default="1,8,64")
    parser.add_argument("--ticks", type=int, default=2048)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--policy", choices=["auto", "random", "uniform", "greedy"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU measurement requested, but JAX is not using a GPU.")
    configs = []
    for raw_env in str(args.envs).split(","):
        for raw_chunk in str(args.chunks).split(","):
            for raw_batch in str(args.batches).split(","):
                configs.append(
                    measure_config(
                        raw_env.strip(),
                        chunk=int(raw_chunk),
                        batch=int(raw_batch),
                        ticks=int(args.ticks),
                        window=int(args.window),
                        seeds=int(args.seeds),
                        repetitions=int(args.repetitions),
                        policy_name=str(args.policy),
                    )
                )
    report = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
        "policy": str(args.policy),
        "note": (
            "Steps/s include environment stepping, drift, metrics and the "
            "policy for vmapped lifetimes of identical shape. Warm-up and "
            "device synchronization are excluded from timed regions; each "
            "seed runs a freshly initialized batch. Not the >=10x claim "
            "comparison, which needs the external baseline on this hardware."
        ),
        "configs": configs,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
