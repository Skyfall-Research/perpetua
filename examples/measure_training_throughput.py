"""Measure lifetime training throughput for the framework's own PPO runners.

Claim-1 comparison harness for the throughput hypothesis (docs/measurement.md):

- ``host``: CartPole through the CPU bridge — the same task as the
  AsyncVectorEnv + PyTorch PPO baseline. ``--bridge eager`` is the host lift
  with ``HostPPORunner``; ``--bridge compiled`` puts the simulator in an
  ordered host callback so the compiled ``PPORunner`` drives it.
  ``--envpool N`` uses EnvPool's batched C++ CartPole with N slots instead of
  one Gymnasium env, and ``--overlap`` steps half the slots while the learner
  acts for the other half (non-deterministic, opt-in).
- ``native``: the compiled native ``PPORunner`` on a JAX environment, vmapped
  over ``--batch`` lifetimes of identical shape. Different dynamics from the
  baseline task; reported separately and labeled as such.

Each measurement warms up (compiling the runner), then runs updates for a
fixed wall-clock duration, synchronizing at every update boundary, and
reports environment transitions per second with IQM/mean/median intervals
from ``perpetua.reporting``. Native runs use the framework's deterministic
compiler options, which may cost throughput.
"""

import argparse
import json
import time
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import PPO, HostPPORunner, PPOConfig, PPORunner


def _config() -> PPOConfig:
    return PPOConfig(rollout_steps=128, epochs=4, minibatches=4)


def _intervals(values: list[float]) -> dict[str, float]:
    scores = jnp.asarray(values, dtype=jnp.float32)[:, None]
    report = pp.reporting.report(scores, key=jax.random.key(0))
    return {
        "iqm": float(report["iqm"].point),
        "iqm_low": float(report["iqm"].low),
        "iqm_high": float(report["iqm"].high),
        "mean": float(report["mean"].point),
        "mean_low": float(report["mean"].low),
        "mean_high": float(report["mean"].high),
    }


def _host_env(bridge: str, envpool: int, overlap: bool, threads: int) -> Any:
    compiled = bridge == "compiled"
    if envpool:
        return pp.make(
            "envpool/CartPole-v1",
            num_envs=envpool,
            num_threads=threads,
            compiled=compiled,
            overlap=overlap,
        )
    if overlap:
        raise ValueError("--overlap needs --envpool.")
    return GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=500), compiled=compiled)


def measure_host(
    *,
    duration: float,
    seeds: int,
    bridge: str = "eager",
    envpool: int = 0,
    overlap: bool = False,
    threads: int = 8,
    total_steps: int = 0,
) -> dict[str, object]:
    """CartPole through the CPU bridge with the same PPO shape.

    Host simulators are single-use, so each seed gets a fresh lifetime and
    runner. Steps count simulator transitions: every slot per tick, or half the
    slots per tick with ``overlap``.
    """
    config = _config()
    per_seed: list[float] = []
    returns: list[float] = []
    walls: list[float] = []
    slots = envpool or 1
    per_tick = slots // 2 if overlap else slots
    for seed_index in range(seeds):
        env = _host_env(bridge, envpool, overlap, threads)
        runner_type = PPORunner if bridge == "compiled" else HostPPORunner
        trainer = runner_type(env, PPO(action_size=2, config=config))
        state = trainer.init(seed=seed_index, window_size=128)
        state = jax.block_until_ready(trainer(state))
        updates = 0
        started = time.perf_counter()
        while _running(started, duration, updates * config.rollout_steps * per_tick, total_steps):
            state = jax.block_until_ready(trainer(state))
            updates += 1
        seconds = time.perf_counter() - started
        walls.append(seconds)
        per_seed.append(updates * config.rollout_steps * per_tick / seconds)
        returns.append(float(jnp.asarray(state.lifetime.metrics.mean).mean()))
        env.close()
    return {
        "wall_seconds_per_seed": walls,
        "total_steps_target": total_steps or None,
        "stack": f"lifetime-ppo-host-{bridge}"
        + (f"-envpool{envpool}" if envpool else "")
        + ("-overlap" if overlap else ""),
        "env": "CartPole-v1",
        "batch": slots,
        "rollout_steps": config.rollout_steps,
        "seeds": seeds,
        "duration_seconds": duration,
        "steps_per_second_per_seed": per_seed,
        "best_steps_per_second": max(per_seed),
        **_intervals(per_seed),
        "lifetime_mean_reward_per_tick_last_seed": returns[-1],
    }


def _native_env(env_id: str, drift_name: str) -> tuple[pp.Environment[Any], int]:
    """Return a native environment and its discrete action count."""
    if drift_name == "cyclic" and env_id == "gymnax/CartPole-v1":
        drift: pp.drift.Drift = pp.drift.Cyclic([(9.8,), (4.9,)], period=256)
    elif drift_name == "cyclic" and env_id == "ContinualGrid-v0":
        drift = pp.drift.Cyclic([(7, 7), (0, 0)], period=64)
    else:
        drift = pp.drift.Stationary()
    if env_id == "gymnax/CartPole-v1":
        return pp.make(env_id, drift=drift), 2
    if env_id == "ContinualGrid-v0":
        return pp.make(env_id, size=8, drift=drift), 5
    raise ValueError(f"Unsupported native environment: {env_id}")


def _running(started: float, duration: float, steps: int, total_steps: int) -> bool:
    """Keep going for ``duration`` seconds, or until ``total_steps`` if one is set."""
    if total_steps:
        return steps < total_steps
    return time.perf_counter() - started < duration


def measure_native(
    *,
    env_id: str,
    drift_name: str,
    batch: int,
    duration: float,
    seeds: int,
    total_steps: int = 0,
) -> dict[str, object]:
    """``batch`` vmapped native lifetimes with the compiled PPORunner."""
    config = _config()
    env, action_size = _native_env(env_id, drift_name)
    trainer = PPORunner(env, PPO(action_size=action_size, config=config))

    def initialize(seed: jax.Array) -> Any:
        return trainer.init(seed=seed, window_size=128)

    init_batch = jax.jit(jax.vmap(initialize))
    # The runner's internal jit owns the deterministic compiler options, so
    # batching uses vmap over it directly rather than nesting another jit.
    run_batch = jax.vmap(trainer)
    seeds_batch = jnp.arange(batch, dtype=jnp.int32)
    state = jax.block_until_ready(run_batch(init_batch(seeds_batch)))
    per_seed: list[float] = []
    returns: list[float] = []
    walls: list[float] = []
    for seed_index in range(seeds):
        state = init_batch(seeds_batch + jnp.int32(seed_index * batch))
        state = jax.block_until_ready(run_batch(state))
        updates = 0
        started = time.perf_counter()
        while _running(started, duration, updates * config.rollout_steps * batch, total_steps):
            state = jax.block_until_ready(run_batch(state))
            updates += 1
        seconds = time.perf_counter() - started
        walls.append(seconds)
        per_seed.append(updates * config.rollout_steps * batch / seconds)
        returns.append(float(jnp.asarray(state.lifetime.metrics.mean).mean()))
    return {
        "stack": "lifetime-ppo-native-vmap",
        "wall_seconds_per_seed": walls,
        "total_steps_target": total_steps or None,
        "env": env_id,
        "drift": drift_name,
        "batch": batch,
        "rollout_steps": config.rollout_steps,
        "seeds": seeds,
        "duration_seconds": duration,
        "steps_per_second_per_seed": per_seed,
        "best_steps_per_second": max(per_seed),
        **_intervals(per_seed),
        "lifetime_mean_reward_per_tick_last_seed": returns[-1],
        "note": (
            "Compiled with deterministic GPU options, which may cost "
            "throughput. With --env gymnax/CartPole-v1 the dynamics match the "
            "CartPole baseline task; the lifetime never resets while the "
            "baseline resets per episode."
        ),
    }


def main() -> None:
    """Run the requested lifetime measurements and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["host", "native"], required=True)
    parser.add_argument("--env", default="gymnax/CartPole-v1")
    parser.add_argument("--drift", choices=["stationary", "cyclic"], default="stationary")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--bridge", choices=["eager", "compiled"], default="eager")
    parser.add_argument("--envpool", type=int, default=0)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="run to this many environment steps instead of for --duration seconds",
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU measurement requested, but JAX is not using a GPU.")
    if args.backend == "host" and args.batch != 1:
        raise ValueError("The host backend measures a single CartPole lifetime.")
    measure = (
        (
            lambda **kwargs: measure_host(
                bridge=str(args.bridge),
                envpool=int(args.envpool),
                overlap=bool(args.overlap),
                threads=int(args.threads),
                total_steps=int(args.total_steps),
                **kwargs,
            )
        )
        if args.backend == "host"
        else lambda **kwargs: measure_native(
            env_id=str(args.env),
            drift_name=str(args.drift),
            batch=int(args.batch),
            total_steps=int(args.total_steps),
            **kwargs,
        )
    )
    report = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
        "measurement": measure(duration=float(args.duration), seeds=int(args.seeds)),
        "note": (
            "Fixed-duration wall-clock measurement synchronizing at each "
            "update boundary; counts environment transitions. Not a learning "
            "quality benchmark."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
