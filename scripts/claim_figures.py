"""Figure data for three measured claims: drift switches, the reset path, glue.

- ``switching``: per-phase wall time and compile events over a lifetime whose
  drift switches every phase, against naive task switching (each task value is
  a constant of its own jitted phase, so every new value compiles).
- ``reset``: per-step FLOPs and steps/s with and without the both-branch
  auto-reset, over every gymnax environment, unbatched and under ``vmap``.
- ``matrix``: which learner runs on which benchmark through the same runner
  call: a few updates across one task switch, health-checked. It shows that
  the pair runs, not that it learns.

Each subcommand prints one JSON report. Timings are microbenchmarks on one
machine: quote them with the backend and treat ratios as indicative.
"""

import argparse
import contextlib
import functools
import json
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp

SWITCHING = {
    # 100 distinct goals need a 10 x 10 grid.
    "ContinualGrid-v0": ({"size": 10}, [(i // 10, i % 10) for i in range(100)]),
    "mjx/Pendulum-v0": ({}, [(round(5.0 + 0.1 * i, 1),) for i in range(100)]),
}


def _rest_action(env: pp.Environment[Any]) -> jax.Array:
    space = env.action_space
    return jnp.asarray(np.zeros(cast(tuple[int, ...], space.shape), dtype=space.dtype))


def measure_switching(env_id: str, *, phases: int, ticks: int) -> dict[str, object]:
    """Per-phase seconds and compile events: drifting lifetime vs naive switching."""
    kwargs, values = SWITCHING[env_id]
    values = values[:phases]
    if len(values) < phases:
        raise ValueError(f"{env_id} has {len(values)} distinct task values; asked for {phases}.")
    env = pp.make(env_id, drift=pp.drift.Cyclic(values, ticks), **kwargs)
    action = _rest_action(env)
    lifetime_phase = jax.jit(
        lambda s, k: pp.measurement.transition_scan(env, s, action, k, ticks=ticks)
    )

    @functools.cache
    def naive_phase(value: Any) -> Any:
        task = pp.make(env_id, drift=pp.drift.Cyclic([value], ticks), **kwargs)
        return jax.jit(
            lambda k: pp.measurement.transition_scan(task, task.init(k)[0], action, k, ticks=ticks)
        )

    keys = jax.random.split(jax.random.key(0), phases)
    rows: dict[str, dict[str, list[float]]] = {}
    with pp.measurement.count_compilations() as counter:
        for name in ("lifetime", "naive"):
            # The lifetime's one init is charged to its first phase.
            seconds, compiles = [], []
            state: Any = None
            for phase in range(phases):
                before, started = counter.count, time.perf_counter()
                if name == "naive":
                    jax.block_until_ready(naive_phase(values[phase])(keys[phase]))
                else:
                    if phase == 0:
                        state = jax.jit(lambda k: env.init(k)[0])(keys[0])
                    state, _ = jax.block_until_ready(lifetime_phase(state, keys[phase]))
                seconds.append(time.perf_counter() - started)
                compiles.append(counter.count - before)
            rows[name] = {"seconds": seconds, "compile_events": compiles}
    return {
        "env": env_id,
        "phases": phases,
        "ticks_per_phase": ticks,
        **rows,
        "note": (
            "Both paths run the same compiled scan of env.step with a fixed zero "
            "action. The lifetime carries one state through every phase; the naive "
            "path builds a task with the phase's value as a compile-time constant "
            "and initializes it, so a value seen before would reuse its compiled "
            "program. Every value here is new. Compile events include JAX's "
            "internal primitive compilations."
        ),
    }


def measure_reset(
    env_id: str, *, ticks: int, limit: int, batch: int, repetitions: int
) -> dict[str, object]:
    """Steps/s (and unbatched FLOPs) for the lifetime vs both-branch auto-reset."""
    env = pp.make(env_id)
    action = _rest_action(env)
    scan = pp.measurement.transition_scan

    def run(limit_: int | None) -> Any:
        def one(key: jax.Array) -> Any:
            state = env.init(key)[0]
            return scan(env, state, action, key, ticks=ticks, autoreset_limit=limit_)

        return jax.jit(one if batch == 1 else jax.vmap(one))

    key = jax.random.key(0)
    keys = key if batch == 1 else jax.random.split(key, batch)
    result: dict[str, object] = {"env": env_id, "batch": batch, "ticks": ticks, "limit": limit}
    for name, fn in (("lifetime", run(None)), ("autoreset", run(limit))):
        jax.block_until_ready(fn(keys))
        timings = []
        for _ in range(repetitions):
            started = time.perf_counter()
            jax.block_until_ready(fn(keys))
            timings.append(time.perf_counter() - started)
        result[f"{name}_steps_per_second"] = ticks * batch / min(timings)
        result[f"{name}_steps_per_second_worst"] = ticks * batch / max(timings)
    if batch == 1:
        state = env.init(key)[0]
        flops = pp.measurement.xla_flops
        result["step_flops"] = flops(lambda s, k: env.step(s, action, k)[0], state, key)
        result["init_flops"] = flops(lambda k: env.init(k)[0], key)
        # The episode counter is traced: as a constant the reset is compiled away.
        result["autoreset_flops"] = flops(
            lambda s, k, t: pp.measurement.autoreset_step(env, s, action, k, limit=limit, ticks=t)[
                0
            ],
            state,
            key,
            jnp.int32(0),
        )
    return result


LEARNERS = ("PPO", "IPPO", "DQN", "IQL", "SAC")


def benchmarks(period: int) -> dict[str, tuple[Callable[[], Any], str]]:
    """Two-task training lifetimes (where a sequence exists) and their torso."""

    def coom() -> Any:
        from perpetua.adapters.coom import COOMLifetime, sequence_tasks

        envs = [COOMLifetime(scenario=s, task=t) for s, t in sequence_tasks("CO8")[:2]]
        return pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic([0.0, 1.0], period))

    return {
        "MEAL": (lambda: pp.presets.meal_sequence(sequence_length=2, period=period), "cnn"),
        "Continual World": (
            lambda: pp.presets.continual_world(pp.presets.CW10[:2], period=period),
            "mlp",
        ),
        "COOM": (coom, "cnn"),
        "Switching Atari": (
            lambda: pp.presets.switching_atari(
                ["Breakout-v5", "Asterix-v5"], num_envs=2, num_threads=1, period=period
            ),
            "cnn",
        ),
        "AgarCL": (lambda: pp.make("agarcl/agario-screen-v0", flatten=True), "mlp"),
        "MORPHEUS (mock)": (lambda: pp.presets.morpheus_regimes(period=period), "mlp"),
    }


def _runner(learner: str, env: Any, torso: Any, steps: int) -> Any:
    """The learner's runner for ``env``, or None when the action type does not fit."""
    from perpetua import agents

    space = env.action_space
    discrete = hasattr(space, "nvec")
    host = env.metadata.get("execution") == "host"
    if learner == "SAC":
        if discrete:
            return None
        config = agents.SACConfig(
            rollout_steps=steps, buffer_capacity=128, batch_size=8, gradient_steps=1
        )
        agent = agents.SAC(action_low=space.low[0], action_high=space.high[0], config=config)
        return (agents.HostSACRunner if host else agents.SACRunner)(env, agent)
    if not discrete:
        return None
    size = int(space.nvec[0])
    if learner in ("PPO", "IPPO"):
        ppo = agents.PPOConfig(
            torso=torso, rollout_steps=steps, epochs=1, minibatches=1, hidden_size=32
        )
        agent = (agents.PPO if learner == "PPO" else agents.IPPO)(
            action_size=size, config=ppo, observation_space=env.observation_space
        )
        return (agents.HostPPORunner if host else agents.PPORunner)(env, agent)
    dqn = agents.DQNConfig(
        torso=torso,
        rollout_steps=steps,
        buffer_capacity=128,
        batch_size=8,
        gradient_steps=1,
        hidden_size=32,
    )
    agent = (agents.DQN if learner == "DQN" else agents.IQL)(
        action_size=size, config=dqn, observation_space=env.observation_space
    )
    return (agents.HostDQNRunner if host else agents.DQNRunner)(env, agent)


def measure_matrix(*, updates: int, steps: int, only: list[str] | None) -> list[dict[str, object]]:
    """Run every learner on every benchmark for ``updates`` updates per task."""
    cells: list[dict[str, object]] = []
    for name, (factory, torso) in benchmarks(updates * steps).items():
        if only and name not in only:
            continue
        for learner in LEARNERS:
            cell: dict[str, object] = {"benchmark": name, "learner": learner}
            env = None
            try:
                started = time.perf_counter()
                env = factory()
                runner = _runner(learner, env, torso, steps)
                if runner is None:
                    cell |= {"status": "not applicable", "detail": "action type"}
                else:
                    state = runner.init(seed=0, window_size=steps)
                    for _ in range(2 * updates):
                        state = runner(state)
                    healthy = bool(state.healthy)
                    cell |= {
                        "status": "ran" if healthy else "failed",
                        "detail": "" if healthy else "non-finite state",
                        "runner": type(runner).__name__,
                        "ticks": 2 * updates * steps,
                        "seconds": time.perf_counter() - started,
                    }
            except Exception as error:  # noqa: BLE001 - the grid reports every failure
                cell |= {"status": "failed", "detail": f"{type(error).__name__}: {error}"[:300]}
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
            cells.append(cell)
            print(json.dumps(cell), file=sys.stderr, flush=True)
    return cells


def gymnax_ids() -> list[str]:
    import gymnax

    return [f"gymnax/{name}" for name in gymnax.registered_envs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    switching = commands.add_parser("switching")
    switching.add_argument("--env", choices=sorted(SWITCHING), default="ContinualGrid-v0")
    switching.add_argument("--phases", type=int, default=100)
    switching.add_argument("--ticks", type=int, default=64)
    reset = commands.add_parser("reset")
    reset.add_argument("--envs", nargs="*", default=None)
    reset.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    reset.add_argument("--ticks", type=int, default=2048)
    reset.add_argument("--limit", type=int, default=64)
    reset.add_argument("--repetitions", type=int, default=3)
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--updates", type=int, default=2, help="updates per task")
    matrix.add_argument("--steps", type=int, default=8, help="rollout steps per update")
    matrix.add_argument("--only", nargs="*", default=None, help="benchmark names")
    args = parser.parse_args()
    report: dict[str, object] = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
    }
    if args.command == "switching":
        report |= measure_switching(args.env, phases=args.phases, ticks=args.ticks)
    elif args.command == "matrix":
        # Benchmarks such as MEAL print to stdout; keep stdout pure JSON.
        with contextlib.redirect_stdout(sys.stderr):
            cells = measure_matrix(updates=args.updates, steps=args.steps, only=args.only)
        report |= {"updates_per_task": args.updates, "rollout_steps": args.steps, "cells": cells}
    else:
        rows: list[dict[str, object]] = []
        for env_id in args.envs or gymnax_ids():
            for batch in args.batches:
                try:
                    rows.append(
                        measure_reset(
                            env_id,
                            ticks=args.ticks,
                            limit=args.limit,
                            batch=batch,
                            repetitions=args.repetitions,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - the sweep reports every failure
                    rows.append({"env": env_id, "batch": batch, "error": repr(error)[:200]})
        report["rows"] = rows
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
