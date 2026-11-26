"""Measure the speed and cost claims locally (compile, reset, glue).

- **B1 — no recompilation across drift**: compile events and stall time over a
  >=100-phase drift lifetime, plus a naive-JAX baseline (a new task constant
  per phase).
- **B2 — removing the reset path saves compute**: per-step FLOPs (XLA static
  estimate) and steps/s for the lifetime transition versus the gymnax-style
  both-branch auto-reset on identical dynamics.
- **B3 — integration cost**: glue-code line counts for the shipped adapters
  and examples. The external baselines (Continual World, COOM, AgarCL) are
  not cloned here; that comparison is a separate, larger measurement.

Timings are transition-function microbenchmarks with warm-up and device
synchronization; they are not the end-to-end throughput claim, and no speed
claim should be made from a single run.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp


def measure_compiles(*, phases: int, ticks_per_phase: int, chunk: int) -> dict[str, object]:
    """Compile events and wall time over a drifting lifetime vs naive switching."""
    if phases < 1 or ticks_per_phase < 1 or chunk < 1:
        raise ValueError("phases, ticks_per_phase and chunk must be positive.")
    env = pp.make(
        "ContinualGrid-v0", size=4, drift=pp.drift.Cyclic([(3, 3), (0, 0)], ticks_per_phase)
    )
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=chunk)
    carry = pp.runtime.init_lifetime(env, seed=0, window_size=chunk)
    with pp.measurement.count_compilations() as cold:
        started = time.perf_counter()
        carry = jax.block_until_ready(runner(carry))
        cold_seconds = time.perf_counter() - started
    total_ticks = phases * ticks_per_phase
    calls = max(total_ticks // chunk, 1)
    with pp.measurement.count_compilations() as warm:
        started = time.perf_counter()
        for _ in range(calls):
            carry = jax.block_until_ready(runner(carry))
        warm_seconds = time.perf_counter() - started
    lifetime = {
        "cold_compile_events": cold.count,
        "cold_seconds": cold_seconds,
        "warm_compile_events": warm.count,
        "warm_seconds": warm_seconds,
        "warm_chunks": calls,
        "ticks": int(carry.state.t.to_int()),
        "drift_elapsed": int(carry.state.drift["goal"].elapsed),
    }
    naive = _naive_switching(phases=phases, ticks=8)
    return {
        "phases": phases,
        "ticks_per_phase": ticks_per_phase,
        "lifetime": lifetime,
        "naive_new_constant_per_task": naive,
        "note": (
            "Lifetime drift is data: the same compiled runner covers every "
            "phase. The naive baseline builds one jitted transition per task "
            "constant, mirroring 'a new task with a new shape or constant "
            "triggers re-compilation'."
        ),
    }


def _naive_task_step(goal: tuple[int, int]) -> Any:
    """A freshly jitted transition with the task constant baked in."""
    goal_array = jnp.asarray(goal, dtype=jnp.int32)
    moves = jnp.asarray([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)

    @jax.jit
    def step(position: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_position = jnp.clip(position + moves[action], 0, 3)
        reached = jnp.all(next_position == goal_array)
        return next_position, reached

    return step


def _naive_switching(*, phases: int, ticks: int) -> dict[str, object]:
    """Time and count compiles for the naive new-constant-per-phase baseline."""
    with pp.measurement.count_compilations() as counter:
        started = time.perf_counter()
        for phase in range(phases):
            step = _naive_task_step((phase % 4, (phase // 4) % 4))
            position = jnp.asarray([0, 0], dtype=jnp.int32)
            reached = jnp.asarray(False)
            for tick in range(ticks):
                position, reached = step(position, jnp.int32(tick % 5))
            jax.block_until_ready((position, reached))
        seconds = time.perf_counter() - started
    return {
        "phases": phases,
        "ticks_per_phase": ticks,
        "compile_events": counter.count,
        "seconds": seconds,
    }


def _rest_action(env: pp.Environment[Any]) -> jax.Array:
    """A fixed zero action matching the environment's action space."""
    space = env.action_space
    shape = cast(tuple[int, ...], space.shape)
    return jnp.asarray(np.zeros(shape, dtype=space.dtype))


def measure_reset_path(
    *, env_id: str, ticks: int, limit: int, repetitions: int = 3
) -> dict[str, object]:
    """Per-step FLOPs and steps/s for lifetime vs both-branch autoreset."""
    if ticks < 1 or limit < 1 or repetitions < 1:
        raise ValueError("ticks, limit and repetitions must be positive.")
    env = pp.make(env_id)
    state = env.init(jax.random.key(0))[0]
    action = _rest_action(env)
    step_flops = pp.measurement.xla_flops(
        lambda s, k: env.step(s, action, k)[0], state, jax.random.key(1)
    )
    init_flops = pp.measurement.xla_flops(lambda k: env.init(k)[0], jax.random.key(1))
    # The episode counter is a traced argument: as a constant, the compiler
    # proves the limit is never reached and drops the reset from the estimate.
    autoreset_flops = pp.measurement.xla_flops(
        lambda s, k, t: pp.measurement.autoreset_step(env, s, action, k, limit=limit, ticks=t)[0],
        state,
        jax.random.key(1),
        jnp.int32(0),
    )
    lifetime = jax.jit(lambda s, k: pp.measurement.transition_scan(env, s, action, k, ticks=ticks))
    autoreset = jax.jit(
        lambda s, k: pp.measurement.transition_scan(
            env, s, action, k, ticks=ticks, autoreset_limit=limit
        )
    )

    def init_scan(key: jax.Array) -> jax.Array:
        def body(carry: jax.Array, _: None) -> tuple[jax.Array, Any]:
            carry, reset_key = jax.random.split(carry)
            fresh = env.init(reset_key)[0]
            return carry, fresh

        return jax.lax.scan(body, key, None, length=ticks)[1]

    reset = jax.jit(init_scan)
    jax.block_until_ready(lifetime(state, jax.random.key(2)))
    jax.block_until_ready(autoreset(state, jax.random.key(2)))
    jax.block_until_ready(reset(jax.random.key(2)))

    def timed(fn: Any, *args: Any) -> list[float]:
        timings = []
        for _ in range(repetitions):
            started = time.perf_counter()
            jax.block_until_ready(fn(*args))
            timings.append(time.perf_counter() - started)
        return timings

    lifetime_timings = timed(lifetime, state, jax.random.key(3))
    autoreset_timings = timed(autoreset, state, jax.random.key(3))
    reset_timings = timed(reset, jax.random.key(3))
    lifetime_seconds = min(lifetime_timings)
    autoreset_seconds = min(autoreset_timings)
    reset_seconds = min(reset_timings)
    return {
        "env": env_id,
        "ticks": ticks,
        "autoreset_limit": limit,
        "step_flops": step_flops,
        "init_flops": init_flops,
        "autoreset_step_flops": autoreset_flops,
        "reset_path_flops_overhead": (
            autoreset_flops / step_flops - 1.0 if step_flops > 0 else None
        ),
        "lifetime_seconds": lifetime_seconds,
        "autoreset_seconds": autoreset_seconds,
        "reset_seconds": reset_seconds,
        "lifetime_seconds_worst": max(lifetime_timings),
        "autoreset_seconds_worst": max(autoreset_timings),
        "reset_seconds_worst": max(reset_timings),
        "lifetime_steps_per_second": ticks / lifetime_seconds,
        "autoreset_steps_per_second": ticks / autoreset_seconds,
        "reset_ticks_per_second": ticks / reset_seconds,
        "repetitions": repetitions,
        "note": (
            "Transition-function microbenchmark with a fixed zero action and "
            "warm-up; policy, metrics and host orchestration are excluded and "
            "identical in both paths. FLOPs are XLA's static cost estimate and "
            "can miss allocation/sampling work the wall clock includes; "
            "'reset' times a compiled loop that materializes one env.init per "
            "tick, which is what the autoreset baseline recomputes every step. "
            "Seconds are best-of-N with the worst case reported; treat ratios "
            "as indicative, not precise."
        ),
    }


def measure_glue() -> dict[str, object]:
    """Glue-code line counts for shipped adapters and examples."""
    root = Path(__file__).resolve().parents[1]
    groups: dict[str, dict[str, dict[str, int]]] = {}
    for name, pattern in (
        ("adapters", root / "src" / "perpetua" / "adapters"),
        ("examples", root / "examples"),
    ):
        files: dict[str, dict[str, int]] = {}
        for path in sorted(pattern.glob("*.py")):
            files[path.name] = pp.measurement.code_lines(path)._asdict()
        groups[name] = files
    totals = {
        name: {
            "files": len(files),
            "code": sum(counts["code"] for counts in files.values()),
        }
        for name, files in groups.items()
    }
    return {
        "groups": groups,
        "totals": totals,
        "note": (
            "Counts cover this repository's shipped integration code. The "
            "external baselines (Continual World, COOM, AgarCL) are not "
            "cloned here; comparing against them is a separate measurement "
            "(scripts/measure_integration_cost.py)."
        ),
    }


def main() -> None:
    """Run the local measurement set and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", type=int, default=100)
    parser.add_argument("--ticks-per-phase", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--b2-ticks", type=int, default=4096)
    parser.add_argument("--autoreset-limit", type=int, default=64)
    args = parser.parse_args()
    report = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
        "b1_compile_events": measure_compiles(
            phases=int(args.phases),
            ticks_per_phase=int(args.ticks_per_phase),
            chunk=int(args.chunk),
        ),
        "b2_reset_path": measure_reset_path(
            env_id="ContinualGrid-v0",
            ticks=int(args.b2_ticks),
            limit=int(args.autoreset_limit),
        ),
        "b2_reset_path_pendulum": measure_reset_path(
            env_id="gymnax/Pendulum-v1",
            ticks=int(args.b2_ticks),
            limit=int(args.autoreset_limit),
        ),
        "b3_glue": measure_glue(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
