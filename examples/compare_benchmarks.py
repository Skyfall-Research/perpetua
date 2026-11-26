"""Compare continual benchmarks with the same lifetime metrics.

For each benchmark, train with the same learner configuration per action type,
save the policy at each task boundary, and evaluate every saved policy on
identical fresh lifetimes built with the training seeds. Reports the final
policy's mean reward and stratified-bootstrap IQM over seeds x tasks,
forgetting (``pp.evaluation.forgetting`` per seed: the best score on each task
from policies saved after learning it, minus the final policy's score), and
ticks to recover after each declared switch from the training reward trace.
AgarCL has no task sequence and reports neither; MORPHEUS uses the paper's regimes as tasks.
Sequence benchmarks also report the paper's six-metric protocol on the training trace.

Small smoke example:
    uv run python examples/compare_benchmarks.py --benchmark meal \
      --tasks 2 --updates-per-task 1 --rollout-steps 4 --eval-ticks 8 --seeds 0

Full measurement options are explicit so no above-random claim is implied.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp
from perpetua.agents.encoders import Torso

warnings.filterwarnings("ignore")


@dataclass
class Spec:
    make_training: Callable[[int, int], Any]
    # Evaluation factories must rebuild exactly the training tasks (same seeds).
    make_tasks: Callable[[int], list[Callable[[], Any]]]
    learner: str
    torso: Torso = "mlp"


def make_spec(name: str) -> Spec:
    if name == "meal":

        def tasks(count: int) -> list[Callable[[], Any]]:
            import meal

            if count == 1:
                return [lambda: pp.make("meal/overcooked-v0", seed=0)]
            backends = meal.make_sequence(
                sequence_length=count, num_agents=2, difficulty="medium", seed=0
            )
            return [
                lambda backend=backend: pp.make("meal/overcooked-v0", backend=backend, seed=0)
                for backend in backends
            ]

        return Spec(
            lambda period, count: (
                pp.presets.meal_sequence(sequence_length=count, seed=0, period=period)
                if count > 1
                else pp.make("meal/overcooked-v0", seed=0)
            ),
            tasks,
            "ppo",
            "cnn",
        )
    if name == "continual_world":
        names = pp.presets.CW10
        return Spec(
            lambda period, count: (
                pp.presets.continual_world(names[:count], period=period)
                if count > 1
                else pp.make("metaworld/MT1-v0", env_name=names[0])
            ),
            # continual_world seeds task i with seed + i (seed=0 here).
            lambda count: [
                lambda name=name, index=index: pp.make(
                    "metaworld/MT1-v0", env_name=name, seed=index
                )
                for index, name in enumerate(names[:count])
            ],
            "sac",
        )
    if name == "coom":

        def tasks(count: int) -> list[Callable[[], Any]]:
            from perpetua.adapters.coom import sequence_tasks

            pairs = sequence_tasks("CO8")[:count]
            return [
                lambda scenario=scenario, task=task: pp.make(
                    "coom/COOM-v0", scenario=scenario, task=task
                )
                for scenario, task in pairs
            ]

        return Spec(
            lambda period, count: (
                pp.presets.coom_sequence("CO8", period=period)
                if count == 8
                else _coom_subset(period, count)
            ),
            tasks,
            "ppo",
            "cnn",
        )
    if name == "atari":
        games = ["Breakout-v5", "Asterix-v5"]
        return Spec(
            lambda period, count: (
                pp.presets.switching_atari(games[:count], num_envs=2, num_threads=1, period=period)
                if count > 1
                else pp.make("envpool/Atari-v0", task=games[0], num_envs=2, num_threads=1)
            ),
            lambda count: [
                lambda game=game: pp.make("envpool/Atari-v0", task=game, num_envs=2, num_threads=1)
                for game in games[:count]
            ],
            "ppo",
            "cnn",
        )
    if name == "agarcl":
        return Spec(
            lambda period, count: pp.make("agarcl/agario-screen-v0", flatten=True),
            lambda count: [
                lambda: pp.make("agarcl/agario-screen-v0", flatten=True) for _ in range(count)
            ],
            "sac",
        )
    if name == "morpheus":
        from perpetua.adapters.morpheus import PAPER_REGIMES

        def regime_world(schedule: list[float], period: int, count: int) -> Any:
            drift = {"regime": pp.drift.Cyclic(schedule, period)}
            return pp.make("morpheus/Morpheus-v0", regimes=PAPER_REGIMES[:count], drift=drift)

        # The paper's regimes in order; evaluation holds one regime fixed per lifetime.
        return Spec(
            lambda period, count: regime_world([float(i) for i in range(count)], period, count),
            lambda count: [
                lambda index=index: regime_world([float(index)], 1, count) for index in range(count)
            ],
            "ppo",
        )
    raise ValueError(f"Unknown benchmark {name!r}.")


def _coom_subset(period: int, count: int) -> Any:
    from perpetua.adapters.coom import COOMLifetime, sequence_tasks

    envs = [
        COOMLifetime(scenario=scenario, task=task)
        for scenario, task in sequence_tasks("CO8")[:count]
    ]
    if count == 1:
        return envs[0]
    return pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic(list(map(float, range(count))), period))


def learner_for(spec: Spec, env: Any, rollout_steps: int) -> tuple[Any, Any]:
    if spec.learner == "sac":
        from perpetua.agents import SAC, SACConfig

        space = env.action_space
        config = SACConfig(
            rollout_steps=rollout_steps, buffer_capacity=128, batch_size=8, gradient_steps=1
        )
        return SAC(action_low=space.low[0], action_high=space.high[0], config=config), config
    from perpetua.agents import PPO, PPOConfig

    config = PPOConfig(
        torso=spec.torso, rollout_steps=rollout_steps, epochs=1, minibatches=1, hidden_size=32
    )
    agent = PPO(
        action_size=int(env.action_space.nvec[0]),
        config=config,
        observation_space=env.observation_space,
    )
    return agent, config


def train(
    benchmark: str, *, tasks: int, updates_per_task: int, rollout_steps: int
) -> tuple[Any, list[Any], np.ndarray, list[Callable[[], Any]], Spec]:
    """Train through the sequence, saving the policy at the end of each task.

    Returns the agent, the saved policy states, the per-tick training reward
    trace (mean over slots) and fresh-lifetime factories for every task.
    """
    spec = make_spec(benchmark)
    period = updates_per_task * rollout_steps
    env = spec.make_training(period, tasks)
    agent, _ = learner_for(spec, env, rollout_steps)
    runner: Any
    if spec.learner == "sac":
        from perpetua.agents import HostSACRunner

        runner = HostSACRunner(env, agent)
    elif env.metadata.get("execution") == "host":
        from perpetua.agents import HostPPORunner

        runner = HostPPORunner(env, agent)
    else:
        from perpetua.agents import PPORunner

        runner = PPORunner(env, agent)
    try:
        state: Any = runner.init(seed=0, window_size=rollout_steps)
        saved: list[Any] = []
        trace: list[np.ndarray] = []
        for _ in range(tasks):
            for _ in range(updates_per_task):
                state = runner(state)
                recent = pp.evaluation.recent_rewards(state.lifetime.metrics, steps=rollout_steps)
                trace.append(np.asarray(recent).mean(axis=1))
            saved.append(state.agent)
        assert bool(state.healthy), f"Unhealthy training on {benchmark}."
        return agent, saved, np.concatenate(trace), spec.make_tasks(tasks), spec
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def evaluate_native(
    task: Callable[[], Any], agent: Any, states: list[Any], seeds: list[int], ticks: int
) -> jax.Array:
    env = task()
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)

    def act(state: Any, obs: jax.Array, info: Any, key: jax.Array) -> jax.Array:
        return agent.act(state, obs, info, key).action

    return pp.evaluation.revisit(
        env,
        act,
        stacked,
        seeds=jnp.asarray(seeds, dtype=jnp.int32),
        ticks=ticks,
    ).T  # (seeds, policies)


def evaluate_host(
    task: Callable[[], Any], agent: Any, states: list[Any], seeds: list[int], ticks: int
) -> np.ndarray:
    columns = []
    for state in states:

        def act(obs: jax.Array, info: Any, key: jax.Array, state: Any = state) -> jax.Array:
            return agent.act(state, obs, info, key).action

        per_seed = []
        for seed in seeds:
            env = task()
            try:
                runner = pp.runtime.HostRunner(env, act, ticks=ticks)
                carry = runner(pp.runtime.init_lifetime(env, seed=seed, window_size=ticks))
                per_seed.append(float(np.mean(np.asarray(carry.metrics.mean))))
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
        columns.append(per_seed)
    return np.asarray(columns).T  # (seeds, policies)


def evaluate_success(
    task: Callable[[], Any], agent: Any, states: list[Any], seeds: list[int], ticks: int
) -> np.ndarray:
    """Fraction of evaluation ticks on which Meta-World reports task success.

    Read through the Oracle (``success`` is privileged), on the same fresh
    lifetimes as the reward evaluation.
    """
    columns = []
    for state in states:
        policy = jax.jit(
            lambda obs, info, key, state=state: agent.act(state, obs, info, key).action
        )
        per_seed = []
        for seed in seeds:
            inner = task()
            env = pp.Oracle(inner, expose=["success"])
            try:
                lifetime = pp.runtime.init_lifetime(env, seed=seed, window_size=ticks)
                total = 0.0
                for _ in range(ticks):
                    lifetime = pp.runtime.tick(env, policy, lifetime)
                    total += float(cast(Any, lifetime.info)["oracle"]["success"][0])
                per_seed.append(total / ticks)
            finally:
                inner.close()
        columns.append(per_seed)
    return np.asarray(columns).T  # (seeds, policies)


def switch_recovery(trace: np.ndarray, tasks: int, period: int) -> list[int | None]:
    """Ticks to recover after each declared switch, judged inside the new task.

    Switches are flagged on ticks ``k * period`` (trace index ``k * period - 1``).
    The final tick's wrap-around switch back to task 0 is excluded. Each switch
    is scored on the previous and new task's segment only; ``None`` means the
    windowed mean did not recover within the new task.
    """
    window = max(1, period // 4)
    recoveries: list[int | None] = []
    for k in range(1, tasks):
        segment = trace[(k - 1) * period : (k + 1) * period]
        change = period - 1
        ticks = int(
            pp.evaluation.recovery_ticks(
                jnp.asarray(segment), jnp.asarray([change]), window=window
            )[0]
        )
        recoveries.append(None if ticks >= len(segment) - change else ticks)
    return recoveries


def protocol_summary(
    benchmark: str, trace: np.ndarray, tasks: int, period: int
) -> dict[str, Any] | None:
    """The MORPHEUS paper's six metrics on the training trace (``pp.protocol``).

    The transition from tick ``i`` runs under the phase active at ``i``, so trace index ``i``
    is labelled with task ``i // period``. MORPHEUS labels its revisit with the regime it repeats.
    """
    if tasks < 2:
        return None
    indices = [(index // period) % tasks for index in range(len(trace))]
    if benchmark == "morpheus":
        from perpetua.adapters.morpheus import PAPER_REGIMES, regime_labels

        labels, origins = regime_labels(PAPER_REGIMES, indices)
    else:
        labels, origins = [f"task{index}" for index in indices], None
    result = pp.protocol.report(
        pp.protocol.ProtocolTrace.from_arrays(trace.tolist(), labels, origins=origins)
    )
    return {
        "per_regime": {
            name: {
                key: entry[key]
                for key in (
                    "per_configuration_reward",
                    "within_trace_forgetting",
                    "tail_mean_reward",
                )
            }
            | {"stability": entry["stability"]["pooled"]}
            for name, entry in result["per_regime"].items()
        },
        "shifts": [
            {key: shift[key] for key in ("to", "adaptation_speed", "recovery_time")}
            for shift in result["shifts"]
        ],
    }


def report(benchmark: str, args: argparse.Namespace) -> dict[str, Any]:
    if benchmark == "agarcl" and args.tasks != 1:
        raise ValueError("agarcl has no task sequence; use --tasks 1.")
    if benchmark == "morpheus" and not 1 <= args.tasks <= 6:
        raise ValueError("morpheus has six paper regimes; use --tasks 1..6.")
    agent, saved, trace, task_factories, spec = train(
        benchmark,
        tasks=args.tasks,
        updates_per_task=args.updates_per_task,
        rollout_steps=args.rollout_steps,
    )
    evaluate = evaluate_native if benchmark == "meal" else evaluate_host
    per_task = [
        evaluate(task, agent, saved, args.seeds, args.eval_ticks) for task in task_factories
    ]
    # scores[seed, policy, task]: policy k was saved at the end of task k.
    scores = np.stack(per_task, axis=-1)
    final = scores[:, -1, :]
    forgetting = (
        [float(pp.evaluation.forgetting(matrix)) for matrix in scores] if args.tasks > 1 else None
    )
    summary = pp.reporting.report(jnp.asarray(final, dtype=jnp.float32), key=jax.random.key(0))
    period = args.updates_per_task * args.rollout_steps
    success = None
    if benchmark == "continual_world":
        # Continual World's metrics are defined on task success, not reward:
        # average performance is the final policy's success over tasks, and
        # forgetting uses the same definition as for reward.
        rates = np.stack(
            [
                evaluate_success(task, agent, saved, args.seeds, args.eval_ticks)
                for task in task_factories
            ],
            axis=-1,
        )
        success = {
            "definition": "fraction of evaluation ticks with Meta-World info['success']",
            "average_performance": float(np.mean(rates[:, -1, :])),
            "forgetting": (
                float(np.mean([float(pp.evaluation.forgetting(m)) for m in rates]))
                if args.tasks > 1
                else None
            ),
            "forward_transfer": None,  # needs single-task reference runs; not measured
            "rates": rates.tolist(),
        }
    return {
        "success": success,
        "benchmark": benchmark,
        "provenance": pp.reporting.provenance(),
        "learner": spec.learner,
        "tasks": args.tasks,
        "updates_per_task": args.updates_per_task,
        "rollout_steps": args.rollout_steps,
        "eval_ticks": args.eval_ticks,
        "seeds": args.seeds,
        "mean_reward": float(np.mean(final)),
        "iqm": float(summary["iqm"].point),
        "iqm_low": float(summary["iqm"].low),
        "iqm_high": float(summary["iqm"].high),
        "forgetting": None if forgetting is None else float(np.mean(forgetting)),
        "forgetting_per_seed": forgetting,
        "training_mean_reward": float(np.mean(trace)),
        "recovery_window": max(1, period // 4),
        "recovery_ticks": switch_recovery(trace, args.tasks, period) if args.tasks > 1 else None,
        "protocol": protocol_summary(benchmark, trace, args.tasks, period),
        "scores": scores.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["meal", "continual_world", "coom", "atari", "agarcl", "morpheus"],
        required=True,
    )
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--updates-per-task", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--eval-ticks", type=int, default=64)
    parser.add_argument(
        "--seeds", type=lambda value: [int(x) for x in value.split(",")], default=[0, 1]
    )
    args = parser.parse_args()
    if args.tasks < 1 or args.updates_per_task < 1 or args.rollout_steps < 1 or args.eval_ticks < 1:
        parser.error("tasks, updates, rollout steps and evaluation ticks must be positive.")
    # Benchmarks such as MEAL print progress to stdout; keep stdout pure JSON.
    with contextlib.redirect_stdout(sys.stderr):
        result = report(args.benchmark, args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
