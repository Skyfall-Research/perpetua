"""Train PPO through COOM's CO8 sequence as one lifetime and log it as JSON lines.

A real-budget companion to ``examples/compare_benchmarks.py`` for one
benchmark: 200k ticks per task by default, through the compiled CPU bridge (the
simulator step is a host callback inside the compiled runner). The log holds
one ``run`` header, an ``eval`` record every ``--eval-every`` ticks with the
training reward, and a ``boundary_eval`` record per task at every task
boundary: the saved policy's reward per tick over ``--eval-ticks`` ticks of a
fresh lifetime of that task. The untrained policy is scored the same way at
tick 0, which gives each task a floor to rescale against.

The adapter exposes reward only, so these are not the success scores of the
COOM paper, and the learner is Perpetua's PPO, not a published agent.

    uv run python scripts/setup_coom.py
    uv run --extra training python scripts/coom_lifetime.py \\
        --seed 0 --out results/co8-ppo-seed0.jsonl
    uv run --with matplotlib python scripts/plot_claim_figures.py reward-lifetime \\
        results/co8-ppo-seed*.jsonl --out lifetime-coom.png
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import jax

import perpetua as pp
from perpetua import agents
from perpetua.adapters.coom import COOMLifetime, sequence_tasks


def _emit(out: Any, record: dict[str, Any]) -> None:
    out.write(json.dumps(record) + "\n")
    out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", type=int, default=8, help="leading tasks of CO8")
    parser.add_argument("--steps-per-task", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-ticks", type=int, default=4_000)
    parser.add_argument("--window", type=int, default=1_000)
    args = parser.parse_args()
    if args.steps_per_task % args.rollout_steps or args.eval_every % args.rollout_steps:
        raise ValueError(
            f"--steps-per-task and --eval-every must be multiples of {args.rollout_steps}."
        )

    pairs = sequence_tasks("CO8")[: args.tasks]
    envs = [COOMLifetime(scenario=s, task=t, compiled=True) for s, t in pairs]
    codes = [float(index) for index in range(len(envs))]
    env = pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic(codes, args.steps_per_task))
    config = agents.PPOConfig(torso="cnn", rollout_steps=args.rollout_steps)
    agent = agents.PPO(action_size=int(cast(Any, env.action_space).nvec[0]), config=config)
    runner = agents.PPORunner(env, agent)
    started = time.perf_counter()

    def boundary(out: Any, agent_state: Any, tick: int) -> None:
        def act(obs: jax.Array, info: Any, key: jax.Array) -> jax.Array:
            return agent.act(agent_state, obs, info, key).action

        for index, (scenario, task) in enumerate(pairs):
            fresh = COOMLifetime(scenario=scenario, task=task)
            try:
                lifetime = pp.runtime.init_lifetime(
                    fresh, seed=1000 + args.seed, window_size=args.eval_ticks
                )
                carry = pp.runtime.HostRunner(fresh, act, ticks=args.eval_ticks)(lifetime)
                reward = float(carry.metrics.mean[0])
            finally:
                fresh.close()
            _emit(
                out,
                {
                    "kind": "boundary_eval",
                    "tick": tick,
                    "task": index,
                    "reward": reward,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                },
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tick = 0
    with out_path.open("w") as out:
        _emit(
            out,
            {
                "kind": "run",
                "benchmark": "COOM CO8",
                "seed": args.seed,
                "tasks": [f"{scenario}/{task}" for scenario, task in pairs],
                "steps_per_task": args.steps_per_task,
                "eval_every": args.eval_every,
                "eval_ticks": args.eval_ticks,
                "window": args.window,
                "learner": {
                    "algorithm": "ppo",
                    "torso": config.torso,
                    "rollout_steps": config.rollout_steps,
                    "epochs": config.epochs,
                    "minibatches": config.minibatches,
                    "hidden_size": config.hidden_size,
                    "learning_rate": config.learning_rate,
                },
                "bridge": "compiled",
                "backend": jax.default_backend(),
                "jax_version": jax.__version__,
                "provenance": pp.reporting.provenance(),
            },
        )
        try:
            state: Any = runner.init(seed=args.seed, window_size=args.window)
            boundary(out, state.agent, 0)
            for index in range(len(pairs)):
                for _ in range(args.steps_per_task // args.rollout_steps):
                    state = runner(state)
                    tick += args.rollout_steps
                    if tick % args.eval_every == 0:
                        metrics = state.lifetime.metrics
                        _emit(
                            out,
                            {
                                "kind": "eval",
                                "tick": tick,
                                "training_task": index,
                                "window_reward": float(metrics.window_mean[0]),
                                "lifetime_reward": float(metrics.mean[0]),
                                "healthy": bool(state.healthy),
                                "elapsed_s": round(time.perf_counter() - started, 1),
                            },
                        )
                boundary(out, state.agent, tick)
        finally:
            cast(Any, env).close()
        healthy = bool(state.healthy)
        _emit(
            out,
            {
                "kind": "done",
                "tick": tick,
                "healthy": healthy,
                "wall_s": round(time.perf_counter() - started, 1),
            },
        )
    if not healthy:
        sys.exit(1)


if __name__ == "__main__":
    main()
