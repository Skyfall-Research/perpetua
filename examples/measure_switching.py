"""Measure switching throughput and pixel-learning stress.

Two small measurements:

* ``switching``: ticks/second for a switching preset (MinAtar compiled, Atari
  host) and the cost of frequent versus infrequent task switches.
* ``pixels``: PPO with the CNN torso on MinAtar Breakout versus the random
  policy, reporting mean reward per tick over the same tick budget.

These are measurements, not benchmark claims: conditions and limits are
printed with the results.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any

import jax
import jax.numpy as jnp

import perpetua as pp

warnings.filterwarnings("ignore")


def switching(
    backend: str, period: int, ticks: int, chunks: int, num_envs: int, num_threads: int
) -> dict[str, Any]:
    if backend == "atari":
        env = pp.presets.switching_atari(
            ["Breakout-v5", "Asterix-v5"], num_envs=num_envs, num_threads=num_threads, period=period
        )
        runner = pp.runtime.HostRunner(env, pp.policies.random, ticks=ticks)
    else:
        env = pp.presets.switching_minatar(period=period)
        runner = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)
    try:
        carry = pp.runtime.init_lifetime(env, seed=0, window_size=32)
        carry = jax.block_until_ready(runner(carry))  # warm up / compile
        start = time.perf_counter()
        for _ in range(chunks):
            carry = runner(carry)
        jax.block_until_ready(carry)
        elapsed = time.perf_counter() - start
        total = chunks * ticks * env.num_agents
        return {
            "backend": backend,
            "period": period,
            "ticks_per_second": total / elapsed,
            "provenance": pp.reporting.provenance(),
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def pixels(
    updates: int, rollout_steps: int, seeds: tuple[int, ...], eval_ticks: int | None
) -> dict[str, object]:
    """Compare final PPO and random policies on identical fresh lifetimes."""

    from perpetua.agents import PPO, PPOConfig
    from perpetua.agents.ppo import PPORunner

    eval_ticks = updates * rollout_steps if eval_ticks is None else eval_ticks
    learned_scores: list[float] = []
    random_scores: list[float] = []
    for seed in seeds:
        env = pp.make("gymnax/Breakout-MinAtar")
        config = PPOConfig(
            torso="cnn", rollout_steps=rollout_steps, epochs=1, minibatches=1, hidden_size=32
        )
        trainer = PPORunner(env, PPO(action_size=3, config=config))
        state = trainer.init(seed=seed)
        for _ in range(updates):
            state = trainer(state)
        agent = PPO(action_size=3, config=config)

        def learned(
            obs: jax.Array, info: Any, key: jax.Array, agent: Any = agent, state: Any = state
        ) -> jax.Array:
            return agent.act(state.agent, obs, info, key).action

        learned_carry = pp.runtime.Runner(env, learned, ticks=eval_ticks)(
            pp.runtime.init_lifetime(env, seed=seed + 10_000, window_size=eval_ticks)
        )
        random_carry = pp.runtime.Runner(env, pp.policies.random, ticks=eval_ticks)(
            pp.runtime.init_lifetime(env, seed=seed + 10_000, window_size=eval_ticks)
        )
        learned_scores.append(float(learned_carry.metrics.mean[0]))
        random_scores.append(float(random_carry.metrics.mean[0]))
    scores = {
        "ppo": learned_scores,
        "random": random_scores,
    }
    reports = {
        name: pp.reporting.report(
            jnp.asarray([values], dtype=jnp.float32).T,
            key=jax.random.key(0),
        )
        for name, values in scores.items()
    }
    return {
        "provenance": pp.reporting.provenance(),
        "updates": updates,
        "training_ticks_per_seed": updates * rollout_steps,
        "eval_ticks_per_seed": eval_ticks,
        "seeds": list(seeds),
        "scores": scores,
        "reports": {
            name: {
                "iqm": float(report["iqm"].point),
                "iqm_low": float(report["iqm"].low),
                "iqm_high": float(report["iqm"].high),
                "mean": float(report["mean"].point),
                "mean_low": float(report["mean"].low),
                "mean_high": float(report["mean"].high),
            }
            for name, report in reports.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    switch_parser = sub.add_parser("switching")
    switch_parser.add_argument("--backend", choices=["atari", "minatar"], default="minatar")
    switch_parser.add_argument("--period", type=int, default=64)
    switch_parser.add_argument("--ticks", type=int, default=128)
    switch_parser.add_argument("--chunks", type=int, default=20)
    switch_parser.add_argument("--num-envs", type=int, default=2)
    switch_parser.add_argument("--num-threads", type=int, default=1)
    pixel_parser = sub.add_parser("pixels")
    pixel_parser.add_argument("--updates", type=int, default=16)
    pixel_parser.add_argument("--rollout-steps", type=int, default=128)
    pixel_parser.add_argument("--eval-ticks", type=int, default=None)
    pixel_parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()
    if args.command == "switching":
        print(
            json.dumps(
                switching(
                    args.backend,
                    args.period,
                    args.ticks,
                    args.chunks,
                    args.num_envs,
                    args.num_threads,
                ),
                indent=2,
            )
        )
    else:
        seeds = tuple(int(value) for value in args.seeds.split(","))
        print(
            json.dumps(pixels(args.updates, args.rollout_steps, seeds, args.eval_ticks), indent=2)
        )


if __name__ == "__main__":
    main()
