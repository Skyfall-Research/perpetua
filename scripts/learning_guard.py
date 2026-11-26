"""Learning regression guard: PPO, DQN and SAC against seeded random references.

Each learner trains on the configuration its unit test uses, for several seeds,
and is compared with a random policy run for the same number of ticks from the
same seed. The pass rule per algorithm is:

- the IQM of per-seed (learned - random) has a 95% bootstrap lower bound > 0;
- at least 80% of seeds clear the unit test's own margin.

Scores are the learned lifetime's final reward window against the random
reference's lifetime mean (random is stationary, so its lifetime mean is the
lower-variance estimate of the same quantity). Prints one JSON object per
line (the last is the summary); exits non-zero if any algorithm fails.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import jax
import numpy as np
from gymnasium import spaces

import perpetua as pp
from perpetua.agents import DQN, PPO, SAC, DQNConfig, DQNRunner, PPORunner, SACConfig, SACRunner

Score = tuple[float, float]


def _mean(values: Any) -> float:
    return float(np.asarray(values).mean())


def ppo(seed: int) -> Score:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    trainer = PPORunner(env, PPO(action_size=5))
    state = trainer.init(seed=seed)
    for _ in range(64):
        state = trainer(state)
    assert bool(state.healthy)
    ticks = state.lifetime.state.t.to_int()
    reference = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(
        pp.runtime.init_lifetime(env, seed=seed)
    )
    return _mean(state.lifetime.metrics.window_mean), _mean(reference.metrics.mean)


def ppo_bean_world(seed: int) -> Score:
    env = pp.make("BeanWorld-v0", size=64, num_agents=4)
    trainer = PPORunner(env, PPO(action_size=5))
    state = trainer.init(seed=seed, window_size=2048)
    for _ in range(400):
        state = trainer(state)
    assert bool(state.healthy)
    ticks = state.lifetime.state.t.to_int()
    reference = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(
        pp.runtime.init_lifetime(env, seed=seed, window_size=2048)
    )
    return _mean(state.lifetime.metrics.window_mean), _mean(reference.metrics.mean)


def torch_ppo(seed: int) -> Score:
    from perpetua.torch_bridge import TorchPPO, TorchRunner

    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    runner = TorchRunner(env, TorchPPO(env.observation_space, env.action_space, seed=seed))
    state = runner.init(seed=seed)
    for _ in range(64):
        state = runner(state)
    assert runner.healthy
    ticks = state.lifetime.state.t.to_int()
    reference = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(
        pp.runtime.init_lifetime(env, seed=seed)
    )
    return _mean(state.lifetime.metrics.window_mean), _mean(reference.metrics.mean)


def dqn(seed: int) -> Score:
    env = pp.make("ContinualGrid-v0", size=4)
    config = DQNConfig(
        rollout_steps=32,
        buffer_capacity=4096,
        batch_size=64,
        gradient_steps=8,
        hidden_size=64,
        epsilon=0.1,
        target_update_interval=64,
    )
    trainer = DQNRunner(env, DQN(action_size=5, config=config))
    state = trainer.init(seed=seed, window_size=128)
    for _ in range(200):
        state = trainer(state)
    assert bool(state.healthy)
    ticks = state.lifetime.state.t.to_int()
    reference = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(
        pp.runtime.init_lifetime(env, seed=seed, window_size=128)
    )
    return _mean(state.lifetime.metrics.window_mean), _mean(reference.metrics.mean)


def _sac(seed: int, *, autotune: bool, env_id: str = "gymnax/Pendulum-v1") -> Score:
    env = pp.make(env_id)
    box = cast(spaces.Box, env.action_space)
    agent = SAC(
        action_low=box.low[0],
        action_high=box.high[0],
        config=SACConfig(
            rollout_steps=128,
            buffer_capacity=65536,
            batch_size=256,
            gradient_steps=128,
            autotune_alpha=autotune,
        ),
    )
    trainer = SACRunner(env, agent)
    state = trainer.init(seed=seed, window_size=1024)
    for _ in range(150):
        state = trainer(state)
    assert bool(state.healthy)

    def uniform_torque(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        del obs, info
        return jax.random.uniform(key, (1, 1), minval=-2.0, maxval=2.0)

    ticks = state.lifetime.state.t.to_int()
    reference = pp.runtime.Runner(env, uniform_torque, ticks=ticks)(
        pp.runtime.init_lifetime(env, seed=seed, window_size=1024)
    )
    return _mean(state.lifetime.metrics.window_mean), _mean(reference.metrics.mean)


# name -> (train, per-seed pass rule from the unit test, rule description)
ALGORITHMS: dict[str, tuple[Callable[[int], Score], Callable[[float, float], bool], str]] = {
    "ppo": (ppo, lambda learned, rand: learned > rand + 0.1, "learned > random + 0.1"),
    "ppo-bean-world": (
        ppo_bean_world,
        lambda learned, rand: learned > rand + 0.05,
        "learned > random + 0.05",
    ),
    "dqn": (dqn, lambda learned, rand: learned > 2 * rand, "learned > 2 x random"),
    "torch-ppo": (torch_ppo, lambda learned, rand: learned > rand + 0.1, "learned > random + 0.1"),
    "sac": (
        lambda seed: _sac(seed, autotune=False),
        lambda learned, rand: learned > -1.0 and learned > rand + 2.0,
        "learned > -1 and learned > random + 2",
    ),
    "sac-mjx-pendulum": (
        lambda seed: _sac(seed, autotune=False, env_id="mjx/Pendulum-v0"),
        lambda learned, rand: learned > -1.0 and learned > rand + 2.0,
        "learned > -1 and learned > random + 2",
    ),
    "sac-autotune": (
        lambda seed: _sac(seed, autotune=True),
        lambda learned, rand: learned > -1.0 and learned > rand + 2.0,
        "learned > -1 and learned > random + 2",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        default=",".join(name for name in ALGORITHMS if name != "torch-ppo"),
        help="comma-separated; torch-ppo needs the optional torch extra",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--min-pass-fraction", type=float, default=0.8)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU run requested, but JAX is not using a GPU.")
    results: dict[str, object] = {}
    passed = True
    for name in str(args.algorithms).split(","):
        train, rule, description = ALGORITHMS[name.strip()]
        learned, random = [], []
        started = time.perf_counter()
        for seed in range(args.seeds):
            score = train(seed)
            learned.append(score[0])
            random.append(score[1])
            print(
                json.dumps(
                    {"algorithm": name, "seed": seed, "learned": score[0], "random": score[1]}
                ),
                flush=True,
            )
        difference = np.asarray(learned) - np.asarray(random)
        intervals = pp.reporting.report(
            np.asarray(difference, dtype=np.float32)[:, None], key=jax.random.key(0)
        )
        seed_passes = sum(rule(a, b) for a, b in zip(learned, random, strict=True))
        algorithm_passed = (
            float(intervals["iqm"].low) > 0 and seed_passes >= args.min_pass_fraction * args.seeds
        )
        passed &= algorithm_passed
        results[name] = {
            "seeds": args.seeds,
            "per_seed_rule": description,
            "seeds_passing_rule": seed_passes,
            "learned": learned,
            "random": random,
            "difference_iqm": float(intervals["iqm"].point),
            "difference_iqm_ci95": [float(intervals["iqm"].low), float(intervals["iqm"].high)],
            "difference_mean": float(intervals["mean"].point),
            "wall_s": round(time.perf_counter() - started, 1),
            "passed": algorithm_passed,
        }
    print(
        json.dumps(
            {
                "summary": True,
                "backend": jax.default_backend(),
                "jax_version": jax.__version__,
                "provenance": pp.reporting.provenance(),
                "results": results,
                "passed": passed,
            }
        ),
        flush=True,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
