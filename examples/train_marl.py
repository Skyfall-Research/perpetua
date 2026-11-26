"""Run continuing IPPO on an interacting native grid or a real PettingZoo CPU world."""

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np

import perpetua as pp
from perpetua.agents import IPPO, HostPPORunner, PPOConfig, PPORunner
from perpetua.core import Environment


@contextlib.contextmanager
def quiet_stdout() -> Iterator[None]:
    """Send stdout to stderr at the descriptor level during noisy imports.

    Some JaxMARL dependency replaces ``sys.stdout`` mid-import, defeating
    object-level redirection; descriptor duplication survives that. Buffered
    chatter must flush before the descriptor is restored.
    """
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def run_marl(
    *,
    backend: str = "native",
    updates: int = 16,
    seed: int = 0,
    checkpoint: Path | None = None,
    resume: Path | None = None,
    require_gpu: bool = False,
) -> dict[str, object]:
    """Train online; the finite CLI budget is not a lifetime/episode boundary."""
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer.")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU execution requested, but JAX is not using a GPU.")
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    env: Environment[Any]
    trainer: PPORunner[Any] | HostPPORunner[Any]
    config = PPOConfig(rollout_steps=32)
    if backend == "native":
        env = pp.make(
            "CooperativeGrid-v0",
            size=2,
            drift=pp.drift.Cyclic([(1.0, 1.0), (0.0, 0.0)], period=128),
        )
        trainer = PPORunner(env, IPPO(action_size=5, config=config))
    elif backend == "partner":
        env = pp.make(
            "PartnerGrid-v0",
            size=3,
            num_learners=1,
            num_partners=1,
            partner_drift=pp.drift.Cyclic([(1.0,), (0.0,)], period=128),
        )
        trainer = PPORunner(env, IPPO(action_size=5, config=config))
    elif backend == "opponent":
        env = pp.make(
            "OpponentGrid-v0",
            size=3,
            num_learners=1,
            num_opponents=1,
            opponent_drift=pp.drift.Cyclic([(1.0,), (0.0,)], period=128),
        )
        trainer = PPORunner(env, IPPO(action_size=5, config=config))
    elif backend == "jaxmarl":
        # JaxMARL prints import-time chatter; keep CLI stdout machine-readable.
        with quiet_stdout():
            env = pp.make(
                "jaxmarl/MPE_simple_spread-v3",
                drift=pp.drift.Cyclic([(1.0,), (0.5,)], period=128),
            )
        trainer = PPORunner(env, IPPO(action_size=5, config=config))
    elif backend == "jaxmarl-scripted":
        with quiet_stdout():
            env = pp.make(
                "jaxmarl/MPE_simple_spread-v3",
                num_scripted=1,
                scripted_behavior="partner",
                scripted_drift=pp.drift.Cyclic([(1.0,), (0.0,)], period=128),
            )
        trainer = PPORunner(env, IPPO(action_size=5, config=config))
    elif backend in {"pettingzoo", "pettingzoo-snapshot"}:
        from pettingzoo import EnvSpec, ParallelEnv, make

        from perpetua.adapters import PettingZooLifetime

        spec = EnvSpec(
            id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
        )
        raw = cast(ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=128))
        env = PettingZooLifetime(
            raw,
            drift=pp.drift.Cyclic([(1.0,), (0.5,)], period=128),
            snapshot=backend == "pettingzoo-snapshot",
        )
        trainer = HostPPORunner(env, IPPO(action_size=5, config=config))
    else:
        raise ValueError(f"Unsupported MARL backend: {backend}")
    try:
        if (checkpoint is not None or resume is not None) and env.metadata.get(
            "checkpointable"
        ) is False:
            raise NotImplementedError("This PettingZoo simulator has no complete checkpoint codec.")
        state = trainer.init(seed=seed, window_size=128)
        if resume is not None:
            state = pp.checkpoint.restore(resume, state, expected_metadata=trainer.metadata)
        for _ in range(updates):
            state = trainer(state)
            if not bool(state.healthy):
                raise RuntimeError("Non-finite continuing MARL training state.")
        if checkpoint is not None:
            pp.checkpoint.save(checkpoint, state, metadata=trainer.metadata)
        steps = state.agent.gradient_steps
        return {
            "backend": backend,
            "policy_device": jax.default_backend(),
            "provenance": pp.reporting.provenance(),
            "algorithm": "ippo",
            "agents": env.num_agents,
            "lifetime_ticks": state.lifetime.state.t.to_int(),
            "updates": state.updates.to_int(),
            "gradient_steps_per_agent": [
                pp.Clock(steps.hi[i], steps.lo[i]).to_int() for i in range(env.num_agents)
            ],
            "mean_reward": np.asarray(state.lifetime.metrics.mean).tolist(),
            "loss_per_agent": np.asarray(state.statistics.loss).tolist(),
            "healthy": bool(state.healthy),
            "metadata": trainer.metadata,
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def main() -> None:
    """Select an execution path without replacing the shared IPPO learner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=[
            "native",
            "partner",
            "opponent",
            "jaxmarl",
            "jaxmarl-scripted",
            "pettingzoo",
            "pettingzoo-snapshot",
        ],
        default="native",
    )
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_marl(
                backend=args.backend,
                updates=args.updates,
                seed=args.seed,
                checkpoint=args.checkpoint,
                resume=args.resume,
                require_gpu=args.require_gpu,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
