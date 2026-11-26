"""Run the same PPO learner on JAX, Gymnasium CPU or batched C++ CartPole."""

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import numpy as np

import perpetua as pp
from perpetua.adapters import GymnasiumLifetime
from perpetua.agents import PPO, HostPPORunner, PPORunner
from perpetua.core import Environment


def run_control(
    *,
    backend: str = "gymnax",
    updates: int = 16,
    num_envs: int = 8,
    require_gpu: bool = False,
    checkpoint: Path | None = None,
    resume: Path | None = None,
) -> dict[str, object]:
    """Run finite integration checks, not a throughput or cross-backend benchmark."""
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer.")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU policy execution requested, but JAX is not using a GPU.")
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    # This CLI selects heterogeneous state types; each implementation remains typed.
    env: Environment[Any]
    trainer: PPORunner[Any] | HostPPORunner[Any]
    if backend == "gymnax":
        env = pp.make("gymnax/CartPole-v1", drift=pp.drift.Cyclic([(9.8,), (5.0,)], period=128))
        trainer = PPORunner(env, PPO(action_size=2))
    elif backend in {"envpool", "envpool-snapshot"}:
        env = pp.make(
            "envpool/CartPole-v1",
            num_envs=num_envs,
            num_threads=2,
            snapshot=backend == "envpool-snapshot",
            drift=pp.drift.Cyclic([(1.0,), (0.5,)], period=128),
        )
        trainer = HostPPORunner(env, PPO(action_size=2))
    elif backend == "gymnasium":
        env = GymnasiumLifetime(gym.make("CartPole-v1"))
        trainer = HostPPORunner(env, PPO(action_size=2))
    else:
        raise ValueError(f"Unsupported control backend: {backend}")
    try:
        if (checkpoint is not None or resume is not None) and env.metadata.get(
            "checkpointable"
        ) is False:
            raise NotImplementedError("This simulator backend has no complete checkpoint support.")
        state = trainer.init(seed=0, window_size=128)
        if resume is not None:
            state = pp.checkpoint.restore(resume, state, expected_metadata=trainer.metadata)
        for _ in range(updates):
            state = trainer(state)
            if not bool(state.healthy):
                raise RuntimeError("Non-finite control training state.")
        if checkpoint is not None:
            pp.checkpoint.save(checkpoint, state, metadata=trainer.metadata)
        return {
            "backend": backend,
            "policy_device": jax.default_backend(),
            "provenance": pp.reporting.provenance(),
            "simulator": {
                "gymnax": "JAX",
                "envpool": "batched C++ CPU",
                "envpool-snapshot": "snapshot-backed C++ CPU",
                "gymnasium": "snapshot-backed CPU",
            }[backend],
            "slots": env.num_agents,
            "lifetime_ticks": state.lifetime.state.t.to_int(),
            "updates": state.updates.to_int(),
            "gradient_steps": state.agent.gradient_steps.to_int(),
            "mean_reward": np.asarray(state.lifetime.metrics.mean).tolist(),
            "healthy": bool(state.healthy),
            "metadata": trainer.metadata,
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def main() -> None:
    """Choose the simulator path while retaining the same PPO implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["gymnax", "envpool", "gymnasium", "envpool-snapshot"],
        default="gymnax",
    )
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_control(
                backend=str(args.backend),
                updates=int(args.updates),
                num_envs=int(args.num_envs),
                require_gpu=bool(args.require_gpu),
                checkpoint=args.checkpoint,
                resume=args.resume,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
