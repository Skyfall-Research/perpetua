"""Train continuing PPO and optionally save or resume its complete lifetime."""

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner


def train(
    *,
    updates: int = 128,
    seed: int = 0,
    drift_name: str = "stationary",
    checkpoint: Path | None = None,
    resume: Path | None = None,
    require_gpu: bool = False,
    progress: bool = False,
) -> dict[str, object]:
    """Run online updates; reward metrics accrue on the training lifetime itself."""
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer.")
    if checkpoint is not None and checkpoint.expanduser().exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU training requested, but JAX is not using a GPU.")
    drifts: dict[str, pp.drift.Drift] = {
        "stationary": pp.drift.Stationary(),
        "cyclic": pp.drift.Cyclic([(2, 2), (0, 0)], period=4096),
        "random-walk": pp.drift.RandomWalk(low=0, high=2, scale=0.05),
    }
    if drift_name not in drifts:
        raise ValueError(f"Unknown drift: {drift_name}")
    env = pp.make("ContinualGrid-v0", size=3, num_agents=4, drift=drifts[drift_name])
    trainer = PPORunner(env, PPO(action_size=5, config=PPOConfig()))
    state = trainer.init(seed=seed, window_size=1024)
    if resume is not None:
        state = pp.checkpoint.restore(resume, state, expected_metadata=trainer.metadata)
    for update in range(1, updates + 1):
        state = trainer(state)
        if not bool(state.healthy):
            raise RuntimeError("Non-finite PPO/lifetime state detected at an update boundary.")
        if progress and update % max(1, updates // 10) == 0:
            window = float(np.asarray(state.lifetime.metrics.window_mean).mean())
            print(f"update {update}/{updates}: windowed reward {window:.3f}", file=sys.stderr)
    if checkpoint is not None:
        pp.checkpoint.save(checkpoint, state, metadata=trainer.metadata)
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": pp.reporting.provenance(),
        "algorithm": "ppo",
        "initialization_seed": seed if resume is None else None,
        "resumed_from": str(resume) if resume is not None else None,
        "lifetime_ticks": state.lifetime.state.t.to_int(),
        "updates": state.updates.to_int(),
        "gradient_steps": state.agent.gradient_steps.to_int(),
        "mean_reward": np.asarray(state.lifetime.metrics.mean).tolist(),
        "windowed_reward": np.asarray(state.lifetime.metrics.window_mean).tolist(),
        "hyperparameters": state.agent.hyperparameters.as_dict(),
        "loss": float(state.statistics.loss),
        "healthy": bool(state.healthy),
        "metadata": trainer.metadata,
    }


def main() -> None:
    """Train additional updates, optionally continuing from an existing checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=128, help="number of learner updates to run")
    parser.add_argument("--seed", type=int, default=0, help="seed of a new lifetime")
    parser.add_argument(
        "--drift",
        choices=["stationary", "cyclic", "random-walk"],
        default="stationary",
        help="how the environment changes during the lifetime",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="directory to save the final state to (must not exist)"
    )
    parser.add_argument("--resume", type=Path, help="checkpoint directory to continue from")
    parser.add_argument(
        "--require-gpu", action="store_true", help="fail unless JAX is running on a GPU"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not print progress lines to stderr"
    )
    args = parser.parse_args()
    report = train(
        updates=int(args.updates),
        seed=int(args.seed),
        drift_name=str(args.drift),
        checkpoint=args.checkpoint,
        resume=args.resume,
        require_gpu=bool(args.require_gpu),
        progress=not args.quiet,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
