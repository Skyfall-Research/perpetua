"""Run/replay compute-aware IPPO with full held-command and pending-rollout checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp
from perpetua.agents import IPPO, ComputePPORunner, PPOConfig
from perpetua.compute import MeasuredTimingTrace, TimingTrace
from perpetua.core import Environment


def compare(left: object, right: object) -> None:
    """Compare returned arrays/keys, not only potentially degenerate summary metrics."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_array_equal(a, b)


def run_compute(
    *,
    backend: str = "native",
    mode: str = "recorded",
    chunks: int = 3,
    ticks: int = 2,
    quantum_ns: int = 1_000_000,
    checkpoint: Path | None = None,
    resume: Path | None = None,
    replay: Path | None = None,
    require_gpu: bool = False,
) -> dict[str, object]:
    """Keep one replayable last chunk, not an unbounded lifetime timing log."""
    if type(chunks) is not int or chunks < 1:
        raise ValueError("chunks must be a positive integer.")
    if mode not in {"recorded", "measured"}:
        raise ValueError("mode must be recorded or measured.")
    if resume is not None and replay is not None:
        raise ValueError("Select resume or replay, not both.")
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU execution requested, but JAX is not using a GPU.")
    env: Environment[Any]
    action_size = 5
    if backend == "native":
        env = pp.make("CooperativeGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 7))
    elif backend == "gymnasium":
        from perpetua.adapters import GymnasiumLifetime

        env = GymnasiumLifetime(gym.make("CartPole-v1", max_episode_steps=3))
        action_size = 2
    elif backend in {"pettingzoo", "pettingzoo-snapshot"}:
        from pettingzoo import EnvSpec, ParallelEnv, make

        from perpetua.adapters import PettingZooLifetime

        spec = EnvSpec(
            id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
        )
        env = PettingZooLifetime(
            cast(ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=3)),
            snapshot=backend == "pettingzoo-snapshot",
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    try:
        if (
            checkpoint is not None or resume is not None or replay is not None
        ) and env.metadata.get("checkpointable") is False:
            raise NotImplementedError("This backend has no complete lifetime checkpoint codec.")
        config = PPOConfig(rollout_steps=8, epochs=1, minibatches=1, hidden_size=16)
        runner = ComputePPORunner[Any](
            env, IPPO(action_size=action_size, config=config), decisions=4, quantum_ns=quantum_ns
        )
        trace_type = MeasuredTimingTrace if mode == "measured" else TimingTrace
        profile = trace_type.from_ticks([ticks] * 4, quantum_ns=quantum_ns)
        metadata = {
            **runner.metadata,
            "cost_source": mode,
            "recorded_profile_ticks": ticks if mode == "recorded" else None,
        }
        state = runner.init(seed=0, window_size=16)
        payload = {"state": state, "start": state, "trace": profile}
        source = replay if replay is not None else resume
        if source is not None:
            payload = pp.checkpoint.restore(source, payload, expected_metadata=metadata)
            state = payload["state"]
        exact = False
        if replay is not None:
            state = runner(payload["start"], payload["trace"])
            compare(state, payload["state"])
            exact = True
        else:
            for _ in range(chunks):
                start = state
                if mode == "measured":
                    state, trace = runner.measure(state)
                else:
                    trace = profile
                    state = runner(state, trace)
                if not bool(state.training.healthy) or not bool(state.timing.healthy):
                    raise RuntimeError("Unhealthy timed training state.")
                payload = {"state": state, "start": start, "trace": trace}
        if checkpoint is not None:
            pp.checkpoint.save(checkpoint, payload, metadata=metadata)
        return {
            "backend": backend,
            "mode": mode,
            "device": jax.default_backend(),
            "provenance": pp.reporting.provenance(),
            "world_ticks": state.training.lifetime.state.t.to_int(),
            "decisions": state.timing.decisions.to_int(),
            "compute_ns": state.timing.compute_ns.to_int(),
            "updates": state.training.updates.to_int(),
            "pending_samples": int(state.filled),
            "mean_reward": np.asarray(state.training.lifetime.metrics.mean).tolist(),
            "healthy": bool(state.training.healthy & state.timing.healthy),
            "exact_replay": exact,
            "metadata": metadata,
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def main() -> None:
    """Select a cost model; wall measurements are not a throughput benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["native", "gymnasium", "pettingzoo", "pettingzoo-snapshot"],
        default="native",
    )
    parser.add_argument("--mode", choices=["recorded", "measured"], default="recorded")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--quantum-ns", type=int, default=1_000_000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_compute(
                backend=args.backend,
                mode=args.mode,
                chunks=args.chunks,
                ticks=args.ticks,
                quantum_ns=args.quantum_ns,
                checkpoint=args.checkpoint,
                resume=args.resume,
                replay=args.replay,
                require_gpu=args.require_gpu,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
