"""Measure buffer donation on a learner with a large replay buffer.

Runs DQN on ``BeanWorld-v0`` (150 observation features) with a replay buffer
of ``--capacity`` transitions, once with ``donate=False`` and once with
``donate=True`` in separate processes, and reports device memory in use, peak
device memory and seconds per update after warm-up. Prints one JSON object.
"""

import argparse
import json
import subprocess
import sys
import time

import jax
import numpy as np

from perpetua.reporting import provenance


def measure(capacity: int, updates: int, donate: bool) -> dict[str, object]:
    import perpetua as pp
    from perpetua.agents import DQN, DQNConfig, DQNRunner

    env = pp.make("BeanWorld-v0", size=64, num_agents=4)
    config = DQNConfig(rollout_steps=64, buffer_capacity=capacity, batch_size=256, gradient_steps=4)
    runner = DQNRunner(env, DQN(action_size=5, config=config), donate=donate)
    state = runner(runner.init(seed=0, window_size=128))  # warm-up and compile
    jax.block_until_ready(state)
    started = time.perf_counter()
    for _ in range(updates):
        state = runner(state)
    jax.block_until_ready(state)
    seconds = (time.perf_counter() - started) / updates
    stats = jax.local_devices()[0].memory_stats() or {}
    replay_bytes = sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(state.replay))
    return {
        "donate": donate,
        "seconds_per_update": seconds,
        "device_bytes_in_use": stats.get("bytes_in_use"),
        "device_peak_bytes": stats.get("peak_bytes_in_use"),
        "replay_bytes": replay_bytes,
        "healthy": bool(state.healthy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--child", choices=["on", "off"])
    args = parser.parse_args()
    if args.child is not None:
        print(json.dumps(measure(args.capacity, args.updates, args.child == "on")))
        return
    results = []
    for mode in ("off", "on"):
        output = subprocess.run(
            [
                sys.executable,
                __file__,
                "--capacity",
                str(args.capacity),
                "--updates",
                str(args.updates),
                "--child",
                mode,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        results.append(json.loads(output.strip().splitlines()[-1]))
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "jax_version": jax.__version__,
                "provenance": provenance(),
                "capacity": args.capacity,
                "updates": args.updates,
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
