"""COOM worker process; runs inside COOM's own pinned legacy environment.

This file must import only the standard library, NumPy and COOM: it is
executed by the legacy interpreter (Gymnasium 0.28.1), never imported by
Perpetua. Requests are JSON lines on stdin; each reply is a JSON line on a
private copy of stdout, followed by the observation as raw float32 bytes when
the header says so. File descriptor 1 is redirected to stderr so that ViZDoom
or COOM prints can never corrupt the protocol.

Each task is built directly with ``make_env``, which avoids COOM's broken
``make_sequence`` default (no action-space function) and its wrappers' lack of
``reset(seed=...)``: the seed goes to ViZDoom through ``doom_kwargs`` instead.
The four stacked RGB frames are merged into channels, ``(84, 84, 12)``.
"""

# Runs in COOM's runtime, so COOM cannot resolve in Perpetua's type checker.
# pyright: basic, reportMissingImports=false

import itertools
import json
import os
import sys
from typing import Any

import numpy as np

_PROTOCOL = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

from COOM.env.builder import build_multi_discrete_actions, make_env  # noqa: E402
from COOM.utils.config import Scenario, Sequence, sequence_scenarios, sequence_tasks  # noqa: E402


def frames(obs: object) -> np.ndarray:
    stacked = np.asarray(obs, dtype=np.float32)  # (frames, H, W, C)
    return np.ascontiguousarray(
        stacked.transpose(1, 2, 0, 3).reshape(stacked.shape[1], stacked.shape[2], -1)
    )


def reply(header: dict, obs: np.ndarray | None = None) -> None:
    if obs is not None:
        header = {**header, "shape": list(obs.shape), "bytes": obs.nbytes}
    _PROTOCOL.write((json.dumps(header) + "\n").encode())
    if obs is not None:
        _PROTOCOL.write(obs.tobytes())
    _PROTOCOL.flush()


def build(request: dict) -> Any:
    doom_kwargs = {
        "env": request["task"],
        "action_space_fn": build_multi_discrete_actions,
        "seed": int(request["seed"]),
        "render": False,
    }
    return make_env(Scenario[request["scenario"].upper()], request["task"], 0, None, doom_kwargs)


def main() -> None:
    env: Any = None
    for line in sys.stdin:
        request = json.loads(line)
        command = request["command"]
        try:
            if command == "sequence":
                name = Sequence[request["sequence"].upper()]
                pairs = itertools.product(sequence_scenarios[name], sequence_tasks[name])
                reply({"tasks": [[scenario.name.lower(), task] for scenario, task in pairs]})
            elif command == "init":
                if env is not None:
                    closing, env = env, None
                    closing.close()
                env = build(request)
                obs, _ = env.reset()
                reply({"actions": int(env.action_space.n)}, frames(obs))
            elif command == "step":
                if env is None:
                    raise RuntimeError("COOM worker has no environment.")
                obs, reward, terminated, truncated, _ = env.step(int(request["action"]))
                respawn = bool(terminated or truncated)
                if respawn:
                    obs, _ = env.reset()
                reply({"reward": float(reward), "respawn": respawn}, frames(obs))
            elif command == "close":
                break
            else:
                raise ValueError(f"Unknown COOM worker command {command!r}.")
        except Exception as error:  # noqa: BLE001 - report to the parent instead of hanging it
            reply({"error": f"{type(error).__name__}: {error}"})
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
