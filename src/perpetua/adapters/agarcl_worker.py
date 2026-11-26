"""AgarCL worker; runs only inside the pinned AgarCL Python environment.

The parent sends JSON lines; a request may be followed by ``bytes`` raw bytes
(the ``replay`` action log, float32 rows of x, y, discrete). Responses are a
JSON header followed by raw observation bytes (uint8 screen or int32 grid).

``reset(seed=...)`` is ignored by AgarCL, so the seed is applied through
``seed`` before reset; later respawns keep the RNG stream. The grid uses the
low-level binding because upstream's ``agario-grid-v0`` wrapper is broken (an
undefined ``args`` and an 11-for-10 constructor call). Upstream also builds
grid frames once per step with ``tick_index=0``, so frames are only written
when ``num_frames == ticks_per_step``; frame 0 then holds the post-step state,
and only its channels are returned.
"""

# Runs in AgarCL's own environment, whose packages are absent here.
# pyright: basic, reportMissingImports=false, reportUnusedImport=false
# pyright: reportAttributeAccessIssue=false

import json
import os
import sys
import types
from typing import Any

import numpy as np

_PROTOCOL = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

import agarcl  # noqa: E402
import gym_agario  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from gym_agario.AgarioEnv import AgarioEnv  # noqa: E402


def _world(request: dict[str, Any]) -> dict[str, Any]:
    """Upstream world settings; omitted ones keep upstream's "normal" defaults."""
    world: dict[str, Any] = {
        "num_bots": int(request.get("num_bots", 0)),
        "mode": int(request.get("mode", 0)),
    }
    for name in ("arena_size", "num_pellets", "num_viruses"):
        if request.get(name) is not None:
            world[name] = int(request[name])
    return world


class Screen:
    def __init__(self, request: dict[str, Any]) -> None:
        self.respawn_after = request.get("respawn_after")
        self.ticks = 0
        self.env = gym.make(
            "agario-screen-v0",
            screen_len=int(request.get("screen_len", 84)),
            agent_view=bool(request.get("agent_view", False)),
            add_noise=False,
            **_world(request),
        )

    def reset(self, seed: int) -> np.ndarray:
        self.ticks = 0
        self.env.unwrapped.seed(seed)
        obs, _ = self.env.reset()
        return np.asarray(obs, dtype=np.uint8)

    def step(self, dx: float, dy: float, discrete: int) -> tuple[np.ndarray, float, bool]:
        action = (np.asarray([dx, dy], dtype=np.float32), discrete)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.ticks += 1
        respawn = bool(
            terminated
            or truncated
            or (self.respawn_after is not None and self.ticks >= int(self.respawn_after))
        )
        if respawn:
            self.ticks = 0
            obs, _ = self.env.reset()
        return np.asarray(obs, dtype=np.uint8), float(reward), respawn

    def close(self) -> None:
        self.env.close()


class Grid:
    def __init__(self, request: dict[str, Any]) -> None:
        self.respawn_after = request.get("respawn_after")
        self.ticks = 0
        # Upstream's own "normal" defaults; the binding takes the first ten.
        args = AgarioEnv._get_env_args(types.SimpleNamespace(), _world(request))
        ticks_per_step = int(args[1])
        self.env = agarcl.GridEnvironment(*args[:10])
        self.env.configure_observation(
            {"num_frames": ticks_per_step, "grid_size": int(request.get("grid_size", 128))}
        )
        self.channels = int(self.env.observation_shape()[0]) // ticks_per_step

    def _obs(self) -> np.ndarray:
        frame = np.asarray(self.env.get_state()[0])[: self.channels]
        return np.ascontiguousarray(np.transpose(frame, (1, 2, 0)), dtype=np.int32)

    def reset(self, seed: int) -> np.ndarray:
        self.ticks = 0
        self.env.seed(seed)
        self.env.reset()
        return self._obs()

    def step(self, dx: float, dy: float, discrete: int) -> tuple[np.ndarray, float, bool]:
        self.env.take_actions([(dx, dy, discrete)])
        reward = float(self.env.step()[0])
        self.ticks += 1
        respawn = bool(
            self.env.dones()[0]
            or (self.respawn_after is not None and self.ticks >= int(self.respawn_after))
        )
        if respawn:
            self.ticks = 0
            self.env.reset()
        return self._obs(), reward, respawn

    def close(self) -> None:
        self.env.close()


def reply(header: dict[str, Any], obs: np.ndarray | None = None) -> None:
    if obs is not None:
        obs = np.ascontiguousarray(obs)
        header = {
            **header,
            "shape": list(obs.shape),
            "bytes": obs.nbytes,
            "dtype": str(obs.dtype),
        }
    _PROTOCOL.write((json.dumps(header) + "\n").encode())
    if obs is not None:
        _PROTOCOL.write(obs.tobytes())
    _PROTOCOL.flush()


def main() -> None:
    sim: Any = None
    stdin = sys.stdin.buffer
    while line := stdin.readline():
        request = json.loads(line)
        command = request["command"]
        try:
            payload = stdin.read(int(request["bytes"])) if "bytes" in request else b""
            if len(payload) != int(request.get("bytes", 0)):
                raise RuntimeError("AgarCL worker received a truncated request payload.")
            if command == "init":
                if sim is not None:
                    closing, sim = sim, None
                    closing.close()
                sim = Grid(request) if request.get("obs_type") == "grid" else Screen(request)
                reply({"actions": 3}, sim.reset(int(request["seed"])))
            elif command in ("step", "replay"):
                if sim is None:
                    raise RuntimeError("AgarCL worker has no environment.")
                if command == "step":
                    rows = [(*request["move"], request["discrete"])]
                else:
                    rows = np.frombuffer(payload, dtype=np.float32).reshape(-1, 3).tolist()
                obs, total, respawns = None, 0.0, 0
                reward, respawn = 0.0, False
                for dx, dy, discrete in rows:
                    obs, reward, respawn = sim.step(float(dx), float(dy), int(discrete))
                    total += reward
                    respawns += respawn
                if obs is None:
                    raise ValueError("replay needs at least one action.")
                header = {"reward": reward, "respawn": respawn}
                if command == "replay":
                    header |= {"ticks": len(rows), "total_reward": total, "respawns": respawns}
                reply(header, obs)
            elif command == "close":
                break
            else:
                raise ValueError(f"Unknown AgarCL worker command {command!r}.")
        except Exception as error:  # noqa: BLE001 - report to the parent instead of hanging it
            reply({"error": f"{type(error).__name__}: {error}"})
    if sim is not None:
        sim.close()


if __name__ == "__main__":
    main()
