"""LineWorld: a minimal custom Perpetua environment with hidden goal drift.

Reference implementation for docs/custom-environments.md: one agent on a line
of length 4, reward 1 for reaching the goal, same-tick respawn at 0, and a
hidden goal parameter that drifts. The environment only describes parameters
and dynamics; ``DriftEnv`` adds the drift process, its RNG stream, declared
switch events, the lifetime clock, Oracle fields and metadata.
"""

import argparse
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

import perpetua as pp
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv


class LineParams(NamedTuple):
    """Everything that may drift. Raw values; ``derive`` makes them effective."""

    goal: jax.Array  # () float32, rounded to a cell by derive


class LineState(NamedTuple):
    """The world only: no drift, RNG stream or clock (the framework owns those)."""

    position: jax.Array  # () int32 in [0, length - 1]
    respawn: jax.Array  # () bool, this transition only


class LineWorld:
    """A continuing line world; respawns are events, never resets."""

    default_drift_field = "goal"

    def __init__(self, *, length: int = 4) -> None:
        if type(length) is not int or length < 2:
            raise ValueError("length must be an integer >= 2.")
        self.length = length
        self.num_agents = 1
        self.observation_space = spaces.Box(
            low=np.zeros((1, 2), dtype=np.float32),
            high=np.full((1, 2), float(length - 1), dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.asarray([3], dtype=np.int64))
        self.metadata: pp.Metadata = {
            "env_id": "line-world-v0",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {"length": length, "respawn_policy": "same-tick-on-goal"},
        }

    @property
    def default_params(self) -> LineParams:
        """The goal starts at the far end of the line."""
        return LineParams(jnp.asarray(float(self.length - 1), dtype=jnp.float32))

    def derive(self, params: LineParams) -> LineParams:
        """Raw drift values become an in-bounds goal cell."""
        return LineParams(jnp.clip(jnp.rint(params.goal), 0, self.length - 1))

    def init(self, key: jax.Array, params: LineParams) -> tuple[LineState, jax.Array, EnvInfo]:
        """Start one lifetime: no reset exists, so this runs exactly once."""
        del key  # deterministic start
        state = LineState(jnp.int32(0), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(
        self, state: LineState, action: Any, key: jax.Array, params: LineParams
    ) -> tuple[LineState, jax.Array]:
        """Apply one action and resolve reward and respawn under ``params``."""
        action = jnp.asarray(action, dtype=jnp.int32)
        if action.shape != (self.num_agents,):
            raise ValueError(f"actions must have shape ({self.num_agents},).")
        del key  # deterministic dynamics; the wrapper owns drift randomness
        move = jnp.clip(action[0], 0, 2) - 1  # 0 left, 1 stay, 2 right
        position = jnp.clip(state.position + move, 0, self.length - 1)
        reached = position == params.goal.astype(jnp.int32)
        reward = jnp.asarray([jnp.where(reached, 1.0, 0.0)], dtype=jnp.float32)
        position = jnp.where(reached, jnp.int32(0), position)  # same-tick respawn
        return LineState(position, jnp.asarray(reached)), reward

    def observe(self, state: LineState, params: LineParams) -> tuple[jax.Array, EnvInfo]:
        """Return the public observation and the common contract info."""
        obs = jnp.stack((state.position, params.goal.astype(jnp.int32)))[None, :]
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 3), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return obs.astype(jnp.float32), info

    def privileged(self, state: LineState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}


def make(**kwargs: Any) -> DriftEnv[LineState, LineParams]:
    """Build LineWorld with optional drift, as a registered factory would."""
    drift = kwargs.pop("drift", None)
    return DriftEnv(LineWorld(**kwargs), drift, env_id="line-world-v0")


def main() -> None:
    """Exercise the contract: stepping, respawn events, compiled lifetime, Oracle."""
    env = make(drift=pp.drift.Cyclic([3.0, 1.0], period=8))
    state, obs, info = env.init(jax.random.key(0))
    assert env.observation_space.contains(np.asarray(obs))
    assert set(info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
    for tick in range(4):
        state, obs, reward, info = env.step(state, jnp.asarray([2]), jax.random.key(tick))
        print(f"tick {tick}: reward {float(reward[0]):.1f} respawn {bool(info['respawn'][0])}")

    runner = pp.runtime.Runner(env, pp.policies.random, ticks=32)
    carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=32))
    print("ticks", carry.state.t.to_int(), "mean reward", float(carry.metrics.mean[0]))

    composed = pp.compose(env, pp.drift.RandomWalk(low=1, high=3, scale=0.1))
    configuration = cast(dict[str, Any], composed.metadata.get("configuration"))
    print("composed drift:", cast(dict[str, Any], configuration["drift"])["type"])

    oracle = pp.Oracle(env, expose=["drift_parameters", "boundary"])
    state, obs, info = oracle.init(jax.random.key(0))
    assert oracle.metadata["oracle_exposure"] == ("drift_parameters", "boundary")
    assert "oracle" in info


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Run the LineWorld custom environment through the Perpetua contract: "
        "stepping, respawn events, a compiled lifetime and the Oracle. Takes no options."
    ).parse_args()
    main()
