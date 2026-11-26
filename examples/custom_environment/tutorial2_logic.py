"""Tutorial step 2: the environment logic.

A forager walks a ``size x size`` grid. Reaching the berry earns +1, and the
forager respawns on a random cell on the same tick: a respawn is an event, not
a reset. The berry's position is a parameter, so a drift process can move it
without touching this code.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

import perpetua as pp
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv

# stay, up, down, left, right
MOVES = jnp.asarray([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


class BerryParams(NamedTuple):
    berry: jax.Array  # float32 (2,): row, column; rounded to a cell by derive


class BerryState(NamedTuple):
    position: jax.Array  # int32 (2,)
    respawn: jax.Array  # bool (), this transition only


class BerryField:
    """One forager on a grid collects a berry and respawns, forever."""

    default_drift_field = "berry"  # what a bare drift process drives

    def __init__(self, *, size: int = 5) -> None:
        if type(size) is not int or size < 2:
            raise ValueError("size must be an integer >= 2.")
        self.size = size
        self.num_agents = 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1, 4), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.asarray([5], dtype=np.int64))
        self.metadata: pp.Metadata = {
            "env_id": "tutorial/BerryField-v0",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {"size": size},
        }

    @property
    def default_params(self) -> BerryParams:
        corner = float(self.size - 1)
        return BerryParams(jnp.asarray([corner, corner], dtype=jnp.float32))

    def derive(self, params: BerryParams) -> BerryParams:
        # Any drift value becomes an in-bounds cell; drift smaller than half a
        # cell does not move the berry, but a Cyclic phase change still fires.
        return BerryParams(jnp.clip(jnp.rint(params.berry), 0, self.size - 1))

    def init(self, key: jax.Array, params: BerryParams) -> tuple[BerryState, jax.Array, EnvInfo]:
        state = BerryState(self._spawn(key, params), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(
        self, state: BerryState, action: Any, key: jax.Array, params: BerryParams
    ) -> tuple[BerryState, jax.Array]:
        move = MOVES[jnp.clip(jnp.asarray(action, dtype=jnp.int32)[0], 0, 4)]
        position = jnp.clip(state.position + move, 0, self.size - 1)
        reached = jnp.all(position == params.berry.astype(jnp.int32))
        position = jnp.where(reached, self._spawn(key, params), position)
        reward = jnp.where(reached, 1.0, 0.0).astype(jnp.float32)[None]
        return BerryState(position, reached), reward

    def observe(self, state: BerryState, params: BerryParams) -> tuple[jax.Array, EnvInfo]:
        cells = jnp.concatenate([state.position.astype(jnp.float32), params.berry])
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 5), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return (cells / (self.size - 1))[None], info

    def privileged(self, state: BerryState) -> dict[str, jax.Array]:
        del state
        return {}

    def _spawn(self, key: jax.Array, params: BerryParams) -> jax.Array:
        """A uniform cell other than the berry's."""
        cells = self.size * self.size
        berry = params.berry.astype(jnp.int32)
        index = jax.random.randint(key, (), 0, cells - 1)
        index = jnp.where(index >= berry[0] * self.size + berry[1], index + 1, index)
        return jnp.stack([index // self.size, index % self.size]).astype(jnp.int32)


def make(**kwargs: Any) -> DriftEnv[BerryState, BerryParams]:
    """The factory: ``DriftEnv`` adds drift, the clock and Oracle fields."""
    drift = kwargs.pop("drift", None)
    return DriftEnv(BerryField(**kwargs), drift, env_id="tutorial/BerryField-v0")


if __name__ == "__main__":
    env = make(drift=pp.drift.Cyclic([(4.0, 4.0), (0.0, 0.0)], period=64))
    state, obs, info = env.init(jax.random.key(0))  # once per lifetime
    for tick in range(3):
        action = pp.policies.random(obs, info, jax.random.key(100 + tick))
        state, obs, reward, info = env.step(state, action, jax.random.key(tick))
    carry = pp.runtime.Runner(env, pp.policies.random, ticks=256)(
        pp.runtime.init_lifetime(env, seed=0, window_size=256)
    )
    print("clock", carry.state.t.to_int(), "mean reward", float(carry.metrics.mean[0]))
