"""Tutorial step 3: drift and evolving actions.

Step 2's world gains fences: a ``(4,)`` parameter, one value per move
direction (1 open, 0 closed). A closed fence masks its move in
``info["action_mask"]`` and blocks it in ``step``. Listing ``fences`` in
``mask_params`` tells the framework that changing it changes the action set,
so it raises ``info["space_change"]`` on exactly those ticks. Shapes never
change, so the lifetime still compiles once.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

import perpetua as pp
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv

MOVES = jnp.asarray([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


class BerryParams(NamedTuple):
    berry: jax.Array  # float32 (2,)
    fences: jax.Array  # float32 (4,): up, down, left, right; 1 open, 0 closed


class BerryState(NamedTuple):
    position: jax.Array  # int32 (2,)
    respawn: jax.Array  # bool ()


class BerryField:
    """A forager collects a berry while fences open and close."""

    default_drift_field = "berry"
    mask_params = ("fences",)  # changing these changes the action mask

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
        return BerryParams(
            jnp.asarray([corner, corner], dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32)
        )

    def derive(self, params: BerryParams) -> BerryParams:
        return BerryParams(
            jnp.clip(jnp.rint(params.berry), 0, self.size - 1),
            (params.fences >= 0.5).astype(jnp.float32),
        )

    def init(self, key: jax.Array, params: BerryParams) -> tuple[BerryState, jax.Array, EnvInfo]:
        state = BerryState(self._spawn(key, params), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(
        self, state: BerryState, action: Any, key: jax.Array, params: BerryParams
    ) -> tuple[BerryState, jax.Array]:
        choice = jnp.clip(jnp.asarray(action, dtype=jnp.int32)[0], 0, 4)
        allowed = self._allowed(params)[choice]
        move = jnp.where(allowed, MOVES[choice], 0)  # a fenced move becomes stay
        position = jnp.clip(state.position + move, 0, self.size - 1)
        reached = jnp.all(position == params.berry.astype(jnp.int32))
        position = jnp.where(reached, self._spawn(key, params), position)
        reward = jnp.where(reached, 1.0, 0.0).astype(jnp.float32)[None]
        return BerryState(position, reached), reward

    def observe(self, state: BerryState, params: BerryParams) -> tuple[jax.Array, EnvInfo]:
        cells = jnp.concatenate([state.position.astype(jnp.float32), params.berry])
        info: EnvInfo = {
            "action_mask": self._allowed(params)[None],
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return (cells / (self.size - 1))[None], info

    def privileged(self, state: BerryState) -> dict[str, jax.Array]:
        del state
        return {}

    def _allowed(self, params: BerryParams) -> jax.Array:
        """Stay is always allowed; each move needs its fence open."""
        return jnp.concatenate([jnp.ones(1, dtype=jnp.bool_), params.fences > 0.5])

    def _spawn(self, key: jax.Array, params: BerryParams) -> jax.Array:
        cells = self.size * self.size
        berry = params.berry.astype(jnp.int32)
        index = jax.random.randint(key, (), 0, cells - 1)
        index = jnp.where(index >= berry[0] * self.size + berry[1], index + 1, index)
        return jnp.stack([index // self.size, index % self.size]).astype(jnp.int32)


def make(**kwargs: Any) -> DriftEnv[BerryState, BerryParams]:
    drift = kwargs.pop("drift", None)
    return DriftEnv(BerryField(**kwargs), drift, env_id="tutorial/BerryField-v0")


if __name__ == "__main__":
    env = make(
        drift={
            "berry": pp.drift.Cyclic([(4.0, 4.0), (0.0, 0.0)], period=64),
            # Every 16 ticks, close the left and right fences, then reopen them.
            "fences": pp.drift.Cyclic([(1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 0.0)], period=16),
        }
    )
    state, obs, info = env.init(jax.random.key(0))
    changes = []
    for tick in range(48):
        action = pp.policies.random(obs, info, jax.random.key(100 + tick))  # respects the mask
        state, obs, reward, info = env.step(state, action, jax.random.key(tick))
        if bool(np.asarray(info.get("space_change"))[0]):
            changes.append(info["t"].to_int())
    print("space_change at ticks", changes)
    print("mask now", np.asarray(info["action_mask"][0]).tolist())
    print("hidden from the agent:", sorted(set(env.oracle_fields) - set(info)))
