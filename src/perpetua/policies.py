"""Stateless reference policies restricted to agent-visible observations/info."""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp

from perpetua.core import Info

__all__ = [
    "Policy",
    "random",
    "uniform",
    "greedy",
]

type Policy = Callable[[jax.Array, Info, jax.Array], jax.Array]


def random(obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
    """Sample uniformly among valid actions using a fresh policy key."""
    del obs
    logits = jnp.where(info["action_mask"], 0.0, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def uniform(space: gym.spaces.Space[Any]) -> Policy:
    """A uniform random policy for a lifetime's action space.

    Discrete (``MultiDiscrete``) slots sample among valid actions like
    :func:`random`; continuous (``Box``) slots sample uniformly within bounds.
    """
    if isinstance(space, gym.spaces.MultiDiscrete):
        return random
    if not isinstance(space, gym.spaces.Box) or not space.is_bounded():
        raise ValueError("uniform needs a MultiDiscrete or bounded Box action space.")
    low = jnp.asarray(space.low, dtype=jnp.float32)
    high = jnp.asarray(space.high, dtype=jnp.float32)

    def sample(obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
        del obs, info
        return jax.random.uniform(key, low.shape, minval=low, maxval=high)

    return sample


def greedy(obs: jax.Array, info: Info, key: jax.Array) -> jax.Array:
    """Move toward the observed grid goal, prioritizing row over column."""
    del key
    row = obs[:, 2] - obs[:, 0]
    column = obs[:, 3] - obs[:, 1]
    action = jnp.where(
        row > 0, 2, jnp.where(row < 0, 1, jnp.where(column > 0, 4, jnp.where(column < 0, 3, 0)))
    )
    valid = info["action_mask"][jnp.arange(obs.shape[0]), action]
    return jnp.where(valid & info["active_mask"], action, 0).astype(jnp.int32)
