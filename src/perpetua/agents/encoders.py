"""A small convolutional torso for pixel observations (MinAtar and Atari).

The torso turns ``(batch, H, W, C)`` observations into ``(batch, features)``
vectors, so the existing actor/critic and Q heads work unchanged. It is a
Nature-CNN-style encoder sized for MinAtar's 10x10 frames, not a claim about
Atari-scale performance.
"""

# JAX/Flax annotations are partial.
# pyright: reportUnknownMemberType=false

import math
import numbers
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from perpetua.agents.plasticity import crelu

__all__ = [
    "Activation",
    "Torso",
    "CNNTorso",
]

#: Hidden-layer activations of the PPO and DQN networks.
type Activation = Literal["tanh", "crelu"]
#: Observation encoders of the PPO and DQN networks.
type Torso = Literal["mlp", "cnn"]


def activate(value: jax.Array, activation: Activation) -> jax.Array:
    """The same activation choice as the MLP torsos (tanh or CReLU)."""
    return crelu(value) if activation == "crelu" else nn.tanh(value)


def check_pixel_scale(pixel_scale: object) -> None:
    """Refuse a configured ``pixel_scale`` other than ``None`` or a finite positive number."""
    if pixel_scale is None:
        return
    if not isinstance(pixel_scale, bool) and isinstance(pixel_scale, numbers.Real):
        value = float(pixel_scale)
        if math.isfinite(value) and value > 0:
            return
    raise ValueError("pixel_scale must be None or a finite positive number.")


def derive_pixel_scale(observation_space: spaces.Space[Any]) -> float:
    """What an observation space says its values should be divided by: 255 or 1.

    255 for raw 8-bit pixels, which is a ``Box`` whose bounds are 0 and 255 in
    every entry, whatever its dtype (EnvPool Atari's uint8 frames are lifted
    to float32, and AgarCL declares its screen as float32). 1 for every other
    space: bounds of 0 and 1 (MinAtar), unbounded spaces (MEAL Overcooked,
    COOM), or a ``uint8`` space with other bounds.
    """
    if not isinstance(observation_space, spaces.Box):
        return 1.0
    pixels = np.all(observation_space.low == 0) and np.all(observation_space.high == 255)
    return 255.0 if pixels else 1.0


def resolve_pixel_scale(
    configured: float | None, observation_space: spaces.Space[Any] | None
) -> float:
    """The divisor a learner's CNN torso uses for its whole life.

    A configured number wins. Otherwise it is derived once from the
    observation space (:func:`derive_pixel_scale`), and is 1 without one.
    """
    if configured is not None:
        return float(configured)
    return 1.0 if observation_space is None else derive_pixel_scale(observation_space)


class CNNTorso(nn.Module):
    """Three 3x3 convolutions, then a dense feature layer.

    Two of the convolutions stride by 2, so a 10x10 frame becomes 3x3 before
    flattening. ``hidden_size`` is the output feature width; CReLU doubles it,
    exactly as in the MLP torso.

    Observations are channel-last, ``(..., H, W, C)``, and are divided by
    ``pixel_scale``. The scale is fixed when the module is built and never
    depends on the values of a frame: 255 brings raw 8-bit pixels into
    [0, 1], and the default of 1 passes the input through unchanged. PPO and
    DQN choose it from their config's ``pixel_scale`` and the environment's
    observation space (:func:`resolve_pixel_scale`).
    """

    hidden_size: int
    activation: Activation = "tanh"
    pixel_scale: float = 1.0

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.astype(jnp.float32)
        if x.ndim < 4:
            raise ValueError("The CNN torso expects (batch, H, W, C) observations.")
        if self.pixel_scale != 1.0:
            x = x / self.pixel_scale
        for features, strides in ((16, 1), (32, 2), (32, 2)):
            x = nn.Conv(
                features,
                (3, 3),
                strides=(strides, strides),
                padding="SAME",
                kernel_init=nn.initializers.orthogonal(math.sqrt(2)),
                bias_init=nn.initializers.zeros_init(),
            )(x)
            x = activate(x, self.activation)
        # Preserve every leading axis (batch, agent slots, ...) and flatten
        # only the spatial/channel axes.
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2)),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        return activate(x, self.activation)
