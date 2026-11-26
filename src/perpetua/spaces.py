"""Stable compiled shapes with evolving action semantics.

Perpetua lifetimes never change observation or action *shapes*: an evolving
space changes what an action *means*, through masks, and announces the change
with a public event. This module owns the static descriptors, validates
stability across configurations and derives the event from consecutive masks.
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.typing import ArrayLike

from perpetua.core import JSONValue

__all__ = [
    "space_signature",
    "validate_stable",
    "mask_change",
    "SpaceManager",
]


def space_signature(space: spaces.Space[object]) -> dict[str, JSONValue]:
    """Summarize the parts of a space that must stay fixed during a lifetime.

    Shapes, dtypes and discrete cardinalities are recorded; bounds and
    semantics may evolve (masks carry what changed).
    """
    shape = space.shape if space.shape is not None else ()
    signature: dict[str, JSONValue] = {
        "type": type(space).__name__,
        "shape": [int(dimension) for dimension in shape],
        "dtype": str(np.dtype(space.dtype)) if space.dtype is not None else "none",
    }
    if isinstance(space, spaces.Discrete):
        signature["n"] = int(cast(Any, space).n)
    if isinstance(space, spaces.Dict):
        children = cast("dict[str, spaces.Space[object]]", cast(Any, space).spaces)
        signature["spaces"] = {
            str(name): space_signature(child) for name, child in children.items()
        }
    if isinstance(space, spaces.Tuple):
        members = cast("tuple[spaces.Space[object], ...]", cast(Any, space).spaces)
        signature["spaces"] = [space_signature(child) for child in members]
    nvec = getattr(cast(Any, space), "nvec", None)
    if nvec is not None:
        signature["nvec"] = [int(value) for value in np.asarray(nvec).ravel()]
    return signature


def validate_stable(
    reference: Mapping[str, spaces.Space[object]],
    current: Mapping[str, spaces.Space[object]],
) -> None:
    """Raise if any space's shape, dtype or cardinality changed.

    Evolving spaces keep identical compiled shapes; semantics change through
    masks and events, never by resizing a descriptor mid-lifetime.
    """
    if set(reference) != set(current):
        raise ValueError("Space names must match between configurations.")
    for name, space in reference.items():
        if space_signature(space) != space_signature(current[name]):
            raise ValueError(
                f"Space {name!r} changed shape, dtype or cardinality; "
                "evolving spaces must keep compiled shapes fixed."
            )


def mask_change(previous: ArrayLike, current: ArrayLike) -> jax.Array:
    """Per-agent bool: this mask differs from the previous one.

    A pure device-side function over two public ``(agents, actions)`` masks,
    so the event exposes nothing beyond what policies already observe.
    """
    before = jnp.asarray(previous)
    after = jnp.asarray(current)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("Masks must have matching (agents, actions) shapes.")
    return jnp.any(before != after, axis=-1)


class SpaceManager:
    """Own a lifetime's static spaces and validate evolving semantics.

    The manager records the observation/action space signatures of a lifetime
    (JSON-ready for checkpoint metadata), refuses configurations whose compiled
    shapes differ, and can derive a space-change event from two masks. It is
    an optional helper: :class:`~perpetua.params.DriftEnv` computes
    ``info['space_change']`` itself, by comparing each agent's mask under the
    previous and current parameters through :func:`mask_change`. It performs
    no per-tick host work.
    """

    def __init__(
        self, *, observation_space: spaces.Space[object], action_space: spaces.Space[object]
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self._signature: dict[str, JSONValue] = {
            "observation": space_signature(observation_space),
            "action": space_signature(action_space),
        }

    @property
    def configuration(self) -> dict[str, JSONValue]:
        """JSON-ready space signature recorded with checkpoints and metadata."""
        return {"spaces": self._signature, "evolving": "action-mask"}

    def check(
        self, *, observation_space: spaces.Space[object], action_space: spaces.Space[object]
    ) -> None:
        """Raise if a new pair of descriptors changes the compiled shapes."""
        validate_stable(
            {"observation": self.observation_space, "action": self.action_space},
            {"observation": observation_space, "action": action_space},
        )

    def event(self, previous_mask: ArrayLike, current_mask: ArrayLike) -> jax.Array:
        """Public space-change event for one transition (device-side bools).

        True per agent when the valid-action set changed. Pure function of
        public masks.
        """
        return mask_change(previous_mask, current_mask)
