"""Tutorial step 1: the shape of a Perpetua environment.

Every method below is filled in by step 2. An environment is a pure function of
state and parameters; there is no ``reset`` and no ``done``.
"""

from typing import Any, NamedTuple

import jax
from gymnasium import spaces

import perpetua as pp
from perpetua.core import EnvInfo


class BerryParams(NamedTuple):
    """Everything that may drift over the lifetime, as float arrays."""


class BerryState(NamedTuple):
    """The world only. The framework owns drift, its RNG stream and the clock."""


class BerryField:
    """One forager on a grid collects a berry and respawns, forever."""

    num_agents: int
    observation_space: spaces.Box
    action_space: spaces.MultiDiscrete
    metadata: pp.Metadata

    @property
    def default_params(self) -> BerryParams:
        """Starting values of every drifting parameter."""
        raise NotImplementedError

    def derive(self, params: BerryParams) -> BerryParams:
        """Turn raw drift values into the effective parameters (optional)."""
        raise NotImplementedError

    def init(self, key: jax.Array, params: BerryParams) -> tuple[BerryState, jax.Array, EnvInfo]:
        """Start the lifetime; called exactly once."""
        raise NotImplementedError

    def step(
        self, state: BerryState, action: Any, key: jax.Array, params: BerryParams
    ) -> tuple[BerryState, jax.Array]:
        """Advance the world one tick and return the reward."""
        raise NotImplementedError

    def observe(self, state: BerryState, params: BerryParams) -> tuple[jax.Array, EnvInfo]:
        """What the agent sees, plus action/active masks and respawn events."""
        raise NotImplementedError

    def privileged(self, state: BerryState) -> dict[str, jax.Array]:
        """World-side diagnostics only an Oracle may expose."""
        raise NotImplementedError


if __name__ == "__main__":
    print("BerryField methods:", [name for name in vars(BerryField) if not name.startswith("_")])
