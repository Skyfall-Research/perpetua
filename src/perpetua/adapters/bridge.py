"""A compiled CPU bridge: host simulators inside one compiled lifetime program.

The plain host lift runs every tick as eager Python: drift, key splits, metric
updates and the policy are separate device dispatches around the simulator
call, which makes it dispatch-bound. Here the simulator call
is a ``jax.experimental.io_callback``, so the whole tick, and a whole
learner rollout, compiles into one program and only the batched simulator step
leaves the device. The ordinary compiled runners (``Runner``, ``PPORunner``,
``DQNRunner``, ``SACRunner``) then drive host simulators unchanged.

The simulator is still one mutable object: states are single-use (a stale or
forked state is refused by a generation check inside the callback), lifetimes
cannot be vmapped or checkpointed, and batching comes from the backend's own
slots (for example EnvPool's ``num_envs``).
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

from typing import Any, NamedTuple, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.typing import ArrayLike
from numpy.typing import NDArray

from perpetua.adapters.host import (
    CPUBackend,
    CPUWorld,
    HostParams,
    _initial_active,  # pyright: ignore[reportPrivateUsage]
    _seed,  # pyright: ignore[reportPrivateUsage]
)
from perpetua.core import EnvInfo

# Reached through ``compiled=True`` on the host lifetimes; nothing here is public.
__all__: list[str] = []


class CallbackState(NamedTuple):
    """The latest batch plus the generation the simulator must be at."""

    generation: jax.Array
    obs: jax.Array
    active: jax.Array
    respawn: jax.Array


class CallbackWorld(CPUWorld):
    """A :class:`~perpetua.adapters.host.CPUWorld` whose step is traceable."""

    def __init__(self, backend: CPUBackend, *, env_id: str) -> None:
        try:
            jax.devices("cpu")
        except RuntimeError as error:
            raise RuntimeError(
                "compiled=True runs the simulator in a host callback, which needs JAX's CPU "
                "platform: include cpu in JAX_PLATFORMS, for example JAX_PLATFORMS=cuda,cpu."
            ) from error
        super().__init__(backend, env_id=env_id)
        self.metadata = {
            **self.metadata,
            "execution": "jax",
            "checkpointable": False,
            "forkable": False,
            "configuration": {
                **self.metadata.get("configuration", {}),
                "bridge": "io_callback; single-use state; not vmappable",
            },
        }
        shape = (self.num_agents, *backend.observation_space.shape)
        self._results = (
            jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct((self.num_agents,), jnp.float32),
            jax.ShapeDtypeStruct((self.num_agents,), jnp.bool_),
            jax.ShapeDtypeStruct((self.num_agents,), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.uint32),
        )
        self._steps = 0

    def init(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: jax.Array, params: HostParams
    ) -> tuple[CallbackState, jax.Array, EnvInfo]:
        """Initialize the simulator once (eagerly) with a seed derived from ``key``."""
        if self._closed or self._initialized:
            raise RuntimeError("Create a new CPU backend for a new lifetime.")
        seed_key, _, _ = jax.random.split(key, 3)
        obs = self.backend.init(_seed(seed_key))
        self._initialized = True
        state = CallbackState(
            jnp.uint32(0),
            jnp.asarray(np.array(obs, dtype=np.float32, copy=True)),
            _initial_active(self.backend, self.num_agents),
            jnp.zeros(self.num_agents, dtype=jnp.bool_),
        )
        obs_out, info = self.observe(state, params)
        return state, obs_out, info

    def observe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: CallbackState, params: HostParams
    ) -> tuple[jax.Array, EnvInfo]:
        """Expose the latest batch; inactive reset slots select placeholder action zero."""
        del params
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            mask = jnp.broadcast_to(
                state.active[:, None], (self.num_agents, int(self.action_space.nvec[0]))
            )
        else:
            mask = jnp.ones((self.num_agents, 1), dtype=jnp.bool_)
        mask = mask.at[:, 0].set(True)
        return state.obs, {
            "action_mask": mask,
            "active_mask": state.active,
            "respawn": state.respawn,
        }

    def _host_step(
        self, action: NDArray[Any], reset_seed: NDArray[Any], generation: NDArray[Any]
    ) -> tuple[NDArray[Any], ...]:
        if self._closed or not self._initialized:
            raise RuntimeError("CPU backend is not running.")
        # The state's counter is uint32 and wraps; compare modulo 2**32.
        if int(generation) != self._steps % 2**32:
            raise ValueError(
                "CPU state is stale or forked: this simulator is at generation "
                f"{self._steps % 2**32}, the state at {int(generation)}."
            )
        actions = self._actions(np.asarray(action))
        # Advance before stepping, as CPUWorld.step does: if the simulator
        # fails part-way, the consumed state must not be accepted again.
        self._steps += 1
        result = self.backend.step(actions, int(reset_seed))
        return (
            np.asarray(result.obs, dtype=np.float32).reshape(self._results[0].shape),
            np.asarray(result.reward, dtype=np.float32).reshape(-1),
            np.asarray(result.active, dtype=np.bool_).reshape(-1),
            np.asarray(result.respawn, dtype=np.bool_).reshape(-1),
            np.asarray(self._steps % 2**32, dtype=np.uint32),
        )

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: CallbackState, action: ArrayLike, key: jax.Array, params: HostParams
    ) -> tuple[CallbackState, jax.Array]:
        """Step the simulator through a host callback; traceable.

        Calls are sequenced by data, not by an effect token: the callback
        returns the next generation, which the next call consumes, so each
        call depends on the one before it even when the actions do not. An
        ordered callback would poison every later call in the process once
        one raised, which a refused stale state must not do.
        """
        if self.backend.uses_reset_seed:
            reset_seed = jax.random.bits(key, (), jnp.uint32) & jnp.uint32(0x7FFFFFFF)
        else:
            reset_seed = jnp.uint32(0)
        obs, reward, active, respawn, generation = cast(
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            io_callback(
                self._host_step,
                self._results,
                jnp.asarray(action),
                reset_seed,
                state.generation,
                ordered=False,
            ),
        )
        next_state = CallbackState(generation, obs, active, respawn)
        return next_state, reward * params.reward_multiplier[0]

    def privileged(self, state: CallbackState) -> dict[str, jax.Array]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}
