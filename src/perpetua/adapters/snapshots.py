"""Versioned numeric snapshots for explicitly supported Gymnasium backends."""

# This version-pinned codec must access Gymnasium's private reset/wrapper state.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from importlib.metadata import version
from typing import Any, NamedTuple, cast

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker, TimeLimit
from numpy.typing import NDArray

__all__ = [
    "CartPoleSnapshot",
    "EnvPoolSnapshot",
    "MPE2Snapshot",
    "ReplaySnapshot",
    "BackendSnapshot",
]


class CartPoleSnapshot(NamedTuple):
    """Numeric CartPole state; all leaves are NumPy arrays, never Python objects.

    Scalars: uint32 version, int64 RNG seed and post-termination counter.
    Physics (4,) and parameters (10,) are float64; options (2,) are int64.
    PCG64 state/increment limbs plus cache flags occupy uint32 RNG (10,).
    For N wrappers: int64 configuration (N, 2), int64 flags/counters (N, 5),
    float32 previous observations (N, 4) and bool presence mask (N,).
    """

    version: NDArray[np.uint32]
    physics: NDArray[np.float64]
    parameters: NDArray[np.float64]
    options: NDArray[np.int64]
    rng: NDArray[np.uint32]
    rng_seed: NDArray[np.int64]
    steps_beyond_terminated: NDArray[np.int64]
    wrapper_configuration: NDArray[np.int64]
    wrappers: NDArray[np.int64]
    previous_obs: NDArray[np.float32]
    previous_present: NDArray[np.bool_]


class EnvPoolSnapshot(NamedTuple):
    """Numeric, quiescent EnvPool CartPole state for N synchronous CPU slots.

    Configuration is uint32 (4,): codec version, slots, threads, episode limit.
    Physics is float64 (N, 4); RNG is uint32 (N, 625), including each MT19937
    index. Int64 counters (N, 5) contain seed, base/physics step counts, done and
    force-reset flags. Bool pending (N,) belongs to Perpetua's reset-credit lift.
    """

    configuration: NDArray[np.uint32]
    physics: NDArray[np.float64]
    rng: NDArray[np.uint32]
    counters: NDArray[np.int64]
    pending: NDArray[np.bool_]


class MPE2Snapshot(NamedTuple):
    """Headless simple_spread snapshot; all fields are owned numeric NumPy arrays.

    For N agents/N landmarks: float64 physics (2*N, 4), agent communication and
    action state (N, 6), bool action-presence (N, 2), int64 queued actions (N,),
    uint32 PCG64 words (10,), int32 agent lists (2, N)/selector order (N,), int64
    counters (6,), float64 rewards/cumulative rewards (N, 2), bool done flags
    (N, 2)/dictionary presence (N, 5), bool wrapper state (wrappers, 2), scalar
    float64 camera range, bool active (N,) and action masks (N, actions).
    Configuration is canonical uint8 JSON; version is scalar uint32.
    """

    version: NDArray[np.uint32]
    configuration: NDArray[np.uint8]
    physics: NDArray[np.float64]
    agent_state: NDArray[np.float64]
    action_present: NDArray[np.bool_]
    current_actions: NDArray[np.int64]
    rng: NDArray[np.uint32]
    agents: NDArray[np.int32]
    selector: NDArray[np.int32]
    counters: NDArray[np.int64]
    rewards: NDArray[np.float64]
    done: NDArray[np.bool_]
    present: NDArray[np.bool_]
    wrappers: NDArray[np.bool_]
    camera: NDArray[np.float64]
    active: NDArray[np.bool_]
    mask: NDArray[np.bool_]


class ReplaySnapshot(NamedTuple):
    """A deterministic simulator restored by replaying its seeded action log.

    Scalars: uint32 version, int64 seed and int64 length (ticks logged). The
    float32 log is a fixed ``(capacity, width)`` buffer whose first ``length``
    rows are the actions taken; fixed capacity keeps checkpoint shapes stable.
    The buffer is shared append-only between successive states, and
    copied before any branch writes to it.
    """

    version: NDArray[np.uint32]
    seed: NDArray[np.int64]
    length: NDArray[np.int64]
    actions: NDArray[np.float32]


type BackendSnapshot = CartPoleSnapshot | EnvPoolSnapshot | MPE2Snapshot | ReplaySnapshot


_PARAMETERS = (
    "gravity",
    "masscart",
    "masspole",
    "total_mass",
    "length",
    "polemass_length",
    "force_mag",
    "tau",
    "theta_threshold_radians",
    "x_threshold",
)
_WRAPPERS = (TimeLimit, OrderEnforcing, PassiveEnvChecker)


def _words(value: int) -> list[int]:
    if not 0 <= value < 2**128:
        raise ValueError("PCG64 state must fit in 128 unsigned bits.")
    return [(value >> (32 * index)) & 0xFFFFFFFF for index in range(4)]


def _integer(words: NDArray[np.uint32]) -> int:
    return sum(int(word) << (32 * index) for index, word in enumerate(words))


class CartPoleCodec:
    """Snapshot/restore headless CartPole with known wrappers and PCG64 RNG.

    This codec deliberately rejects subclasses, unknown wrappers, render
    resources and unsupported RNGs rather than claiming generic serialization.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        if not self.supports(env):
            raise ValueError("Unsupported CartPole snapshot backend or wrapper chain.")
        self.env = env
        self.base = cast(Any, env.unwrapped)
        self.wrappers: list[Any] = []
        current = env
        while isinstance(current, gym.Wrapper):
            self.wrappers.append(cast(Any, current))
            current = current.env

    @staticmethod
    def supports(env: gym.Env[Any, Any]) -> bool:
        """Headless ``CartPoleEnv`` in known wrappers only, on exactly gymnasium 1.4.0."""
        if type(env.unwrapped) is not CartPoleEnv or env.unwrapped.render_mode is not None:
            return False
        if version("gymnasium") != "1.4.0":
            return False
        current = env
        while isinstance(current, gym.Wrapper):
            if type(cast(Any, current)) not in _WRAPPERS:
                return False
            current = current.env
        return True

    @property
    def configuration(self) -> dict[str, Any]:
        """Static backend identity/settings checked by the checkpoint manifest."""
        wrappers: list[dict[str, Any]] = []
        for wrapper in self.wrappers:
            config: dict[str, Any] = {"type": type(wrapper).__name__}
            if type(wrapper) is TimeLimit:
                config["max_episode_steps"] = int(wrapper._max_episode_steps)
            if type(wrapper) is OrderEnforcing:
                config["disable_render_order_enforcing"] = bool(
                    wrapper._disable_render_order_enforcing
                )
            wrappers.append(config)
        return {
            "codec": "gymnasium-cartpole-pcg64-v1",
            "gymnasium": version("gymnasium"),
            "parameters": {name: float(getattr(self.base, name)) for name in _PARAMETERS},
            "integrator": self.base.kinematics_integrator,
            "sutton_barto_reward": bool(self.base._sutton_barto_reward),
            "wrappers": wrappers,
        }

    def snapshot(self) -> CartPoleSnapshot:
        """Copy all dynamics/reset state without changing the NumPy RNG stream."""
        if self.base.state is None:
            raise RuntimeError("Initialize CartPole before capturing a snapshot.")
        rng = self.base.np_random.bit_generator.state
        if rng["bit_generator"] != "PCG64":
            raise NotImplementedError("The CartPole snapshot codec supports PCG64 only.")
        if self.base.kinematics_integrator not in {"euler", "semi-implicit euler"}:
            raise ValueError("Unsupported CartPole integration mode.")
        flags = np.zeros((len(self.wrappers), 5), dtype=np.int64)
        previous = np.zeros((len(self.wrappers), 4), dtype=np.float32)
        present = np.zeros(len(self.wrappers), dtype=np.bool_)
        for index, wrapper in enumerate(self.wrappers):
            if type(wrapper) is TimeLimit:
                flags[index, 0] = -1 if wrapper._elapsed_steps is None else wrapper._elapsed_steps
            elif type(wrapper) is OrderEnforcing:
                flags[index, 0] = int(wrapper._has_reset)
            else:
                flags[index] = [
                    wrapper.checked_reset,
                    wrapper.checked_step,
                    wrapper.checked_render,
                    wrapper.close_called,
                    wrapper.checked_data_reuse,
                ]
                if wrapper._previous_data is not None:
                    obs, info = wrapper._previous_data
                    if info:
                        raise ValueError("Unsupported checker diagnostic state.")
                    previous[index] = np.asarray(obs, dtype=np.float32)
                    present[index] = True
        words = _words(int(rng["state"]["state"])) + _words(int(rng["state"]["inc"]))
        words += [int(rng["has_uint32"]), int(rng["uinteger"])]
        return CartPoleSnapshot(
            np.asarray(1, dtype=np.uint32),
            np.asarray(self.base.state, dtype=np.float64).copy(),
            np.asarray([getattr(self.base, name) for name in _PARAMETERS], dtype=np.float64),
            np.asarray(
                [self.base.kinematics_integrator != "euler", self.base._sutton_barto_reward],
                dtype=np.int64,
            ),
            np.asarray(words, dtype=np.uint32),
            np.asarray(self.base.np_random_seed, dtype=np.int64),
            np.asarray(
                -1
                if self.base.steps_beyond_terminated is None
                else self.base.steps_beyond_terminated,
                dtype=np.int64,
            ),
            self._wrapper_configuration(),
            flags,
            previous,
            present,
        )

    def restore(self, snapshot: CartPoleSnapshot) -> None:
        """Validate numeric data before applying it to this compatible backend."""
        if type(snapshot) is not CartPoleSnapshot:
            raise ValueError("Unsupported CartPole snapshot version.")
        # Runtime-loaded fields must be validated despite the declared snapshot type.
        version_value = cast(object, snapshot.version)
        if (
            not isinstance(version_value, np.ndarray)
            or version_value.shape != ()
            or version_value.dtype != np.dtype(np.uint32)
            or int(cast(NDArray[np.uint32], version_value)) != 1
        ):
            raise ValueError("Unsupported CartPole snapshot version.")
        expected = (
            ((), np.uint32),
            ((4,), np.float64),
            ((10,), np.float64),
            ((2,), np.int64),
            ((10,), np.uint32),
            ((), np.int64),
            ((), np.int64),
            ((len(self.wrappers), 2), np.int64),
            ((len(self.wrappers), 5), np.int64),
            ((len(self.wrappers), 4), np.float32),
            ((len(self.wrappers),), np.bool_),
        )
        for value, (shape, dtype) in zip(cast(tuple[object, ...], snapshot), expected, strict=True):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.dtype(dtype)
            ):
                raise ValueError("CartPole snapshot shape/dtype does not match the backend.")
        if not np.array_equal(snapshot.wrapper_configuration, self._wrapper_configuration()):
            raise ValueError("CartPole snapshot wrapper configuration does not match the backend.")
        if not np.all(np.isfinite(snapshot.physics)) or not np.all(
            np.isfinite(snapshot.parameters)
        ):
            raise ValueError("CartPole snapshot physics/parameters must be finite.")
        if np.any((snapshot.options < 0) | (snapshot.options > 1)) or snapshot.rng[8] > 1:
            raise ValueError("Invalid CartPole snapshot options/RNG flags.")
        if int(snapshot.steps_beyond_terminated) < -1 or int(snapshot.rng_seed) < -1:
            raise ValueError("Invalid CartPole snapshot counters/seed.")
        for index, wrapper in enumerate(self.wrappers):
            flags = snapshot.wrappers[index]
            if type(wrapper) is TimeLimit:
                valid = flags[0] >= -1 and np.all(flags[1:] == 0)
            elif type(wrapper) is OrderEnforcing:
                valid = 0 <= flags[0] <= 1 and np.all(flags[1:] == 0)
            else:
                valid = np.all((flags >= 0) & (flags <= 1))
            if not valid:
                raise ValueError("Invalid CartPole wrapper snapshot flags.")
        rng = {
            "bit_generator": "PCG64",
            "state": {"state": _integer(snapshot.rng[:4]), "inc": _integer(snapshot.rng[4:8])},
            "has_uint32": int(snapshot.rng[8]),
            "uinteger": int(snapshot.rng[9]),
        }
        generator = np.random.Generator(np.random.PCG64(0))
        generator.bit_generator.state = rng
        self.base.state = snapshot.physics.copy()
        for name, value in zip(_PARAMETERS, snapshot.parameters, strict=True):
            setattr(self.base, name, float(value))
        self.base.kinematics_integrator = "semi-implicit euler" if snapshot.options[0] else "euler"
        self.base._sutton_barto_reward = bool(snapshot.options[1])
        self.base._np_random = generator
        self.base._np_random_seed = int(snapshot.rng_seed)
        beyond = int(snapshot.steps_beyond_terminated)
        self.base.steps_beyond_terminated = None if beyond == -1 else beyond
        for index, wrapper in enumerate(self.wrappers):
            flags = snapshot.wrappers[index]
            if type(wrapper) is TimeLimit:
                wrapper._elapsed_steps = None if flags[0] == -1 else int(flags[0])
            elif type(wrapper) is OrderEnforcing:
                wrapper._has_reset = bool(flags[0])
            else:
                (
                    wrapper.checked_reset,
                    wrapper.checked_step,
                    wrapper.checked_render,
                    wrapper.close_called,
                    wrapper.checked_data_reuse,
                ) = map(bool, flags)
                wrapper._previous_data = (
                    (snapshot.previous_obs[index].copy(), {})
                    if snapshot.previous_present[index]
                    else None
                )

    def _wrapper_configuration(self) -> NDArray[np.int64]:
        configuration = np.zeros((len(self.wrappers), 2), dtype=np.int64)
        for index, wrapper in enumerate(self.wrappers):
            configuration[index, 0] = _WRAPPERS.index(type(wrapper))
            if type(wrapper) is TimeLimit:
                configuration[index, 1] = wrapper._max_episode_steps
            elif type(wrapper) is OrderEnforcing:
                configuration[index, 1] = wrapper._disable_render_order_enforcing
        return configuration
