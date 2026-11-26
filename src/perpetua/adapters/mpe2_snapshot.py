"""Version-pinned numeric codec for headless MPE2 simple_spread parallel worlds."""

# The complete codec intentionally reads pinned private AEC/physics bookkeeping.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

import json
from importlib.metadata import version
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pettingzoo.utils.conversions import aec_to_parallel_wrapper  # pyright: ignore[reportMissingTypeStubs]
from pettingzoo.utils.wrappers import (  # pyright: ignore[reportMissingTypeStubs]
    AssertOutOfBoundsWrapper,
    OrderEnforcingWrapper,
)

from perpetua.adapters.snapshots import MPE2Snapshot, _integer, _words

# Reached through ``PettingZooLifetime(snapshot=True)``; nothing here is public.
__all__: list[str] = []


class MPE2Codec:
    """Capture real float64 physics, owned PCG64 and known parallel/AEC state.

    Noise using process-global RNG, scripted callbacks, unknown wrappers,
    benchmark-info payloads and rendering are deliberately unsupported.
    Static settings are checked before capture/restore, never silently replaced.
    """

    def __init__(self, env: Any) -> None:
        if not self.supports(env):
            raise ValueError("Unsupported MPE2 snapshot environment/settings/wrappers.")
        self.env = env
        self.base = env.unwrapped
        self.names: tuple[str, ...] = tuple(self.base.possible_agents)
        self.n = len(self.names)
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.wrappers: list[Any] = []
        current = env.aec_env
        while current is not self.base:
            self.wrappers.append(current)
            current = current.env
        self.configuration = self._configuration()
        self._signature = np.frombuffer(
            json.dumps(self.configuration, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8
        ).copy()

    @staticmethod
    def supports(env: Any) -> bool:
        """Headless discrete MPE2 simple_spread on exactly mpe2 1.1.1 with pettingzoo 1.27.0."""
        from mpe2._mpe_utils.core import Agent, Landmark  # pyright: ignore[reportMissingTypeStubs]
        from mpe2.simple_spread.simple_spread import (  # pyright: ignore[reportMissingTypeStubs]
            ExtendedWorld,
            Scenario,
            raw_env,
        )

        if version("mpe2") != "1.1.1" or version("pettingzoo") != "1.27.0":
            return False
        if type(env) is not aec_to_parallel_wrapper:
            return False
        current = cast(Any, env.aec_env)
        while type(current) in {OrderEnforcingWrapper, AssertOutOfBoundsWrapper}:
            current = current.env
        if type(current) is not raw_env or type(current.scenario) is not Scenario:
            return False
        if (
            type(current.world) is not ExtendedWorld
            or current.render_mode is not None
            or current.renderOn
            or current.continuous_actions
            or current.benchmark_data
        ):
            return False
        world = current.world
        if (world.dim_p, world.dim_c, world.dim_color) != (2, 2, 3):
            return False
        if len(world.agents) != len(world.landmarks) or not world.agents:
            return False
        if any(
            type(agent) is not Agent
            or agent.action_callback is not None
            or agent.u_noise not in {None, 0.0}
            or agent.c_noise not in {None, 0.0}
            or not agent.movable
            or not agent.silent
            for agent in world.agents
        ):
            return False
        if any(type(entity) is not Landmark for entity in world.landmarks):
            return False
        return type(current.np_random.bit_generator) is np.random.PCG64

    def _configuration(self) -> dict[str, Any]:
        world, scenario = self.base.world, self.base.scenario
        entities = []
        for entity in world.entities:
            settings = {
                name: getattr(entity, name)
                for name in (
                    "name",
                    "size",
                    "movable",
                    "collide",
                    "density",
                    "initial_mass",
                    "max_speed",
                    "accel",
                )
            }
            if entity in world.agents:
                settings.update(
                    {
                        name: getattr(entity, name)
                        for name in (
                            "silent",
                            "blind",
                            "u_noise",
                            "c_noise",
                            "u_range",
                        )
                    }
                )
            entities.append(settings)
        return {
            "codec": "mpe2-simple-spread-pcg64-v1",
            "mpe2": version("mpe2"),
            "pettingzoo": version("pettingzoo"),
            "agents": list(self.base.possible_agents),
            "wrappers": [type(wrapper).__name__ for wrapper in self.wrappers],
            "max_cycles": self.base.max_cycles,
            "local_ratio": self.base.local_ratio,
            "dynamic_rescaling": self.base.dynamic_rescaling,
            "curriculum": scenario.curriculum,
            "terminate_on_success": scenario.terminate_on_success,
            "num_agent_neighbors": scenario.num_agent_neighbors,
            "num_landmark_neighbors": scenario.num_landmark_neighbors,
            "world": {
                name: getattr(world, name)
                for name in (
                    "dim_p",
                    "dim_c",
                    "dim_color",
                    "dt",
                    "damping",
                    "contact_force",
                    "contact_margin",
                    "collaborative",
                )
            },
            "entities": entities,
        }

    def _running(self) -> None:
        if not hasattr(self.base, "agent_selection"):
            raise RuntimeError("Initialize MPE2 before snapshot/restore.")
        if not self.supports(self.env) or self._configuration() != self.configuration:
            raise ValueError("MPE2 snapshot configuration changed after construction.")

    def _order(self, names: list[str]) -> NDArray[np.int32]:
        if (
            len(names) > self.n
            or len(set(names)) != len(names)
            or not set(names).issubset(self.indices)
        ):
            raise ValueError("Invalid MPE2 snapshot agent ordering.")
        result = np.full(self.n, -1, dtype=np.int32)
        result[: len(names)] = [self.indices[name] for name in names]
        return result

    def _array(self, value: object) -> NDArray[np.float64]:
        if not isinstance(value, np.ndarray) or value.shape != (2,) or value.dtype != np.float64:
            raise ValueError("MPE2 snapshot dynamics must retain float64 two-vectors.")
        return cast(NDArray[np.float64], value)

    def snapshot(self, active: NDArray[np.bool_], mask: NDArray[np.bool_]) -> MPE2Snapshot:
        """Copy all known numeric state without advancing simulator or RNG."""
        self._running()
        physics = np.stack(
            [
                np.concatenate((self._array(entity.state._p_pos), self._array(entity.state._p_vel)))
                for entity in self.base.world.entities
            ]
        )
        agent_state = np.zeros((self.n, 6), dtype=np.float64)
        action_present = np.zeros((self.n, 2), dtype=np.bool_)
        for i, agent in enumerate(self.base.world.agents):
            agent_state[i, :2] = self._array(agent.state._c)
            for j, value in enumerate((agent.action._u, agent.action._c)):
                if value is not None:
                    agent_state[i, 2 + 2 * j : 4 + 2 * j] = self._array(value)
                    action_present[i, j] = True
        current_actions = np.asarray(
            [-1 if value is None else value for value in self.base.current_actions], dtype=np.int64
        )
        rng = self.base.np_random.bit_generator.state
        words = _words(int(rng["state"]["state"])) + _words(int(rng["state"]["inc"]))
        words += [int(rng["has_uint32"]), int(rng["uinteger"])]
        rewards = np.zeros((self.n, 2), dtype=np.float64)
        done = np.zeros((self.n, 2), dtype=np.bool_)
        present = np.zeros((self.n, 5), dtype=np.bool_)
        for i, name in enumerate(self.names):
            for j, values in enumerate(
                (
                    self.base.rewards,
                    self.base._cumulative_rewards,
                    self.base.terminations,
                    self.base.truncations,
                    self.base.infos,
                )
            ):
                present[i, j] = name in values
                if name in values:
                    if j < 2:
                        rewards[i, j] = values[name]
                    elif j < 4:
                        done[i, j - 2] = values[name]
                    elif values[name]:
                        raise ValueError("Unsupported MPE2 snapshot info payload.")
        selector = self.base._agent_selector
        skip = self.base.__dict__.get("_skip_agent_selection", "absent")
        skip_index = -2 if skip == "absent" else -1 if skip is None else self.indices[skip]
        counters = np.asarray(
            [
                self.base.steps,
                selector._current_agent,
                self.indices[selector.selected_agent],
                self.indices[self.base.agent_selection],
                skip_index,
                self.base.scenario.curriculum_stage,
            ],
            dtype=np.int64,
        )
        wrappers = np.zeros((len(self.wrappers), 2), dtype=np.bool_)
        for i, wrapper in enumerate(self.wrappers):
            if type(wrapper) is OrderEnforcingWrapper:
                wrappers[i] = [wrapper._has_reset, wrapper._has_updated]
        result = MPE2Snapshot(
            np.asarray(1, dtype=np.uint32),
            self._signature.copy(),
            physics,
            agent_state,
            action_present,
            current_actions,
            np.asarray(words, dtype=np.uint32),
            np.stack((self._order(self.base.agents), self._order(self.env.agents))),
            self._order(selector.agent_order),
            counters,
            rewards,
            done,
            present,
            wrappers,
            np.asarray(self.base.original_cam_range, dtype=np.float64),
            active.copy(),
            mask.copy(),
        )
        self._validate(result)
        return result

    def _decode_order(self, value: NDArray[np.int32]) -> list[str]:
        valid = value[value >= 0]
        if (
            np.any(value < -1)
            or np.any(valid >= self.n)
            or len(set(valid.tolist())) != len(valid)
            or not np.array_equal(value[: len(valid)], valid)
        ):
            raise ValueError("Invalid MPE2 snapshot agent order.")
        return [self.names[int(i)] for i in valid]

    def _validate(self, snapshot: MPE2Snapshot) -> np.random.Generator:
        if type(snapshot) is not MPE2Snapshot:
            raise ValueError("Unsupported MPE2 snapshot version/type.")
        version_value = cast(object, snapshot.version)
        if (
            not isinstance(version_value, np.ndarray)
            or version_value.shape != ()
            or version_value.dtype != np.uint32
            or int(cast(NDArray[np.uint32], version_value)) != 1
        ):
            raise ValueError("Unsupported MPE2 snapshot version.")
        configuration = cast(object, snapshot.configuration)
        if (
            not isinstance(configuration, np.ndarray)
            or configuration.dtype != np.uint8
            or not np.array_equal(cast(NDArray[np.uint8], configuration), self._signature)
        ):
            raise ValueError("MPE2 snapshot configuration does not match the backend.")
        expected = [
            ((), np.uint32),
            (self._signature.shape, np.uint8),
            ((2 * self.n, 4), np.float64),
            ((self.n, 6), np.float64),
            ((self.n, 2), np.bool_),
            ((self.n,), np.int64),
            ((10,), np.uint32),
            ((2, self.n), np.int32),
            ((self.n,), np.int32),
            ((6,), np.int64),
            ((self.n, 2), np.float64),
            ((self.n, 2), np.bool_),
            ((self.n, 5), np.bool_),
            ((len(self.wrappers), 2), np.bool_),
            ((), np.float64),
            ((self.n,), np.bool_),
            ((self.n, 5), np.bool_),
        ]
        for value, (shape, dtype) in zip(cast(tuple[object, ...], snapshot), expected, strict=True):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.dtype(dtype)
            ):
                raise ValueError("MPE2 snapshot shape/dtype does not match the backend.")
        if any(
            not np.all(np.isfinite(value))
            for value in (snapshot.physics, snapshot.agent_state, snapshot.rewards, snapshot.camera)
        ):
            raise ValueError("MPE2 snapshot numeric state must be finite.")
        raw_agents = self._decode_order(snapshot.agents[0])
        self._decode_order(snapshot.agents[1])
        selector = self._decode_order(snapshot.selector)
        c = snapshot.counters
        # MPE2 1.1.1 inherits a no-op curriculum setter/getter; only stage zero exists.
        if (
            not 0 <= c[0] <= self.base.max_cycles
            or len(selector) != self.n
            or not 0 <= c[1] < self.n
            or not 0 <= c[2] < self.n
            or not 0 <= c[3] < self.n
            or not -2 <= c[4] < self.n
            or c[5] != 0
            or selector[(int(c[1]) - 1) % self.n] != self.names[int(c[2])]
        ):
            raise ValueError("Invalid MPE2 snapshot counters/selector.")
        if np.any((snapshot.current_actions < -1) | (snapshot.current_actions > 4)):
            raise ValueError("Invalid MPE2 snapshot pending actions.")
        presence = np.asarray([name in raw_agents for name in self.names], dtype=np.bool_)
        if not np.all(snapshot.present == presence[:, None]):
            raise ValueError("Invalid MPE2 snapshot dictionary membership.")
        listed = np.asarray(
            [name in self._decode_order(snapshot.agents[1]) for name in self.names], dtype=np.bool_
        )
        if np.any(snapshot.active & ~listed) or not np.all(np.any(snapshot.mask, axis=1)):
            raise ValueError("Invalid MPE2 snapshot active/action masks.")
        for i, wrapper in enumerate(self.wrappers):
            if type(wrapper) is AssertOutOfBoundsWrapper and np.any(snapshot.wrappers[i]):
                raise ValueError("Invalid stateless MPE2 wrapper flags.")
        if snapshot.rng[8] > 1:
            raise ValueError("Invalid PCG64 cache flag.")
        generator = np.random.Generator(np.random.PCG64(0))
        generator.bit_generator.state = {
            "bit_generator": "PCG64",
            "state": {"state": _integer(snapshot.rng[:4]), "inc": _integer(snapshot.rng[4:8])},
            "has_uint32": int(snapshot.rng[8]),
            "uinteger": int(snapshot.rng[9]),
        }
        return generator

    def restore(self, snapshot: MPE2Snapshot) -> None:
        """Validate the whole snapshot before rebinding any simulator state."""
        self._running()
        generator = self._validate(snapshot)
        for entity, values in zip(self.base.world.entities, snapshot.physics, strict=True):
            entity.state.p_pos, entity.state.p_vel = values[:2].copy(), values[2:].copy()
        for i, agent in enumerate(self.base.world.agents):
            agent.state.c = snapshot.agent_state[i, :2].copy()
            agent.action.u = (
                snapshot.agent_state[i, 2:4].copy() if snapshot.action_present[i, 0] else None
            )
            agent.action.c = (
                snapshot.agent_state[i, 4:6].copy() if snapshot.action_present[i, 1] else None
            )
        self.base.current_actions = [
            None if value == -1 else int(value) for value in snapshot.current_actions
        ]
        self.base.np_random = generator
        self.base.agents = self._decode_order(snapshot.agents[0])
        self.env.agents = self._decode_order(snapshot.agents[1])
        selector = self.base._agent_selector
        selector.agent_order = self._decode_order(snapshot.selector)
        c = snapshot.counters
        self.base.steps, selector._current_agent = int(c[0]), int(c[1])
        selector.selected_agent = self.names[int(c[2])]
        self.base.agent_selection = self.names[int(c[3])]
        if c[4] == -2:
            self.base.__dict__.pop("_skip_agent_selection", None)
        else:
            self.base._skip_agent_selection = None if c[4] == -1 else self.names[int(c[4])]
        self.base.scenario.curriculum_stage = int(c[5])
        self.base.rewards = {
            name: float(snapshot.rewards[i, 0])
            for i, name in enumerate(self.names)
            if snapshot.present[i, 0]
        }
        self.base._cumulative_rewards = {
            name: float(snapshot.rewards[i, 1])
            for i, name in enumerate(self.names)
            if snapshot.present[i, 1]
        }
        self.base.terminations = {
            name: bool(snapshot.done[i, 0])
            for i, name in enumerate(self.names)
            if snapshot.present[i, 2]
        }
        self.base.truncations = {
            name: bool(snapshot.done[i, 1])
            for i, name in enumerate(self.names)
            if snapshot.present[i, 3]
        }
        self.base.infos = {name: {} for i, name in enumerate(self.names) if snapshot.present[i, 4]}
        self.base.original_cam_range = float(snapshot.camera)
        for wrapper, flags in zip(self.wrappers, snapshot.wrappers, strict=True):
            if type(wrapper) is OrderEnforcingWrapper:
                wrapper._has_reset, wrapper._has_updated = map(bool, flags)
