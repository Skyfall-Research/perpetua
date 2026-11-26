"""Explicit parallel PettingZoo views and host-controlled continual lifts."""

# Optional simulator/JAX interfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable, Mapping
from importlib.metadata import version
from typing import Any, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from perpetua.adapters.host import BackendResult, CPULifetime, CPUState, CPUWorld, HostParams
from perpetua.adapters.snapshots import BackendSnapshot, MPE2Snapshot
from perpetua.core import EnvInfo, Environment, Info, JSONValue, requires_extra
from perpetua.drift import Drift
from perpetua.runtime import seed_key

with requires_extra("marl", "perpetua.adapters.pettingzoo"):
    from pettingzoo import AECEnv, ParallelEnv  # pyright: ignore[reportMissingTypeStubs]

    from perpetua.adapters.mpe2_snapshot import MPE2Codec

__all__ = [
    "ParallelAction",
    "PettingZooAdapter",
    "PettingZooLifetime",
    "AECLifetime",
]

type ParallelAction = int | np.integer[Any]


class PettingZooAdapter[State](ParallelEnv[str, NDArray[Any], ParallelAction]):
    """View fixed native slots through PettingZoo's parallel dictionary API.

    Per-agent Box observations retain their native dtype; compatibility action
    masks are int8 (actions,). Inactive slots remain named, selecting action zero.
    A horizon truncates a reporting segment, never the world. The first reset
    without a seed starts the lifetime from seed 0; later unseeded resets
    observe the current state, and reseeding is refused. Reset options are ignored
    and cannot reconfigure native dynamics. No privileged global state is exposed.
    """

    def __init__(self, env: Environment[State], *, horizon: int | None = None) -> None:
        if not isinstance(env.observation_space, gym.spaces.Box) or not isinstance(
            env.action_space, gym.spaces.MultiDiscrete
        ):
            raise ValueError("The parallel view requires Box observations and discrete slots.")
        if (
            not env.observation_space.shape
            or env.observation_space.shape[0] != env.num_agents
            or env.action_space.nvec.shape != (env.num_agents,)
        ):
            raise ValueError("Space descriptors must retain the native agent axis.")
        if np.any(env.action_space.start != 0):
            raise ValueError("The parallel view requires zero-based discrete slots.")
        if horizon is not None and (type(horizon) is not int or horizon < 1):
            raise ValueError("horizon must be a positive integer or None.")
        self.env = env
        self.horizon = horizon
        self.possible_agents = [f"agent_{i}" for i in range(env.num_agents)]
        self.agents: list[str] = []
        self._spaces = {
            name: gym.spaces.Box(
                env.observation_space.low[i],
                env.observation_space.high[i],
                dtype=cast(np.dtype[Any], env.observation_space.dtype),
            )
            for i, name in enumerate(self.possible_agents)
        }
        self._actions = {
            name: gym.spaces.Discrete(int(env.action_space.nvec[i]))
            for i, name in enumerate(self.possible_agents)
        }
        self.metadata = {
            "name": "perpetua_parallel_v0",
            "render_modes": [],
            "perpetua": env.metadata.copy(),
            "synthetic_horizon": horizon,
            "lifetime_preserved_on_reset": True,
            "reset_options": "ignored",
        }
        self._state: State | None = None
        self._key = jax.random.key(0)
        self._segment_steps = 0
        self._closed = False
        self._step = (
            env.step
            if env.metadata.get("execution") == "host"
            else cast(
                Callable[[State, ArrayLike, jax.Array], tuple[State, jax.Array, jax.Array, Info]],
                jax.jit(env.step),
            )
        )

    def observation_space(self, agent: str) -> gym.spaces.Box[Any]:
        """Return the same per-agent Box descriptor for the lifetime."""
        return self._spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Discrete:
        """Return the same per-agent Discrete descriptor for the lifetime."""
        return self._actions[agent]

    def _public(
        self, obs: jax.Array, info: Info
    ) -> tuple[dict[str, NDArray[Any]], dict[str, dict[str, Any]]]:
        observations = {
            name: np.array(obs[i], copy=True) for i, name in enumerate(self.possible_agents)
        }
        infos: dict[str, dict[str, Any]] = {}
        for i, name in enumerate(self.possible_agents):
            infos[name] = {
                "action_mask": np.asarray(
                    info["action_mask"][i, : self._actions[name].n], dtype=np.int8
                ),
                "active_mask": bool(info["active_mask"][i]),
                "respawn": bool(info["respawn"][i]),
                "t": info["t"],
            }
            if "oracle" in info:
                infos[name]["oracle"] = {
                    key: np.asarray(value) for key, value in info["oracle"].items()
                }
        return observations, infos

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, dict[str, Any]]]:
        """Initialize once, or reopen a segment without advancing state/RNG."""
        del options  # Recorded compatibility behavior: options cannot change the world.
        if self._closed:
            raise RuntimeError("The parallel view is closed.")
        if self._state is not None and seed is not None:
            raise ValueError("A seed cannot restart a lifetime; construct a new adapter.")
        if self._state is None:
            key, init_key = jax.random.split(seed_key(0 if seed is None else seed, "reset"))
            self._state, obs, info = self.env.init(init_key)
            self._key = key
        else:
            obs, info = self.env.observe(self._state)
        self._segment_steps = 0
        self.agents = self.possible_agents.copy()
        return self._public(obs, info)

    def step(
        self, actions: Mapping[str, ParallelAction]
    ) -> tuple[
        dict[str, NDArray[Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance one joint native tick and optionally end a synthetic segment."""
        if self._closed or self._state is None:
            raise RuntimeError("Call reset once before stepping an open parallel view.")
        if set(actions) != set(self.agents):
            raise ValueError("Action dictionary must contain exactly the current agents.")
        if not self.agents:
            return {}, {}, {}, {}, {}
        for name, action in actions.items():
            if not self._actions[name].contains(action):
                raise ValueError("Action is outside the agent's action space.")
        key, step_key = jax.random.split(self._key)
        self._state, obs, reward, info = self._step(
            self._state,
            jnp.asarray([actions[name] for name in self.possible_agents], dtype=jnp.int32),
            step_key,
        )
        self._key = key
        self._segment_steps += 1
        observations, infos = self._public(obs, info)
        rewards = {name: float(reward[i]) for i, name in enumerate(self.possible_agents)}
        truncated = self.horizon is not None and self._segment_steps >= self.horizon
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, truncated)
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        """Release an external backend, if present, and close this view."""
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
        self._closed = True


def _box(space: gym.spaces.Space[Any]) -> gym.spaces.Box[Any]:
    if isinstance(space, gym.spaces.Dict):
        entries = cast(dict[str, gym.spaces.Space[Any]], space.spaces)
        if set(entries) == {"observation", "action_mask"}:
            space = entries["observation"]
    if not isinstance(space, gym.spaces.Box):
        raise ValueError("The parallel lift requires Box or masked Box observations.")
    return cast(gym.spaces.Box[Any], space)


class _PettingZooBackend:
    checkpointable: bool

    def __init__(self, env: ParallelEnv[str, Any, Any], *, snapshot: bool) -> None:
        if not isinstance(cast(object, env), ParallelEnv):
            raise ValueError("The lift requires a parallel PettingZoo environment, not AEC.")
        self.env = env
        if type(snapshot) is not bool:
            raise ValueError("snapshot must be a boolean.")
        self.codec = MPE2Codec(env) if snapshot else None
        self.checkpointable = self.codec is not None
        names = cast(tuple[object, ...], tuple(env.possible_agents))
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("possible_agents must contain unique nonempty string slots.")
        self.names = cast(tuple[str, ...], names)
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.num_slots = len(self.names)
        first = _box(env.observation_space(self.names[0]))
        self._shape = first.shape
        action = env.action_space(self.names[0])
        if not isinstance(action, gym.spaces.Discrete) or action.start != 0:
            raise ValueError("The parallel lift requires zero-based Discrete actions.")
        self.action_count = int(action.n)
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(self.action_count)
        for name in self.names:
            box, action = _box(env.observation_space(name)), env.action_space(name)
            if (
                box.shape != first.shape
                or not np.array_equal(box.low, first.low)
                or not np.array_equal(box.high, first.high)
                or not isinstance(action, gym.spaces.Discrete)
                or action.n != self.action_count
                or action.start != 0
            ):
                raise ValueError("The parallel lift requires homogeneous Box/Discrete slots.")
        self.observation_space = gym.spaces.Box(
            np.minimum(np.asarray(first.low, dtype=np.float32).reshape(-1), 0.0),
            np.maximum(np.asarray(first.high, dtype=np.float32).reshape(-1), 0.0),
            dtype=np.float32,
        )
        self.active = np.zeros(self.num_slots, dtype=np.bool_)
        self.mask = np.zeros((self.num_slots, self.action_count), dtype=np.bool_)
        self.uses_reset_seed = not bool(env.metadata.get("lifetime_preserved_on_reset", False))
        max_cycles = getattr(env, "max_cycles", None)
        if max_cycles is None:
            max_cycles = getattr(env.unwrapped, "max_cycles", None)
        self.configuration: dict[str, JSONValue] = {
            "backend": "pettingzoo",
            "version": version("pettingzoo"),
            "env_id": str(env.metadata.get("name", type(env).__name__)),
            "possible_agents": list(self.names),
            "reset_policy": "same-tick-when-all-legacy-agents-finish",
            "reset_rng": "step-key-derived-seed"
            if self.uses_reset_seed
            else "lifetime-preserving-view",
            "legacy_info": "only-action-mask-activity-and-respawn",
            "legacy_max_cycles": max_cycles if isinstance(max_cycles, int) else None,
            "synthetic_horizon": cast(JSONValue, env.metadata.get("synthetic_horizon")),
            "snapshot": self.codec.configuration if self.codec is not None else None,
        }

    def _read(
        self, obs: dict[str, Any], infos: dict[str, dict[str, Any]]
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        if len(self.env.agents) != len(set(self.env.agents)) or not set(self.env.agents).issubset(
            self.indices
        ):
            raise ValueError("PettingZoo introduced an agent outside possible_agents.")
        result = np.zeros((self.num_slots, *self.observation_space.shape), dtype=np.float32)
        active = np.zeros(self.num_slots, dtype=np.bool_)
        reported = np.zeros(self.num_slots, dtype=np.bool_)
        mask = np.zeros((self.num_slots, self.action_count), dtype=np.bool_)
        mask[:, 0] = True
        for name in self.env.agents:
            i = self.indices[name]
            info = infos.get(name, {})
            active[i] = bool(info.get("active_mask", True))
            reported[i] = bool(info.get("respawn", False))
            value = obs[name]
            available = info.get("action_mask")
            if isinstance(value, dict):
                masked = cast(dict[str, Any], value)
                available, value = masked["action_mask"], masked["observation"]
            array = np.asarray(value, dtype=np.float32)
            if array.shape != self._shape:
                raise ValueError("PettingZoo observation shape changed during a lifetime.")
            if active[i]:
                result[i] = array.reshape(-1)
                if available is None:
                    mask[i] = True
                else:
                    bits = np.asarray(available)
                    if bits.shape != (self.action_count,) or not np.all((bits == 0) | (bits == 1)):
                        raise ValueError("Invalid PettingZoo action mask shape/values.")
                    mask[i] = bits.astype(np.bool_)
                    if not np.any(mask[i]):
                        raise ValueError("An active PettingZoo slot must have a valid action.")
        self.active, self.mask = active, mask
        return result, reported

    def init(self, seed: int) -> NDArray[np.float32]:
        obs, info = self.env.reset(seed=seed)
        if not self.env.agents:
            raise ValueError("PettingZoo reset must initialize at least one agent.")
        return self._read(obs, info)[0]

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        previous = self.active.copy()
        actions = {
            name: int(action[self.indices[name]]) if previous[self.indices[name]] else 0
            for name in self.env.agents
        }
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        reward = np.zeros(self.num_slots, dtype=np.float32)
        for name in actions:
            reward[self.indices[name]] = float(rewards[name])
            if name not in terminated or name not in truncated:
                raise ValueError(
                    "PettingZoo transition omitted agent termination/truncation flags."
                )
        if any(
            terminated.get(name, False) or truncated.get(name, False) for name in self.env.agents
        ):
            raise ValueError("PettingZoo retained an agent after termination/truncation.")
        if not self.env.agents:
            obs, infos = (
                self.env.reset(seed=reset_seed) if self.uses_reset_seed else self.env.reset()
            )
            if not self.env.agents:
                raise ValueError("PettingZoo reset must initialize at least one agent.")
            obs, reported = self._read(obs, infos)
            # A lifetime-preserving view only reopened a reporting segment:
            # nothing respawned unless the world it wraps says so.
            respawn = (
                self.active.copy() if self.uses_reset_seed else self.active & (~previous | reported)
            )
        else:
            obs, reported = self._read(obs, infos)
            respawn = self.active & (~previous | reported)
        return BackendResult(obs, reward, self.active.copy(), respawn)

    def snapshot(self) -> MPE2Snapshot:
        if self.codec is None:
            raise NotImplementedError("No complete checkpoint codec for this PettingZoo backend.")
        return self.codec.snapshot(self.active, self.mask)

    def restore(self, snapshot: BackendSnapshot) -> None:
        if self.codec is None:
            raise NotImplementedError("No complete checkpoint codec for this PettingZoo backend.")
        if not isinstance(snapshot, MPE2Snapshot):
            raise ValueError("Expected an MPE2 snapshot.")
        self.codec.restore(snapshot)
        self.active, self.mask = snapshot.active.copy(), snapshot.mask.copy()

    def close(self) -> None:
        self.env.close()


class PettingZooWorld(CPUWorld):
    """CPU world over a PettingZoo backend: real activity and action masks."""

    def __init__(self, backend: Any, *, env_id: str) -> None:
        self._backend = backend
        super().__init__(backend, env_id=env_id)

    def init(self, key: jax.Array, params: HostParams) -> tuple[CPUState, jax.Array, EnvInfo]:
        """Initialize the external world and retain its actual active-slot subset."""
        state, _, _ = super().init(key, params)
        state = state._replace(active=jnp.asarray(self._backend.active))
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: CPUState, params: HostParams) -> tuple[jax.Array, EnvInfo]:
        """Expose masks from this valid mutable generation, never legacy private info."""
        obs, info = super().observe(state, params)
        if self._backend.checkpointable:
            if not isinstance(state.snapshot, MPE2Snapshot):
                raise ValueError("Expected a portable MPE2 snapshot.")
            mask = jnp.asarray(state.snapshot.mask) & state.active[:, None]
        else:
            mask = jnp.asarray(self._backend.mask) & state.active[:, None]
        info["action_mask"] = mask.at[:, 0].set(mask[:, 0] | ~state.active)
        return obs, info

    def step(
        self, state: CPUState, action: ArrayLike, key: jax.Array, params: HostParams
    ) -> tuple[CPUState, jax.Array]:
        """Reject masked inputs before any simulator mutation/generation advance."""
        self._validate(state)
        actions = self._actions(action)
        _, info = self.observe(state, params)
        if np.any(
            np.asarray(state.active)
            & ~np.asarray(info["action_mask"])[np.arange(self.num_agents), actions]
        ):
            raise ValueError("PettingZoo action is outside the current action mask.")
        return super().step(state, actions, key, params)


class PettingZooLifetime(CPULifetime):
    """Lift homogeneous parallel Box/Discrete agents into fixed float32 CPU slots.

    Ordinary/Dict observation masks and optional activity/respawn info are
    preserved. Other legacy info, task IDs and done flags are not public. Full
    legacy-world exhaustion respawns internally; no native clock/learner reset.
    ``snapshot=True`` selects the MPE2 codec, which supports only discrete-action, headless
    MPE2 ``simple_spread`` (``parallel_env``, default wrappers) on exactly mpe2 1.1.1 with
    pettingzoo 1.27.0; any other environment or version raises ``ValueError``. Without it,
    external simulator/RNG state remains opaque, single-use and non-checkpointable.
    """

    world_class = PettingZooWorld

    def __init__(
        self, env: ParallelEnv[str, Any, Any], *, drift: Drift | None = None, snapshot: bool = False
    ) -> None:
        backend = _PettingZooBackend(env, snapshot=snapshot)
        super().__init__(
            backend, env_id=f"pettingzoo/{backend.configuration['env_id']}", drift=drift
        )


class _AECBackend:
    """Cycle-synchronous backend over a homogeneous AEC environment.

    One backend step drives one full AEC cycle in selector order, mirroring
    PettingZoo's official parallel conversion, and accumulates each agent's
    reward across the cycle. A terminated or truncated agent's slot is
    inactive from that tick on, and the agent receives ``None`` whenever the
    selector reaches it, during the cycle or at its end; an agent that dies
    before its turn in a cycle does not act in it, and the action submitted
    for its slot is dropped. Actions of inactive slots are ignored: a live
    agent that reports itself inactive is sent action 0, as in the parallel
    lift. Full legacy exhaustion resets on the same tick. Agent reordering
    is refused rather than guessed, and so is a turn-based game: an
    environment whose ``metadata["is_parallelizable"]`` is ``False`` raises
    ``ValueError`` at construction (one that does not declare the key is
    accepted). Legacy state/RNG stays opaque and single-use.
    """

    checkpointable = False
    uses_reset_seed = True

    def __init__(self, env: AECEnv[str, Any, Any]) -> None:
        if not isinstance(cast(object, env), AECEnv):
            raise ValueError("AECLifetime requires a sequential AEC environment.")
        metadata = cast("dict[str, Any]", getattr(env, "metadata", None) or {})
        if metadata.get("is_parallelizable") is False:
            raise ValueError(
                f"AECLifetime does not support turn-based games: {metadata.get('name', 'this')} "
                "environment declares metadata['is_parallelizable'] = False, so its agents "
                "do not all act in every cycle and one tick cannot be one joint action."
            )
        self.env = env
        names = cast(tuple[object, ...], tuple(env.possible_agents))
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("possible_agents must contain unique nonempty string slots.")
        self.names = cast(tuple[str, ...], names)
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.num_slots = len(self.names)
        first = _box(env.observation_space(self.names[0]))
        self._shape = first.shape
        action = env.action_space(self.names[0])
        if not isinstance(action, gym.spaces.Discrete) or action.start != 0:
            raise ValueError("AECLifetime requires zero-based Discrete actions.")
        self.action_count = int(action.n)
        for name in self.names:
            box, action = _box(env.observation_space(name)), env.action_space(name)
            if (
                box.shape != first.shape
                or not np.array_equal(box.low, first.low)
                or not np.array_equal(box.high, first.high)
                or not isinstance(action, gym.spaces.Discrete)
                or action.n != self.action_count
                or action.start != 0
            ):
                raise ValueError("AECLifetime requires homogeneous Box/Discrete slots.")
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(self.action_count)
        self.observation_space = gym.spaces.Box(
            np.minimum(np.asarray(first.low, dtype=np.float32).reshape(-1), 0.0),
            np.maximum(np.asarray(first.high, dtype=np.float32).reshape(-1), 0.0),
            dtype=np.float32,
        )
        self.active = np.zeros(self.num_slots, dtype=np.bool_)
        self.mask = np.zeros((self.num_slots, self.action_count), dtype=np.bool_)
        max_cycles = getattr(env, "max_cycles", None)
        if max_cycles is None:
            max_cycles = getattr(env.unwrapped, "max_cycles", None)
        self.configuration: dict[str, JSONValue] = {
            "backend": "pettingzoo-aec",
            "version": version("pettingzoo"),
            "env_id": str(env.metadata.get("name", type(env).__name__)),
            "possible_agents": list(self.names),
            "cycle": "one-joint-tick-per-full-aec-cycle",
            "reset_policy": "same-tick-when-all-legacy-agents-finish",
            "reset_rng": "step-key-derived-seed",
            "legacy_info": "only-activity-and-respawn",
            "legacy_max_cycles": max_cycles if isinstance(max_cycles, int) else None,
            "synthetic_horizon": None,
            "snapshot": None,
        }

    def _read(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        live = list(self.env.agents)
        if len(live) != len(set(live)) or not set(live).issubset(self.indices):
            raise ValueError("AEC introduced an agent outside possible_agents.")
        result = np.zeros((self.num_slots, *self.observation_space.shape), dtype=np.float32)
        active = np.zeros(self.num_slots, dtype=np.bool_)
        reported = np.zeros(self.num_slots, dtype=np.bool_)
        mask = np.zeros((self.num_slots, self.action_count), dtype=np.bool_)
        for name in live:
            if self._dead(name):
                continue
            i = self.indices[name]
            info = self.env.infos.get(name, {})
            active[i] = bool(info.get("active_mask", True))
            reported[i] = bool(info.get("respawn", False))
            value = self.env.observe(name)
            available = info.get("action_mask")
            if isinstance(value, dict):
                masked = cast(dict[str, Any], value)
                available, value = masked["action_mask"], masked["observation"]
            array = np.asarray(value, dtype=np.float32)
            if array.shape != self._shape:
                raise ValueError("AEC observation shape changed during a lifetime.")
            if active[i]:
                result[i] = array.reshape(-1)
                if available is None:
                    mask[i] = True
                else:
                    bits = np.asarray(available)
                    if bits.shape != (self.action_count,) or not np.all((bits == 0) | (bits == 1)):
                        raise ValueError("Invalid AEC action mask shape/values.")
                    mask[i] = bits.astype(np.bool_)
                    if not np.any(mask[i]):
                        raise ValueError("An active AEC slot must have a valid action.")
        self.active, self.mask = active, mask
        return result, reported

    def init(self, seed: int) -> NDArray[np.float32]:
        self.env.reset(seed=seed)
        if not self.env.agents:
            raise ValueError("AEC reset must initialize at least one agent.")
        return self._read()[0]

    def _dead(self, name: str) -> bool:
        if name not in self.env.terminations:
            raise ValueError(
                f"The AEC environment left its selector on {name!r}, an agent it has removed."
            )
        return bool(self.env.terminations[name] or self.env.truncations[name])

    def _bury(self) -> None:
        """Give every dead agent the selector has reached its ``None`` step."""
        env = self.env
        while env.agents and self._dead(env.agent_selection):
            env.step(None)

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        previous = self.active.copy()
        rewards = np.zeros(self.num_slots, dtype=np.float32)
        env = self.env
        for name in tuple(env.agents):
            self._bury()
            if name not in env.agents or self._dead(name):
                continue
            if name != env.agent_selection:
                raise ValueError("AEC environments must step in selector order within a cycle.")
            i = self.indices[name]
            env.step(int(action[i]) if previous[i] else 0)
            for other in env.agents:
                rewards[self.indices[other]] += float(env.rewards[other])
        self._bury()
        if not env.agents:
            env.reset(seed=reset_seed)
            if not env.agents:
                raise ValueError("AEC reset must initialize at least one agent.")
            obs, _ = self._read()
            respawn = self.active.copy()
        else:
            obs, reported = self._read()
            respawn = self.active & (~previous | reported)
        return BackendResult(obs, rewards, self.active.copy(), respawn)

    def close(self) -> None:
        self.env.close()

    def snapshot(self) -> BackendSnapshot:
        raise NotImplementedError("No complete checkpoint codec for this AEC backend.")

    def restore(self, snapshot: BackendSnapshot) -> None:
        raise NotImplementedError("No complete checkpoint codec for this AEC backend.")


class AECLifetime(PettingZooLifetime):
    """Lift a homogeneous AEC environment into synchronous fixed-slot host ticks.

    One Perpetua tick drives one full AEC cycle in selector order, mirroring
    PettingZoo's official parallel conversion, and accumulates per-agent
    rewards across the cycle. Slots terminated at a cycle boundary become
    inactive until the legacy world respawns; full exhaustion resets on the
    same tick without touching the Perpetua clock or learner. An agent that
    dies before its turn in a cycle does not act in it and its slot's action
    is dropped. Legacy state/RNG stays opaque and single-use:
    checkpoint attempts are refused, exactly like unsupported parallel worlds.

    Turn-based games (PettingZoo classic's ``connect_four_v3``,
    ``tictactoe_v3`` and the like) are not supported: an environment whose
    ``metadata["is_parallelizable"]`` is ``False`` raises ``ValueError`` at
    construction. An environment that does not declare the key is accepted.
    """

    def __init__(self, env: AECEnv[str, Any, Any], *, drift: Drift | None = None) -> None:
        backend = _AECBackend(env)
        CPULifetime.__init__(
            self, backend, env_id=f"pettingzoo-aec/{backend.configuration['env_id']}", drift=drift
        )
