"""Host-controlled CPU backends with explicit ownership and checkpoint limits."""

# Native extensions and JAX expose some partially annotated interfaces.
# pyright: reportUnknownMemberType=false

import sys
from importlib import import_module
from typing import Any, NamedTuple, Protocol, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from perpetua.adapters.snapshots import (
    BackendSnapshot,
    CartPoleCodec,
    CartPoleSnapshot,
    EnvPoolSnapshot,
)
from perpetua.core import EnvInfo, JSONValue, Metadata
from perpetua.drift import Drift
from perpetua.params import DriftEnv

__all__ = [
    "BackendResult",
    "CPUBackend",
    "CPUState",
    "HostParams",
    "CPUWorld",
    "CPULifetime",
    "GymnasiumLifetime",
    "MetaWorldLifetime",
    "EnvPoolAtari",
    "EnvPoolCartPole",
]


class BackendResult(NamedTuple):
    """One ordered batch of CPU observations, rewards and world events."""

    obs: NDArray[np.float32]
    reward: NDArray[np.float32]
    active: NDArray[np.bool_]
    respawn: NDArray[np.bool_]


class CPUBackend(Protocol):
    """CPU execution boundary; snapshots contain numeric simulator state, never handles."""

    @property
    def checkpointable(self) -> bool: ...

    @property
    def num_slots(self) -> int: ...

    @property
    def action_space(self) -> gym.spaces.Space[Any]: ...

    @property
    def observation_space(self) -> gym.spaces.Box[Any]: ...

    @property
    def configuration(self) -> dict[str, JSONValue]: ...

    @property
    def uses_reset_seed(self) -> bool: ...

    def init(self, seed: int) -> NDArray[np.float32]: ...

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult: ...

    def close(self) -> None: ...

    def snapshot(self) -> BackendSnapshot: ...

    def restore(self, snapshot: BackendSnapshot) -> None: ...


class CPUState(NamedTuple):
    """Portable snapshot state when supported, otherwise an opaque single-use state.

    Drift, its RNG stream and the lifetime clock live in the ``DriftEnv``
    wrapper, not here.
    """

    owner: object | None
    generation: int
    obs: jax.Array
    active: jax.Array
    respawn: jax.Array
    snapshot: BackendSnapshot | None = None


class HostParams(NamedTuple):
    """The host lift's one drifting parameter: a reward multiplier, float32 (1,)."""

    reward_multiplier: jax.Array


def _seed(key: jax.Array) -> int:
    try:
        return int(jax.random.bits(key, shape=(), dtype=jnp.uint32)) & 0x7FFFFFFF
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerIntegerConversionError) as error:
        raise TypeError(
            "Host simulators hold one mutable process and cannot be vmapped or traced "
            "through jit with the lifetime seed: run one lifetime per simulator "
            "(batch with the backend's own slots, for example EnvPool's num_envs)."
        ) from error


def _pool_seed(seed: int, num_envs: int) -> int:
    """EnvPool seeds slot ``i`` with ``seed + i``, which must stay within a C ``int``."""
    limit = 2**31 - num_envs
    return seed if seed <= limit else seed % (limit + 1)


def _initial_active(backend: object, slots: int) -> jax.Array:
    """Slots active at init: all, unless the backend reports ``initial_active``."""
    declared = getattr(backend, "initial_active", None)
    if declared is None:
        return jnp.ones(slots, dtype=jnp.bool_)
    return jnp.asarray(np.asarray(declared, dtype=np.bool_))


class CPUWorld:
    """A synchronous product-world of CPU simulator slots, as a parameter environment.

    Snapshot-backed states restore before stepping and can resume/replay safely.
    Opaque backend states remain single-use. Observations/rewards transfer in
    batches, not per env; snapshots are small numeric CPU arrays. The reward
    multiplier is the drifting parameter; :class:`CPULifetime` wraps this world
    in ``DriftEnv``, which owns drift, its key, switch events and the clock.
    """

    default_drift_field = "reward_multiplier"

    def __init__(self, backend: CPUBackend, *, env_id: str) -> None:
        self.backend = backend
        self.num_agents = backend.num_slots
        low = np.broadcast_to(
            backend.observation_space.low, (self.num_agents, *backend.observation_space.shape)
        )
        high = np.broadcast_to(backend.observation_space.high, low.shape)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        if isinstance(backend.action_space, gym.spaces.Discrete):
            self.action_space: gym.spaces.Space[Any] = gym.spaces.MultiDiscrete(
                np.full(self.num_agents, int(backend.action_space.n), dtype=np.int64)
            )
        elif isinstance(backend.action_space, gym.spaces.Box):
            action_low = np.broadcast_to(
                np.asarray(cast(NDArray[Any], backend.action_space.low), dtype=np.float32),
                (self.num_agents, *backend.action_space.shape),
            )
            action_high = np.broadcast_to(
                np.asarray(cast(NDArray[Any], backend.action_space.high), dtype=np.float32),
                (self.num_agents, *backend.action_space.shape),
            )
            self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)
        else:
            raise ValueError("The CPU lift requires Discrete or Box slot actions.")
        self.metadata: Metadata = {
            "env_id": env_id,
            "oracle_exposure": (),
            "execution": "host",
            "checkpointable": backend.checkpointable,
            "configuration": {
                "backend": backend.configuration,
                "drift_parameter": "reward_multiplier",
                "clock": "synchronous-batch-tick",
            },
        }
        self._owner = object()
        self._generation = 0
        self._initialized = False
        self._closed = False

    @property
    def default_params(self) -> HostParams:
        """Rewards start unscaled."""
        return HostParams(jnp.asarray([1.0], dtype=jnp.float32))

    def derive(self, params: HostParams) -> HostParams:
        """The multiplier is used as drifted; nothing is rounded or derived."""
        return params

    def init(self, key: jax.Array, params: HostParams) -> tuple[CPUState, jax.Array, EnvInfo]:
        """Initialize the external backend once with a derived seed."""
        if self._closed or self._initialized:
            raise RuntimeError("Create a new CPU backend for a new lifetime.")
        if params.reward_multiplier.shape != (1,):
            raise ValueError("CPU reward drift must have parameter shape (1,).")
        seed_key, _, _ = jax.random.split(key, 3)
        obs = self.backend.init(_seed(seed_key))
        self._initialized = True
        state = CPUState(
            None if self.backend.checkpointable else self._owner,
            self._generation,
            jnp.asarray(np.array(obs, copy=True)),
            _initial_active(self.backend, self.num_agents),
            jnp.zeros(self.num_agents, dtype=jnp.bool_),
            self.backend.snapshot() if self.backend.checkpointable else None,
        )
        obs, info = self.observe(state, params)
        return state, obs, info

    def _validate(self, state: CPUState) -> None:
        if self._closed or not self._initialized:
            raise RuntimeError("CPU backend is not running.")
        if self.backend.checkpointable:
            if state.owner is not None or state.snapshot is None:
                raise ValueError("A snapshot-backed CPU state must contain its portable snapshot.")
            return
        if state.owner is not self._owner or state.generation != self._generation:
            raise ValueError(
                "CPU state is stale or belongs to another backend; it cannot be forked."
            )

    def observe(self, state: CPUState, params: HostParams) -> tuple[jax.Array, EnvInfo]:
        """Expose the latest batch; inactive reset slots select placeholder action zero."""
        del params
        self._validate(state)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            mask = jnp.broadcast_to(
                state.active[:, None], (self.num_agents, int(self.action_space.nvec[0]))
            )
        else:
            # Continuous slots expose a single validity flag per action vector.
            mask = jnp.ones((self.num_agents, 1), dtype=jnp.bool_)
        mask = mask.at[:, 0].set(True)
        return state.obs, {
            "action_mask": mask,
            "active_mask": state.active,
            "respawn": state.respawn,
        }

    def step(
        self, state: CPUState, action: ArrayLike, key: jax.Array, params: HostParams
    ) -> tuple[CPUState, jax.Array]:
        """Step one CPU batch and scale its reward by the current multiplier."""
        self._validate(state)
        actions = self._actions(action)
        reset_seed = _seed(key) if self.backend.uses_reset_seed else 0
        if self.backend.checkpointable:
            if state.snapshot is None:
                raise ValueError("Missing CPU backend snapshot.")
            self.backend.restore(state.snapshot)
        # An attempted external transition may mutate even when the backend raises.
        self._generation += 1
        result = self.backend.step(actions, reset_seed)
        reward = jnp.asarray(np.array(result.reward, copy=True)) * params.reward_multiplier[0]
        next_state = CPUState(
            None if self.backend.checkpointable else self._owner,
            state.generation + 1 if self.backend.checkpointable else self._generation,
            jnp.asarray(np.array(result.obs, copy=True)),
            jnp.asarray(result.active),
            jnp.asarray(result.respawn),
            self.backend.snapshot() if self.backend.checkpointable else None,
        )
        return next_state, reward

    def _actions(self, action: ArrayLike) -> NDArray[Any]:
        """Validate before mutable stepping; adapters may additionally check masks."""
        actions = np.asarray(action)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            if actions.shape != (self.num_agents,) or actions.dtype.kind not in {"i", "u"}:
                raise ValueError("CPU actions must be an integer vector matching simulator slots.")
            if np.any(actions < 0) or np.any(actions >= self.action_space.nvec):
                raise ValueError("CPU action is outside the backend action space.")
            return np.ascontiguousarray(actions, dtype=np.int32)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError("The CPU lift requires Discrete or Box slot actions.")
        if actions.shape != self.action_space.shape or actions.dtype.kind != "f":
            raise ValueError("CPU continuous actions must be a float array matching slot vectors.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("CPU continuous actions must be finite.")
        low = np.asarray(cast(NDArray[Any], self.action_space.low), dtype=np.float32)
        high = np.asarray(cast(NDArray[Any], self.action_space.high), dtype=np.float32)
        return np.clip(actions, low, high).astype(np.float32)

    def privileged(self, state: CPUState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        self._validate(state)
        return {}

    def close(self) -> None:
        """Release external simulator resources; stale states cannot be reused."""
        self.backend.close()
        self._closed = True


class CPULifetime(DriftEnv[CPUState, HostParams]):
    """A CPU simulator lift: a :class:`CPUWorld` wrapped in ``DriftEnv``.

    Construct it, or a subclass, directly (``GymnasiumLifetime(env, drift=...)``)
    and use it like any environment; ``world_class`` selects the world a
    subclass needs.
    """

    world_class: type[CPUWorld] = CPUWorld

    def __init__(
        self,
        backend: CPUBackend,
        *,
        env_id: str,
        drift: Drift | None = None,
        compiled: bool = False,
    ) -> None:
        try:
            world_class = self.world_class
            if compiled:
                from perpetua.adapters.bridge import CallbackWorld

                world_class = CallbackWorld
            super().__init__(world_class(backend, env_id=env_id), drift, env_id=env_id)
        except BaseException:
            backend.close()  # e.g. an unknown drift field must not leak the simulator
            raise

    def with_drift(self, drift: Any) -> "CPULifetime":
        """Host lifts own one mutable simulator, so drift cannot be rebound.

        A rebound copy would share the simulator with this lifetime; pass
        ``drift=`` when constructing the lift instead.
        """
        del drift
        raise TypeError(
            f"{type(self).__name__} does not support rebinding drift after construction: "
            "it owns one mutable simulator. Pass drift= when constructing it."
        )


class _GymnasiumBackend:
    num_slots = 1
    uses_reset_seed = True

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        if not isinstance(env.observation_space, gym.spaces.Box) or not isinstance(
            env.action_space, (gym.spaces.Discrete, gym.spaces.Box)
        ):
            raise ValueError(
                "The Gymnasium CPU adapter supports Box observations with Discrete or Box actions."
            )
        self.env = env
        self.last_info: dict[str, Any] = {}  # the latest step's info; empty before one
        self.codec = CartPoleCodec(env) if CartPoleCodec.supports(env) else None
        self.action_space: gym.spaces.Space[Any] = env.action_space
        # The upstream shape is kept and values are cast to float32.
        self.observation_space = gym.spaces.Box(
            np.atleast_1d(np.asarray(cast(NDArray[Any], env.observation_space.low), np.float32)),
            np.atleast_1d(np.asarray(cast(NDArray[Any], env.observation_space.high), np.float32)),
            dtype=np.float32,
        )
        self.configuration: dict[str, JSONValue] = {
            "backend": "gymnasium",
            "env_id": env.spec.id if env.spec else type(env).__name__,
            "legacy_time_limit": env.spec.max_episode_steps if env.spec else None,
            "respawn": "same-tick-on-termination-or-truncation",
            "action_encoding": (
                "discrete"
                if isinstance(env.action_space, gym.spaces.Discrete)
                else f"box-{tuple(int(d) for d in env.action_space.shape)}"
            ),
            "snapshot": self.codec.configuration if self.codec is not None else None,
        }

    @property
    def checkpointable(self) -> bool:
        return self.codec is not None

    def snapshot(self) -> CartPoleSnapshot:
        if self.codec is None:
            raise NotImplementedError("No complete checkpoint codec for this Gymnasium backend.")
        return self.codec.snapshot()

    def restore(self, snapshot: BackendSnapshot) -> None:
        if self.codec is None:
            raise NotImplementedError("No complete checkpoint codec for this Gymnasium backend.")
        if not isinstance(snapshot, CartPoleSnapshot):
            raise ValueError("Expected a Gymnasium CartPole snapshot.")
        self.codec.restore(snapshot)

    def _reset(self, seed: int) -> Any:
        obs, _ = self.env.reset(seed=seed)
        return obs

    def init(self, seed: int) -> NDArray[np.float32]:
        return np.atleast_1d(np.asarray(self._reset(seed), dtype=np.float32))[None]

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        env_action: Any = (
            int(action[0])
            if isinstance(self.action_space, gym.spaces.Discrete)
            else np.asarray(action[0], dtype=np.float32)
        )
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.last_info = dict(info)
        respawn = terminated or truncated
        if respawn:
            obs = self._reset(reset_seed)
        return BackendResult(
            np.atleast_1d(np.asarray(obs, dtype=np.float32))[None],
            np.asarray([reward], dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            np.asarray([respawn], dtype=np.bool_),
        )

    def close(self) -> None:
        self.env.close()


class GymnasiumLifetime(CPULifetime):
    """Lift a Box/Discrete Gymnasium CPU env into a host-controlled lifetime.

    The env is used as given, including any ``TimeLimit`` from ``gym.make``:
    a termination or truncation becomes a same-tick respawn (the env is reset
    on that tick), so the time limit still bounds each episode but never ends
    the lifetime. Lifetimes are checkpointable only for headless CartPole: an
    exact ``CartPoleEnv`` with ``render_mode=None``, wrapped only in
    ``TimeLimit``, ``OrderEnforcing`` and ``PassiveEnvChecker``, on exactly
    gymnasium 1.4.0. Any other env runs but cannot be checkpointed.

    The observation keeps the env's own shape with a leading slot axis, so a
    ``Box(0, 255, (H, W, C), uint8)`` env gives ``(1, H, W, C)``, and every
    value is cast to float32 (``observation_space`` is float32 with the env's
    bounds). Pixel observations therefore need ``torso="cnn"`` in PPO and DQN.
    """

    def __init__(
        self, env: gym.Env[Any, Any], *, drift: Drift | None = None, compiled: bool = False
    ) -> None:
        backend = _GymnasiumBackend(env)
        super().__init__(
            backend,
            env_id=f"gymnasium/{backend.configuration['env_id']}",
            drift=drift,
            compiled=compiled,
        )


class MetaWorldWorld(CPUWorld):
    """A Meta-World world that reports the backend's task-success flag to the Oracle.

    ``success`` is Meta-World's own ``info["success"]`` for the last step, a float32
    ``(1,)``; it is privileged (the task outcome a benchmark scores), so it never
    enters observations or public info.
    """

    oracle_fields = ("success",)

    def privileged(self, state: CPUState) -> dict[str, jax.Array]:
        self._validate(state)
        info = cast(_GymnasiumBackend, self.backend).last_info
        if info and "success" not in info:
            raise RuntimeError("Meta-World step info has no 'success' entry.")
        return {"success": jnp.asarray([float(info.get("success", 0.0))], dtype=jnp.float32)}


class _MetaWorldBackend(_GymnasiumBackend):
    """Meta-World with an optional new goal configuration at every episode start.

    With ``randomize_goals`` each reset first selects one of MT1's 50 generated
    goal and object configurations from the reset seed the framework supplies,
    so episodes vary as in Continual World while equally seeded lifetimes stay
    identical and no extra random state needs checkpointing.

    Upstream declares the goal position (observation entries 36 to 38) as the
    point ``low == high == 0`` while the observations carry the goal there, so
    ``observation_space.contains`` would be False for every observation. Every
    entry whose upstream bounds coincide is declared ``(-inf, inf)`` instead;
    the other entries keep upstream's bounds.
    """

    def __init__(self, env: gym.Env[Any, Any], tasks: list[Any], *, randomize: bool) -> None:
        super().__init__(env)
        self.tasks = tasks
        self.randomize = randomize
        space = self.observation_space
        point = space.low == space.high
        if point.any():
            self.observation_space = gym.spaces.Box(
                np.where(point, -np.inf, space.low).astype(np.float32),
                np.where(point, np.inf, space.high).astype(np.float32),
                dtype=np.float32,
            )

    def _reset(self, seed: int) -> Any:
        if self.randomize:
            # set_task takes effect at the reset that follows it.
            index = int(np.random.default_rng(seed).integers(len(self.tasks)))
            cast(Any, self.env.unwrapped).set_task(self.tasks[index])
        return super()._reset(seed)


class MetaWorldLifetime(CPULifetime):
    """One Farama Meta-World MT1 task as a host-controlled continual lifetime.

    The Oracle field ``success`` exposes Meta-World's task-success flag, which
    the Continual World metrics are defined on (not available with
    ``compiled=True``). By default every episode uses the goal pinned by
    ``task_index``; ``randomize_goals=True`` picks one of MT1's 50 goal and
    object configurations at each episode start, as Continual World does.

    ``seed`` fixes which 50 goal and object configurations MT1 generates, and
    ``task_index`` selects one of them: two lifetimes with different ``seed`` and
    the same ``task_index`` face different goals. It is separate from the seed
    given to ``init_lifetime``, which only drives resets and noise.
    ``max_episode_steps`` is the episode length and may not exceed Meta-World's
    own horizon (``max_path_length``, 500 steps for the MT1 tasks), at which an
    episode always ends; a larger value raises ``ValueError``.
    """

    world_class = MetaWorldWorld

    def __init__(
        self,
        *,
        env_name: str = "hammer-v3",
        seed: int = 0,
        max_episode_steps: int = 200,
        task_index: int = 0,
        drift: Drift | None = None,
        compiled: bool = False,
        randomize_goals: bool = False,
    ) -> None:
        if not env_name or type(seed) is not int or seed < 0:
            raise ValueError("Meta-World needs a task name and a nonnegative integer seed.")
        if type(randomize_goals) is not bool:
            raise ValueError("randomize_goals must be a boolean.")
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer.")
        if type(task_index) is not int or not 0 <= task_index < 50:
            raise ValueError("task_index must select one of MT1's 50 generated goals.")
        try:
            import_module("metaworld")  # Registers the Gymnasium MT1 id.
        except ImportError as error:
            raise ImportError(
                "Install Perpetua's metaworld extra to use Meta-World tasks."
            ) from error
        # Upstream declares the goal entries of its observation as a single point
        # while observing the goal there, so its passive checker would warn on
        # every reset and step; the lifted space (below) declares them correctly.
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            seed=seed,
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,
        )
        horizon = int(cast(Any, env.unwrapped).max_path_length)
        if max_episode_steps > horizon:
            env.close()
            raise ValueError(
                f"max_episode_steps={max_episode_steps} exceeds Meta-World's own horizon of "
                f"{horizon} steps for {env_name}, where every episode ends regardless; "
                f"use a value in [1, {horizon}]."
            )
        # MT1 otherwise samples goals on reset from an independent RNG, which
        # would make two equally seeded lifetimes diverge on their first tick.
        tasks = env.get_wrapper_attr("tasks")
        env.get_wrapper_attr("toggle_sample_tasks_on_reset")(False)
        cast(Any, env.unwrapped).set_task(tasks[task_index])
        backend = _MetaWorldBackend(env, list(tasks), randomize=randomize_goals)
        from importlib.metadata import version

        backend.configuration = {
            **backend.configuration,
            "task": env_name,
            "seed": seed,
            "task_index": task_index,
            "goals": "per-episode" if randomize_goals else "fixed",
            "metaworld": version("metaworld"),
            "horizon": max_episode_steps,
        }
        super().__init__(backend, env_id="metaworld/MT1-v0", drift=drift, compiled=compiled)


class _EnvPoolBackend:
    uses_reset_seed = False
    checkpointable: bool
    num_slots: int
    num_threads: int
    episode_steps: int
    action_count: int
    action_space: gym.spaces.Space[Any]
    observation_space: gym.spaces.Box[Any]
    configuration: dict[str, JSONValue]
    module: Any
    native: Any
    pool: Any
    pending: NDArray[np.bool_]
    initial_active: NDArray[np.bool_] | None = None
    overlap: bool
    current: NDArray[np.int64]
    last_obs: NDArray[np.float32]

    def __init__(
        self,
        *,
        num_envs: int,
        num_threads: int,
        episode_steps: int,
        snapshot: bool,
        overlap: bool = False,
    ) -> None:
        if sys.platform != "linux":
            raise RuntimeError("The EnvPool CPU bridge is currently validated on Linux only.")
        for value in (num_envs, num_threads, episode_steps):
            if type(value) is not int or value < 1:
                raise ValueError("EnvPool slot/thread/episode counts must be positive integers.")
        if type(snapshot) is not bool or type(overlap) is not bool:
            raise ValueError("snapshot and overlap must be booleans.")
        if overlap and (snapshot or num_envs < 2 or num_envs % 2):
            raise ValueError("overlap=True needs an even num_envs >= 2 and snapshot=False.")
        self.overlap = overlap
        try:
            self.module = cast(Any, import_module("envpool"))
        except ImportError as error:
            raise ImportError(
                "Install Perpetua's cpu-bridge extra and EnvPool's runtime dependencies."
            ) from error
        self.num_slots = num_envs
        self.num_threads = num_threads
        self.episode_steps = episode_steps
        spec = self.module.make_spec("CartPole-v1", num_envs=num_envs)
        self.observation_space = cast(gym.spaces.Box[Any], spec.gymnasium_observation_space)
        self.action_count = 2
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(2)
        self.native = None
        self.checkpointable = snapshot
        if snapshot:
            try:
                self.native = cast(Any, import_module("perpetua_envpool_snapshot"))
            except ImportError as error:
                raise ImportError(
                    "Build native/envpool_snapshot for complete EnvPool snapshots."
                ) from error
            if (
                self.native.codec != "envpool-cartpole-mt19937-v1"
                or self.native.upstream_revision != "9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3"
            ):
                raise ValueError("Unsupported EnvPool native snapshot codec/revision.")
        self.configuration: dict[str, JSONValue] = {
            "backend": "envpool",
            "task": "CartPole-v1",
            "num_envs": num_envs,
            "num_threads": num_threads,
            "legacy_time_limit": episode_steps,
            "respawn": "next-tick-native-autoreset",
            "rng": "internal-seeded-EnvPool",
            "overlap": (
                "async half batches: half the slots act per tick while the other half steps"
                if overlap
                else None
            ),
            "snapshot": (
                {"codec": self.native.codec, "upstream_revision": self.native.upstream_revision}
                if snapshot
                else None
            ),
        }
        self.pool: Any = None
        self.pending = np.zeros(num_envs, dtype=np.bool_)
        self.current = np.arange(0, dtype=np.int64)
        self.last_obs = np.zeros((num_envs, *self.observation_space.shape), dtype=np.float32)

    def _order(self, info: dict[str, Any]) -> NDArray[np.int64]:
        ids = np.asarray(info["env_id"])
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.arange(self.num_slots)):
            raise ValueError("EnvPool did not return a complete synchronous simulator batch.")
        return order

    def _receive(self) -> tuple[Any, ...]:
        """One async half batch: (ids, obs, reward, done) for the half that finished."""
        obs, reward, terminated, truncated, info = self.pool.recv()
        ids = np.asarray(info["env_id"], dtype=np.int64)
        done = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        return ids, np.asarray(obs, dtype=np.float32), np.asarray(reward, dtype=np.float32), done

    def _step_overlapped(self, action: NDArray[Any]) -> BackendResult:
        # Send the half that acted, then take whichever half finished stepping
        # meanwhile: that half acts next, while the one just sent steps.
        self.pool.send(np.asarray(action)[self.current].astype(np.int32), self.current)
        ids, obs, reward, done = self._receive()
        self.last_obs[ids] = obs
        rewards = np.zeros(self.num_slots, dtype=np.float32)
        rewards[ids] = reward
        respawn = np.zeros(self.num_slots, dtype=np.bool_)
        respawn[ids] = self.pending[ids]
        self.pending[ids] = done
        active = np.zeros(self.num_slots, dtype=np.bool_)
        active[ids] = ~done
        self.current = ids
        return BackendResult(self.last_obs.copy(), rewards, active, respawn)

    def init(self, seed: int) -> NDArray[np.float32]:
        seed = _pool_seed(seed, self.num_slots)
        if self.overlap:
            self.pool = self.module.make_gymnasium(
                "CartPole-v1",
                num_envs=self.num_slots,
                batch_size=self.num_slots // 2,
                num_threads=self.num_threads,
                seed=seed,
                max_episode_steps=self.episode_steps,
            )
            self.pool.async_reset()
            ids, obs, _, _ = self._receive()
            self.last_obs[ids] = obs
            self.current = ids
            # Only the received half has an observation; the rest start inactive.
            self.initial_active = np.zeros(self.num_slots, dtype=np.bool_)
            self.initial_active[ids] = True
            return self.last_obs.copy()
        if self.checkpointable:
            self.pool = self.native.CartPolePool(
                self.num_slots, self.num_threads, self.episode_steps, seed
            )
        else:
            self.pool = self.module.make_gymnasium(
                "CartPole-v1",
                num_envs=self.num_slots,
                batch_size=self.num_slots,
                num_threads=self.num_threads,
                seed=seed,
                max_episode_steps=self.episode_steps,
                gym_reset_return_info=True,
            )
        obs, info = self.pool.reset()
        return np.asarray(obs, dtype=np.float32)[self._order(info)]

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        del reset_seed  # Simulator/reset RNG belongs to EnvPool, not a fake JAX snapshot.
        if self.overlap:
            return self._step_overlapped(action)
        obs, reward, terminated, truncated, info = self.pool.step(action)
        order = self._order(info)
        respawn = self.pending.copy()
        self.pending = (
            np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        )[order]
        return BackendResult(
            np.asarray(obs, dtype=np.float32)[order],
            np.asarray(reward, dtype=np.float32)[order],
            ~self.pending,
            respawn,
        )

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def snapshot(self) -> EnvPoolSnapshot:
        if not self.checkpointable:
            raise NotImplementedError("EnvPool does not expose a complete checkpoint snapshot.")
        if self.pool is None:
            raise RuntimeError("EnvPool is not running.")
        configuration, physics, rng, counters = self.pool.snapshot()
        return EnvPoolSnapshot(configuration, physics, rng, counters, self.pending.copy())

    def restore(self, snapshot: BackendSnapshot) -> None:
        if not self.checkpointable:
            raise NotImplementedError("EnvPool does not expose a complete checkpoint restore.")
        if self.pool is None:
            raise RuntimeError("EnvPool is not running.")
        if type(snapshot) is not EnvPoolSnapshot:
            raise ValueError("Expected an EnvPool CartPole snapshot.")
        if (
            snapshot.pending.shape != (self.num_slots,)
            or snapshot.pending.dtype != np.bool_
            or snapshot.counters.shape != (self.num_slots, 5)
            or not np.array_equal(snapshot.pending, snapshot.counters[:, 3])
        ):
            raise ValueError("Invalid EnvPool pending reset mask/counters.")
        self.pool.restore(snapshot.configuration, snapshot.physics, snapshot.rng, snapshot.counters)
        self.pending = snapshot.pending.copy()


class _EnvPoolAtariBackend:
    """A batched Atari pool through EnvPool's Gymnasium interface.

    Uses ``full_action_space=True`` so every game shares the same 18-action
    space, which is what makes switching between games possible without
    changing the compiled action shape. EnvPool is Linux-only in this project;
    its simulator state is opaque, so there is no snapshot codec here.
    """

    uses_reset_seed = False
    checkpointable = False
    task: str
    num_slots: int
    num_threads: int
    module: Any
    configuration: dict[str, JSONValue]
    pool: Any
    pending: NDArray[np.bool_]
    observation_space: gym.spaces.Box[Any]
    action_space: gym.spaces.Discrete

    def __init__(self, task: str, *, num_envs: int, num_threads: int) -> None:
        if sys.platform != "linux":
            raise RuntimeError("The EnvPool Atari bridge is currently validated on Linux only.")
        if (
            type(num_envs) is not int
            or num_envs < 1
            or type(num_threads) is not int
            or num_threads < 1
        ):
            raise ValueError("EnvPool Atari slot/thread counts must be positive integers.")
        try:
            self.module = cast(Any, import_module("envpool"))
        except ImportError as error:
            raise ImportError(
                "Install Perpetua's cpu-bridge extra and EnvPool's runtime dependencies."
            ) from error
        self.task = task
        self.num_slots = num_envs
        self.num_threads = num_threads
        self.configuration: dict[str, JSONValue] = {
            "backend": "envpool-atari",
            "task": task,
            "num_envs": num_envs,
            "num_threads": num_threads,
            "full_action_space": True,
            "respawn": "terminated-or-truncated",
            "rng": "internal-seeded-EnvPool",
        }
        self.pool: Any = None
        # EnvPool stacks four frames first, (4, 84, 84); they are moved last so
        # the observation is (H, W, C), as the CNN torso and the other pixel
        # backends expect. Values stay 0-255; a CNN learner given
        # this observation space divides them by 255.
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(18)

    @staticmethod
    def _frames(obs: Any, order: NDArray[np.int64]) -> NDArray[np.float32]:
        stacked = np.asarray(obs, dtype=np.float32)[order]
        return np.ascontiguousarray(np.transpose(stacked, (0, 2, 3, 1)))

    def _order(self, info: dict[str, Any]) -> NDArray[np.int64]:
        ids = np.asarray(info["env_id"])
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.arange(self.num_slots)):
            raise ValueError("EnvPool did not return a complete synchronous simulator batch.")
        return order

    def init(self, seed: int) -> NDArray[np.float32]:
        seed = _pool_seed(seed, self.num_slots)
        self.pool = self.module.make_gymnasium(
            self.task,
            num_envs=self.num_slots,
            batch_size=self.num_slots,
            num_threads=self.num_threads,
            full_action_space=True,
            gym_reset_return_info=True,
            seed=seed,
        )
        obs, info = self.pool.reset()
        self.pending = np.zeros(self.num_slots, dtype=np.bool_)
        return self._frames(obs, self._order(info))

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        del reset_seed  # Simulator and reset RNG belong to EnvPool.
        obs, reward, terminated, truncated, info = self.pool.step(action)
        order = self._order(info)
        # EnvPool resets a finished game on its next step and ignores that
        # step's action. As in the CartPole backend, the finished slot is
        # inactive on the tick that ends its game (so the ignored action is not
        # the learner's), and the respawn is reported with the reset frame.
        respawn = self.pending.copy()
        self.pending = (
            np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        )[order]
        return BackendResult(
            self._frames(obs, order),
            np.asarray(reward, dtype=np.float32)[order],
            ~self.pending,
            respawn,
        )

    def snapshot(self) -> BackendSnapshot:
        """Atari EnvPool state is opaque; there is no snapshot codec."""
        raise ValueError("Atari EnvPool does not support snapshots.")

    def restore(self, snapshot: BackendSnapshot) -> None:
        """Atari EnvPool state is opaque; there is nothing to restore."""
        del snapshot
        raise ValueError("Atari EnvPool does not support snapshots.")

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()


class EnvPoolAtari(CPULifetime):
    """One Atari game as a host-controlled, batched, full-action-space lifetime."""

    def __init__(
        self,
        task: str = "Breakout-v5",
        *,
        num_envs: int = 8,
        num_threads: int = 4,
        drift: Drift | None = None,
        compiled: bool = False,
    ) -> None:
        super().__init__(
            _EnvPoolAtariBackend(task, num_envs=num_envs, num_threads=num_threads),
            env_id=f"envpool/{task}",
            drift=drift,
            compiled=compiled,
        )


class EnvPoolCartPole(CPULifetime):
    """Batched C++ CartPole simulator slots with host orchestration and reward drift.

    ``snapshot=True`` selects the optional headless native binding, preserving
    float64 physics, MT19937 RNGs and pending resets in portable CPU state.
    Ordinary EnvPool remains opaque and single-use, with no per-step snapshots.

    ``episode_steps`` is EnvPool's own CartPole time limit, which is kept: a
    slot whose episode terminates or is truncated reports its reward and is
    inactive on that tick, and EnvPool resets it natively on the next tick
    (ignoring that tick's action), which reports ``respawn``.
    ``overlap=True`` (an even ``num_envs >= 2``, without ``snapshot``) steps
    half the slots asynchronously while the other half acts, so each tick only
    one half reports observations and rewards; the other half is inactive with
    zero reward and its previous observation.
    """

    def __init__(
        self,
        *,
        num_envs: int = 32,
        num_threads: int = 4,
        episode_steps: int = 500,
        snapshot: bool = False,
        drift: Drift | None = None,
        compiled: bool = False,
        overlap: bool = False,
    ) -> None:
        if snapshot and compiled:
            raise ValueError("compiled=True states are single-use; use snapshot=False.")
        backend = _EnvPoolBackend(
            num_envs=num_envs,
            num_threads=num_threads,
            episode_steps=episode_steps,
            snapshot=snapshot,
            overlap=overlap,
        )
        super().__init__(backend, env_id="envpool/CartPole-v1", drift=drift, compiled=compiled)
