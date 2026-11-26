"""Gym-familiar construction with lazy imports for optional integrations."""

import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from perpetua.core import Environment, requires_extra
from perpetua.drift import Drift
from perpetua.envs.continual_grid import ContinualGrid

if TYPE_CHECKING:
    from perpetua.adapters.gymnax import (
        CartPoleParams,
        CartPoleState,
        PendulumParams,
        PendulumState,
    )
    from perpetua.adapters.host import EnvPoolCartPole
    from perpetua.adapters.jaxmarl import JaxMARLState, SimpleSpreadParams
    from perpetua.envs.continual_grid import GridParams, GridState
    from perpetua.envs.door_grid import DoorParams, DoorState
    from perpetua.envs.opponent_grid import OpponentParams, OpponentState
    from perpetua.envs.partner_grid import PartnerParams, PartnerState
    from perpetua.params import DriftEnv

__all__ = [
    "EnvSpec",
    "register",
    "unregister",
    "registered",
    "environments",
    "spec",
    "make",
]


@dataclass(frozen=True)
class EnvSpec:
    """What the registry records about a built-in id, known without constructing it.

    Attributes:
        id: The environment id.
        namespace: The id's prefix (``"gymnax"`` in ``"gymnax/CartPole-v1"``),
            or ``None`` for a native world.
        extra: The optional extra ``make`` needs to construct the id, or ``None``.
        optional_extra: An extra that only some of the id's options need
            (MORPHEUS's live client; its default mock needs none), or ``None``.
        host: Whether it steps a simulator outside JAX, so its
            ``metadata["execution"]`` is ``"host"``.
        platform: The only ``sys.platform`` it runs on, or ``None`` for any.
        setup: The module (run with ``python -m``) that builds its separate
            runtime, or ``None``.
    """

    id: str
    namespace: str | None = None
    extra: str | None = None
    optional_extra: str | None = None
    host: bool = False
    platform: str | None = None
    setup: str | None = None


_USER_ENVIRONMENTS: dict[str, Callable[..., Environment[Any]]] = {}
# One entry per id prefix; ``spec`` fills in the id.
_NAMESPACES = {
    None: EnvSpec(""),
    "gymnax": EnvSpec("", extra="interop"),
    "jaxmarl": EnvSpec("", extra="jaxmarl"),
    "meal": EnvSpec("", extra="meal"),
    "mjx": EnvSpec("", extra="mjx"),
    "metaworld": EnvSpec("", extra="metaworld", host=True),
    "envpool": EnvSpec("", extra="cpu-bridge", host=True, platform="linux"),
    "coom": EnvSpec("", host=True, setup="perpetua.adapters.coom_setup"),
    "agarcl": EnvSpec("", host=True, setup="perpetua.adapters.agarcl_setup"),
    "morpheus": EnvSpec("", optional_extra="morpheus", host=True),
}


def register(
    env_id: str, factory: Callable[..., Environment[Any]], *, overwrite: bool = False
) -> None:
    """Register a third-party environment factory under ``env_id``.

    The factory receives the keyword arguments passed to ``make`` and must
    return an object satisfying the Environment contract (see
    ``docs/custom-environments.md``). Built-in ids and existing third-party
    ids are refused unless ``overwrite=True``.
    """
    if type(env_id) is not str or not env_id:
        raise ValueError("Registered environment ids must be non-empty strings.")
    if not callable(factory):
        raise ValueError("Registered environment factories must be callable.")
    if env_id in _ENVIRONMENTS and not overwrite:
        raise ValueError(
            f"{env_id!r} is a built-in environment; pass overwrite=True to replace it."
        )
    if env_id in _USER_ENVIRONMENTS and not overwrite:
        raise ValueError(f"{env_id!r} is already registered; pass overwrite=True to replace it.")
    _USER_ENVIRONMENTS[env_id] = factory


def unregister(env_id: str) -> None:
    """Remove a third-party registration; built-in ids cannot be removed."""
    if env_id not in _USER_ENVIRONMENTS:
        raise ValueError(f"{env_id!r} is not a registered third-party environment.")
    del _USER_ENVIRONMENTS[env_id]


def registered() -> tuple[str, ...]:
    """Return the sorted third-party environment ids."""
    return tuple(sorted(_USER_ENVIRONMENTS))


def environments(*, builtin: bool = False) -> tuple[str, ...]:
    """Return every id ``make`` accepts: the built-in ids, then third-party ones.

    ``builtin=True`` returns the built-in ids only.
    """
    if builtin:
        return _ENVIRONMENTS
    return (*_ENVIRONMENTS, *(env_id for env_id in registered() if env_id not in _ENVIRONMENTS))


def spec(env_id: str) -> EnvSpec:
    """Return the registry's record for a built-in id.

    Raises ``ValueError`` for any other id: the registry knows nothing about
    a third-party factory.
    """
    if env_id not in _ENVIRONMENTS:
        raise ValueError(f"{env_id!r} is not a built-in environment.")
    namespace = env_id.partition("/")[0] if "/" in env_id else None
    return replace(_NAMESPACES[namespace], id=env_id, namespace=namespace)


# Other MPE tasks run through the generic JaxMARLMPE adapter; simple_spread
# keeps its scripted-partner adapter above.
_JAXMARL_MPE = (
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "MPE_simple_crypto_v3",
)

# Every other gymnax environment runs through the generic parameter adapter.
# CartPole and Pendulum keep their hand-written wrappers on purpose: CartPole
# zeroes the reward on failure ticks (measured: 358 vs 400 over 400 ticks with
# the same 42 respawns) and Pendulum seeds its reset from a split key.
_GYMNAX_GENERIC = (
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
    "Catch-bsuite",
    "DeepSea-bsuite",
    "MemoryChain-bsuite",
    "UmbrellaChain-bsuite",
    "DiscountingChain-bsuite",
    "MNISTBandit-bsuite",
    "FourRooms-misc",
    "MetaMaze-misc",
    "BernoulliBandit-misc",
    "GaussianBandit-misc",
    "Reacher-misc",
    "Swimmer-misc",
    "Pong-misc",
)


_ENVIRONMENTS = (
    "ContinualGrid-v0",
    "CooperativeGrid-v0",
    "DoorGrid-v0",
    "ForagerGrid-v0",
    "BeanWorld-v0",
    "PartnerGrid-v0",
    "OpponentGrid-v0",
    "gymnax/CartPole-v1",
    "gymnax/Pendulum-v1",
    *(f"gymnax/{name}" for name in _GYMNAX_GENERIC),
    "envpool/CartPole-v1",
    "envpool/Atari-v0",
    "meal/overcooked-v0",
    "metaworld/MT1-v0",
    "coom/COOM-v0",
    "morpheus/Morpheus-v0",
    "agarcl/agario-screen-v0",
    "agarcl/agario-grid-v0",
    "agarcl/grand-arena-v0",
    "mjx/Pendulum-v0",
    "mjx/CartPole-v0",
    "jaxmarl/MPE_simple_spread-v3",
    *(f"jaxmarl/{name}" for name in _JAXMARL_MPE),
)


@overload
def make(
    env_id: Literal["DoorGrid-v0"],
    *,
    size: int = 5,
    num_agents: int = 1,
    drift: Drift | Mapping[str, Drift] | None = None,
) -> "DriftEnv[DoorState, DoorParams]": ...


@overload
def make(
    env_id: Literal["CooperativeGrid-v0"],
    *,
    size: int = 5,
    num_agents: int = 2,
    drift: Drift | Mapping[str, Drift] | None = None,
) -> "DriftEnv[GridState, GridParams]": ...


@overload
def make(
    env_id: Literal["PartnerGrid-v0"],
    *,
    size: int = 5,
    num_learners: int = 1,
    num_partners: int = 1,
    drift: Drift | Mapping[str, Drift] | None = None,
    partner_drift: Drift | None = None,
) -> "DriftEnv[PartnerState, PartnerParams]": ...


@overload
def make(
    env_id: Literal["OpponentGrid-v0"],
    *,
    size: int = 5,
    num_learners: int = 1,
    num_opponents: int = 1,
    drift: Drift | Mapping[str, Drift] | None = None,
    opponent_drift: Drift | None = None,
) -> "DriftEnv[OpponentState, OpponentParams]": ...


@overload
def make(
    env_id: Literal["ContinualGrid-v0"],
    *,
    size: int = 5,
    num_agents: int = 1,
    drift: Drift | Mapping[str, Drift] | None = None,
) -> "DriftEnv[GridState, GridParams]": ...


@overload
def make(
    env_id: Literal["gymnax/CartPole-v1"], *, drift: Drift | None = None
) -> "DriftEnv[CartPoleState, CartPoleParams]": ...


@overload
def make(
    env_id: Literal["gymnax/Pendulum-v1"], *, drift: Drift | None = None
) -> "DriftEnv[PendulumState, PendulumParams]": ...


@overload
def make(
    env_id: Literal["envpool/CartPole-v1"],
    *,
    num_envs: int = 32,
    num_threads: int = 4,
    episode_steps: int = 500,
    snapshot: bool = False,
    drift: Drift | None = None,
    compiled: bool = False,
    overlap: bool = False,
) -> "EnvPoolCartPole": ...


@overload
def make(
    env_id: Literal["jaxmarl/MPE_simple_spread-v3"],
    *,
    num_agents: int = 3,
    num_landmarks: int = 3,
    local_ratio: float = 0.5,
    max_steps: int = 25,
    drift: Drift | Mapping[str, Drift] | None = None,
    num_scripted: int = 0,
    scripted_behavior: Literal["partner", "opponent"] = "partner",
    scripted_drift: Drift | None = None,
) -> "DriftEnv[JaxMARLState, SimpleSpreadParams]": ...


@overload
def make(env_id: str, **kwargs: Any) -> Environment[Any]: ...


def _with_field_drift(drift: Any, bare: str, field: str, keyword: str, second: Any) -> Any:
    """Merge ``drift`` (a process for ``bare``, or a mapping) with ``keyword=`` for ``field``."""
    if second is None:
        return drift
    drifts: dict[str, Drift] = {}
    if isinstance(drift, Mapping):
        drifts.update(cast("Mapping[str, Drift]", drift))
    elif drift is not None:
        drifts[bare] = drift
    if field in drifts:
        raise ValueError(
            f"{keyword}= and drift[{field!r}] both set the drift of {field!r}; pass one of them."
        )
    drifts[field] = second
    return drifts


def _marl_drifts(drift: Any, scripted_drift: Any) -> Any:
    """Merge a JaxMARL ``drift`` (one process or a mapping) with ``scripted_drift``."""
    return _with_field_drift(
        drift, "reward_multiplier", "scripted", "scripted_drift", scripted_drift
    )


def make(env_id: str, **kwargs: Any) -> Environment[Any]:
    """Construct a supported environment; optional libraries load only when requested.

    Ids registered with :func:`register` are looked up first, and their
    factory receives every keyword argument. For built-in parameter
    environments, ``drift=`` is a drift process (applied to the environment's
    default drifting field) or a mapping from parameter names to processes;
    without it nothing drifts. ``PartnerGrid-v0`` also takes
    ``partner_drift=`` (the ``partner`` field), ``OpponentGrid-v0`` takes
    ``opponent_drift=`` (the ``opponent`` field) and the JaxMARL ids take
    ``scripted_drift=`` (the ``scripted`` field; a bare ``drift`` there drives
    ``reward_multiplier``). Naming the same field through such a keyword and
    in the ``drift`` mapping raises ``ValueError``. The generic gymnax ids take
    ``drift_fields=``, the float parameter fields that may drift (by default
    every one the adapter supports), and no other keyword arguments; a bare ``drift`` there drives
    the first of those fields (``dt`` for Acrobot, ``min_position`` for
    MountainCar), and an id with none (``gymnax/Pong-misc``, the MinAtar
    ids, ``Catch``, ``MemoryChain``, ``UmbrellaChain`` and ``MNISTBandit``)
    raises ``ValueError`` for any ``drift``. Elsewhere the remaining keyword
    arguments go to the environment's constructor. Raises ``ValueError`` for
    an unknown id.
    """
    if env_id in _USER_ENVIRONMENTS:
        return _USER_ENVIRONMENTS[env_id](**kwargs)
    if env_id not in _ENVIRONMENTS:
        available = environments()
        close = difflib.get_close_matches(env_id, available, n=3, cutoff=0.5)
        hint = f" Did you mean {' or '.join(map(repr, close))}?" if close else ""
        raise ValueError(f"Unknown environment {env_id!r}.{hint} Available: {available}")
    record = spec(env_id)
    extra = record.extra or record.optional_extra
    if extra is None:
        return _build(env_id, **kwargs)
    with requires_extra(extra, env_id):
        return _build(env_id, **kwargs)


def _build(env_id: str, **kwargs: Any) -> Environment[Any]:
    if env_id == "ContinualGrid-v0":
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(ContinualGrid(**kwargs), drift, env_id="ContinualGrid-v0")
    if env_id == "CooperativeGrid-v0":
        from perpetua.envs.cooperative_grid import CooperativeGrid
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(CooperativeGrid(**kwargs), drift, env_id="CooperativeGrid-v0")
    if env_id == "ForagerGrid-v0":
        from perpetua.envs.forager_grid import ForagerGrid
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(ForagerGrid(**kwargs), drift, env_id="ForagerGrid-v0")
    if env_id == "BeanWorld-v0":
        from perpetua.envs.bean_world import BeanWorld
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(BeanWorld(**kwargs), drift, env_id="BeanWorld-v0")
    if env_id == "DoorGrid-v0":
        from perpetua.envs.door_grid import DoorGrid
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(DoorGrid(**kwargs), drift, env_id="DoorGrid-v0")
    if env_id == "PartnerGrid-v0":
        from perpetua.envs.partner_grid import PartnerGrid
        from perpetua.params import DriftEnv

        drifts = _with_field_drift(
            kwargs.pop("drift", None),
            "goal",
            "partner",
            "partner_drift",
            kwargs.pop("partner_drift", None),
        )
        return DriftEnv(cast(Any, PartnerGrid(**kwargs)), drifts, env_id="PartnerGrid-v0")
    if env_id == "OpponentGrid-v0":
        from perpetua.envs.opponent_grid import OpponentGrid
        from perpetua.params import DriftEnv

        drifts = _with_field_drift(
            kwargs.pop("drift", None),
            "goal",
            "opponent",
            "opponent_drift",
            kwargs.pop("opponent_drift", None),
        )
        return DriftEnv(cast(Any, OpponentGrid(**kwargs)), drifts, env_id="OpponentGrid-v0")
    if env_id.startswith("gymnax/") and env_id.split("/", 1)[1] in _GYMNAX_GENERIC:
        from perpetua.adapters.gymnax_params import GymnaxParameters
        from perpetua.params import DriftEnv

        backend = env_id.split("/", 1)[1]
        drift = kwargs.pop("drift", None)
        drift_fields = kwargs.pop("drift_fields", None)
        if kwargs:
            raise TypeError(
                f"{env_id} accepts only drift= and drift_fields=, not {', '.join(sorted(kwargs))}."
            )
        return DriftEnv(GymnaxParameters(backend, drift_fields=drift_fields), drift, env_id=env_id)
    if env_id == "gymnax/CartPole-v1":
        from perpetua.adapters.gymnax import GymnaxCartPole
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(GymnaxCartPole(**kwargs), drift, env_id="gymnax/CartPole-v1")
    if env_id == "gymnax/Pendulum-v1":
        from perpetua.adapters.gymnax import GymnaxPendulum
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(GymnaxPendulum(**kwargs), drift, env_id="gymnax/Pendulum-v1")
    if env_id.startswith("jaxmarl/") and env_id.split("/", 1)[1] in _JAXMARL_MPE:
        from perpetua.adapters.jaxmarl_mpe import JaxMARLMPE
        from perpetua.params import DriftEnv

        drifts = _marl_drifts(kwargs.pop("drift", None), kwargs.pop("scripted_drift", None))
        return DriftEnv(JaxMARLMPE(env_id.split("/", 1)[1], **kwargs), drifts, env_id=env_id)
    if env_id == "jaxmarl/MPE_simple_spread-v3":
        from perpetua.adapters.jaxmarl import JaxMARLSimpleSpread
        from perpetua.params import DriftEnv

        drifts = _marl_drifts(kwargs.pop("drift", None), kwargs.pop("scripted_drift", None))
        return DriftEnv(
            JaxMARLSimpleSpread(**kwargs), drifts, env_id="jaxmarl/MPE_simple_spread-v3"
        )
    if env_id == "envpool/Atari-v0":
        from perpetua.adapters.host import EnvPoolAtari

        return EnvPoolAtari(**kwargs)
    if env_id == "meal/overcooked-v0":
        from perpetua.adapters.meal import MealOvercooked
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        return DriftEnv(MealOvercooked(**kwargs), drift, env_id=env_id)
    if env_id == "metaworld/MT1-v0":
        from perpetua.adapters.host import MetaWorldLifetime

        return MetaWorldLifetime(**kwargs)
    if env_id == "coom/COOM-v0":
        from perpetua.adapters.coom import COOMLifetime

        return COOMLifetime(**kwargs)
    if env_id == "morpheus/Morpheus-v0":
        from perpetua.adapters.morpheus import MorpheusLifetime

        return MorpheusLifetime(**kwargs)
    if env_id in ("agarcl/agario-screen-v0", "agarcl/agario-grid-v0"):
        from perpetua.adapters.agarcl import AgarCLLifetime

        return AgarCLLifetime(obs_type=env_id.split("-")[1], **kwargs)
    if env_id in ("mjx/Pendulum-v0", "mjx/CartPole-v0"):
        from perpetua.adapters.mjx import MJXControl
        from perpetua.params import DriftEnv

        drift = kwargs.pop("drift", None)
        task = "pendulum" if env_id == "mjx/Pendulum-v0" else "cartpole"
        return DriftEnv(MJXControl(task=task, **kwargs), drift, env_id=env_id)
    if env_id == "agarcl/grand-arena-v0":
        from perpetua.adapters.agarcl import GRAND_ARENA, AgarCLLifetime

        return AgarCLLifetime(**{**GRAND_ARENA, **kwargs}, env_id=env_id)
    from perpetua.adapters.host import EnvPoolCartPole

    return EnvPoolCartPole(**kwargs)
