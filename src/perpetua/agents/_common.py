"""Pieces the PPO, DQN and SAC learners and their runners share."""

# JAX exposes some partially annotated transformation helpers.
# pyright: reportUnknownMemberType=false

import math
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from perpetua import compute, runtime
from perpetua.agents.encoders import derive_pixel_scale
from perpetua.core import Environment


def as_float(value: object, message: str) -> float:
    """Return ``value`` as a Python float, or raise ``ValueError`` saying what was received.

    The float counterpart of ``perpetua.core.as_int`` for configuration
    fields: accepts Python ints and floats and NumPy or JAX real scalars
    (zero-dimensional arrays included); rejects bools, strings, traced values
    and arrays with dimensions. A config then holds plain floats, which its
    JSON record in a checkpoint needs.
    """
    received = f"{message} Got {type(value).__name__} {value!r}."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(received)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(cast(Any, value))
    if isinstance(value, (np.ndarray, jax.Array)) and not isinstance(value, jax.core.Tracer):
        array = cast(Any, value)
        real = np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
        if array.ndim == 0 and real:
            return float(array)
    raise ValueError(received)


def set_floats(config: object, names: tuple[str, ...]) -> None:
    """Replace the named fields of a frozen config dataclass by Python floats."""
    for name in names:
        value = as_float(getattr(config, name), f"{name} must be a real number.")
        object.__setattr__(config, name, value)


def as_float32(leaf: Any) -> jax.Array:
    """Hyperparameter override leaf as a float32 array, like ``from_config`` makes."""
    return jnp.asarray(leaf, dtype=jnp.float32)


def check_discrete_actions(
    env: Environment[Any], mask: jax.Array, action_size: int, learner: str
) -> None:
    """Refuse a continuous environment, or an ``action_size`` other than the mask width."""
    space = cast(Any, env).action_space
    if isinstance(space, spaces.Box):
        raise ValueError(
            f"{learner} needs a discrete action space, but this environment's action_space is "
            f"a continuous Box of shape {space.shape}; use pp.agents.SAC for continuous actions."
        )
    width = int(mask.shape[-1])
    if width != action_size:
        raise ValueError(
            f"{learner} was built with action_size={action_size}, but the environment's action "
            f"mask has {width} actions (a task switch pads every task to its largest action "
            "count). Pass action_size=int(env.action_space.nvec[0])."
        )


def check_torso_observations(obs: jax.Array, torso: str, learner: str) -> None:
    """The MLP torso reads ``(agents, features)``; pixels need ``torso="cnn"``."""
    if torso == "mlp" and obs.ndim != 2:
        raise ValueError(
            f"The MLP torso needs (agents, features) observations; {learner} got "
            f"{tuple(obs.shape)}. Use the config's torso='cnn' for pixel observations."
        )


def check_environment_pixel_scale(env: Environment[Any], agent: Any, learner: str) -> None:
    """Refuse a CNN learner whose derived pixel scale is not this environment's.

    A learner built without ``observation_space`` (or with another
    environment's) would otherwise read raw 8-bit pixels unscaled, or divide
    other input by 255. A configured ``pixel_scale`` is the user's choice and
    is not checked.
    """
    if agent.config.torso != "cnn" or agent.config.pixel_scale is not None:
        return
    expected = derive_pixel_scale(cast(Any, env).observation_space)
    if agent.pixel_scale != expected:
        raise ValueError(
            f"{learner}'s CNN torso divides observations by {agent.pixel_scale:g}, but this "
            f"environment's observation_space gives a pixel scale of {expected:g} (255 for "
            "bounds of 0 and 255, else 1). Build the learner with "
            "observation_space=env.observation_space, or set the config's pixel_scale to "
            "choose the scale yourself."
        )


def checked_hyperparameters(hyperparameters: Any) -> dict[str, float] | None:
    """Host values of a hyperparameter tuple that passed the checks every learner shares.

    Every value must be finite, the learning rate and maximum gradient norm
    positive and ``centering_step_size`` in [0, 1]; each learner then checks
    its own fields. ``None`` inside a traced transformation, where the values
    are not concrete and nothing is checked.
    """
    leaves = [getattr(hyperparameters, name) for name in hyperparameters._fields]
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return None
    values = {name: float(getattr(hyperparameters, name)) for name in hyperparameters._fields}
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"hyperparameter {name} must be finite.")
    if values["learning_rate"] <= 0 or values["max_grad_norm"] <= 0:
        raise ValueError("Hyperparameter learning rate and maximum gradient norm must be positive.")
    if not 0 <= values["centering_step_size"] <= 1:
        raise ValueError("centering_step_size must be in [0, 1].")
    return values


def l2_regularized(
    loss: jax.Array, params: Any, coefficient: jax.Array, *, configured: float
) -> jax.Array:
    """``loss`` plus ``coefficient`` times the sum of squares of every parameter leaf.

    The term is gated statically by the ``configured`` coefficient: zero skips
    it entirely, so sweeping the traced ``coefficient`` requires a nonzero
    configured default.
    """
    if configured <= 0.0:
        return loss
    penalty = jnp.sum(jnp.stack([jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(params)]))
    return loss + coefficient * penalty


def periodic_reset[StateT](
    state: StateT, interval: int, reset: Callable[[StateT], StateT]
) -> StateT:
    """Count one update toward the periodic reset and apply ``reset`` when it is due.

    ``state.reset_age`` counts ``update`` calls since the last reset. On the
    call that brings it to ``interval`` the state is replaced by
    ``reset(state)``, which must return ``reset_age`` zero; otherwise only the
    age advances.
    """
    age = cast(Any, state).reset_age + jnp.int32(1)

    def keep_age(state: StateT) -> StateT:
        return cast(Any, state)._replace(reset_age=age)

    return cast(StateT, jax.lax.cond(age >= interval, reset, keep_age, state))


def weighted_mean(values: jax.Array, active: jax.Array) -> jax.Array:
    """The mean of ``values`` over active entries (0 when none is active)."""
    weights = active.astype(jnp.float32)
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def advance_cursor(cursor: jax.Array, capacity: int) -> jax.Array:
    """The insertion count, held in ``[capacity, 2 * capacity)`` once the ring is full.

    Subtracting ``capacity`` keeps ``cursor % capacity`` (the write index) and
    ``min(cursor, capacity)`` (the fill level) unchanged, so an int32 cursor
    never overflows however long the lifetime runs.
    """
    advanced = cursor + jnp.int32(1)
    return jnp.where(advanced >= 2 * capacity, advanced - capacity, advanced).astype(jnp.int32)


class Bridge(NamedTuple):
    """A transition waiting for its slot's next active tick.

    ``reward`` is the discounted reward collected since the slot acted and
    ``discount`` the discount accumulated over the ticks it has been inactive
    since (1 for an ordinary one-tick transition).
    """

    reward: jax.Array
    discount: jax.Array
    open: jax.Array


def init_bridge(num_agents: int) -> Bridge:
    """No transition is waiting."""
    return Bridge(
        jnp.zeros(num_agents, dtype=jnp.float32),
        jnp.ones(num_agents, dtype=jnp.float32),
        jnp.zeros(num_agents, dtype=jnp.bool_),
    )


def bridge_step(
    bridge: Bridge,
    active: jax.Array,
    next_active: jax.Array,
    reward: jax.Array,
    gamma: jax.Array,
) -> tuple[Bridge, jax.Array]:
    """Advance the waiting transitions by one tick.

    A slot that acts this tick starts a transition; one that is inactive adds
    this tick's discounted reward to the transition it left open. Returns the
    new bridge and the slots whose transition completes now, because the slot
    is active again and its next observation is a real one.
    """
    reward = jnp.where(active, reward, bridge.reward + gamma * bridge.discount * reward)
    discount = jnp.where(active, 1.0, bridge.discount * gamma).astype(jnp.float32)
    waiting = active | bridge.open
    return Bridge(reward, discount, waiting & ~next_active), waiting & next_active


def training_health(
    previous: jax.Array, metrics_healthy: jax.Array, trained: Any, statistics: Any
) -> jax.Array:
    """Stay healthy only while the metrics and every learner/statistics leaf are finite.

    Once false it stays false.
    """
    finite = jnp.all(
        jnp.stack(
            [jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves((trained, statistics))]
        )
    )
    return previous & metrics_healthy & finite


def runner_metadata(
    env: Environment[Any],
    agent: Any,
    reference: runtime.Policy | None,
    *,
    host_compiler_options: dict[str, Any] | None = None,
) -> dict[str, object]:
    """What a learner runner records: the environment, the learner and how it compiles.

    A compiled runner records the measured compiler options; a host runner
    passes the options its compiled pieces use.
    """
    options: dict[str, object] = (
        {"compiler_options": compute.measured_compiler_options()}
        if host_compiler_options is None
        else {"host_compiler_options": host_compiler_options.copy()}
    )
    return {
        "environment": env.metadata.copy(),
        "learner": agent.configuration,
        **options,
        **runtime.reference_metadata(reference),
    }


def compile_chunk[CarryT](
    run: Callable[[CarryT], CarryT], *, donate: bool
) -> Callable[[CarryT], CarryT]:
    """Compile one runner chunk with the deterministic options, donating the carry on request.

    The compiler options make this the outermost compiled call: JAX accepts
    them only on a top-level ``jax.jit``, so the chunk can be vmapped but not
    traced inside another ``jax.jit``, which is refused with the remedy.
    """
    compiled = cast(
        Callable[[CarryT], CarryT],
        jax.jit(
            run,
            compiler_options=compute.measured_compiler_options(),
            donate_argnums=(0,) if donate else (),
        ),
    )

    def chunk(carry: CarryT) -> CarryT:
        try:
            return compiled(carry)
        except ValueError as error:
            # JAX's own refusal of compiler options on a nested jit.
            if "compiler_options" not in str(error) or "top-level" not in str(error):
                raise
            raise ValueError(
                "A learner runner is already compiled and must be the outermost compiled "
                "call: do not wrap it in jax.jit. Call runner(carry) directly, or "
                "jax.vmap(runner)(carries) for a batch of lifetimes, without jax.jit around it."
            ) from error

    return chunk
