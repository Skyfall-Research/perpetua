"""Reusable compiled chunks over complete continuing-lifetime carries."""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import warnings
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from perpetua import drift as drift_module
from perpetua import metrics
from perpetua.core import Environment, Info, Metadata, StateT, as_int
from perpetua.params import _REFERENCE_TAG  # pyright: ignore[reportPrivateUsage]

# The public API; the remaining module-level helpers serve the runners and learners.
__all__ = [
    "Lifetime",
    "init_lifetime",
    "assign_schedule",
    "drift_change",
    "drift_switch",
    "shadow_reward",
    "transition",
    "interact",
    "tick",
    "ChunkRunner",
    "Runner",
    "HostRunner",
]

type Policy = Callable[[jax.Array, Info, jax.Array], jax.Array]


class Lifetime[StateT](NamedTuple):
    """Environment, public observation/info, independent RNG streams and metrics.

    Policies here are stateless references, so this is not a complete
    training checkpoint: the learners in :mod:`perpetua.agents` embed a
    ``Lifetime`` in their own carries next to learner, optimizer and replay
    state.
    """

    state: StateT
    obs: jax.Array
    info: Info
    env_key: jax.Array
    agent_key: jax.Array
    metrics: metrics.Metrics


def seed_key(seed: object, caller: str = "init_lifetime") -> jax.Array:
    """``jax.random.key`` for an integer seed, refusing seeds JAX would silently alias."""
    if isinstance(seed, jax.Array) and jnp.issubdtype(seed.dtype, jax.dtypes.prng_key):
        raise TypeError(
            f"{caller} takes an integer seed, not a PRNG key; use env.init(key) to start "
            "an environment from a key."
        )
    if isinstance(seed, jax.Array | np.ndarray):
        return jax.random.key(cast(Any, seed))
    wide = jax.config.jax_enable_x64
    bound = "2**63" if wide else "2**32"
    value = as_int(
        seed,
        f"seed must be an integer in [0, {bound}): JAX keys hold 32 bits unless "
        f"jax_enable_x64 is set, and seeds that differ by a multiple of {bound} would give "
        f"identical lifetimes (reduce a larger seed with seed % {bound}).",
        minimum=0,
        maximum=(2**63 if wide else 2**32) - 1,
    )
    return jax.random.key(value)


def init_lifetime(
    env: Environment[StateT],
    *,
    seed: int | jax.Array = 0,
    window_size: int = 128,
    ema_decay: float = 0.99,
) -> Lifetime[StateT]:
    """Initialize one lifetime; seeds may be vmapped to create independent runs.

    ``seed`` is an integer in ``[0, 2**32)`` (up to ``2**63`` with
    ``jax_enable_x64``); other Python or NumPy integers and non-integers raise
    ``ValueError``, since JAX would wrap an integer and two different seeds
    would give the same lifetime. A typed PRNG key is refused: start an environment from a key
    with ``env.init(key)``. Array seeds (for ``jax.vmap``) are not range
    checked and wrap at 32 bits.
    """
    init_key, env_key, agent_key = jax.random.split(seed_key(seed), 3)
    state, obs, info = env.init(init_key)
    streaming = metrics.init(
        num_agents=env.num_agents, window_size=window_size, ema_decay=ema_decay
    )
    return Lifetime(state, obs, info, env_key, agent_key, streaming)


def assign_schedule[StateT](
    env: Environment[StateT],
    lifetime: Lifetime[StateT],
    data: drift_module.ScheduleData,
    *,
    field: str | None = None,
) -> Lifetime[StateT]:
    """Give one lifetime its own :class:`~perpetua.drift.Schedule` for ``field``.

    Traceable, so ``jax.vmap`` assigns a different schedule to every lifetime
    of a batch. The parameter, its phase and timer restart at the schedule's
    offset, and the observation is refreshed under the new parameter. ``field``
    defaults to the environment's primary field: its ``default_drift_field``
    when declared (even if that field does not drift), otherwise the first
    drifted field. Raises ``ValueError`` when the environment has no drift
    processes or the chosen field is not driven by a ``Schedule``, or when the
    environment refuses the values (a task switch refuses a ``task`` schedule
    that names a task it does not have). Under ``jax.vmap`` or ``jax.jit`` the
    values are traced and cannot be inspected, so check a stacked batch
    beforehand with ``env.check_schedule(field, data)`` where the environment
    provides it.

    It is meant for the start of a lifetime. Called later, it replaces the
    parameter without a switch event (it does not set ``boundary``) and
    without running the ``after_drift`` hook a task switch runs.
    """
    world: Any = env
    while not hasattr(world, "drifts") and hasattr(world, "env"):
        world = world.env
    drifts = getattr(world, "drifts", None)
    if drifts is None:
        raise ValueError("assign_schedule needs an environment with drift processes.")
    name = field if field is not None else getattr(world, "_primary", None)
    if name not in drifts or not isinstance(drifts[name], drift_module.Schedule):
        raise ValueError(f"Field {name!r} is not driven by a pp.drift.Schedule.")
    process = cast(drift_module.Schedule, drifts[name])
    check = getattr(getattr(world, "env", None), "check_schedule", None)
    if check is not None and not isinstance(data.values, Tracer):
        check(name, data)
    state: Any = lifetime.state
    state = state._replace(drift={**state.drift, name: process.start(data)})
    obs, info = env.observe(cast(StateT, state))
    return lifetime._replace(state=cast(StateT, state), obs=obs, info=info)


def drift_switch(state: object) -> jax.Array | None:
    """Return the hidden switch flag a backend state exposes, if any.

    Native and adapter states carry ``drift_changed``; passing it to
    ``metrics.update`` turns on in-carry recovery tracking. Backends without
    the field return ``None`` and the tracking fields stay inert.
    """
    return getattr(state, "drift_changed", None)


def drift_change(state: object) -> jax.Array | None:
    """Deprecated alias of :func:`drift_switch`."""
    warnings.warn(
        "runtime.drift_change is deprecated; use runtime.drift_switch.",
        DeprecationWarning,
        stacklevel=2,
    )
    return drift_switch(state)


def shadow_reward[StateT](
    env: Environment[StateT],
    state: StateT,
    obs: jax.Array,
    info: Info,
    reference: Policy,
    key: jax.Array,
) -> jax.Array:
    """Reward a reference policy earns from the learner's same state and step key.

    One extra environment step from the pre-action state; the shadow state is
    discarded. ``key`` is the learner's step key, so the comparison shares
    transition noise (common random numbers) while the reference's action draw
    uses an independent folded key. Costs one environment step per call and
    never enters the public ``info``. Regret against this reward is a
    same-state advantage, not best-in-hindsight re-planning.
    """
    action = reference(obs, info, jax.random.fold_in(key, _REFERENCE_TAG))
    _, _, reward, _ = env.step(state, action, key)
    return reward


def reference_metadata(reference: Policy | None) -> dict[str, str]:
    """Checkpoint metadata naming the regret reference policy, if any.

    Empty without a reference. With one, strict restores refuse resuming a
    regret accumulator against a different reference. The learner runners in
    :mod:`perpetua.agents` record it; :class:`Runner` and :class:`HostRunner`
    copy the environment's metadata and do not.
    """
    if reference is None:
        return {}
    module = getattr(reference, "__module__", None) or "?"
    name = getattr(reference, "__qualname__", None) or type(reference).__name__
    return {"regret_reference": f"{module}.{name}"}


def check_reference(env: Environment[Any], reference: Policy | None) -> None:
    """Refuse a regret reference the backend cannot shadow-step.

    A shadow step re-steps the pre-action state. JAX backends are functional
    and snapshot-capable host backends restore their snapshot on every step,
    so both are exact; opaque mutable simulators cannot fork a state, so they
    are refused here, before any simulator step happens. So are compiled
    bridges (``compiled=True`` host lifts), whose single-use state declares
    ``forkable`` False in its metadata.
    """
    if reference is None:
        return
    metadata = env.metadata
    if metadata.get("forkable", True) is False or (
        metadata.get("execution") == "host" and not metadata.get("checkpointable", False)
    ):
        raise ValueError(
            "Regret needs a shadow step from the same state; this host backend has no "
            "snapshot support, so it cannot provide one. Compiled (compiled=True) host "
            "lifts keep one single-use simulator state; drop reference= for them, or use a "
            "snapshot-capable host backend with HostRunner."
        )


def compile_reference(
    reference: Policy | None, compiler_options: dict[str, bool | int] | None = None
) -> Policy | None:
    """Compile a reference policy for host loops (``None`` passes through)."""
    if reference is None:
        return None
    return cast(Policy, jax.jit(reference, compiler_options=compiler_options))


def transition(
    env: Environment[StateT],
    lifetime: Lifetime[StateT],
    action: jax.Array,
    *,
    reference: Policy | None = None,
    update: metrics.Updater = metrics.update,
) -> tuple[Lifetime[StateT], jax.Array]:
    """Step the world once with a chosen action and measure it on the same clock.

    The single per-tick body every runner shares: it splits the environment
    key, steps, and updates the metrics carry with the reward, the declared
    drift switch (recovery tracking) and, with ``reference``, the reference
    policy's same-state reward (regret; one extra environment step, see
    :func:`shadow_reward`). The agent key is left untouched, so callers that
    hold an action across ticks (compute-aware time) can use it directly.
    ``update`` lets host loops pass a compiled ``metrics.update``. Returns
    the next lifetime and this tick's reward ``(agents,)`` float32.
    """
    env_key, step_key = jax.random.split(lifetime.env_key)
    state, obs, reward, info = env.step(lifetime.state, action, step_key)
    reference_reward = (
        None
        if reference is None
        else shadow_reward(env, lifetime.state, lifetime.obs, lifetime.info, reference, step_key)
    )
    measured = update(
        lifetime.metrics, reward, switch=drift_switch(state), reference_reward=reference_reward
    )
    return Lifetime(state, obs, info, env_key, lifetime.agent_key, measured), reward


def interact[StateT, OutT](
    env: Environment[StateT],
    lifetime: Lifetime[StateT],
    act: Callable[[jax.Array, Info, jax.Array], OutT],
    *,
    action: Callable[[OutT], jax.Array] | None = None,
    reference: Policy | None = None,
    update: metrics.Updater = metrics.update,
) -> tuple[Lifetime[StateT], OutT, jax.Array]:
    """Act from public data, then :func:`transition` — one full agent tick.

    ``act(obs, info, key)`` may return any pytree (for example a sampled
    action with log-probabilities); ``action`` extracts the environment action
    from it (identity by default). The agent key is split here and the
    environment key in ``transition``, so learners and scripted policies
    consume RNG identically. Returns the next lifetime, the policy output and
    this tick's reward.
    """
    agent_key, act_key = jax.random.split(lifetime.agent_key)
    output = act(lifetime.obs, lifetime.info, act_key)
    chosen = cast(jax.Array, output) if action is None else action(output)
    advanced, reward = transition(
        env, lifetime._replace(agent_key=agent_key), chosen, reference=reference, update=update
    )
    return advanced, output, reward


def tick(
    env: Environment[StateT],
    policy: Policy,
    carry: Lifetime[StateT],
    *,
    reference: Policy | None = None,
) -> Lifetime[StateT]:
    """Act from public data, then step and measure reward on the same clock.

    With ``reference``, every tick also computes the reference policy's
    same-state reward (one extra environment step, see :func:`shadow_reward`)
    and the metrics carry accumulates cumulative and windowed regret against
    it. Without a reference the carry, the RNG streams and the cost are
    unchanged.
    """
    return interact(env, carry, policy, reference=reference)[0]


def unalias[T](tree: T) -> T:
    """Copy repeated array objects so every leaf owns its buffer.

    Initial carries often reuse one zeros array for several fields; a donated
    call must not receive the same buffer twice. Later carries returned by a
    compiled call already own distinct buffers, so this copies nothing then.
    """
    seen: set[int] = set()

    def own(leaf: Any) -> Any:
        if not isinstance(leaf, jax.Array):
            return leaf
        if id(leaf) in seen:
            return jnp.copy(leaf)
        seen.add(id(leaf))
        return leaf

    return cast(T, jax.tree.map(own, tree))


class ChunkRunner[CarryT](Protocol):
    """Runner contract: one call advances a complete carry by one bounded chunk.

    Every runner in the library provides these three members, and anything
    that drives or wraps "any runner" (:func:`perpetua.checkpoint.replay`,
    :class:`perpetua.probes.ProbedRunner`) relies on nothing else. A custom
    runner or wrapper is accepted wherever it provides them; a wrapper forwards
    ``donates`` from the runner it wraps.

    ``runner(carry)`` returns the next carry of the same structure, with no
    reset and no done; the caller keeps only the returned carry and may
    checkpoint it. The carry is complete for every runner but one:
    :class:`perpetua.torch_bridge.TorchRunner` keeps the learner's state in
    its PyTorch agent, outside the carry, so the generic
    :func:`perpetua.checkpoint.save`, ``restore`` and ``replay`` refuse it and
    its own ``save`` and ``restore`` write and load the two together. A custom
    runner that keeps learner state outside its carry has the same limit and
    is not detected. :class:`perpetua.compute.ComputeRunner` and
    :class:`perpetua.agents.ComputePPORunner` take the chunk's timing trace as
    a second argument (``runner(state, trace)``), so they provide ``metadata``
    and ``donates`` but need the trace bound before they are driven through
    this call.
    """

    @property
    def metadata(self) -> Mapping[str, object]:
        """JSON-compatible configuration of the run, compared on checkpoint restore."""
        ...

    @property
    def donates(self) -> bool:
        """Whether a call consumes (donates) the buffers of the carry passed in.

        When True the input carry must not be read after the call; copy a
        carry that has to outlive the next call.
        """
        ...

    def __call__(self, carry: CarryT, /) -> CarryT:
        """Advance exactly one chunk and return the new carry."""
        ...


class Runner[StateT]:
    """Compile a fixed-size chunk once and reuse it through hidden drift.

    Call repeatedly without reinitializing the carry. ``jax.vmap(runner)`` batches
    lifetimes with identical shapes. Reporting metadata records Oracle exposure;
    it stays on the host and is not passed to the policy or scan. With
    ``reference`` each tick costs one extra environment step for the same-state
    reference reward (see :func:`shadow_reward`); the reference is not
    recorded in ``metadata``. Raises ``ValueError`` for a host environment
    (use :class:`HostRunner`) and unless ``ticks`` is an integer >= 1.
    ``donate=True`` donates the input carry's buffers to the compiled call, so
    large states (replay buffers, windows) are updated in place instead of
    copied. The carry passed in is then invalid: keep only the returned one.
    Results are unchanged.
    """

    def __init__(
        self,
        env: Environment[StateT],
        policy: Policy,
        *,
        ticks: int = 256,
        reference: Policy | None = None,
        donate: bool = False,
    ) -> None:
        if env.metadata.get("execution") == "host":
            raise ValueError("Use HostRunner for mutable CPU simulators, not the compiled Runner.")
        ticks = as_int(ticks, "ticks must be a positive integer.", minimum=1)
        check_reference(env, reference)
        self.metadata: Metadata = env.metadata.copy()

        def run(carry: Lifetime[StateT]) -> Lifetime[StateT]:
            def advance(state: Lifetime[StateT], _: None) -> tuple[Lifetime[StateT], None]:
                return tick(env, policy, state, reference=reference), None

            final, _ = jax.lax.scan(advance, carry, None, length=ticks)
            return final

        self._run = cast(
            Callable[[Lifetime[StateT]], Lifetime[StateT]],
            jax.jit(run, donate_argnums=(0,) if donate else ()),
        )
        self._donate = donate

    @property
    def donates(self) -> bool:
        """Whether a call consumes the carry passed in (``donate=True``)."""
        return self._donate

    def __call__(self, carry: Lifetime[StateT]) -> Lifetime[StateT]:
        """Advance exactly one chunk; no resets or per-tick host callbacks."""
        return self._run(unalias(carry) if self.donates else carry)


class HostRunner[StateT]:
    """Bounded host-controlled chunks: a Python loop over ticks with a compiled policy.

    Each call runs ``ticks`` ticks one at a time in Python (the chunk itself is
    not compiled); the policy and ``reference`` are jitted. Works for host and
    JAX environments. Raises ``ValueError`` unless ``ticks`` is an integer
    >= 1, or for a ``reference`` on a host backend without snapshot support.
    ``metadata`` is a copy of the environment's and does not record the
    reference.
    """

    def __init__(
        self,
        env: Environment[StateT],
        policy: Policy,
        *,
        ticks: int = 256,
        reference: Policy | None = None,
    ) -> None:
        ticks = as_int(ticks, "ticks must be a positive integer.", minimum=1)
        check_reference(env, reference)
        self.env = env
        self.ticks = ticks
        self.metadata = env.metadata.copy()
        self.policy = cast(Policy, jax.jit(policy))
        self.reference = compile_reference(reference)

    @property
    def donates(self) -> bool:
        """Always False: a call leaves the carry passed in readable."""
        return False

    def __call__(self, state: Lifetime[StateT]) -> Lifetime[StateT]:
        """Advance one chunk without process pools or episode-return bookkeeping."""
        for _ in range(self.ticks):
            state = tick(self.env, self.policy, state, reference=self.reference)
        return state
