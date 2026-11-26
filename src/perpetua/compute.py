"""Opt-in compute-aware time: recorded tick schedules and synchronized host measurement."""

# JAX transformation/device interfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import math
from collections.abc import Callable
from time import perf_counter_ns
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from numpy.typing import ArrayLike

from perpetua import runtime
from perpetua.core import Clock, Environment, as_int

# The public API; the remaining module-level helpers serve the runners and learners.
__all__ = [
    "tick_count",
    "TimingTrace",
    "MeasuredTimingTrace",
    "ComputeTiming",
    "TimedLifetime",
    "AdvanceResult",
    "init_timing",
    "advance",
    "advance_host",
    "record_trace",
    "ComputeRunner",
]


def tick_count(compute_ns: int, quantum_ns: int) -> int:
    """Quantize nonnegative uint64 nanoseconds to at least one world tick.

    Integer ceil avoids float32/float64 boundary rounding. A single decision
    interval is limited to uint32 ticks; oversized costs are refused, never clipped.
    """
    compute_ns = as_int(
        compute_ns, "compute_ns must be an integer in [0, 2**64).", minimum=0, maximum=2**64 - 1
    )
    quantum_ns = as_int(
        quantum_ns, "quantum_ns must be a positive uint64 integer.", minimum=1, maximum=2**64 - 1
    )
    ticks = max(1, (compute_ns + quantum_ns - 1) // quantum_ns)
    if ticks >= 2**32:
        raise OverflowError("A decision's elapsed tick count must fit in uint32.")
    return ticks


class TimingTrace(NamedTuple):
    """Bounded external-input trace: uint32 ticks (decisions,) and uint64 ns limbs.

    Tick counts are authoritative replay inputs; measured costs must be retained
    with checkpoints/segments for deterministic replay of wall-timed executions.
    """

    ticks: jax.Array
    compute_ns: Clock

    @classmethod
    def from_ticks(cls, ticks: ArrayLike, *, quantum_ns: int) -> "TimingTrace":
        """Validate a host schedule and record exact synthetic compute costs."""
        quantum_ns = as_int(
            quantum_ns,
            "quantum_ns must be a positive uint64 integer.",
            minimum=1,
            maximum=2**64 - 1,
        )
        values = np.asarray(ticks)
        if values.ndim != 1 or not values.size:
            raise ValueError("ticks must be a nonempty one-dimensional decision schedule.")
        if values.dtype.kind not in {"i", "u"}:
            raise TypeError("ticks must have integer dtype.")
        counts = [int(value) for value in values]
        if any(not 1 <= value < 2**32 for value in counts):
            raise ValueError("Decision ticks must be in [1, 2**32).")
        costs = [value * quantum_ns for value in counts]
        if any(value >= 2**64 for value in costs):
            raise OverflowError("Synthetic compute cost must fit in uint64 nanoseconds.")
        return cls(
            jnp.asarray(counts, dtype=jnp.uint32),
            Clock(
                jnp.asarray([value >> 32 for value in costs], dtype=jnp.uint32),
                jnp.asarray([value & 0xFFFFFFFF for value in costs], dtype=jnp.uint32),
            ),
        )


class MeasuredTimingTrace(TimingTrace):
    """Numeric wall-cost trace requiring the measured execution partition on replay.

    Recorded schedules may fuse a whole native chunk. Measured replay instead
    preserves separate policy/work and world-step kernels, avoiding different
    GPU floating-point optimizations. The trace type is checkpointed as schema.
    """


class ComputeTiming(NamedTuple):
    """Held int32 command (agents,), exact decision/ns clocks and scalar timing health."""

    held_action: jax.Array
    decisions: Clock
    compute_ns: Clock
    last_ticks: jax.Array
    healthy: jax.Array


class TimedLifetime[State](NamedTuple):
    """Complete stateless-policy lifetime and compute controller resume state."""

    lifetime: runtime.Lifetime[State]
    timing: ComputeTiming


class AdvanceResult[State](NamedTuple):
    """Latest lifetime and float32 interval reward summed over its world ticks.

    Tick ``i`` of the interval is weighted ``gamma**i``; with the default
    ``gamma=1.0`` (as ``ComputeRunner`` uses) the sum is undiscounted.
    """

    lifetime: runtime.Lifetime[State]
    reward: jax.Array
    healthy: jax.Array


def init_timing(num_agents: int) -> ComputeTiming:
    """Initialize the held command to action 0 for every slot, without advancing time or RNG.

    There is no earlier command on the first decision, so its ``k - 1`` held
    ticks run action 0 (repaired to the first valid action if 0 is masked).
    Action 0 is only a no-op where the environment makes it one: in
    ``CartPole`` it pushes left.
    """
    return ComputeTiming(
        jnp.zeros(num_agents, dtype=jnp.int32),
        Clock.from_int(0),
        Clock.from_int(0),
        jnp.uint32(0),
        jnp.asarray(True),
    )


def _repair(
    action: jax.Array, mask: jax.Array, active_mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    if action.shape != mask.shape[:1] or action.dtype.kind not in {"i", "u"}:
        raise ValueError("Commands must be integer vectors matching agent slots.")
    safe = jnp.clip(action, 0, mask.shape[1] - 1).astype(jnp.int32)
    valid = (action >= 0) & (action < mask.shape[1]) & mask[jnp.arange(mask.shape[0]), safe]
    first = jnp.argmax(mask, axis=-1).astype(jnp.int32)
    return jnp.where(valid, safe, first), jnp.all(jnp.any(mask, axis=-1) | ~active_mask)


def _world_tick[State](
    env: Environment[State], life: runtime.Lifetime[State], action: jax.Array
) -> tuple[runtime.Lifetime[State], jax.Array, jax.Array]:
    selected, valid = _repair(action, life.info["action_mask"], life.info["active_mask"])

    def step(life: runtime.Lifetime[State]) -> tuple[runtime.Lifetime[State], jax.Array, jax.Array]:
        life, reward = runtime.transition(env, life, selected)
        return life, reward, life.metrics.healthy

    def failed(
        life: runtime.Lifetime[State],
    ) -> tuple[runtime.Lifetime[State], jax.Array, jax.Array]:
        return life, jnp.zeros(env.num_agents, dtype=jnp.float32), jnp.asarray(False)

    if env.metadata.get("execution") == "host":
        return step(life) if bool(valid) else failed(life)
    return cast(
        tuple[runtime.Lifetime[State], jax.Array, jax.Array],
        jax.lax.cond(valid, step, failed, life),
    )


def _as_ticks(ticks: jax.Array | int) -> jax.Array:
    """A scalar uint32 tick count from an array or a Python/NumPy integer."""
    if not isinstance(ticks, jax.Array | np.ndarray):
        return jnp.uint32(
            as_int(
                ticks,
                "ticks must be a scalar uint32 count (an integer in [0, 2**32)).",
                minimum=0,
                maximum=2**32 - 1,
            )
        )
    if ticks.shape != () or ticks.dtype != jnp.uint32:
        raise ValueError(
            f"ticks must be a scalar uint32 count; got shape {tuple(ticks.shape)} "
            f"and dtype {ticks.dtype}."
        )
    return jnp.asarray(ticks)


def advance[State](
    env: Environment[State],
    lifetime: runtime.Lifetime[State],
    held_action: jax.Array,
    action: jax.Array,
    ticks: jax.Array | int,
    *,
    gamma: float = 1.0,
) -> AdvanceResult[State]:
    """Pure native catch-up: held command for k-1 ticks, new command on tick k.

    Commands are repaired against each current public mask, choosing its first
    valid entry (action 0 when it is valid, which is not necessarily "stay").
    Environment/drift keys and metrics advance every world tick; agent sampling
    keys advance only at decisions. ``ticks`` is a scalar uint32 array or a
    Python (or NumPy) integer, such as :func:`tick_count` returns; zero returns
    an unchanged, unhealthy result.
    """
    if env.metadata.get("execution") == "host":
        raise ValueError("Use advance_host for mutable CPU simulators.")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1].")
    ticks = _as_ticks(ticks)

    def condition(
        carry: tuple[jax.Array, runtime.Lifetime[State], jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        return (carry[0] < ticks) & carry[4]

    def body(
        carry: tuple[jax.Array, runtime.Lifetime[State], jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, runtime.Lifetime[State], jax.Array, jax.Array, jax.Array]:
        i, life, total, discount, healthy = carry
        command = jnp.where(i == ticks - jnp.uint32(1), action, held_action)
        life, reward, valid = _world_tick(env, life, command)
        return i + jnp.uint32(1), life, total + discount * reward, discount * gamma, healthy & valid

    _, life, total, _, healthy = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.uint32(0),
            lifetime,
            jnp.zeros(env.num_agents, dtype=jnp.float32),
            jnp.asarray(1.0),
            ticks > 0,
        ),
    )
    return AdvanceResult(life, total, healthy)


def advance_host[State](
    env: Environment[State],
    lifetime: runtime.Lifetime[State],
    held_action: jax.Array,
    action: jax.Array,
    ticks: jax.Array | int,
    *,
    gamma: float = 1.0,
) -> AdvanceResult[State]:
    """Explicit host catch-up with the same timing/reward contract as advance."""
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1].")
    ticks = _as_ticks(ticks)
    total, discount, healthy = jnp.zeros(env.num_agents, dtype=jnp.float32), 1.0, ticks > 0
    for i in range(int(ticks)):
        lifetime, reward, valid = _world_tick(
            env, lifetime, action if i == int(ticks) - 1 else held_action
        )
        total = total + discount * reward
        discount *= gamma
        healthy = healthy & valid
        if not bool(healthy):
            break
    return AdvanceResult(lifetime, total, healthy)


def require_discrete_actions(env: Environment[Any], who: str) -> None:
    """Refuse a continuous-action environment: compute-aware time holds an int32 command."""
    if not hasattr(env.action_space, "nvec"):
        raise ValueError(
            f"{who} needs a discrete (MultiDiscrete) action space, because the held command "
            f"is an int32 vector; {env.metadata.get('env_id', type(env).__name__)} has "
            f"{type(env.action_space).__name__}. Use the synchronous Runner or PPORunner for "
            "continuous actions."
        )


def check_trace(trace: TimingTrace, decisions: int, quantum_ns: int) -> None:
    """Check a trace's shapes, dtypes and quantum before a runner replays it.

    Every tick count must be what :func:`tick_count` gives for its recorded
    cost under the runner's ``quantum_ns``, so a trace built for another
    quantum is refused instead of charging costs that disagree with the ticks
    run. A zero tick count is left to the replay, which marks the carry
    unhealthy. The values are read on the host here, once per call and before
    the compiled replay; a trace that is itself traced (the runner called
    under ``jax.jit`` or ``jax.vmap`` with the trace as an argument) gets the
    static checks only.
    """
    fields = (
        ("ticks", trace.ticks),
        ("compute_ns.hi", trace.compute_ns.hi),
        ("compute_ns.lo", trace.compute_ns.lo),
    )
    for name, value in fields:
        if value.shape != (decisions,) or value.dtype != jnp.uint32:
            raise ValueError(
                f"Timing trace field {name} must be a uint32 array of shape ({decisions},) "
                f"(the runner's decisions); got shape {tuple(value.shape)} and dtype "
                f"{value.dtype}."
            )
    if any(isinstance(value, Tracer) for _, value in fields):
        return
    ticks = np.asarray(trace.ticks).astype(np.uint64)
    costs = (np.asarray(trace.compute_ns.hi).astype(np.uint64) << np.uint64(32)) | np.asarray(
        trace.compute_ns.lo
    ).astype(np.uint64)
    quantum = np.uint64(quantum_ns)
    expected = np.maximum(costs // quantum + (costs % quantum != 0), np.uint64(1))
    wrong = np.flatnonzero((ticks != 0) & (ticks != expected))
    if wrong.size:
        index = int(wrong[0])
        raise ValueError(
            f"Timing trace decision {index} records {int(ticks[index])} ticks for "
            f"{int(costs[index])} ns, but this runner's quantum_ns={quantum_ns} gives "
            f"{int(expected[index])}. Build the trace with the runner's quantum_ns."
        )


def record_trace(costs: list[int], *, quantum_ns: int) -> MeasuredTimingTrace:
    """Convert completed synchronized host measurements into a numeric replay trace."""
    costs = [
        as_int(value, "compute_ns must be an integer in [0, 2**64).", minimum=0, maximum=2**64 - 1)
        for value in costs
    ]
    return MeasuredTimingTrace(
        jnp.asarray([tick_count(value, quantum_ns) for value in costs], dtype=jnp.uint32),
        Clock(
            jnp.asarray([value >> 32 for value in costs], dtype=jnp.uint32),
            jnp.asarray([value & 0xFFFFFFFF for value in costs], dtype=jnp.uint32),
        ),
    )


def next_timing(
    timing: ComputeTiming, action: jax.Array, ticks: jax.Array, cost: Clock, healthy: jax.Array
) -> ComputeTiming:
    """Advance controller accounting after a completed decision interval."""
    return ComputeTiming(
        action.astype(jnp.int32),
        timing.decisions.increment(),
        timing.compute_ns.add(cost),
        ticks,
        timing.healthy & healthy,
    )


def measured_compiler_options() -> dict[str, bool | int]:
    """Stable GPU kernel options for all checkpointable/replay paths, never global flags."""
    return {"xla_gpu_exclude_nondeterministic_ops": True, "xla_gpu_autotune_level": 0}


def configuration(*, decisions: int, quantum_ns: int, work: str) -> dict[str, object]:
    """Version the time contract independently from environment/learner metadata."""
    decisions = as_int(decisions, "decisions must be a positive integer.", minimum=1)
    tick_count(0, quantum_ns)
    return {
        "version": 1,
        "decisions_per_chunk": decisions,
        "quantum_ns": quantum_ns,
        "charged_work": work,
        "arrival": "held-command-until-final-charged-tick",
        "mask_repair": "first-current-valid-action",
        "warmup": "excluded",
        "measurement": "synchronized-perf-counter-nanoseconds",
        "replay_partition": "compiled-recorded; separate-kernels-measured",
        "measured_compiler_options": measured_compiler_options(),
    }


class ComputeRunner[State]:
    """Stateless-policy compute-aware chunks; native replay compiles once.

    measure() synchronizes and times policy work outside tick loops, returning
    the trace required for deterministic replay. Dynamics/transfer/logging time
    is not charged as agent compute. This is a latency model, not OS real time.

    Needs a discrete action space (``MultiDiscrete``): the held command is an
    int32 vector, so a continuous-action environment raises ``ValueError``
    here. The first decision holds action 0 for its ``k - 1`` ticks (see
    :func:`init_timing`).
    """

    def __init__(
        self,
        env: Environment[State],
        policy: runtime.Policy,
        *,
        decisions: int = 32,
        quantum_ns: int = 1_000_000,
    ) -> None:
        require_discrete_actions(env, "ComputeRunner")
        decisions = as_int(decisions, "decisions must be a positive integer.", minimum=1)
        self.env, self.decisions, self.quantum_ns = env, decisions, quantum_ns
        self.metadata: dict[str, object] = {
            "environment": env.metadata.copy(),
            "controller": configuration(
                decisions=decisions, quantum_ns=quantum_ns, work="policy-call"
            ),
        }
        self._policy = cast(runtime.Policy, jax.jit(policy))
        self._measured_policy = cast(
            runtime.Policy, jax.jit(policy, compiler_options=measured_compiler_options())
        )
        self._host = env.metadata.get("execution") == "host"

        def catch_up(
            life: runtime.Lifetime[State], held: jax.Array, action: jax.Array, ticks: jax.Array
        ) -> AdvanceResult[State]:
            return (advance_host if self._host else advance)(env, life, held, action, ticks)

        self._advance = (
            catch_up
            if self._host
            else cast(
                Callable[
                    [runtime.Lifetime[State], jax.Array, jax.Array, jax.Array], AdvanceResult[State]
                ],
                jax.jit(catch_up),
            )
        )
        self._measured_advance = (
            catch_up
            if self._host
            else cast(
                Callable[
                    [runtime.Lifetime[State], jax.Array, jax.Array, jax.Array], AdvanceResult[State]
                ],
                jax.jit(catch_up, compiler_options=measured_compiler_options()),
            )
        )

        def decision(
            state: TimedLifetime[State], ticks: jax.Array, cost: Clock, *, measured: bool = False
        ) -> TimedLifetime[State]:
            agent_key, act_key = jax.random.split(state.lifetime.agent_key)
            policy_call = self._measured_policy if measured else self._policy
            action = policy_call(state.lifetime.obs, state.lifetime.info, act_key)
            life = state.lifetime._replace(agent_key=agent_key)
            step = self._measured_advance if measured else self._advance
            result = step(life, state.timing.held_action, action, ticks)
            return TimedLifetime(
                result.lifetime, next_timing(state.timing, action, ticks, cost, result.healthy)
            )

        self._decision = decision

        def run(state: TimedLifetime[State], trace: TimingTrace) -> TimedLifetime[State]:
            def step(
                state: TimedLifetime[State], item: tuple[jax.Array, Clock]
            ) -> tuple[TimedLifetime[State], None]:
                ticks, cost = item
                state = state._replace(
                    timing=state.timing._replace(healthy=state.timing.healthy & (ticks > 0))
                )

                def perform(state: TimedLifetime[State]) -> TimedLifetime[State]:
                    return decision(state, ticks, cost)

                def unchanged(state: TimedLifetime[State]) -> TimedLifetime[State]:
                    return state

                state = cast(
                    TimedLifetime[State],
                    jax.lax.cond(state.timing.healthy, perform, unchanged, state),
                )
                return state, None

            return jax.lax.scan(step, state, (trace.ticks, trace.compute_ns))[0]

        self._run = cast(
            Callable[[TimedLifetime[State], TimingTrace], TimedLifetime[State]], jax.jit(run)
        )

    def init(self, *, seed: int | jax.Array = 0, window_size: int = 128) -> TimedLifetime[State]:
        """Initialize/prime policy compilation without consuming lifetime state/RNG."""
        life = runtime.init_lifetime(self.env, seed=seed, window_size=window_size)
        jax.block_until_ready(self._policy(life.obs, life.info, jax.random.key(0)))
        jax.block_until_ready(self._measured_policy(life.obs, life.info, jax.random.key(1)))
        return TimedLifetime(life, init_timing(self.env.num_agents))

    @property
    def donates(self) -> bool:
        """Always False: a call leaves the carry passed in readable."""
        return False

    def __call__(self, state: TimedLifetime[State], trace: TimingTrace) -> TimedLifetime[State]:
        """Replay a supplied bounded schedule; no wall timing is used."""
        check_trace(trace, self.decisions, self.quantum_ns)
        if not self._host and not isinstance(trace, MeasuredTimingTrace):
            return self._run(state, trace)
        for i in range(self.decisions):
            if not bool(state.timing.healthy) or int(trace.ticks[i]) == 0:
                return state._replace(timing=state.timing._replace(healthy=jnp.asarray(False)))
            state = self._decision(
                state,
                trace.ticks[i],
                Clock(trace.compute_ns.hi[i], trace.compute_ns.lo[i]),
                measured=isinstance(trace, MeasuredTimingTrace),
            )
        return state

    def measure(
        self, state: TimedLifetime[State]
    ) -> tuple[TimedLifetime[State], MeasuredTimingTrace]:
        """Charge synchronized policy cost and return numeric timing inputs for replay."""
        costs: list[int] = []
        for _ in range(self.decisions):
            if not bool(state.timing.healthy):
                raise RuntimeError("Cannot measure an unhealthy compute controller.")
            agent_key, act_key = jax.random.split(state.lifetime.agent_key)
            start = perf_counter_ns()
            action = cast(
                jax.Array,
                jax.block_until_ready(
                    self._measured_policy(state.lifetime.obs, state.lifetime.info, act_key)
                ),
            )
            cost = perf_counter_ns() - start
            ticks = jnp.uint32(tick_count(cost, self.quantum_ns))
            life = state.lifetime._replace(agent_key=agent_key)
            result = self._measured_advance(life, state.timing.held_action, action, ticks)
            state = TimedLifetime(
                result.lifetime,
                next_timing(state.timing, action, ticks, Clock.from_int(cost), result.healthy),
            )
            costs.append(cost)
        return state, record_trace(costs, quantum_ns=self.quantum_ns)
