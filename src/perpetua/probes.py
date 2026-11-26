"""Learner health probes: dormant-unit ratio and effective rank.

The functions are on-demand diagnostics. :class:`ProbedRunner` puts them on the
lifetime's own clock: it evaluates a learner's ``probe`` every few updates on
recent observations and keeps a bounded history in the carry, so a collapse is
visible while the run is going, not only at the end.
"""

# JAX device members are partially annotated.
# pyright: reportUnknownMemberType=false

import inspect
import math
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp

from perpetua.core import as_int
from perpetua.runtime import ChunkRunner

__all__ = [
    "dormant_unit_ratio",
    "effective_rank",
    "ProbeTrace",
    "Probed",
    "LearnerRunner",
    "ProbedRunner",
]


def _check_batch(activations: jax.Array) -> None:
    if activations.ndim != 2 or activations.shape[0] < 1 or activations.shape[1] < 1:
        raise ValueError(
            f"Probe activations must have shape (batch, units); got {tuple(activations.shape)}."
        )
    if activations.shape[0] < 2:
        raise ValueError(
            "Probes need a batch of at least two rows: one row has zero variance, so every "
            "unit would read as dormant and the rank as 1. Probe several observations, for "
            "example the last few ticks' (state.lifetime.obs stacked with earlier ones)."
        )


def _row_weights(activations: jax.Array, weights: jax.Array | None) -> jax.Array | None:
    if weights is None:
        return None
    weights = jnp.asarray(weights, dtype=activations.dtype)
    if weights.shape != activations.shape[:1]:
        raise ValueError(
            f"Probe row weights must have shape ({activations.shape[0]},), one per batch row; "
            f"got {tuple(weights.shape)}."
        )
    return weights


def _counted_rows(activations: jax.Array, weights: jax.Array) -> jax.Array:
    """Zero the rows of weight 0, so a non-finite value there cannot reach the statistic."""
    return jnp.where(weights[:, None] > 0, activations, 0.0)


def _weighted_mean(activations: jax.Array, weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    total = jnp.sum(weights)
    total = jnp.where(total > 0, total, 1.0)
    return jnp.sum(activations * weights[:, None], axis=0) / total, total


def _usable(activations: jax.Array, weights: jax.Array | None) -> jax.Array:
    """Whether the counted activations, and the weights if given, are finite and nonnegative."""
    usable = jnp.all(jnp.isfinite(activations))
    if weights is not None:
        usable = usable & jnp.all(jnp.isfinite(weights) & (weights >= 0))
    return usable


def dormant_unit_ratio(
    activations: jax.Array, *, threshold: float = 1e-3, weights: jax.Array | None = None
) -> jax.Array:
    """Fraction of units whose activation variance across the batch is tiny.

    ``activations`` has shape (batch, units); the ratio is in [0, 1]. A unit is
    dormant when its variance falls below the absolute threshold. This is a
    variance test, not the normalised mean absolute activation score of Sokar
    et al. (2023), so a unit stuck at a nonzero constant also counts. The batch
    needs at least two rows: the variance over one row is zero, so every unit
    would count as dormant. Any non-finite activation gives NaN.

    Optional nonnegative ``weights`` (batch,) weight the rows of the variance;
    a row of weight 0 is left out, so 0/1 weights give the statistic of the
    rows marked 1 at a fixed batch shape. With fewer than two such rows every
    unit counts as dormant (for a positive threshold). A negative or
    non-finite weight gives NaN.
    """
    _check_batch(activations)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Dormant-unit threshold must be finite and nonnegative.")
    rows = _row_weights(activations, weights)
    if rows is None:
        variance = jnp.var(activations, axis=0)
    else:
        activations = _counted_rows(activations, rows)
        mean, total = _weighted_mean(activations, rows)
        variance = jnp.sum(jnp.square(activations - mean) * rows[:, None], axis=0) / total
    ratio = jnp.mean((variance < threshold).astype(jnp.float32))
    return jnp.where(_usable(activations, rows), ratio, jnp.nan)


def effective_rank(activations: jax.Array, *, weights: jax.Array | None = None) -> jax.Array:
    """Entropy-based effective rank (Roy & Vetterli, 2007) of the centered activations.

    Uses the singular values of the centered (batch, units) matrix; a rank-one
    representation scores ~1, and isotropic directions approach the smaller of
    the batch/unit counts. The batch needs at least two rows. Activations with
    no variance at all score 1, the measure's lower bound, and any non-finite
    activation gives NaN.

    Optional nonnegative ``weights`` (batch,) weight the rows as in
    ``dormant_unit_ratio``: the rows are centered on their weighted mean and
    scaled by the square root of their weight, so a row of weight 0 adds
    nothing to the singular values. A negative or non-finite weight gives NaN.
    """
    _check_batch(activations)
    rows = _row_weights(activations, weights)
    if rows is None:
        centered = activations - jnp.mean(activations, axis=0, keepdims=True)
    else:
        activations = _counted_rows(activations, rows)
        mean, _ = _weighted_mean(activations, rows)
        centered = (activations - mean) * jnp.sqrt(rows)[:, None]
    singular = jnp.linalg.svd(centered, compute_uv=False)
    total = jnp.sum(singular)
    probabilities = jnp.where(total > 0, singular / jnp.where(total > 0, total, 1.0), 0.0)
    entropy = -jnp.sum(jnp.where(probabilities > 0, probabilities * jnp.log(probabilities), 0.0))
    usable = _usable(activations, rows) & jnp.isfinite(total)
    return jnp.where(usable, jnp.exp(entropy), jnp.nan)


class ProbeTrace(NamedTuple):
    """Bounded probe history carried next to a learner's training state.

    ``obs`` holds the observations seen at the last ``memory`` update
    boundaries (the probe batch); until ``memory`` updates have passed, the
    remaining entries are copies of the observation at tracking start.
    ``active`` holds the ``active_mask`` that came with each of them, shape
    (memory, agent slots). ``latest`` holds the newest value of each probe (zeros before the first
    evaluation); ``history`` a ring of the last ``window`` evaluations, written
    at ``cursor`` (zeros where not yet written); ``evaluations`` counts them and
    ``updates`` counts the runner calls since tracking started.
    """

    obs: jax.Array
    active: jax.Array
    latest: dict[str, jax.Array]
    history: dict[str, jax.Array]
    cursor: jax.Array
    evaluations: jax.Array
    updates: jax.Array


class Probed[State](NamedTuple):
    """A learner's training state plus its probe trace; checkpoint it as one pytree."""

    state: State
    trace: ProbeTrace


class LearnerRunner[State](ChunkRunner[State], Protocol):
    """What :class:`ProbedRunner` needs: a runner that also has a probing agent and ``init``."""

    @property
    def agent(self) -> Any: ...

    @property
    def init(self) -> Callable[..., State]: ...


class ProbedRunner[State]:
    """Run a learner runner and probe its network every ``every`` updates.

    Works with any runner whose ``agent`` has ``probe(agent_state, obs, *,
    threshold, weights)`` (both passed by keyword) and whose state has
    ``agent`` and ``lifetime`` (PPO, DQN and SAC, compiled or host). The probe
    batch is the agent-slot observations from the last ``memory`` update
    boundaries, so it costs no extra environment steps and never touches the
    learner's RNG: training is bit-identical with and without probing. Memory
    is O(memory + window), not lifetime length.

    Only rows whose slot was active count: an inactive slot's observation is a
    placeholder (all zeros for the scripted slots of ``PartnerGrid-v0``), and
    counting those rows would read as dormancy the network does not have. The
    rows keep their place in the batch and the runner passes ``weights`` (one
    per row, 0 for an inactive slot) to the learner's ``probe``, which hands
    them to ``dormant_unit_ratio`` and ``effective_rank``. A custom ``probe``
    without a ``weights`` parameter is called without them and counts every
    row. With fewer than two active rows in the batch the readings are 1 (all
    dormant, for a positive threshold) and rank 1.

    Until ``memory`` updates have passed, the probe batch is padded with copies
    of the observation at tracking start, so the first readings overstate
    dormancy and understate rank; discard evaluations before then.
    """

    def __init__(
        self,
        runner: LearnerRunner[State],
        *,
        every: int = 1,
        window: int = 128,
        memory: int = 32,
        threshold: float = 1e-3,
    ) -> None:
        every = as_int(every, "every must be a positive integer.", minimum=1)
        window = as_int(window, "window must be a positive integer.", minimum=1)
        memory = as_int(memory, "memory must be a positive integer.", minimum=1)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative.")
        probe = getattr(getattr(runner, "agent", None), "probe", None)
        if probe is None:
            raise ValueError("The runner's agent has no probe method.")
        accepted = inspect.signature(probe).parameters.values()
        weighted = any(
            p.name == "weights" or p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted
        )
        self.runner = runner
        self.window = window
        self.memory = memory
        self.metadata: dict[str, object] = {
            **dict(runner.metadata),
            "probes": {"every": every, "window": window, "memory": memory, "threshold": threshold},
        }

        def evaluate(agent_state: Any, obs: jax.Array, active: jax.Array) -> dict[str, jax.Array]:
            batch = obs.reshape((-1, *obs.shape[2:]))
            options: dict[str, Any] = {"threshold": threshold}
            if weighted:
                options["weights"] = active.reshape(-1).astype(jnp.float32)
            values = cast(dict[str, jax.Array], probe(agent_state, batch, **options))
            return {name: jnp.asarray(value, dtype=jnp.float32) for name, value in values.items()}

        def record(state: Any, trace: ProbeTrace) -> ProbeTrace:
            obs = jnp.roll(trace.obs, -1, axis=0).at[-1].set(state.lifetime.obs)
            active = (
                jnp.roll(trace.active, -1, axis=0).at[-1].set(state.lifetime.info["active_mask"])
            )
            updates = trace.updates + jnp.int32(1)

            def measure(_: None) -> ProbeTrace:
                latest = evaluate(state.agent, obs, active)
                history = {
                    name: trace.history[name].at[trace.cursor].set(value)
                    for name, value in latest.items()
                }
                return ProbeTrace(
                    obs,
                    active,
                    latest,
                    history,
                    (trace.cursor + 1) % window,
                    trace.evaluations + jnp.int32(1),
                    updates,
                )

            def skip(_: None) -> ProbeTrace:
                return trace._replace(obs=obs, active=active, updates=updates)

            return cast(ProbeTrace, jax.lax.cond(updates % every == 0, measure, skip, None))

        self._evaluate = evaluate
        self._record: Callable[[Any, ProbeTrace], ProbeTrace] = jax.jit(record)

    def track(self, state: State) -> Probed[State]:
        """Start tracking an existing training state (fresh or restored)."""
        lifetime = cast(Any, state).lifetime
        rows = self.memory * lifetime.obs.shape[0]
        if rows < 2:
            raise ValueError(
                f"ProbedRunner(memory={self.memory}) probes {rows} observation row with a "
                "single agent; probes need at least two rows, so pass memory=2 or more."
            )
        obs = jnp.broadcast_to(lifetime.obs, (self.memory, *lifetime.obs.shape))
        mask = jnp.asarray(lifetime.info["active_mask"], dtype=jnp.bool_)
        active = jnp.broadcast_to(mask, (self.memory, *mask.shape))
        shapes = jax.eval_shape(self._evaluate, cast(Any, state).agent, obs, active)
        latest = {name: jnp.zeros(shape.shape, jnp.float32) for name, shape in shapes.items()}
        history = {
            name: jnp.zeros((self.window, *shape.shape), jnp.float32)
            for name, shape in shapes.items()
        }
        zero = jnp.int32(0)
        return Probed(
            state,
            ProbeTrace(jnp.array(obs), jnp.array(active), latest, history, zero, zero, zero),
        )

    def init(self, **kwargs: Any) -> Probed[State]:
        """Initialise the learner (same arguments as its runner) and tracking."""
        return self.track(self.runner.init(**kwargs))

    @property
    def donates(self) -> bool:
        """Whether the wrapped runner consumes the learner state passed in."""
        return self.runner.donates

    def __call__(self, carry: Probed[State]) -> Probed[State]:
        """Advance one learner update, then record the probes if they are due."""
        state = self.runner(carry.state)
        return Probed(state, self._record(state, carry.trace))
