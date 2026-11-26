"""Compute-aware PPO/IPPO with bounded pending rollouts and full decision-work timing."""

# JAX transformation/device interfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from time import perf_counter_ns
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from perpetua import compute, runtime
from perpetua.agents._common import check_environment_pixel_scale
from perpetua.agents.ppo import (
    IPPO,
    PPO,
    Action,
    PPOState,
    PPOStats,
    Rollout,
    TrainingState,
    init_training,
    training_health,
    validate_minibatches,
)
from perpetua.compute import AdvanceResult, ComputeTiming, MeasuredTimingTrace, TimingTrace
from perpetua.core import Clock, Environment, Info

__all__ = [
    "TimedTrainingState",
    "ComputePPORunner",
]


class TimedTrainingState[State](NamedTuple):
    """Full training/timing state plus a fixed-size unconsumed on-policy buffer.

    The uint32 elapsed_ticks buffer (time, agents) and int32 filled scalar are
    checkpointed together with all world/model/optimizer/RNG/metric state.
    A full buffer is optimized at the next decision, not at a hidden time gap.
    """

    training: TrainingState[State]
    timing: ComputeTiming
    rollout: Rollout
    filled: jax.Array


class _WorkState(NamedTuple):
    agent: PPOState
    optimizer_key: jax.Array
    updates: Clock
    statistics: PPOStats
    agent_key: jax.Array
    healthy: jax.Array
    filled: jax.Array


def _empty_rollout(obs: jax.Array, mask: jax.Array, steps: int) -> Rollout:
    shape = (steps, obs.shape[0])
    return Rollout(
        jnp.zeros((steps, *obs.shape), dtype=obs.dtype),
        jnp.zeros((steps, *mask.shape), dtype=jnp.bool_),
        jnp.zeros(shape, dtype=jnp.bool_),
        jnp.zeros(shape, dtype=jnp.int32),
        jnp.zeros(shape),
        jnp.zeros(shape),
        jnp.zeros(shape),
        jnp.zeros(shape, dtype=jnp.uint32),
    )


class ComputePPORunner[State]:
    """Timed decisions on native or explicit host worlds, using the same PPO/IPPO.

    On a full pending buffer, optimize then act from the current observation.
    This complete work unit is timed/synchronized by measure(); its latency
    belongs to that new decision's semi-Markov interval. Every physics tick is
    measured online.

    The catch-up between decisions discounts each interval's reward with the
    static ``agent.config.gamma``, while GAE and reward centering use the
    traced ``hyperparameters.gamma``; the two agree unless a lifetime's
    hyperparameters override gamma. ``init`` takes no ``hyperparameters``
    argument, so the configured defaults are used.
    """

    def __init__(
        self,
        env: Environment[State],
        agent: PPO | IPPO,
        *,
        decisions: int | None = None,
        quantum_ns: int = 1_000_000,
    ) -> None:
        compute.require_discrete_actions(env, "ComputePPORunner")
        validate_minibatches(agent, env.num_agents)
        check_environment_pixel_scale(env, agent, type(agent).__name__)
        self.env, self.agent = env, agent
        self.decisions = agent.config.rollout_steps if decisions is None else decisions
        self.quantum_ns = quantum_ns
        self._host = env.metadata.get("execution") == "host"
        self.metadata: dict[str, object] = {
            "environment": env.metadata.copy(),
            "learner": agent.configuration,
            "controller": compute.configuration(
                decisions=self.decisions,
                quantum_ns=quantum_ns,
                work="optimize-full-buffer-then-act",
            ),
            "discount_clock": "world-ticks; discounted-interval-reward",
        }

        def work(
            carry: _WorkState, obs: jax.Array, info: Info, rollout: Rollout
        ) -> tuple[_WorkState, Action]:
            def optimize(carry: _WorkState) -> _WorkState:
                _, last_value = agent.evaluate(carry.agent, obs)
                trained, key, statistics = agent.update(
                    carry.agent, rollout, last_value, carry.optimizer_key
                )
                healthy = training_health(carry.healthy, jnp.asarray(True), trained, statistics)
                return _WorkState(
                    trained,
                    key,
                    carry.updates.increment(),
                    statistics,
                    carry.agent_key,
                    healthy,
                    jnp.int32(0),
                )

            def unchanged(carry: _WorkState) -> _WorkState:
                return carry

            carry = cast(
                _WorkState,
                jax.lax.cond(
                    carry.filled == agent.config.rollout_steps, optimize, unchanged, carry
                ),
            )
            agent_key, act_key = jax.random.split(carry.agent_key)
            sampled = agent.act(carry.agent, obs, info, act_key)
            healthy = (
                carry.healthy
                & jnp.all(jnp.isfinite(sampled.value))
                & jnp.all(jnp.isfinite(sampled.log_prob))
            )
            return carry._replace(agent_key=agent_key, healthy=healthy), sampled

        self._work = cast(
            Callable[[_WorkState, jax.Array, Info, Rollout], tuple[_WorkState, Action]],
            jax.jit(work),
        )
        self._measured_work = cast(
            Callable[[_WorkState, jax.Array, Info, Rollout], tuple[_WorkState, Action]],
            jax.jit(work, compiler_options=compute.measured_compiler_options()),
        )

        def catch_up(
            life: runtime.Lifetime[State], held: jax.Array, action: jax.Array, ticks: jax.Array
        ) -> AdvanceResult[State]:
            return (compute.advance_host if self._host else compute.advance)(
                env, life, held, action, ticks, gamma=agent.config.gamma
            )

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
                jax.jit(catch_up, compiler_options=compute.measured_compiler_options()),
            )
        )

        def run(state: TimedTrainingState[State], trace: TimingTrace) -> TimedTrainingState[State]:
            def decision(
                state: TimedTrainingState[State], item: tuple[jax.Array, Clock]
            ) -> tuple[TimedTrainingState[State], None]:
                ticks, cost = item
                healthy = self._valid(state) & (ticks > 0)
                state = state._replace(timing=state.timing._replace(healthy=healthy))

                def perform(state: TimedTrainingState[State]) -> TimedTrainingState[State]:
                    updated, sampled = self._perform_work(state)

                    def finish(state: TimedTrainingState[State]) -> TimedTrainingState[State]:
                        return self._finish(state, sampled, ticks, cost)

                    def failed(state: TimedTrainingState[State]) -> TimedTrainingState[State]:
                        return state._replace(
                            timing=state.timing._replace(healthy=jnp.asarray(False))
                        )

                    return cast(
                        TimedTrainingState[State],
                        jax.lax.cond(updated.training.healthy, finish, failed, updated),
                    )

                def unchanged(state: TimedTrainingState[State]) -> TimedTrainingState[State]:
                    return state

                state = cast(
                    TimedTrainingState[State], jax.lax.cond(healthy, perform, unchanged, state)
                )
                return state, None

            return jax.lax.scan(decision, state, (trace.ticks, trace.compute_ns))[0]

        self._run = cast(
            Callable[[TimedTrainingState[State], TimingTrace], TimedTrainingState[State]],
            jax.jit(run, compiler_options=compute.measured_compiler_options()),
        )

    def _work_state(self, state: TimedTrainingState[State]) -> _WorkState:
        t = state.training
        return _WorkState(
            t.agent,
            t.optimizer_key,
            t.updates,
            t.statistics,
            t.lifetime.agent_key,
            t.healthy,
            state.filled,
        )

    def _perform_work(
        self, state: TimedTrainingState[State], *, measured: bool = False
    ) -> tuple[TimedTrainingState[State], Action]:
        call = self._measured_work if measured else self._work
        work, sampled = call(
            self._work_state(state),
            state.training.lifetime.obs,
            state.training.lifetime.info,
            state.rollout,
        )
        return self._apply_work(state, work), sampled

    def _apply_work(
        self, state: TimedTrainingState[State], work: _WorkState
    ) -> TimedTrainingState[State]:
        training = state.training._replace(
            agent=work.agent,
            optimizer_key=work.optimizer_key,
            updates=work.updates,
            statistics=work.statistics,
            healthy=work.healthy,
            lifetime=state.training.lifetime._replace(agent_key=work.agent_key),
        )
        return state._replace(training=training, filled=work.filled)

    def _finish(
        self,
        state: TimedTrainingState[State],
        sampled: Action,
        ticks: jax.Array,
        cost: Clock,
        *,
        measured: bool = False,
    ) -> TimedTrainingState[State]:
        life = state.training.lifetime
        step = self._measured_advance if measured else self._advance
        result = step(life, state.timing.held_action, sampled.action, ticks)
        transition = Rollout(
            life.obs,
            life.info["action_mask"],
            life.info["active_mask"],
            sampled.action,
            sampled.value,
            result.reward,
            sampled.log_prob,
            jnp.broadcast_to(ticks, (self.env.num_agents,)),
        )

        def insert(buffer: jax.Array, value: jax.Array) -> jax.Array:
            return buffer.at[state.filled].set(value)

        rollout = jax.tree.map(insert, state.rollout, transition)
        healthy = state.training.healthy & result.healthy
        return TimedTrainingState(
            state.training._replace(lifetime=result.lifetime, healthy=healthy),
            compute.next_timing(state.timing, sampled.action, ticks, cost, healthy),
            rollout,
            state.filled + jnp.int32(1),
        )

    def _valid(self, state: TimedTrainingState[State]) -> jax.Array:
        elapsed = state.rollout.elapsed_ticks
        if elapsed is None:
            raise ValueError("A compute rollout must contain elapsed world tick counts.")
        filled = state.filled
        populated = jnp.arange(self.agent.config.rollout_steps)[:, None] < filled
        return (
            state.timing.healthy
            & state.training.healthy
            & (filled >= 0)
            & (filled <= self.agent.config.rollout_steps)
            & jnp.all(~populated | (elapsed > 0))
        )

    def init(
        self, *, seed: int | jax.Array = 0, window_size: int = 128
    ) -> TimedTrainingState[State]:
        """Initialize/prime compilation using throwaway keys, not lifetime RNG.

        The learner starts from the configured hyperparameter defaults.
        """
        training = init_training(self.env, self.agent, seed=seed, window_size=window_size)
        state = TimedTrainingState(
            training,
            compute.init_timing(self.env.num_agents),
            _empty_rollout(
                training.lifetime.obs,
                training.lifetime.info["action_mask"],
                self.agent.config.rollout_steps,
            ),
            jnp.int32(0),
        )
        prime = self._work_state(state)._replace(
            agent_key=jax.random.key(0), optimizer_key=jax.random.key(1)
        )
        jax.block_until_ready(
            self._work(prime, training.lifetime.obs, training.lifetime.info, state.rollout)
        )
        prime = prime._replace(agent_key=jax.random.key(2), optimizer_key=jax.random.key(3))
        jax.block_until_ready(
            self._measured_work(prime, training.lifetime.obs, training.lifetime.info, state.rollout)
        )
        return state

    @property
    def donates(self) -> bool:
        """Always False: a call leaves the carry passed in readable."""
        return False

    def __call__(
        self, state: TimedTrainingState[State], trace: TimingTrace
    ) -> TimedTrainingState[State]:
        """Replay known costs/ticks with bounded on-policy data and no host timing."""
        compute.check_trace(trace, self.decisions, self.quantum_ns)
        if not self._host and not isinstance(trace, MeasuredTimingTrace):
            return self._run(state, trace)
        for i in range(self.decisions):
            if not bool(self._valid(state)) or int(trace.ticks[i]) == 0:
                return state._replace(timing=state.timing._replace(healthy=jnp.asarray(False)))
            state, sampled = self._perform_work(
                state, measured=isinstance(trace, MeasuredTimingTrace)
            )
            if not bool(state.training.healthy):
                return state._replace(timing=state.timing._replace(healthy=jnp.asarray(False)))
            state = self._finish(
                state,
                sampled,
                trace.ticks[i],
                Clock(trace.compute_ns.hi[i], trace.compute_ns.lo[i]),
                measured=isinstance(trace, MeasuredTimingTrace),
            )
        return state

    def measure(
        self, state: TimedTrainingState[State]
    ) -> tuple[TimedTrainingState[State], MeasuredTimingTrace]:
        """Measure synchronized optimization+inference; retain timings for replay."""
        costs: list[int] = []
        for _ in range(self.decisions):
            if not bool(self._valid(state)):
                raise RuntimeError("Cannot measure an unhealthy or invalid compute-training state.")
            start = perf_counter_ns()
            work, sampled = cast(
                tuple[_WorkState, Action],
                jax.block_until_ready(
                    self._measured_work(
                        self._work_state(state),
                        state.training.lifetime.obs,
                        state.training.lifetime.info,
                        state.rollout,
                    )
                ),
            )
            cost = perf_counter_ns() - start
            ticks = jnp.uint32(compute.tick_count(cost, self.quantum_ns))
            state = self._apply_work(state, work)
            if not bool(work.healthy):
                raise RuntimeError("Non-finite timed PPO decision work.")
            state = self._finish(state, sampled, ticks, Clock.from_int(cost), measured=True)
            costs.append(cost)
        return state, compute.record_trace(costs, quantum_ns=self.quantum_ns)
