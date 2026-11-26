"""Health probes and parallel revisit evaluation must be exact and reproducible."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import DQN, IQL, PPO, SAC, DQNConfig, PPOConfig, SACConfig
from perpetua.agents.ppo import PPOState
from perpetua.envs.continual_grid import GridState
from perpetua.params import DriftEnvState

type Act = Callable[[PPOState, jax.Array, pp.Info, jax.Array], jax.Array]


def test_probe_functions_exact_and_validated() -> None:
    activations = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 2.0]], dtype=jnp.float32)
    ratio = pp.probes.dormant_unit_ratio(activations)
    assert float(ratio) == 0.5
    constant = jnp.ones((4, 3), dtype=jnp.float32)
    assert float(pp.probes.dormant_unit_ratio(constant)) == 1.0
    varying = jnp.asarray([[0.0], [1.0], [2.0]], dtype=jnp.float32)
    assert float(pp.probes.dormant_unit_ratio(varying, threshold=0.0)) == 0.0
    rank_one = jnp.outer(jnp.asarray([1.0, 2.0, -1.0, 0.5]), jnp.asarray([1.0, -1.0, 2.0]))
    assert abs(float(pp.probes.effective_rank(rank_one)) - 1.0) < 1e-4
    with pytest.raises(ValueError, match=r"Probe activations must have shape \(batch, units\)"):
        pp.probes.dormant_unit_ratio(jnp.zeros((3,)))
    with pytest.raises(ValueError, match=r"Dormant-unit threshold must be finite and"):
        pp.probes.dormant_unit_ratio(activations, threshold=-1.0)
    with pytest.raises(ValueError, match=r"Probe activations must have shape \(batch, units\)"):
        pp.probes.effective_rank(jnp.zeros((0, 2)))
    with pytest.raises(ValueError, match=r"at least two rows.*every unit would read as dormant"):
        pp.probes.dormant_unit_ratio(jnp.ones((1, 4)))
    with pytest.raises(ValueError, match=r"at least two rows"):
        pp.probes.effective_rank(jnp.ones((1, 4)))


def test_ppo_probe_reports_live_units_and_rank() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    agent = PPO(action_size=5, config=PPOConfig(hidden_size=16))
    state = agent.init(obs, jax.random.key(1))
    batch = jnp.broadcast_to(obs, (32, obs.shape[0], obs.shape[1])).reshape(32, -1)
    report = agent.probe(state, batch)
    assert set(report) == {"dormant_actor", "dormant_critic", "rank_actor", "rank_critic"}
    assert 0.0 <= float(report["dormant_actor"]) <= 1.0
    assert 0.0 <= float(report["dormant_critic"]) <= 1.0
    assert 1.0 <= float(report["rank_actor"]) <= 32.0
    assert 1.0 <= float(report["rank_critic"]) <= 32.0
    with pytest.raises(ValueError, match=r"Dormant-unit threshold must be finite and"):
        agent.probe(state, batch, threshold=-1.0)


def test_dqn_iql_and_sac_probes_report_live_units_and_rank() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    batch = jnp.broadcast_to(obs, (32, obs.shape[0], obs.shape[1])).reshape(32, -1)

    dqn = DQN(action_size=5, config=DQNConfig(hidden_size=16))
    state = dqn.init(obs, jax.random.key(1))
    report = dqn.probe(state, batch)
    assert set(report) == {"dormant_q", "rank_q"}
    assert 0.0 <= float(report["dormant_q"]) <= 1.0
    assert 1.0 <= float(report["rank_q"]) <= 32.0
    np.testing.assert_array_equal(
        np.asarray(dqn.probe(state, batch)["rank_q"]), np.asarray(report["rank_q"])
    )
    with pytest.raises(ValueError, match=r"Dormant-unit threshold must be finite and"):
        dqn.probe(state, batch, threshold=-1.0)

    iql = IQL(action_size=5, config=DQNConfig(hidden_size=16))
    stacked = iql.init(jnp.broadcast_to(obs, (2, obs.shape[1])), jax.random.key(2))
    per_slot = iql.probe(stacked, batch)
    assert set(per_slot) == {"dormant_q", "rank_q"}
    assert per_slot["dormant_q"].shape == (2,)
    assert per_slot["rank_q"].shape == (2,)

    pendulum = pp.make("gymnax/Pendulum-v1")
    pendulum_obs = pendulum.init(jax.random.key(0))[1]
    sac = SAC(
        action_low=pendulum.action_space.low[0],
        action_high=pendulum.action_space.high[0],
        config=SACConfig(hidden_size=16),
    )
    sac_state = sac.init(pendulum_obs, jax.random.key(3))
    sac_batch = jnp.broadcast_to(
        pendulum_obs, (32, pendulum_obs.shape[0], pendulum_obs.shape[1])
    ).reshape(32, -1)
    sac_report = sac.probe(sac_state, sac_batch)
    assert set(sac_report) == {"dormant_actor", "dormant_critic", "rank_actor", "rank_critic"}
    assert 0.0 <= float(sac_report["dormant_actor"]) <= 1.0
    assert 0.0 <= float(sac_report["dormant_critic"]) <= 1.0
    assert 1.0 <= float(sac_report["rank_actor"]) <= 32.0
    assert 1.0 <= float(sac_report["rank_critic"]) <= 32.0


def test_revisit_is_exact_reproducible_and_matches_a_sequential_rollout() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    obs = env.init(jax.random.key(0))[1]
    agent = PPO(action_size=5, config=PPOConfig(hidden_size=8))

    def act(state: PPOState, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        return agent.act(state, obs, info, key).action

    def stack(*leaves: jax.Array) -> jax.Array:
        return jnp.stack(leaves)

    states: PPOState = jax.tree.map(
        stack,
        agent.init(obs, jax.random.key(1)),
        agent.init(obs, jax.random.key(2)),
        agent.init(obs, jax.random.key(3)),
    )
    seeds = jnp.asarray([0, 1])
    key = jax.random.key(7)
    result = pp.evaluation.revisit(env, act, states, seeds=seeds, ticks=8)
    assert result.shape == (3, 2)
    again = pp.evaluation.revisit(env, act, states, seeds=seeds, ticks=8)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(again))

    def duplicate(leaf: jax.Array) -> jax.Array:
        return jnp.stack([leaf[0], leaf[0]])

    identical: PPOState = jax.tree.map(duplicate, states)
    twin = pp.evaluation.revisit(env, act, identical, seeds=seeds, ticks=8)
    np.testing.assert_array_equal(np.asarray(twin[0]), np.asarray(twin[1]))

    def first_of(leaf: jax.Array) -> jax.Array:
        return leaf[0]

    def single(state: PPOState, seed: jax.Array, step_key: jax.Array) -> jax.Array:
        return rollout_once(env, act, state, seed, step_key, ticks=8)

    keys = jax.random.split(key, (3, 2))
    value = cast(jax.Array, jax.jit(single)(jax.tree.map(first_of, states), seeds[1], keys[0, 1]))
    assert float(value) == float(result[0, 1])


def rollout_once(
    env: pp.Environment[DriftEnvState[GridState]],
    act: Act,
    state: PPOState,
    seed: jax.Array,
    key: jax.Array,
    *,
    ticks: int,
) -> jax.Array:
    lifetime = pp.runtime.init_lifetime(env, seed=seed, window_size=ticks)

    def policy(obs: jax.Array, info: pp.Info, act_key: jax.Array) -> jax.Array:
        return act(state, obs, info, act_key)

    def step(
        carry: pp.runtime.Lifetime[DriftEnvState[GridState]], _: None
    ) -> tuple[pp.runtime.Lifetime[DriftEnvState[GridState]], None]:
        return pp.runtime.tick(env, policy, carry), None

    final, _ = jax.lax.scan(step, lifetime, None, length=ticks)
    return jnp.mean(final.metrics.mean)


def test_revisit_ranks_reference_policies_exactly() -> None:
    env = pp.make("ContinualGrid-v0", size=3)

    def greedy_act(state: jax.Array, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        del state
        return pp.policies.greedy(obs, info, key)

    def random_act(state: jax.Array, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        del state
        return pp.policies.random(obs, info, key)

    states = jnp.zeros((2,))
    seeds = jnp.asarray([0])
    greedy = pp.evaluation.revisit(env, greedy_act, states, seeds=seeds, ticks=32)
    random = pp.evaluation.revisit(env, random_act, states, seeds=seeds, ticks=32)
    assert float(greedy[0, 0]) > float(random[0, 0])


def test_revisit_validates_inputs() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    states = jnp.zeros((2,))

    def act(state: jax.Array, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        del state
        return pp.policies.random(obs, info, key)

    with pytest.raises(ValueError, match=r"Revisit seeds must be a nonempty one-dimensional"):
        pp.evaluation.revisit(env, act, states, seeds=jnp.asarray([[0]]), ticks=4)
    with pytest.raises(ValueError, match=r"Revisit seeds must be a nonempty one-dimensional"):
        pp.evaluation.revisit(
            env,
            act,
            states,
            seeds=jnp.asarray([], dtype=jnp.int32),
            ticks=4,
        )
    with pytest.raises(ValueError, match=r"Revisit seeds must be a nonempty one-dimensional"):
        pp.evaluation.revisit(env, act, states, seeds=jnp.asarray([0.5]), ticks=4)
    with pytest.raises(ValueError, match=r"ticks must be a positive integer"):
        pp.evaluation.revisit(env, act, states, seeds=jnp.asarray([0]), ticks=0)


def test_recovery_ticks_measures_post_change_recovery() -> None:
    rewards = jnp.concatenate((jnp.full(16, 1.0), jnp.zeros(6), jnp.full(16, 1.0)))
    ticks = pp.evaluation.recovery_ticks(rewards, jnp.asarray([16]), window=4, fraction=0.9)
    assert int(ticks[0]) == 6
    none = pp.evaluation.recovery_ticks(jnp.full(8, -2.0), jnp.asarray([2]), window=2)
    assert int(none[0]) == 0
    empty = pp.evaluation.recovery_ticks(rewards, jnp.asarray([], dtype=jnp.int32), window=4)
    assert empty.shape == (0,)
    with pytest.raises(ValueError, match=r"Switch indices must lie inside the reward trace"):
        pp.evaluation.recovery_ticks(rewards, jnp.asarray([99]), window=4)
    with pytest.raises(ValueError, match=r"window must be a positive integer"):
        pp.evaluation.recovery_ticks(rewards, jnp.asarray([16]), window=0)
    with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\]"):
        pp.evaluation.recovery_ticks(rewards, jnp.asarray([16]), fraction=1.5)


def test_forgetting_matches_hand_computed_phase_matrix() -> None:
    # scores[k, j]: policy saved after phase k, evaluated on phase j.
    scores = jnp.asarray(
        [
            [1.0, 0.5, 0.4],
            [0.9, 0.8, 0.3],
            [0.2, 0.6, 0.7],
        ]
    )
    # Phase 0: best earlier score max(1.0, 0.9) minus final 0.2 = 0.8.
    # Phase 1: best earlier score 0.8 minus final 0.6 = 0.2. The last phase
    # has no later policy to forget it, so the mean is over phases 0 and 1.
    assert abs(float(pp.evaluation.forgetting(scores)) - 0.5) < 1e-6
    with pytest.raises(ValueError, match=r"Forgetting requires a square \(phases, phases\)"):
        pp.evaluation.forgetting(jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"Forgetting requires a square \(phases, phases\)"):
        pp.evaluation.forgetting(jnp.zeros((0, 0)))
    with pytest.raises(ValueError, match=r"Forgetting requires a square \(phases, phases\)"):
        pp.evaluation.forgetting(jnp.zeros((1, 1)))


def test_forgetting_separates_retained_from_forgotten_policies() -> None:
    learned = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    retained = jnp.asarray([*learned, [1.0, 1.0, 1.0]])
    forgotten = jnp.asarray([*learned, [0.0, 0.0, 1.0]])
    assert float(pp.evaluation.forgetting(retained)) == 0.0
    assert float(pp.evaluation.forgetting(forgotten)) == 1.0
    # Scores the policy never had before learning a phase do not count.
    late = jnp.asarray([[0.0, 0.0], [0.0, 1.0]])
    assert float(pp.evaluation.forgetting(late)) == 0.0
    # Improvement on an earlier phase is negative forgetting (backward transfer).
    improved = jnp.asarray([[0.5, 0.0], [0.9, 1.0]])
    assert abs(float(pp.evaluation.forgetting(improved)) + 0.4) < 1e-6


def test_recovery_ticks_on_a_real_cyclic_goal_lifetime() -> None:
    env = pp.make("ContinualGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 8))
    state, obs, info = env.init(jax.random.key(0))
    rewards = []
    changes = []
    for tick in range(32):
        state, obs, reward, info = env.step(
            state, pp.policies.greedy(obs, info, jax.random.key(tick)), jax.random.key(tick)
        )
        rewards.append(float(reward[0]))
        if bool(state.drift_changed):
            changes.append(tick)
    assert changes == [7, 15, 23, 31]
    trace = jnp.asarray(rewards)
    ticks = pp.evaluation.recovery_ticks(trace, jnp.asarray(changes[:2]), window=4, fraction=0.5)
    # Recovery is bounded within the remaining trace; exact timing depends on
    # the travel phase, so only the range is asserted here.
    assert np.all(np.asarray(ticks) >= 0)
    assert np.all(np.asarray(ticks) < 8)


def test_recent_rewards_reconstructs_the_metrics_ring() -> None:
    state = pp.metrics.init(num_agents=1, window_size=4)
    for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        state = pp.metrics.update(state, jnp.asarray([value]))
    np.testing.assert_array_equal(
        np.asarray(pp.evaluation.recent_rewards(state, steps=3)), [[4.0], [5.0], [6.0]]
    )
    np.testing.assert_array_equal(
        np.asarray(pp.evaluation.recent_rewards(state, steps=4)), [[3.0], [4.0], [5.0], [6.0]]
    )
    with pytest.raises(ValueError, match=r"steps must not exceed the metrics window size"):
        pp.evaluation.recent_rewards(state, steps=5)
    with pytest.raises(ValueError, match=r"steps must be a positive integer"):
        pp.evaluation.recent_rewards(state, steps=0)
    fresh = pp.metrics.init(num_agents=1, window_size=4)
    with pytest.raises(ValueError, match=r"steps must not exceed the populated metrics"):
        pp.evaluation.recent_rewards(fresh, steps=1)
    two = pp.metrics.init(num_agents=2, window_size=4)
    two = pp.metrics.update(two, jnp.asarray([1.0, -1.0]))
    assert pp.evaluation.recent_rewards(two, steps=1).shape == (1, 2)


def test_metrics_track_recovery_in_the_carry() -> None:
    state = pp.metrics.init(num_agents=1, window_size=4)
    for _ in range(4):
        state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(False))
    assert not bool(state.has_switch)
    assert int(state.since_switch) == 0
    assert int(state.recovered_at) == -1
    state = pp.metrics.update(state, jnp.asarray([0.0]), switch=jnp.asarray(True))
    assert bool(state.has_switch)
    assert int(state.since_switch) == 0
    assert int(state.recovered_at) == -1
    np.testing.assert_allclose(np.asarray(state.switch_baseline), 1.0)
    # Window [1,1,1,0] -> [1,1,0,1] -> [1,0,1,1] -> [0,1,1,1] -> [1,1,1,1]:
    # the 10% band (>= 0.9) is first met four ticks after the change, the
    # first tick whose window holds only post-change rewards.
    for _ in range(3):
        state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(False))
    assert int(state.recovered_at) == -1
    state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(False))
    assert int(state.recovered_at) == 4
    state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(False))
    assert int(state.recovered_at) == 4
    assert int(state.since_switch) == 5
    with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\]"):
        pp.metrics.update(state, jnp.asarray([1.0]), fraction=1.5)


def test_a_switch_before_any_reward_is_not_tracked() -> None:
    # An empty window has no mean: a baseline of 0.0 made any non-negative
    # reward stream read as recovered and any negative one as never recovering.
    for reward in (1.0, -1.0):
        state = pp.metrics.init(num_agents=1, window_size=2)
        state = pp.metrics.update(state, jnp.asarray([reward]), switch=jnp.asarray(True))
        for _ in range(5):
            state = pp.metrics.update(state, jnp.asarray([reward]), switch=jnp.asarray(False))
        assert not bool(state.has_switch)
        assert int(state.since_switch) == 0
        assert int(state.recovered_at) == -1
        # The next switch has a window and is tracked as usual.
        state = pp.metrics.update(state, jnp.asarray([reward]), switch=jnp.asarray(True))
        np.testing.assert_allclose(np.asarray(state.switch_baseline), reward)
        for _ in range(3):
            state = pp.metrics.update(state, jnp.asarray([reward]), switch=jnp.asarray(False))
        assert bool(state.has_switch)
        assert int(state.recovered_at) == 2


def test_ticks_since_a_switch_saturate_instead_of_wrapping() -> None:
    state = pp.metrics.init(num_agents=1, window_size=4)
    state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(False))
    state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(True))
    state = state._replace(since_switch=jnp.int32(2**31 - 3))
    seen: list[int] = []
    for _ in range(4):
        state = pp.metrics.update(state, jnp.asarray([-5.0]), switch=jnp.asarray(False))
        seen.append(int(state.since_switch))
    assert seen == [2**31 - 2, 2**31 - 1, 2**31 - 1, 2**31 - 1]
    assert state.since_switch.dtype == jnp.int32
    assert int(state.recovered_at) == -1
    # A recovery that late reports the saturated count, not a negative one.
    for _ in range(4):
        state = pp.metrics.update(state, jnp.asarray([1.0]), switch=jnp.asarray(False))
    assert int(state.recovered_at) == 2**31 - 1


def test_metrics_update_without_change_leaves_recovery_inert() -> None:
    state = pp.metrics.init(num_agents=1, window_size=4)
    for _ in range(8):
        state = pp.metrics.update(state, jnp.asarray([1.0]))
    assert not bool(state.has_switch)
    assert int(state.since_switch) == 0
    assert int(state.recovered_at) == -1


def test_in_carry_recovery_matches_post_hoc_recovery_ticks() -> None:
    rewards = jnp.concatenate((jnp.full(16, 1.0), jnp.zeros(6), jnp.full(16, 1.0)))
    post = pp.evaluation.recovery_ticks(rewards, jnp.asarray([16]), window=4, fraction=0.9)
    assert int(post[0]) == 6
    state = pp.metrics.init(num_agents=1, window_size=4)
    for tick in range(rewards.shape[0]):
        state = pp.metrics.update(state, rewards[tick : tick + 1], switch=tick == 16)
    # The post-hoc measure reports the first forward window inside the band;
    # the in-carry tracker reports the first trailing one, a window later.
    assert int(state.recovered_at) == int(post[0]) + 4 - 1


def test_recovery_is_tracked_inside_the_compiled_lifetime() -> None:
    env = pp.make("ContinualGrid-v0", size=3, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 8))
    runner = pp.runtime.Runner(env, pp.policies.greedy, ticks=36)
    carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
    metrics_state = carry.metrics
    # Changes land on ticks 7, 15, 23 and 31; four ticks later the trailing
    # window holds only post-change rewards and reports the recovery.
    assert bool(metrics_state.has_switch)
    assert int(metrics_state.since_switch) == 4
    assert int(metrics_state.recovered_at) == 4
    assert np.isfinite(float(metrics_state.switch_baseline))
    assert bool(metrics_state.healthy)
    # A window as long as the phase never fills with post-change rewards, so
    # recovery is not claimed before the next change resets the tracker.
    slow = pp.runtime.Runner(env, pp.policies.greedy, ticks=36)(
        pp.runtime.init_lifetime(env, seed=0, window_size=8)
    )
    assert int(slow.metrics.since_switch) == 4
    assert int(slow.metrics.recovered_at) == -1


def test_gradual_drift_declares_no_switch_so_recovery_does_not_apply() -> None:
    walk = pp.drift.RandomWalk(low=0.0, high=4.0, scale=0.5)
    before = walk.init(jnp.asarray([2.0, 2.0]), jax.random.key(0))
    after = walk.step(before, jax.random.key(1))
    assert bool(jnp.any(after.value != before.value))
    assert not bool(pp.drift.switched(before, after))
    env = pp.make("ContinualGrid-v0", size=5, drift=walk)
    carry = pp.runtime.Runner(env, pp.policies.greedy, ticks=512)(
        pp.runtime.init_lifetime(env, seed=0, window_size=8)
    )
    # has_switch false means "not applicable", not "never recovered".
    assert not bool(carry.metrics.has_switch)
    assert int(carry.metrics.recovered_at) == -1
    assert not bool(carry.state.drift_changed)


def test_cyclic_switch_is_declared_even_when_the_parameter_is_unchanged() -> None:
    cyclic = pp.drift.Cyclic([(1.0,), (1.0,)], 2)
    state = cyclic.init(jnp.asarray([1.0]), jax.random.key(0))
    first = cyclic.step(state, jax.random.key(1))
    second = cyclic.step(first, jax.random.key(2))
    assert not bool(pp.drift.switched(state, first))
    assert bool(pp.drift.switched(first, second))


def test_recovery_tracking_stays_inert_without_drift() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    carry = pp.runtime.Runner(env, pp.policies.greedy, ticks=16)(
        pp.runtime.init_lifetime(env, seed=0, window_size=4)
    )
    assert not bool(carry.metrics.has_switch)
    assert int(carry.metrics.since_switch) == 0
    assert int(carry.metrics.recovered_at) == -1


def test_metrics_accumulate_cumulative_and_windowed_regret() -> None:
    state = pp.metrics.init(num_agents=1, window_size=2)
    for reward, reference in ((1.0, 1.0), (2.0, 1.0), (3.0, 1.0)):
        state = pp.metrics.update(
            state, jnp.asarray([reward]), reference_reward=jnp.asarray([reference])
        )
    # Regret is reference - reward: per tick [0, -1, -2] (the learner beat the
    # reference), cumulative -3, window (2 entries) [-1, -2].
    np.testing.assert_allclose(np.asarray(state.regret_total), [-3.0])
    np.testing.assert_allclose(np.asarray(state.regret_mean), [-1.0])
    np.testing.assert_allclose(np.asarray(state.regret_window_mean), [-1.5])
    with pytest.raises(ValueError, match=r"Reference reward shape must match the configured"):
        pp.metrics.update(state, jnp.asarray([1.0]), reference_reward=jnp.asarray([1.0, 2.0]))


def test_metrics_regret_stays_inert_without_a_reference() -> None:
    state = pp.metrics.init(num_agents=1, window_size=2)
    for _ in range(4):
        state = pp.metrics.update(state, jnp.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(state.regret_total), [0.0])
    np.testing.assert_allclose(np.asarray(state.regret_mean), [0.0])
    np.testing.assert_allclose(np.asarray(state.regret_window_mean), [0.0])


def test_shadow_reward_matches_a_manual_reference_step() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    state, obs, info = env.init(jax.random.key(0))
    key = jax.random.key(5)
    shadow = pp.runtime.shadow_reward(env, state, obs, info, pp.policies.greedy, key)
    seen: list[jax.Array] = []

    def recording(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        seen.append(key)
        return pp.policies.greedy(obs, info, key)

    pp.runtime.shadow_reward(env, state, obs, info, recording, key)
    action = pp.policies.greedy(obs, info, seen[0])
    _, _, manual, _ = env.step(state, action, key)
    np.testing.assert_array_equal(np.asarray(shadow), np.asarray(manual))


def test_shadow_reference_key_is_not_a_child_of_a_plain_split_of_the_step_key() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    state, obs, info = env.init(jax.random.key(0))
    key = jax.random.key(5)
    seen: list[jax.Array] = []

    def recording(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        seen.append(key)
        return jnp.zeros((1,), dtype=jnp.int32)

    pp.runtime.shadow_reward(env, state, obs, info, recording, key)
    reference = np.asarray(jax.random.key_data(seen[0]))
    for width in (2, 3, 4, 8):
        for child in jax.random.split(key, width):
            assert not np.array_equal(np.asarray(jax.random.key_data(child)), reference)


def test_runner_regret_is_exact_and_does_not_perturb_the_learner() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    init = pp.runtime.init_lifetime(env, seed=0, window_size=8)
    plain = pp.runtime.Runner(env, pp.policies.greedy, ticks=32)(init)
    same = pp.runtime.Runner(env, pp.policies.greedy, ticks=32, reference=pp.policies.greedy)(init)
    # An identical reference earns exactly the learner's reward from the same
    # state, and the shadow step leaves the learner's trajectory untouched.
    np.testing.assert_allclose(np.asarray(same.metrics.regret_total), [0.0])
    np.testing.assert_array_equal(
        np.asarray(same.state.env.positions), np.asarray(plain.state.env.positions)
    )
    np.testing.assert_allclose(np.asarray(same.metrics.total), np.asarray(plain.metrics.total))
    against_random = pp.runtime.Runner(
        env, pp.policies.greedy, ticks=32, reference=pp.policies.random
    )(init)
    # Greedy beats random, so its regret against random is negative.
    assert float(against_random.metrics.regret_total[0]) < 0.0
    assert float(against_random.metrics.regret_window_mean[0]) <= 0.0
    assert bool(against_random.metrics.healthy)


def test_recovery_ticks_matches_a_direct_search_on_a_long_trace() -> None:
    # Longer than one scoring batch, so batches are stitched together.
    rng = np.random.default_rng(0)
    length, window, fraction = 2600, 16, 0.9
    rewards = rng.normal(size=length).astype(np.float32)
    rewards[1000:1400] -= 3.0  # a dip after the first change
    changes = np.asarray([1000, 2000, 2590])

    def direct(change: int) -> int:
        def mean(start: int) -> np.float32:
            index = np.minimum(start + np.arange(window), length - 1)
            return np.float32(jnp.mean(jnp.asarray(rewards[index])))

        reference = mean(max(change - window, 0))
        threshold = reference - np.float32(1.0 - fraction) * abs(reference)
        for offset in range(length):
            start = change + offset
            if start + window <= length and mean(start) >= threshold:
                return offset
        return length - change

    ticks = pp.evaluation.recovery_ticks(rewards, changes, window=window, fraction=fraction)
    assert [int(tick) for tick in ticks] == [direct(int(change)) for change in changes]


def test_change_names_are_deprecated_aliases_of_the_switch_names() -> None:
    state = pp.metrics.init(num_agents=1, window_size=2)
    reward = jnp.asarray([1.0])
    with pytest.warns(DeprecationWarning, match="pass switch="):
        old = pp.metrics.update(state, reward, change=jnp.asarray(True))
    new = pp.metrics.update(state, reward, switch=jnp.asarray(True))
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True)
    )
    for name in ("since", "has"):
        with pytest.warns(DeprecationWarning, match=f"use Metrics.{name}_switch"):
            value = getattr(new, f"{name}_change")
        assert np.array_equal(np.asarray(value), np.asarray(getattr(new, f"{name}_switch")))
    with pytest.warns(DeprecationWarning, match="use Metrics.switch_baseline"):
        assert float(new.change_baseline) == float(new.switch_baseline)
    with pytest.raises(TypeError, match="switch= only"):
        pp.metrics.update(state, reward, switch=jnp.asarray(True), change=jnp.asarray(True))
    with pytest.warns(DeprecationWarning, match="runtime.drift_switch"):
        assert pp.runtime.drift_change(object()) is None


def test_probes_report_nan_for_non_finite_activations_and_keep_tiny_scales() -> None:
    for bad in (jnp.nan, jnp.inf):
        for activations in (jnp.full((4, 3), bad), jnp.eye(4, 3).at[0, 0].set(bad)):
            assert np.isnan(float(pp.probes.dormant_unit_ratio(activations)))
            assert np.isnan(float(pp.probes.effective_rank(activations)))
    rank_one = jnp.outer(jnp.asarray([1.0, 2.0, -1.0, 0.5]), jnp.asarray([1.0, -1.0, 2.0]))
    for scale in (1.0, 1e-14):
        assert abs(float(pp.probes.effective_rank(rank_one * scale)) - 1.0) < 1e-4, scale
    assert float(pp.probes.effective_rank(jnp.ones((4, 3)))) == 1.0


def test_recovery_ticks_takes_unsigned_and_narrow_switch_indices() -> None:
    # The reference of a switch before tick 32 is the low start of the trace; an
    # index that wrapped below zero would read the high end instead.
    rewards = jnp.concatenate((jnp.zeros(43), jnp.full(300, 1.0)))
    expected = np.asarray(pp.evaluation.recovery_ticks(rewards, np.asarray([5, 40]), window=32))
    np.testing.assert_array_equal(expected, [0, 0])
    narrow = [np.asarray([5, 40], dtype=dtype) for dtype in (np.uint8, np.uint32, np.int8)]
    for switches in (
        *narrow,
        np.asarray([5, 40], dtype=np.uint64),
        jnp.asarray([5, 40], jnp.uint32),
    ):
        ticks = pp.evaluation.recovery_ticks(rewards, switches, window=32)
        assert ticks.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(ticks), expected)
    for outside in (np.asarray([2**64 - 1], dtype=np.uint64), np.asarray([-1], dtype=np.int8)):
        with pytest.raises(ValueError, match=r"Switch indices must lie inside the reward trace"):
            pp.evaluation.recovery_ticks(rewards, outside, window=32)
    with pytest.raises(ValueError, match=r"one-dimensional integer array"):
        pp.evaluation.recovery_ticks(rewards, np.asarray([5.0]), window=32)


def test_metrics_update_refuses_a_switch_that_is_not_a_scalar_flag() -> None:
    state = pp.metrics.init(num_agents=2, window_size=4)
    reward = jnp.asarray([1.0, 0.0])
    for bad in (2.5, float("nan"), jnp.asarray([True, False]), jnp.asarray(1.0)):
        with pytest.raises(ValueError, match=r"switch must be a scalar boolean"):
            pp.metrics.update(state, reward, switch=bad)  # pyright: ignore[reportArgumentType]
    # A switch is tracked once the window holds a reward.
    state = pp.metrics.update(state, reward)
    for flag in (True, 1, np.bool_(True), jnp.asarray(1, dtype=jnp.uint8)):
        updated = pp.metrics.update(state, reward, switch=flag)  # pyright: ignore[reportArgumentType]
        assert bool(updated.has_switch)
        assert updated.has_switch.shape == ()
    update = cast(pp.metrics.Updater, jax.jit(pp.metrics.update))
    compiled = update(state, reward, switch=jnp.asarray(True))
    assert bool(compiled.has_switch)


def random_act(state: jax.Array, obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
    del state
    return pp.policies.random(obs, info, key)


def slot_means(env: pp.Environment[Any], seed: int, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot lifetime means and which slots were ever active, from a plain loop."""
    lifetime = pp.runtime.init_lifetime(env, seed=seed, window_size=1)

    def tick(carry: pp.runtime.Lifetime[Any]) -> pp.runtime.Lifetime[Any]:
        return pp.runtime.tick(env, pp.policies.random, carry)

    step = cast(Callable[[pp.runtime.Lifetime[Any]], pp.runtime.Lifetime[Any]], jax.jit(tick))
    seen = np.asarray(lifetime.info["active_mask"])
    for _ in range(ticks):
        lifetime = step(lifetime)
        seen = seen | np.asarray(lifetime.info["active_mask"])
    return np.asarray(lifetime.metrics.mean), seen


def test_revisit_scores_only_the_learner_slots_of_a_zero_sum_world() -> None:
    env = pp.make("OpponentGrid-v0", size=3)
    result = pp.evaluation.revisit(
        env, random_act, jnp.zeros((1,)), seeds=jnp.asarray([0, 1]), ticks=128
    )
    for column, seed in enumerate((0, 1)):
        means, seen = slot_means(env, seed, 128)
        assert seen.tolist() == [True, False]
        assert means[0] == -means[1] != 0.0  # all slots would average to zero
        assert float(result[0, column]) == float(means[0])


@pytest.mark.parametrize(
    "kwargs", [{"size": 3}, {"size": 4, "num_agents": 3}], ids=["single-agent", "all-active"]
)
def test_revisit_still_averages_every_slot_when_all_of_them_act(kwargs: dict[str, int]) -> None:
    env = pp.make("ContinualGrid-v0", **kwargs)
    result = pp.evaluation.revisit(
        env, random_act, jnp.zeros((1,)), seeds=jnp.asarray([0, 5]), ticks=64
    )
    for column, seed in enumerate((0, 5)):
        means, seen = slot_means(env, seed, 64)
        assert seen.all()
        # The value before learner slots were singled out: the mean over all slots.
        assert float(result[0, column]) == float(jnp.mean(jnp.asarray(means)))
