"""Reward centering: offset invariance, convergence and bounded values."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from perpetua.agents.dqn import DQN, DQNConfig, init_replay, insert
from perpetua.agents.ppo import PPO, PPOConfig, PPOState, Rollout


def rollout(reward: float, *, steps: int = 4, features: int = 3) -> Rollout:
    """A tiny on-policy rollout with constant reward and zero values."""
    obs = jnp.zeros((steps, 1, features), dtype=jnp.float32)
    return Rollout(
        obs,
        jnp.ones((steps, 1, 3), dtype=jnp.bool_),
        jnp.ones((steps, 1), dtype=jnp.bool_),
        jnp.zeros((steps, 1), dtype=jnp.int32),
        jnp.zeros((steps, 1), dtype=jnp.float32),
        jnp.full((steps, 1), reward, dtype=jnp.float32),
        jnp.zeros((steps, 1), dtype=jnp.float32),
    )


def test_ppo_offset_invariance_with_matching_rate() -> None:
    agent = PPO(action_size=3, config=PPOConfig(reward_centering=True, epochs=1, minibatches=1))
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))
    base, _, _ = agent.update(state, rollout(1.0), jnp.zeros(1), jax.random.key(1))
    shifted = state._replace(reward_rate=jnp.asarray(7.0))
    other, _, _ = agent.update(shifted, rollout(8.0), jnp.zeros(1), jax.random.key(1))
    for a, b in zip(jax.tree.leaves(base.params), jax.tree.leaves(other.params), strict=True):
        np.testing.assert_allclose(
            np.asarray(cast("jax.Array", a)), np.asarray(cast("jax.Array", b)), atol=1e-6
        )


def test_dqn_offset_invariance_with_matching_rate() -> None:
    agent = DQN(action_size=3, config=DQNConfig(reward_centering=True))
    obs = jnp.zeros((1, 3), dtype=jnp.float32)
    state = agent.init(obs, jax.random.key(0))
    replay = init_replay(num_agents=1, obs_dim=3, action_size=3, capacity=4)
    transition = (
        obs,
        jnp.ones((1, 3), dtype=jnp.bool_),
        jnp.ones((1,), dtype=jnp.bool_),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.asarray([1.0]),
        obs,
        jnp.ones((1, 3), dtype=jnp.bool_),
    )
    replay = insert(replay, transition)
    replay = replay._replace(reward=jnp.asarray([1.0]))
    base, _ = agent.update(state, replay, jax.random.key(2))
    shifted = state._replace(reward_rate=jnp.asarray(7.0))
    replay_shifted = replay._replace(reward=jnp.asarray([8.0]))
    other, _ = agent.update(shifted, replay_shifted, jax.random.key(2))
    for a, b in zip(jax.tree.leaves(base.params), jax.tree.leaves(other.params), strict=True):
        np.testing.assert_allclose(
            np.asarray(cast("jax.Array", a)), np.asarray(cast("jax.Array", b)), atol=1e-6
        )


def test_reward_rate_converges_to_the_average_reward() -> None:
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=0.9),
    )
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))
    # Zero parameters and zero learning rate keep values at zero, so the TD
    # error is exactly the reward offset: r_bar_{n+1} = r_bar_n + beta*(1 - r_bar_n).
    state = state._replace(
        params=jax.tree.map(jnp.zeros_like, state.params),
        hyperparameters=state.hyperparameters._replace(
            learning_rate=jnp.asarray(0.0), centering_step_size=jnp.asarray(0.1)
        ),
    )
    data = rollout(1.0)
    update = jax.jit(agent.update)
    for _ in range(100):
        state, _, _ = update(state, data, jnp.zeros(1), jax.random.key(1))
    expected = 1.0 - 0.9**100
    np.testing.assert_allclose(np.asarray(state.reward_rate), expected, atol=1e-5)
    assert abs(float(state.reward_rate) - 1.0) < 1e-3


def test_centered_advantages_stay_small_at_high_discount() -> None:
    from perpetua.agents.ppo import compute_gae

    rewards = jnp.ones((256, 1), dtype=jnp.float32)
    values = jnp.zeros((256, 1), dtype=jnp.float32)
    last = jnp.zeros((1,), dtype=jnp.float32)
    _, plain_targets = compute_gae(
        rewards=rewards, values=values, last_value=last, gamma=0.999, gae_lambda=1.0
    )
    _, centered_targets = compute_gae(
        rewards=rewards - 0.999, values=values, last_value=last, gamma=0.999, gae_lambda=1.0
    )
    # Over 256 steps at gamma 0.999 the uncentered return is hundreds,
    # while the centered one stays of order one.
    assert float(jnp.max(jnp.abs(plain_targets))) > 100.0
    assert float(jnp.max(jnp.abs(centered_targets))) < 5.0


def timed_rollout(rewards: list[float], ticks: list[int]) -> Rollout:
    """A rollout whose decisions span different numbers of world ticks."""
    base = rollout(0.0, steps=len(rewards))
    return base._replace(
        reward=jnp.asarray(rewards, dtype=jnp.float32)[:, None],
        elapsed_ticks=jnp.asarray(ticks, dtype=jnp.uint32)[:, None],
    )


def test_discounted_tick_count() -> None:
    from perpetua.agents.ppo import discounted_tick_count

    ticks = jnp.asarray([1, 2, 5], dtype=jnp.uint32)
    np.testing.assert_allclose(
        np.asarray(discounted_tick_count(jnp.asarray(0.9), ticks)),
        [1.0, 1.9, (1 - 0.9**5) / 0.1],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(discounted_tick_count(jnp.asarray(1.0), ticks)), [1, 2, 5]
    )


def test_ppo_offset_invariance_holds_when_decisions_span_different_tick_counts() -> None:
    # Adding c to every world tick's reward adds c * (1 - gamma**k) / (1 - gamma)
    # to a k-tick decision's discounted reward. With the rate shifted by c,
    # centering must remove exactly that, whatever k is.
    gamma, offset = 0.9, 7.0
    ticks = [1, 3, 2, 5]
    base_rewards = [0.5, 1.0, -0.25, 2.0]
    counts = [(1 - gamma**k) / (1 - gamma) for k in ticks]
    shifted_rewards = [r + offset * c for r, c in zip(base_rewards, counts, strict=True)]
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=gamma),
    )
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))
    base, _, _ = agent.update(
        state, timed_rollout(base_rewards, ticks), jnp.zeros(1), jax.random.key(1)
    )
    shifted = state._replace(reward_rate=jnp.asarray(offset))
    other, _, _ = agent.update(
        shifted, timed_rollout(shifted_rewards, ticks), jnp.zeros(1), jax.random.key(1)
    )
    for a, b in zip(jax.tree.leaves(base.params), jax.tree.leaves(other.params), strict=True):
        np.testing.assert_allclose(
            np.asarray(cast("jax.Array", a)), np.asarray(cast("jax.Array", b)), atol=1e-5
        )
    np.testing.assert_allclose(
        float(other.reward_rate) - float(base.reward_rate), offset, atol=1e-5
    )


def test_one_tick_decisions_center_exactly_as_before() -> None:
    agent = PPO(action_size=3, config=PPOConfig(reward_centering=True, epochs=1, minibatches=1))
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))._replace(reward_rate=jnp.asarray(0.3))
    plain, _, _ = agent.update(state, rollout(1.0), jnp.zeros(1), jax.random.key(1))
    timed, _, _ = agent.update(
        state, timed_rollout([1.0] * 4, [1] * 4), jnp.zeros(1), jax.random.key(1)
    )
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(timed), strict=True):
        a, b = np.asarray(cast("jax.Array", a)), np.asarray(cast("jax.Array", b))
        if jax.default_backend() == "cpu":
            np.testing.assert_array_equal(a, b)
        else:  # a GPU may differ in the last float32 bit (one ulp seen on an H200)
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def test_rate_estimates_the_per_tick_reward_under_compute_aware_time() -> None:
    # Constant reward 1 per tick over decisions of 1-5 ticks: the rate must
    # converge to 1 per tick, not to the reward per decision.
    gamma, ticks = 0.9, [1, 3, 2, 5]
    counts = [(1 - gamma**k) / (1 - gamma) for k in ticks]
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=gamma),
    )
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))
    state = state._replace(
        params=jax.tree.map(jnp.zeros_like, state.params),
        hyperparameters=state.hyperparameters._replace(
            learning_rate=jnp.asarray(0.0), centering_step_size=jnp.asarray(0.1)
        ),
    )
    data = timed_rollout(counts, ticks)  # each decision: the discounted sum of 1s
    update = jax.jit(agent.update)
    for _ in range(200):
        state, _, _ = update(state, data, jnp.zeros(1), jax.random.key(1))
    assert abs(float(state.reward_rate) - 1.0) < 1e-4


def test_compute_aware_ppo_trains_with_reward_centering() -> None:
    import perpetua as pp
    from perpetua.agents import ComputePPORunner
    from perpetua.compute import TimingTrace

    env = pp.make("ContinualGrid-v0", size=3)
    config = PPOConfig(
        rollout_steps=4, epochs=1, minibatches=1, hidden_size=8, reward_centering=True
    )
    runner = ComputePPORunner(env, PPO(action_size=5, config=config), decisions=4, quantum_ns=100)
    state = runner.init(seed=0, window_size=4)
    for _ in range(3):
        state = runner(state, TimingTrace.from_ticks([1, 3, 2, 4], quantum_ns=100))
    assert bool(state.training.healthy)
    assert state.training.updates.to_int() >= 2
    assert np.isfinite(float(state.training.agent.reward_rate))


def sitting_out_rollout(placeholder: float) -> Rollout:
    # One slot, reward 1 per tick, inactive on tick 1 with a placeholder value.
    data = rollout(1.0)
    return data._replace(
        active=jnp.asarray([[True], [False], [True], [True]]),
        value=jnp.asarray([[0.0], [placeholder], [0.0], [0.0]]),
    )


def test_rate_ignores_the_placeholder_value_of_an_inactive_tick() -> None:
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=0.9),
    )
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))
    state = state._replace(
        hyperparameters=state.hyperparameters._replace(centering_step_size=jnp.asarray(0.1))
    )
    rates = []
    for placeholder in (0.0, 5.0, 123.0, -7.0):
        updated, _, _ = agent.update(
            state, sitting_out_rollout(placeholder), jnp.zeros(1), jax.random.key(1)
        )
        rates.append(float(updated.reward_rate))
    # Bridged errors on the active ticks: 1 + 0.9 * 1 across the inactive tick,
    # a span of 1.9 discounted ticks, then 1 and 1. Per tick each is 1.
    np.testing.assert_allclose(rates, [0.1] * 4, atol=1e-6)


def frozen(agent: PPO, slots: int = 1) -> PPOState:
    """Zero values and no learning, so the TD error is the centered reward alone."""
    state = agent.init(jnp.zeros((slots, 3)), jax.random.key(0))
    return state._replace(
        params=jax.tree.map(jnp.zeros_like, state.params),
        hyperparameters=state.hyperparameters._replace(
            learning_rate=jnp.asarray(0.0), centering_step_size=jnp.asarray(0.1)
        ),
    )


def test_rate_converges_when_a_slot_acts_every_32nd_tick() -> None:
    # The bridged error spans 27.5 discounted ticks; undivided, the effective
    # step 0.1 * 27.5 exceeds 2 and the rate oscillated to -3.8e14 in 60 updates.
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=0.99),
    )
    state = frozen(agent)
    data = rollout(1.0, steps=64)._replace(active=(jnp.arange(64) % 32 == 0)[:, None])
    rates = []
    update = jax.jit(agent.update)
    for _ in range(60):
        state, _, _ = update(state, data, jnp.zeros(1), jax.random.key(1))
        rates.append(float(state.reward_rate))
    np.testing.assert_allclose(rates, [1.0 - 0.9 ** (n + 1) for n in range(60)], atol=1e-5)


def test_span_and_tick_count_compose_to_the_ticks_of_the_whole_span() -> None:
    # Decisions of 2, 3, 1, 4, 2 world ticks; the slot sits out steps 1, 2 and 4.
    gamma, rate = 0.9, 0.25
    ticks = [2, 3, 1, 4, 2, 1]
    on = [True, False, False, True, False, True]
    rewards = [0.5, 1.0, -0.25, 2.0, 0.75, 1.5]
    values = [0.3, 9.0, -9.0, -0.4, 9.0, 0.8]
    last = 0.6
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=gamma),
    )
    state = agent.init(jnp.zeros((1, 3)), jax.random.key(0))._replace(reward_rate=jnp.asarray(rate))
    state = state._replace(
        hyperparameters=state.hyperparameters._replace(centering_step_size=jnp.asarray(0.1))
    )
    data = timed_rollout(rewards, ticks)._replace(
        active=jnp.asarray(on)[:, None], value=jnp.asarray(values, dtype=jnp.float32)[:, None]
    )
    updated, _, _ = agent.update(state, data, jnp.asarray([last]), jax.random.key(1))

    def count(k: int) -> float:
        return (1 - gamma**k) / (1 - gamma)

    # Each span written out: the tick-centered rewards, discounted, then the
    # value of the next acting step; divided by the ticks of the whole span.
    first = (
        (rewards[0] - rate * count(2))
        + gamma**2 * (rewards[1] - rate * count(3))
        + gamma**5 * (rewards[2] - rate * count(1))
        + gamma**6 * values[3]
        - values[0]
    ) / count(6)
    second = (
        (rewards[3] - rate * count(4))
        + gamma**4 * (rewards[4] - rate * count(2))
        + gamma**6 * values[5]
        - values[3]
    ) / count(6)
    third = (rewards[5] - rate * count(1) + gamma * last - values[5]) / count(1)
    expected = rate + 0.1 * (first + second + third) / 3
    np.testing.assert_allclose(float(updated.reward_rate), expected, rtol=1e-5)


def test_rate_estimates_the_per_tick_reward_with_long_decisions_and_inactive_steps() -> None:
    gamma, ticks = 0.9, [1, 3, 2, 5, 4, 2]
    counts = [(1 - gamma**k) / (1 - gamma) for k in ticks]
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=gamma),
    )
    state = frozen(agent)
    data = timed_rollout(counts, ticks)._replace(
        active=jnp.asarray([True, False, False, True, True, False])[:, None]
    )
    update = jax.jit(agent.update)
    for _ in range(200):
        state, _, _ = update(state, data, jnp.zeros(1), jax.random.key(1))
    assert abs(float(state.reward_rate) - 1.0) < 1e-4


def test_rate_is_unchanged_when_no_slot_sits_out() -> None:
    # Reference rates recorded from the update before spans were divided out
    # (macOS CPU, float32): with every slot active the divisor is the same.
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=2, minibatches=2, gamma=0.9),
    )
    state = agent.init(jnp.zeros((2, 3)), jax.random.key(0))._replace(reward_rate=jnp.asarray(0.25))
    obs_key, value_key, reward_key, tick_key = jax.random.split(jax.random.key(7), 4)
    plain = Rollout(
        jax.random.normal(obs_key, (8, 2, 3)),
        jnp.ones((8, 2, 3), dtype=jnp.bool_),
        jnp.ones((8, 2), dtype=jnp.bool_),
        jnp.zeros((8, 2), dtype=jnp.int32),
        jax.random.normal(value_key, (8, 2)),
        jax.random.normal(reward_key, (8, 2)),
        jnp.full((8, 2), -1.0986123),
    )
    timed = plain._replace(
        elapsed_ticks=jax.random.randint(tick_key, (8, 2), 1, 5).astype(jnp.uint32)
    )
    recorded = {"plain": "0x1.f6274c0000000p-3", "timed": "0x1.fa2d500000000p-3"}
    for name, data in (("plain", plain), ("timed", timed)):
        updated, _, _ = agent.update(state, data, jnp.asarray([0.7, -0.2]), jax.random.key(1))
        rate, reference = float(updated.reward_rate), float.fromhex(recorded[name])
        if jax.default_backend() == "cpu":
            assert rate == reference
        else:
            np.testing.assert_allclose(rate, reference, rtol=1e-6)


def test_rate_matches_the_one_step_td_error_when_every_slot_is_active() -> None:
    agent = PPO(
        action_size=3,
        config=PPOConfig(reward_centering=True, epochs=1, minibatches=1, gamma=0.9),
    )
    state = agent.init(jnp.zeros((2, 3)), jax.random.key(0))._replace(reward_rate=jnp.asarray(0.25))
    values = jnp.asarray([[0.5, -1.0], [2.0, 0.0], [1.5, 0.3], [-0.5, 1.0]])
    rewards = jnp.asarray([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0], [1.0, -1.0]])
    last = jnp.asarray([0.7, -0.2])
    data = Rollout(
        jnp.zeros((4, 2, 3)),
        jnp.ones((4, 2, 3), dtype=jnp.bool_),
        jnp.ones((4, 2), dtype=jnp.bool_),
        jnp.zeros((4, 2), dtype=jnp.int32),
        values,
        rewards,
        jnp.zeros((4, 2)),
    )
    updated, _, _ = agent.update(state, data, last, jax.random.key(1))
    next_values = jnp.concatenate([values[1:], last[None]], axis=0)
    td = rewards - 0.25 + 0.9 * next_values - values
    expected = 0.25 + float(state.hyperparameters.centering_step_size) * float(jnp.mean(td))
    np.testing.assert_allclose(float(updated.reward_rate), expected, rtol=1e-6)
