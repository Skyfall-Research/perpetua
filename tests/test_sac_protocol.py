"""Opt-in SAC options for task-sequence protocols such as Continual World.

Network depth, LayerNorm and leaky ReLU; uniform exploration and delayed
learning after each switch; replay, optimizer and temperature resets on the
Oracle's ``boundary``; respawn-crossing transitions kept out of TD targets;
and the one-hot task observation.
"""

# JAX and Flax typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import math
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import SAC, HostSACRunner, SACConfig, SACRunner

SMALL: dict[str, Any] = {
    "rollout_steps": 4,
    "buffer_capacity": 64,
    "batch_size": 4,
    "gradient_steps": 2,
}


def pendulum(period: int = 8, *, oracle: bool = True) -> Any:
    env = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (12.0,)], period))
    return pp.Oracle(env, expose=["boundary", "task_id"]) if oracle else env


def sac(env: Any, **config: Any) -> SAC:
    box = env.action_space
    return SAC(
        action_low=box.low[0],
        action_high=box.high[0],
        config=SACConfig(**cast(dict[str, Any], {**SMALL, "hidden_size": 8, **config})),
    )


def test_network_depth_layer_norm_and_leaky_relu() -> None:
    env = pendulum(oracle=False)
    agent = sac(env, hidden_layers=4, layer_norm=True, activation="leaky_relu")
    state = SACRunner(env, agent).init(seed=0)
    actor = cast(Any, state.agent.params.actor)["params"]
    assert sorted(actor) == [*(f"Dense_{i}" for i in range(6)), "LayerNorm_0"]
    critic = cast(Any, state.agent.params.q1)["params"]
    assert "LayerNorm_0" in critic
    assert "Dense_4" in critic
    with pytest.raises(ValueError, match=r"hidden_layers must be a positive integer\."):
        SACConfig(hidden_layers=0)
    with pytest.raises(ValueError, match=r"activation must be 'tanh', 'crelu' or"):
        SACConfig(activation="relu6")  # pyright: ignore[reportArgumentType]


def test_boundary_reset_requires_the_oracle_to_expose_the_switch() -> None:
    env = pendulum(oracle=False)
    with pytest.raises(ValueError, match="Oracle"):
        SACRunner(env, sac(env, task_boundary_reset=True)).init(seed=0)


def test_boundary_resets_replay_optimizer_and_temperature_but_keeps_weights() -> None:
    # Gravity switches every 8 ticks; chunks are 4 ticks, learning waits 3.
    env = pendulum(period=8)
    agent = sac(env, task_boundary_reset=True, autotune_alpha=True, alpha=math.e, update_after=3)
    runner = SACRunner(env, agent)
    state = runner.init(seed=0)
    seen = []
    for _ in range(4):
        state = runner(state)
        seen.append(
            (
                state.lifetime.state.t.to_int(),
                int(state.replay.cursor),
                int(state.since_boundary),
                round(float(state.agent.log_alpha), 4),
                state.agent.gradient_steps.to_int(),
            )
        )
    # At tick 8 the switch empties the replay, restarts the counter and the
    # temperature (log alpha back to 1), and that chunk does not learn.
    assert [row[:3] for row in seen] == [(4, 4, 3), (8, 0, 0), (12, 4, 3), (16, 0, 0)]
    assert seen[1][3] == 1.0
    assert seen[0][3] != 1.0
    assert [row[4] for row in seen] == [2, 2, 4, 4]


def test_begin_task_restarts_optimizer_and_alpha_only() -> None:
    env = pendulum(oracle=False)
    agent = sac(env, autotune_alpha=True, alpha=math.e)
    runner = SACRunner(env, agent)
    state = runner(runner(runner.init(seed=0)))
    fresh = agent.begin_task(state.agent)
    assert float(fresh.log_alpha) == pytest.approx(1.0)
    for a, b in zip(
        jax.tree.leaves(fresh.params), jax.tree.leaves(state.agent.params), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    first = jax.tree.leaves(fresh.opt_state.actor)
    initial = jax.tree.leaves(agent.actor_optimizer.init(fresh.params.actor))
    for a, b in zip(first, initial, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_random_steps_explore_uniformly_then_follow_the_policy() -> None:
    env = pendulum(oracle=False)
    agent = sac(env, random_steps=8, rollout_steps=8, gradient_steps=1)
    runner = SACRunner(env, agent)
    explored = runner(runner.init(seed=0))
    plain_agent = sac(env, rollout_steps=8, gradient_steps=1)
    plain = SACRunner(env, plain_agent)(SACRunner(env, plain_agent).init(seed=0))
    actions = np.asarray(explored.replay.action[:8])
    assert np.all(np.abs(actions) <= 2.0)
    assert not np.allclose(actions, np.asarray(plain.replay.action[:8]))
    # A newly initialized policy is close to zero-mean; uniform actions spread out.
    assert np.std(actions) > 0.5


def test_update_after_delays_learning() -> None:
    env = pendulum(oracle=False)
    agent = sac(env, update_after=6)
    runner = SACRunner(env, agent)
    state = runner(runner.init(seed=0))  # 4 ticks < 6: no gradient steps
    assert state.agent.gradient_steps.to_int() == 0
    state = runner(state)  # 8 ticks: learning starts
    assert state.agent.gradient_steps.to_int() == SMALL["gradient_steps"]


class Blink:
    """A one-slot continuous world that respawns on every third tick."""

    num_agents = 1

    def __init__(self) -> None:
        from gymnasium import spaces

        self.observation_space = spaces.Box(0.0, 3.0, shape=(1, 1), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1, 1), dtype=np.float32)
        self.metadata: pp.Metadata = {"env_id": "blink-v0", "oracle_exposure": ()}

    @property
    def default_params(self) -> Any:
        return {"scale": jnp.asarray(1.0, dtype=jnp.float32)}

    def init(self, key: jax.Array, params: Any) -> Any:
        state = (jnp.int32(0), jnp.asarray(False))
        return (state, *self.observe(state, params))

    def step(self, state: Any, action: Any, key: jax.Array, params: Any) -> Any:
        count = (state[0] + 1) % 3
        return (count, count == 0), jnp.ones(1, dtype=jnp.float32)

    def observe(self, state: Any, params: Any) -> Any:
        info = {
            "action_mask": jnp.ones((1, 1), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state[1][None],
        }
        return state[0].astype(jnp.float32).reshape(1, 1), info

    def privileged(self, state: Any) -> dict[str, jax.Array]:
        return {}


def test_mask_respawns_keeps_respawn_transitions_out_of_td_targets() -> None:
    from perpetua.params import DriftEnv

    env = DriftEnv(cast(Any, Blink()), None, env_id="blink-v0")
    masked = SACRunner(env, sac(env, mask_respawns=True, rollout_steps=9, gradient_steps=1))
    state = masked(masked.init(seed=0))
    active = np.asarray(state.replay.active[:9])[:, 0]
    # Ticks 3, 6 and 9 respawn: the transitions into them are inactive.
    assert active.tolist() == [True, True, False] * 3
    plain = SACRunner(env, sac(env, rollout_steps=9, gradient_steps=1))
    assert bool(np.all(np.asarray(plain(plain.init(seed=0)).replay.active[:9])))


class Pause(Blink):
    """Blink on a six-tick cycle: inactive on counts 2 and 3, respawn flagged on 3."""

    def step(self, state: Any, action: Any, key: jax.Array, params: Any) -> Any:
        count = (state[0] + 1) % 6
        return (count, count == 3), jnp.ones(1, dtype=jnp.float32)

    def observe(self, state: Any, params: Any) -> Any:
        obs, info = super().observe(state, params)
        info["active_mask"] = ((state[0] != 2) & (state[0] != 3))[None]
        return obs, info


def test_mask_respawns_sees_a_respawn_flagged_while_the_slot_waits() -> None:
    from perpetua.params import DriftEnv

    env = DriftEnv(cast(Any, Pause()), None, env_id="pause-v0")
    masked = SACRunner(env, sac(env, mask_respawns=True, rollout_steps=12, gradient_steps=1))
    state = masked(masked.init(seed=0))
    # Per cycle: 0->1 is stored; the action at count 1 waits through counts 2
    # and 3 and completes at 4, having crossed the respawn flagged at 3.
    assert (
        np.asarray(state.replay.active[:12])[:, 0].tolist()
        == [
            True,
            False,
            False,
            False,
            True,
            True,
        ]
        * 2
    )
    assert not bool(np.any(np.asarray(state.pending.respawned)))
    plain = SACRunner(env, sac(env, rollout_steps=12, gradient_steps=1))
    assert (
        np.asarray(plain(plain.init(seed=0)).replay.active[:12])[:, 0].tolist()
        == [
            True,
            False,
            False,
            True,
            True,
            True,
        ]
        * 2
    )


def test_log_probability_is_exact_when_the_policy_is_nearly_deterministic() -> None:
    env = pendulum(oracle=False)
    agent = sac(env)
    actor = SACRunner(env, agent).init(seed=0).agent.params.actor
    obs = jnp.zeros((4, 3))
    key = jax.random.key(3)
    mean = 0.37

    def narrowed(log_std: jax.Array) -> Any:
        # Zero weights: the heads output their biases, a mean and a log-std.
        def leaf(path: Any, value: jax.Array) -> jax.Array:
            name = jax.tree_util.keystr(path)
            if "Dense_2" in name and "bias" in name:
                return jnp.full_like(value, mean)
            if "Dense_3" in name and "bias" in name:
                return jnp.zeros_like(value) + log_std
            return jnp.zeros_like(value)

        return jax.tree_util.tree_map_with_path(leaf, actor)

    def log_prob(log_std: jax.Array) -> jax.Array:
        return agent._squashed(narrowed(log_std), obs, key).log_prob  # pyright: ignore[reportPrivateUsage]

    log_std = jnp.asarray(-17.0, dtype=jnp.float32)
    noise = np.asarray(jax.random.normal(key, (4, 1), dtype=jnp.float32), dtype=np.float64)
    pre_tanh = mean + math.exp(-17.0) * noise
    correction = 2.0 * (math.log(2.0) - pre_tanh - np.log1p(np.exp(-2.0 * pre_tanh)))
    scale = float(np.asarray(env.action_space.high)[0, 0])
    reference = (
        -0.5 * noise**2 + 17.0 - 0.5 * math.log(2.0 * math.pi) - correction - math.log(scale)
    )[:, 0]
    np.testing.assert_allclose(np.asarray(log_prob(log_std)), reference, rtol=1e-6)
    # d/d log_std is -1 per sample: the noise term does not depend on it and
    # the tanh correction moves by std * noise, which is negligible here.

    def total(value: jax.Array) -> jax.Array:
        return log_prob(value).sum()

    gradient = jax.grad(total)(log_std)
    np.testing.assert_allclose(float(gradient), -4.0, rtol=1e-5)


def test_one_hot_observation_appends_an_exposed_field() -> None:
    env = pendulum(period=4)
    wrapped = pp.OneHotObservation(env, field="task_id", size=3)
    assert wrapped.observation_space.shape == (1, 3 + 3)
    state, obs, info = wrapped.init(jax.random.key(0))
    np.testing.assert_array_equal(np.asarray(obs[0, -3:]), [1.0, 0.0, 0.0])
    configuration = cast(dict[str, Any], cast(Any, wrapped.metadata)["configuration"])
    assert configuration["observation_appends"] == [
        {"field": "task_id", "encoding": "one_hot", "size": 3}
    ]
    assert "task_id" in wrapped.metadata["oracle_exposure"]
    with pytest.raises(ValueError, match="privileged"):
        pp.OneHotObservation(pendulum(oracle=False), field="task_id", size=3)
    pp.testing.check_env(lambda: pp.OneHotObservation(pendulum(period=4), field="task_id", size=3))
    del state, info


def test_squashed_log_probability_keeps_its_gradient_when_tanh_saturates() -> None:
    env = pendulum(oracle=False)
    agent = sac(env)
    state = SACRunner(env, agent).init(seed=0)
    actor = state.agent.params.actor
    obs = jnp.zeros((1, 3))
    # Moderate pre-tanh values: the exact correction matches log(1 - tanh**2).
    u = jnp.linspace(-3.0, 3.0, 13)
    exact = 2.0 * (math.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    np.testing.assert_allclose(
        np.asarray(exact), np.log1p(-(np.tanh(np.asarray(u)) ** 2)), atol=1e-5
    )

    def log_prob(shift: jax.Array) -> jax.Array:
        # Push the policy mean far into tanh saturation through its output bias.
        def moved(path: Any, leaf: jax.Array) -> jax.Array:
            name = jax.tree_util.keystr(path)
            return leaf + shift if "Dense_2" in name and "bias" in name else leaf

        params = jax.tree_util.tree_map_with_path(moved, actor)
        sampled = agent._squashed(params, obs, jax.random.key(0))  # pyright: ignore[reportPrivateUsage]
        return sampled.log_prob.sum()

    gradient = float(jax.grad(log_prob)(jnp.asarray(500.0)))
    assert np.isfinite(gradient)
    assert abs(gradient) > 1.0


@pytest.mark.parametrize("host", [False, True])
def test_emptied_replay_is_not_learned_from_without_update_after(host: bool) -> None:
    # Same schedule as above, but learning does not wait (update_after=0): the
    # chunk whose last tick empties the replay must still not learn.
    env = pendulum(period=8)
    agent = sac(env, task_boundary_reset=True, update_after=0)
    runner = HostSACRunner(env, agent) if host else SACRunner(env, agent)
    state = runner.init(seed=0)
    seen = []
    for _ in range(4):
        before = state.agent.params.actor
        state = runner(state)
        unchanged = all(
            bool(jnp.array_equal(old, new))
            for old, new in zip(
                jax.tree.leaves(before), jax.tree.leaves(state.agent.params.actor), strict=True
            )
        )
        seen.append((int(state.replay.cursor), state.agent.gradient_steps.to_int(), unchanged))
    assert seen == [(4, 2, False), (0, 2, True), (4, 4, False), (0, 4, True)]


class Relabelled:
    """Replace the exposed ``task_id`` of an Oracle environment with a fixed value."""

    def __init__(self, env: Any, value: jax.Array, dtype: Any = jnp.float32) -> None:
        self.env, self.value, self.dtype = env, value, dtype
        for name in ("num_agents", "observation_space", "action_space", "oracle_fields"):
            setattr(self, name, getattr(env, name))
        self.metadata = env.metadata

    def init(self, key: jax.Array) -> tuple[Any, jax.Array, Any]:
        state, obs, info = self.env.init(key)
        return state, obs.astype(self.dtype), {**info, "oracle": {"task_id": self.value}}


def test_one_hot_observation_is_float32_and_zero_outside_the_range() -> None:
    def code(value: Any, dtype: Any = jnp.float32) -> jax.Array:
        wrapped = pp.OneHotObservation(
            cast(Any, Relabelled(pendulum(period=4), jnp.asarray(value), dtype)),
            field="task_id",
            size=3,
        )
        obs = wrapped.init(jax.random.key(0))[1]
        assert obs.dtype == wrapped.observation_space.dtype == jnp.float32
        return obs[0, -3:]

    for outside in (-0.5, -1.0, 3.0, 3.5, float("nan"), float("inf"), -7, 3):
        np.testing.assert_array_equal(np.asarray(code(outside)), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(code(2)), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(code(1.0)), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(code(jnp.uint32(2**31 + 1))), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(code(1, jnp.float16)), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(code(1, jnp.int32)), [0.0, 1.0, 0.0])
