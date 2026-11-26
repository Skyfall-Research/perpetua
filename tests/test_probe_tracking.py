"""Health probes on the lifetime clock: bounded history, no effect on training."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import DQN, PPO, SAC, DQNConfig, DQNRunner, PPORunner, SACConfig, SACRunner


def same(left: object, right: object) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def ppo() -> Any:
    return PPORunner(pp.make("ContinualGrid-v0", size=3, num_agents=4), PPO(action_size=5))


def dqn() -> Any:
    config = DQNConfig(rollout_steps=16, buffer_capacity=256, batch_size=16, gradient_steps=2)
    return DQNRunner(pp.make("ContinualGrid-v0", size=4), DQN(action_size=5, config=config))


def sac() -> Any:
    env = pp.make("gymnax/Pendulum-v1")
    config = SACConfig(rollout_steps=16, buffer_capacity=512, batch_size=16, gradient_steps=2)
    agent = SAC(
        action_low=env.action_space.low[0], action_high=env.action_space.high[0], config=config
    )
    return SACRunner(env, agent)


@pytest.mark.parametrize("build", [ppo, dqn, sac])
def test_probes_follow_the_update_clock_without_changing_training(
    build: Callable[[], Any],
) -> None:
    plain = build()
    expected = plain.init(seed=2)
    for _ in range(5):
        expected = plain(expected)
    tracked = pp.probes.ProbedRunner(build(), every=2, window=2, memory=4)
    carry = tracked.init(seed=2)
    for _ in range(5):
        carry = tracked(carry)
    same(expected, carry.state)
    trace = carry.trace
    assert int(trace.updates) == 5
    assert int(trace.evaluations) == 2
    assert trace.obs.shape[0] == 4
    for name, value in trace.latest.items():
        assert np.isfinite(np.asarray(value)).all(), name
        # The ring holds both evaluations and the latest is the second.
        np.testing.assert_array_equal(np.asarray(trace.history[name][1]), np.asarray(value))
    assert any(name.startswith("dormant") for name in trace.latest)
    assert any(name.startswith("rank") for name in trace.latest)
    assert tracked.metadata["probes"] == {"every": 2, "window": 2, "memory": 4, "threshold": 1e-3}


def test_probed_carry_checkpoints_and_resumes(tmp_path: Path) -> None:
    tracked = pp.probes.ProbedRunner(ppo(), every=1, window=3, memory=2)
    carry = tracked(tracked.init(seed=0))
    pp.checkpoint.save(tmp_path / "probed", carry, metadata=tracked.metadata)
    restored = pp.checkpoint.restore(
        tmp_path / "probed", tracked.init(seed=9), expected_metadata=tracked.metadata
    )
    same(tracked(carry), tracked(restored))


def test_one_row_probe_batch_names_the_memory_to_raise() -> None:
    single: Any = PPORunner(pp.make("ContinualGrid-v0", size=3), PPO(action_size=5))
    with pytest.raises(ValueError, match=r"memory=1\).*single agent.*memory=2 or more"):
        pp.probes.ProbedRunner(single, memory=1).init(seed=0)
    assert pp.probes.ProbedRunner(single, memory=2).init(seed=0) is not None


def test_probed_runner_validates_its_settings() -> None:
    with pytest.raises(ValueError, match="every"):
        pp.probes.ProbedRunner(ppo(), every=0)
    runner = pp.runtime.Runner(pp.make("ContinualGrid-v0"), pp.policies.random)
    with pytest.raises(ValueError, match="probe"):
        pp.probes.ProbedRunner(cast(Any, runner))


def test_row_weights_give_the_statistics_of_the_weighted_rows() -> None:
    activations = jax.random.normal(jax.random.key(0), (12, 7)) * jnp.linspace(0.01, 1.0, 7)
    keep = np.asarray([True, False, True, True, False, False, True, True, False, True, True, False])
    weights = jnp.asarray(keep, dtype=jnp.float32)
    subset = activations[np.flatnonzero(keep)]
    assert float(
        pp.probes.dormant_unit_ratio(activations, threshold=0.01, weights=weights)
    ) == float(pp.probes.dormant_unit_ratio(subset, threshold=0.01))
    np.testing.assert_allclose(
        float(pp.probes.effective_rank(activations, weights=weights)),
        float(pp.probes.effective_rank(subset)),
        rtol=1e-5,
    )
    # Zero rows are what an inactive slot contributes; unweighted they change both.
    padded = jnp.where(weights[:, None] > 0, activations, 0.0)
    assert float(pp.probes.effective_rank(padded)) != pytest.approx(
        float(pp.probes.effective_rank(subset)), rel=1e-3
    )
    ones = jnp.ones(12, dtype=jnp.float32)
    for statistic in (pp.probes.dormant_unit_ratio, pp.probes.effective_rank):
        weighted, plain = statistic(activations, weights=ones), statistic(activations)
        if jax.default_backend() == "cpu":
            assert float(weighted) == float(plain)
        else:
            np.testing.assert_allclose(float(weighted), float(plain), rtol=1e-6)
    with pytest.raises(ValueError, match=r"row weights must have shape \(12,\)"):
        pp.probes.dormant_unit_ratio(activations, weights=jnp.ones(3))
    lone = jnp.zeros(12).at[4].set(1.0)
    assert float(pp.probes.dormant_unit_ratio(activations, weights=lone)) == 1.0


def test_probes_leave_out_the_placeholder_rows_of_inactive_slots() -> None:
    env = pp.make("PartnerGrid-v0", size=5, num_learners=1, num_partners=3)
    runner: Any = PPORunner(env, PPO(action_size=5))
    tracked = pp.probes.ProbedRunner(runner, memory=8)
    carry = tracked.init(seed=0)
    for _ in range(10):
        carry = tracked(carry)
    trace = carry.trace
    rows = np.asarray(trace.obs).reshape(32, -1)
    active = np.asarray(trace.active).reshape(-1)
    assert trace.active.shape == (8, 4)
    assert active.sum() == 8
    assert (rows[~active] == 0).all()
    on_active = runner.agent.probe(carry.state.agent, jnp.asarray(rows[active]))
    on_all = runner.agent.probe(carry.state.agent, jnp.asarray(rows))
    for name, value in trace.latest.items():
        np.testing.assert_allclose(float(value), float(on_active[name]), rtol=1e-5)
    # 24 of the 32 rows are zeros: over all rows the actor read as fully alive.
    assert float(on_all["dormant_actor"]) < float(trace.latest["dormant_actor"])


@pytest.mark.parametrize("build", [ppo, dqn, sac])
def test_probes_are_unchanged_when_every_slot_is_active(build: Callable[[], Any]) -> None:
    runner = build()
    tracked = pp.probes.ProbedRunner(runner, every=1, memory=6)
    carry = tracked.init(seed=1)
    for _ in range(7):
        carry = tracked(carry)
    trace = carry.trace
    assert bool(jnp.all(trace.active))
    # What the runner computed before rows carried weights: the probe on all rows.
    batch = trace.obs.reshape((-1, *trace.obs.shape[2:]))

    def probe(agent: Any, obs: jax.Array) -> dict[str, jax.Array]:
        return runner.agent.probe(agent, obs, threshold=1e-3)

    plain = cast(dict[str, jax.Array], jax.jit(probe)(carry.state.agent, batch))
    for name, value in trace.latest.items():
        if jax.default_backend() == "cpu":
            assert float(value) == float(plain[name]), name
        else:
            np.testing.assert_allclose(float(value), float(plain[name]), rtol=1e-4)


def test_fractional_weights_scale_out_and_bad_weights_give_nan() -> None:
    activations = jax.random.normal(jax.random.key(1), (8, 5)) * jnp.linspace(0.05, 1.0, 5)
    for scale in (0.1, 0.01):
        weights = jnp.full(8, scale, dtype=jnp.float32)
        np.testing.assert_allclose(
            float(pp.probes.effective_rank(activations, weights=weights)),
            float(pp.probes.effective_rank(activations)),
            rtol=1e-4,
        )
        assert float(
            pp.probes.dormant_unit_ratio(activations, threshold=0.05, weights=weights)
        ) == float(pp.probes.dormant_unit_ratio(activations, threshold=0.05))
    for bad in (-1.0, float("nan"), float("inf")):
        weights = jnp.ones(8).at[3].set(bad)
        assert np.isnan(float(pp.probes.effective_rank(activations, weights=weights)))
        assert np.isnan(float(pp.probes.dormant_unit_ratio(activations, weights=weights)))
