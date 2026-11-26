"""Compiled lifetimes must retain state, RNG and bounded streaming metrics."""

# JAX 0.11 transformation/device annotations are partial.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import perpetua as pp
from perpetua.envs.continual_grid import ContinualGrid, GridState
from perpetua.params import DriftEnvState


def assert_same_tree(left: object, right: object) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    left_leaves = cast(list[jax.Array], jax.tree.leaves(left))
    right_leaves = cast(list[jax.Array], jax.tree.leaves(right))
    for a, b in zip(left_leaves, right_leaves, strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_chunks_match_uninterrupted_and_eager_execution() -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 3))
    initial = pp.runtime.init_lifetime(env, seed=7, window_size=4)
    short = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    long = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    split = short(short(initial))
    complete = long(initial)
    assert_same_tree(split, complete)
    eager = initial
    for _ in range(8):
        eager = pp.runtime.tick(env, pp.policies.random, eager)
    assert_same_tree(eager, complete)
    assert complete.state.t.to_int() == 8
    assert complete.metrics.ticks.to_int() == 8


def test_shapes_and_single_trace_across_drift_phases() -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 2))
    traces: list[int] = []

    def policy(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        traces.append(1)
        return pp.policies.random(obs, info, key)

    runner = pp.runtime.Runner(env, policy, ticks=3)
    lifetime = pp.runtime.init_lifetime(env, window_size=4)
    shapes = [leaf.shape for leaf in jax.tree.leaves(lifetime)]
    for _ in range(4):
        lifetime = runner(lifetime)
        assert [leaf.shape for leaf in jax.tree.leaves(lifetime)] == shapes
    assert len(traces) == 1
    assert lifetime.state.t.to_int() == 12


def test_lifetimes_batch_without_sharing_rng() -> None:
    env = pp.make("ContinualGrid-v0")
    seeds = jnp.asarray([1, 2, 3], dtype=jnp.uint32)

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[GridState]]:
        return pp.runtime.init_lifetime(env, seed=seed)

    initial = jax.vmap(initialize)(seeds)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    result = jax.vmap(runner)(initial)
    assert result.obs.shape == (3, 1, 4)
    assert result.metrics.window.shape == (3, 128, 1)
    for index, seed in enumerate([1, 2, 3]):
        individual = runner(pp.runtime.init_lifetime(env, seed=seed))

        def select(leaf: jax.Array, index: int = index) -> jax.Array:
            return leaf[index]

        assert_same_tree(jax.tree.map(select, result), individual)


def test_streaming_metrics_match_reference() -> None:
    metrics = pp.metrics.init(num_agents=1, window_size=2, ema_decay=0.5)
    for value in [2.0, 4.0, 6.0]:
        metrics = pp.metrics.update(metrics, jnp.asarray([value]))
    np.testing.assert_allclose(metrics.mean, [4.0])
    np.testing.assert_allclose(metrics.window_mean, [5.0])
    np.testing.assert_allclose(metrics.ema, [4.5])
    np.testing.assert_allclose(metrics.welford_mean, [4.0])
    np.testing.assert_allclose(metrics.reward_variance, [4.0])  # sample variance
    np.testing.assert_allclose(metrics.reward_std, [2.0])
    assert metrics.ticks.to_int() == 3
    assert metrics.window.shape == (2, 1)
    assert bool(metrics.healthy)
    metrics = pp.metrics.update(metrics, jnp.asarray([jnp.nan]))
    assert not bool(metrics.healthy)


@pytest.mark.parametrize("log2_ticks", [22, 25])
def test_reward_variance_stays_accurate_past_float32_integer_range(log2_ticks: int) -> None:
    # Sample variance of a repeating 4-tick pattern has a closed form; the
    # old uncompensated sum was already 5% off at 2**22 ticks and froze at 2**25.
    ticks = 2**log2_ticks
    pattern = jnp.asarray([100.0, 100.5, 101.0, 101.5], dtype=jnp.float32)

    def body(carry: pp.metrics.Metrics, index: jax.Array) -> tuple[pp.metrics.Metrics, None]:
        return pp.metrics.update(carry, pattern[index % 4][None]), None

    @jax.jit
    def run(carry: pp.metrics.Metrics) -> pp.metrics.Metrics:
        return jax.lax.scan(body, carry, jnp.arange(ticks, dtype=jnp.int32))[0]

    metrics = cast(pp.metrics.Metrics, run(pp.metrics.init(num_agents=1, window_size=1)))
    expected = 0.3125 * ticks / (ticks - 1)
    np.testing.assert_allclose(metrics.reward_variance, [expected], rtol=1e-4)
    np.testing.assert_allclose(metrics.reward_std, [expected**0.5], rtol=1e-4)
    np.testing.assert_allclose(metrics.welford_mean, [100.75], rtol=1e-6)
    assert bool(metrics.healthy)


def test_reward_variance_is_never_negative_for_a_constant_reward() -> None:
    metrics = pp.metrics.init(num_agents=1, window_size=1)
    for _ in range(50):
        metrics = pp.metrics.update(metrics, jnp.asarray([0.1]))
    assert 0.0 <= float(metrics.reward_variance[0]) < 1e-12
    assert float(metrics.reward_std[0]) < 1e-6


def test_metric_count_and_environment_clock_both_cross_uint32() -> None:
    env = pp.make("ContinualGrid-v0")
    lifetime = pp.runtime.init_lifetime(env)
    clock = pp.Clock.from_int(2**32 - 1)
    lifetime = lifetime._replace(
        state=lifetime.state._replace(t=clock),
        metrics=lifetime.metrics._replace(ticks=clock),
    )
    lifetime = pp.runtime.Runner(env, pp.policies.random, ticks=2)(lifetime)
    assert lifetime.state.t.to_int() == 2**32 + 1
    assert lifetime.metrics.ticks.to_int() == 2**32 + 1


def test_scripted_policy_collects_reward() -> None:
    env = pp.make("ContinualGrid-v0", size=4)
    lifetime = pp.runtime.init_lifetime(env, seed=0)
    lifetime = pp.runtime.Runner(env, pp.policies.greedy, ticks=20)(lifetime)
    assert float(lifetime.metrics.mean[0]) > 0


def test_runtime_and_metric_configuration_validation() -> None:
    env = pp.make("ContinualGrid-v0")
    with pytest.raises(ValueError, match=r"ticks must be a positive integer\."):
        pp.runtime.Runner(env, pp.policies.random, ticks=0)
    with pytest.raises(ValueError, match=r"window_size must be a positive integer\."):
        pp.metrics.init(num_agents=1, window_size=0)
    with pytest.raises(ValueError, match=r"ema_decay must be finite and in"):
        pp.metrics.init(num_agents=1, ema_decay=1.0)


def test_seeds_outside_32_bits_are_refused_instead_of_aliased() -> None:
    env = pp.make("ContinualGrid-v0")
    first = pp.runtime.init_lifetime(env, seed=0)
    assert_same_tree(first, pp.runtime.init_lifetime(env, seed=cast(Any, np.int64(0))))
    for seed in (2**32, 2**40, -1):
        with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 2\*\*32\).*Got int"):
            pp.runtime.init_lifetime(env, seed=seed)
    with (
        jax.enable_x64(True),
        pytest.raises(ValueError, match=r"\[0, 2\*\*63\).*seed % 2\*\*63"),
    ):
        pp.runtime.init_lifetime(env, seed=2**63)
    for seed in (3.0, np.float64(3.0)):
        with pytest.raises(
            ValueError, match=r"seed must be an integer in \[0, 2\*\*32\).*Got float"
        ):
            pp.runtime.init_lifetime(env, seed=cast(Any, seed))
    with pytest.raises(TypeError, match=r"integer seed, not a PRNG key.*env\.init\(key\)"):
        pp.runtime.init_lifetime(env, seed=jax.random.key(0))  # pyright: ignore[reportArgumentType]

    def initialize(seed: jax.Array) -> Any:
        return pp.runtime.init_lifetime(env, seed=seed)

    batched = jax.vmap(initialize)(jnp.arange(2))
    assert batched.state.t.lo.shape == (2,)


def test_a_compiled_host_lift_refuses_a_regret_reference_when_built() -> None:
    env = pp.make("ContinualGrid-v0")
    bridged = {**env.metadata, "execution": "jax", "checkpointable": False, "forkable": False}

    class Bridged:
        metadata = bridged

    with pytest.raises(ValueError, match=r"Regret needs a shadow step.*compiled=True"):
        pp.runtime.check_reference(cast(Any, Bridged()), pp.policies.random)
    pp.runtime.check_reference(cast(Any, Bridged()), None)
    pp.runtime.check_reference(env, pp.policies.random)


def test_compensated_sum_retains_small_rewards_after_large_total() -> None:
    metrics = pp.metrics.init(num_agents=1, window_size=2)
    metrics = metrics._replace(
        total=jnp.asarray([2**24], dtype=jnp.float32), ticks=pp.Clock.from_int(2**24)
    )
    for _ in range(8):
        metrics = pp.metrics.update(metrics, jnp.asarray([1.0]))
    assert float(metrics.total[0]) == 2**24 + 8
    np.testing.assert_allclose(metrics.mean, [1.0])


def test_batched_metric_means_broadcast_over_agents_not_lifetimes() -> None:
    def initialize(_: jax.Array) -> pp.metrics.Metrics:
        return pp.metrics.init(num_agents=2, window_size=2)

    metrics = jax.vmap(initialize)(jnp.asarray([0, 1]))
    metrics = metrics._replace(
        total=jnp.asarray([[2.0, 4.0], [12.0, 16.0]]),
        ticks=pp.Clock(jnp.zeros(2, dtype=jnp.uint32), jnp.asarray([2, 4], dtype=jnp.uint32)),
        filled=jnp.asarray([1, 2]),
        window_total=jnp.asarray([[1.0, 2.0], [6.0, 8.0]]),
    )
    np.testing.assert_allclose(metrics.mean, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(metrics.window_mean, [[1.0, 2.0], [3.0, 4.0]])


def test_health_detects_accumulator_overflow_from_finite_rewards() -> None:
    metrics = pp.metrics.init(num_agents=1, window_size=2)
    reward = jnp.asarray([np.finfo(np.float32).max])
    metrics = pp.metrics.update(metrics, reward)
    assert bool(metrics.healthy)
    metrics = pp.metrics.update(metrics, reward)
    assert not bool(metrics.healthy)


def test_window_mean_recovers_once_a_large_reward_leaves_the_window() -> None:
    state = pp.metrics.init(num_agents=1, window_size=4)
    for reward in [1e8] + [1.0] * 12:
        state = pp.metrics.update(state, jnp.asarray([reward], dtype=jnp.float32))
    # A running add/subtract keeps the float32 rounding error of 1e8 forever (0.25 here).
    np.testing.assert_array_equal(np.asarray(state.window_mean), [1.0])
    np.testing.assert_array_equal(np.asarray(state.window_total), [4.0])


def test_reference_policy_and_after_drift_hook_get_different_keys() -> None:
    seen: dict[str, jax.Array] = {}

    class Hooked(ContinualGrid):
        def after_drift(
            self, state: GridState, previous: object, current: object, key: jax.Array
        ) -> GridState:
            seen["hook"] = jax.random.key_data(key)
            return state

    def reference(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        seen["reference"] = jax.random.key_data(key)
        return jnp.zeros((1,), dtype=jnp.int32)

    env = pp.params.DriftEnv(Hooked(size=3), pp.drift.Cyclic([(0, 0), (1, 1)], 2))
    carry = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    pp.runtime.tick(env, pp.policies.random, carry, reference=reference)
    assert not np.array_equal(np.asarray(seen["hook"]), np.asarray(seen["reference"]))


def test_framework_keys_are_not_children_of_a_plain_split() -> None:
    """No split of any width that a world makes can equal a key the framework derives."""
    seen: dict[str, jax.Array] = {}

    class Recording(ContinualGrid):
        def init(self, key: jax.Array, params: Any) -> Any:
            seen["init"] = key
            return super().init(key, params)

        def step(
            self, state: GridState, action: ArrayLike, key: jax.Array, params: Any
        ) -> tuple[GridState, jax.Array]:
            seen["step"] = key
            return super().step(state, action, key, params)

        def after_drift(
            self, state: GridState, previous: object, current: object, key: jax.Array
        ) -> GridState:
            seen["hook"] = key
            return state

    class RecordingDrift(pp.drift.Cyclic):
        def init(self, value: jax.Array, key: jax.Array) -> pp.drift.DriftState:
            seen["drift_init"] = key
            return super().init(value, key)

    def reference(obs: jax.Array, info: pp.Info, key: jax.Array) -> jax.Array:
        seen["reference"] = key
        return jnp.zeros((1,), dtype=jnp.int32)

    env = pp.params.DriftEnv(Recording(size=3), RecordingDrift([(0, 0), (1, 1)], 2))
    carry = pp.runtime.init_lifetime(env, seed=0, window_size=4)
    pp.runtime.tick(env, pp.policies.random, carry, reference=reference)

    def data(key: jax.Array) -> np.ndarray:
        return np.asarray(jax.random.key_data(key))

    lifetime = [data(carry.state.drift_key), data(seen["drift_init"])]
    step = [data(seen["hook"]), data(seen["reference"])]
    for width in (2, 3, 4, 8):
        for parent, derived in ((seen["init"], lifetime), (seen["step"], step)):
            for child in jax.random.split(parent, width):
                assert not any(np.array_equal(data(child), other) for other in derived)
    # The world still gets the lifetime key unchanged.
    np.testing.assert_array_equal(
        data(seen["init"]), data(jax.random.split(jax.random.key(0), 3)[0])
    )


def test_oracle_and_one_hot_wrappers_close_the_wrapped_environment() -> None:
    closed: list[bool] = []

    class Closable(ContinualGrid):
        def close(self) -> None:
            closed.append(True)

    inner = pp.params.DriftEnv(Closable(size=3), pp.drift.Cyclic([(0, 0), (1, 1)], 2))
    oracle = pp.Oracle(inner, expose=["task_id"])
    pp.OneHotObservation(oracle, field="task_id", size=2).close()
    assert closed == [True]


def test_environment_without_parameters_is_refused_clearly() -> None:
    class NoParams(ContinualGrid):
        @property
        def default_params(self) -> tuple[()]:  # pyright: ignore[reportIncompatibleMethodOverride]
            return ()

    with pytest.raises(ValueError, match="default_params has no parameter fields"):
        pp.params.DriftEnv(cast(Any, NoParams(size=3)))
