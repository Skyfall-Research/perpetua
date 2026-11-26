"""A world's state keeps the same types from init through every step.

A weakly typed leaf on one side (common with backends that build state from
Python scalars) would make every compiled runner compile its chunk twice.
"""

import jax
import pytest
from helpers import unavailable

import perpetua as pp

NATIVE = [env_id for env_id in pp.environments(builtin=True) if not pp.spec(env_id).host]


@pytest.mark.parametrize("env_id", NATIVE)
def test_state_types_are_stable_across_steps(env_id: str) -> None:
    reason = unavailable(env_id)
    if reason is not None:
        pytest.skip(reason)
    env = pp.make(env_id)
    state, obs, info = env.init(jax.random.key(0))
    action = pp.policies.uniform(env.action_space)(obs, info, jax.random.key(1))
    stepped, *_ = env.step(state, action, jax.random.key(2))
    assert jax.tree.map(jax.typeof, state) == jax.tree.map(jax.typeof, stepped)


def test_a_chunked_runner_compiles_once_across_calls() -> None:
    env = pp.make("gymnax/Pendulum-v1")  # its backend's init state was weakly typed
    runner = pp.runtime.Runner(env, pp.policies.uniform(env.action_space), ticks=4)
    carry = runner(pp.runtime.init_lifetime(env, seed=0))
    with pp.measurement.count_compilations() as compiles:
        runner(carry)
    assert compiles.count == 0
