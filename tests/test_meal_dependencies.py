"""MEAL's distribution dependency must import under the pinned JAX stack."""

# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false

import jax
import jax.numpy as jnp
import pytest


def test_meal_distrax_imports_and_samples() -> None:
    distrax = pytest.importorskip("distrax")
    tfp = pytest.importorskip("tensorflow_probability")
    distribution = distrax.Categorical(logits=jnp.zeros(3))
    assert int(distribution.sample(seed=jax.random.key(0))) in (0, 1, 2)
    from tensorflow_probability.substrates import jax as tfp_jax

    normal = tfp_jax.distributions.Normal(0.0, 1.0)
    assert bool(jnp.isfinite(normal.sample(seed=jax.random.key(1))))
    print(f"distrax {distrax.__version__}, tfp {tfp.__version__}")
