# pp.metrics, pp.evaluation, pp.reporting

Streaming lifetime metrics live in the carry: mean, window, EMA, Welford
variance, same-state regret and recovery after declared switches.
`perpetua.evaluation` adds revisit evaluation and `forgetting` (re-scoring saved
policies on fresh lifetimes), `perpetua.protocol` the MORPHEUS six-metric
protocol on labelled traces (its forgetting is `within_trace_forgetting`, one
training trace's first minus second encounter), and `perpetua.reporting` IQM
and stratified bootstrap intervals. See [evaluation](../evaluation.md).
`pp.reporting.provenance()` names the code revision and software behind a
result; see [reproducibility](../introduction/reproducibility.md#versions).

```python
env = pp.make("ContinualGrid-v0")
runner = pp.runtime.Runner(env, pp.policies.random, ticks=256)
carry = runner(pp.runtime.init_lifetime(env, seed=0))
carry.metrics.mean, carry.metrics.window_mean, carry.metrics.reward_std
scores = jnp.asarray([[0.2, 0.4], [0.3, 0.5], [0.1, 0.6]])  # (seeds, tasks)
pp.reporting.report(scores, key=jax.random.key(0))["iqm"]
pp.reporting.provenance()["revision"]
```

```{eval-rst}
.. automodule:: perpetua.metrics
   :members:

.. automodule:: perpetua.evaluation
   :members:

.. automodule:: perpetua.protocol
   :members:

.. automodule:: perpetua.reporting
   :members:

.. automodule:: perpetua.measurement
   :members:
```
