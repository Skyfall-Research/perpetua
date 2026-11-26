# pp.testing

`check_env` runs every lifetime-contract check on fresh environments from a
factory and raises `ContractError` naming the first failure it detects; the
checks assume a `DriftEnv`-style state, and errors outside that shape surface
unchanged. See
[testing environments](../introduction/testing.md) for what each check asserts.

```python
import perpetua as pp

pp.testing.check_env(lambda: pp.make("ContinualGrid-v0", size=4))
```

`reward_action_sensitivity` is a diagnostic, not a contract check: the
fraction of ticks, along a random lifetime, on which the reward differs
between actions stepped from the same state. Where it is `0.0`,
[same-state regret](../evaluation.md#where-same-state-regret-reads-zero)
carries no information.

```python
pp.testing.reward_action_sensitivity(pp.make("ContinualGrid-v0"), ticks=200)
```

```{eval-rst}
.. automodule:: perpetua.testing
   :members:
```
