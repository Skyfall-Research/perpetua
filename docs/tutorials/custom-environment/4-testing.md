# Tutorial: testing your environment

`pp.testing.check_env` is the counterpart of PettingZoo's `api_test`. Give it a
factory that builds a fresh environment; it runs every lifetime-contract check
and raises `ContractError` naming the first thing that breaks (see
[testing environments](../../introduction/testing.md) for the list).

The last step checks the world, registers it under an id, and trains a
continuing PPO learner in it while the berry moves and a fence opens and
closes:

```{literalinclude} ../../../examples/custom_environment/tutorial4_testing.py
:language: python
```

```text
check_env passed
after 10240 ticks: PPO 0.0761 vs random 0.0178 per tick
```

Across eight seeds at this budget, PPO's lifetime mean was 3–4× the random
policy's on the same seed (0.055–0.085 against 0.018–0.021 per tick).

In a package, put the same call in your test suite:

```text
def test_berry_field_honours_the_contract():
    pp.testing.check_env(lambda: make(size=5))
```
