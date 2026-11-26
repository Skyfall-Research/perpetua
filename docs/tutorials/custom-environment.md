---
orphan: true
---

# A custom environment in one page

Environments describe parameters and dynamics; the framework owns drift. The
example below is the same shape as `examples/custom_env.py`.

```python
from typing import NamedTuple
import jax
import jax.numpy as jnp
import perpetua as pp
from perpetua.core import EnvInfo
from perpetua.params import DriftEnv


class LineParams(NamedTuple):
    goal: jax.Array


class LineState(NamedTuple):
    position: jax.Array
    respawn: jax.Array


class LineWorld:
    default_drift_field = "goal"
    num_agents = 1

    def __init__(self, length: int = 4) -> None:
        self.length = length
        from gymnasium import spaces
        import numpy as np

        self.observation_space = spaces.Box(0.0, float(length - 1), (1, 2), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.asarray([3], dtype=np.int64))
        self.metadata = {
            "env_id": "line-world-v0",
            "oracle_exposure": (),
            "configuration": {"length": length},
        }

    @property
    def default_params(self) -> LineParams:
        return LineParams(jnp.asarray(float(self.length - 1), dtype=jnp.float32))

    def derive(self, params: LineParams) -> LineParams:
        return LineParams(jnp.clip(jnp.rint(params.goal), 0, self.length - 1))

    def init(self, key, params):
        state = LineState(jnp.int32(0), jnp.asarray(False))
        obs, info = self.observe(state, params)
        return state, obs, info

    def step(self, state, action, key, params):
        del key
        move = jnp.clip(action[0], 0, 2) - 1
        position = jnp.clip(state.position + move, 0, self.length - 1)
        reached = position == params.goal.astype(jnp.int32)
        reward = jnp.asarray([jnp.where(reached, 1.0, 0.0)], dtype=jnp.float32)
        return LineState(jnp.where(reached, 0, position), jnp.asarray(reached)), reward

    def observe(self, state, params):
        obs = jnp.stack((state.position, params.goal.astype(jnp.int32)))[None, :].astype(
            jnp.float32
        )
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 3), dtype=jnp.bool_),
            "active_mask": jnp.ones((1,), dtype=jnp.bool_),
            "respawn": state.respawn[None],
        }
        return obs, info

    def privileged(self, state):
        return {}


env = DriftEnv(LineWorld(), pp.drift.Cyclic([3.0, 1.0], period=4))
state, obs, info = env.init(jax.random.key(0))
state, obs, reward, info = env.step(state, jnp.asarray([2]), jax.random.key(1))
assert reward.shape == (1,)
print("reward", float(reward[0]))
```

`derive` is applied by the framework; `init`, `step` and `observe` receive the
already-derived parameters and must not derive again.
