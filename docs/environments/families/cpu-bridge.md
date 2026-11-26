# CPU bridge

Batched C++ simulators through EnvPool (Linux only), stepped by the host or, with `compiled=True`, from inside the compiled runners.

```bash
uv sync --locked --extra cpu-bridge
```

The Agents column counts EnvPool's parallel copies of the simulator (`num_envs`): independent environments stepped together, not interacting agents. ALE is GPL-2.0 and is never vendored; it is installed only with EnvPool.

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [EnvPool CartPole](../envpool-cartpole-v1.md) | `envpool/CartPole-v1` | Discrete | `32` | `(32, 4)` | [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |
| [EnvPool Atari](../envpool-atari-v0.md) | `envpool/Atari-v0` | Discrete | `8` | `(8, 84, 84, 4)` | [the Arcade Learning Environment](https://ale.farama.org/environments/) |

```{toctree}
:hidden:

../envpool-cartpole-v1
../envpool-atari-v0
```
