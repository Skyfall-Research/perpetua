# Classic and MuJoCo control

Control tasks with drifting physics: gymnax's JAX control suite and MuJoCo models on MJX. Time limits are removed; terminations become respawns.

```bash
uv sync --locked --extra interop   # gymnax
uv sync --locked --extra mjx       # MJX
```

Drift addresses physical parameters by name (for example `gravity`).

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [CartPole](../gymnax-cartpole-v1.md) | `gymnax/CartPole-v1` | Discrete | `1` | `(1, 4)` | [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |
| [Pendulum](../gymnax-pendulum-v1.md) | `gymnax/Pendulum-v1` | Continuous | `1` | `(1, 3)` | [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) |
| [Acrobot](../gymnax-acrobot-v1.md) | `gymnax/Acrobot-v1` | Discrete | `1` | `(1, 6)` | [Gymnasium Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) |
| [Mountain Car](../gymnax-mountaincar-v0.md) | `gymnax/MountainCar-v0` | Discrete | `1` | `(1, 2)` | [Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) |
| [Mountain Car Continuous](../gymnax-mountaincarcontinuous-v0.md) | `gymnax/MountainCarContinuous-v0` | Continuous | `1` | `(1, 2)` | [Gymnasium Continuous Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) |
| [MJX Pendulum](../mjx-pendulum-v0.md) | `mjx/Pendulum-v0` | Continuous | `1` | `(1, 3)` | — |
| [MJX CartPole](../mjx-cartpole-v0.md) | `mjx/CartPole-v0` | Continuous | `1` | `(1, 5)` | — |

```{toctree}
:hidden:

../gymnax-cartpole-v1
../gymnax-pendulum-v1
../gymnax-acrobot-v1
../gymnax-mountaincar-v0
../gymnax-mountaincarcontinuous-v0
../mjx-pendulum-v0
../mjx-cartpole-v0
```
