# MinAtar, bsuite and misc (gymnax)

The rest of gymnax 0.0.9 through the generic `GymnaxParameters` adapter: MinAtar games, bsuite tasks and gymnax's miscellaneous environments.

```bash
uv sync --locked --extra interop
```

The adapter removes the backend time limit, turns terminations into respawns inside the lifetime, and accepts drift on any float parameter field that no other field is precomputed from (`drift_fields=` narrows the set).

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [Asterix (MinAtar)](../gymnax-asterix-minatar.md) | `gymnax/Asterix-MinAtar` | Discrete | `1` | `(1, 10, 10, 4)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [Breakout (MinAtar)](../gymnax-breakout-minatar.md) | `gymnax/Breakout-MinAtar` | Discrete | `1` | `(1, 10, 10, 4)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [Freeway (MinAtar)](../gymnax-freeway-minatar.md) | `gymnax/Freeway-MinAtar` | Discrete | `1` | `(1, 10, 10, 7)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [SpaceInvaders (MinAtar)](../gymnax-spaceinvaders-minatar.md) | `gymnax/SpaceInvaders-MinAtar` | Discrete | `1` | `(1, 10, 10, 6)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [Catch (bsuite)](../gymnax-catch-bsuite.md) | `gymnax/Catch-bsuite` | Discrete | `1` | `(1, 10, 5)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [DeepSea (bsuite)](../gymnax-deepsea-bsuite.md) | `gymnax/DeepSea-bsuite` | Discrete | `1` | `(1, 8, 8)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [MemoryChain (bsuite)](../gymnax-memorychain-bsuite.md) | `gymnax/MemoryChain-bsuite` | Discrete | `1` | `(1, 3)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [UmbrellaChain (bsuite)](../gymnax-umbrellachain-bsuite.md) | `gymnax/UmbrellaChain-bsuite` | Discrete | `1` | `(1, 3)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [DiscountingChain (bsuite)](../gymnax-discountingchain-bsuite.md) | `gymnax/DiscountingChain-bsuite` | Discrete | `1` | `(1, 2)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [MNISTBandit (bsuite)](../gymnax-mnistbandit-bsuite.md) | `gymnax/MNISTBandit-bsuite` | Discrete | `1` | `(1, 28, 28)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [FourRooms (misc)](../gymnax-fourrooms-misc.md) | `gymnax/FourRooms-misc` | Discrete | `1` | `(1, 4)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [MetaMaze (misc)](../gymnax-metamaze-misc.md) | `gymnax/MetaMaze-misc` | Discrete | `1` | `(1, 15)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [BernoulliBandit (misc)](../gymnax-bernoullibandit-misc.md) | `gymnax/BernoulliBandit-misc` | Discrete | `1` | `(1, 4)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [GaussianBandit (misc)](../gymnax-gaussianbandit-misc.md) | `gymnax/GaussianBandit-misc` | Discrete | `1` | `(1, 4)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [Reacher (misc)](../gymnax-reacher-misc.md) | `gymnax/Reacher-misc` | Continuous | `1` | `(1, 8)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [Swimmer (misc)](../gymnax-swimmer-misc.md) | `gymnax/Swimmer-misc` | Continuous | `1` | `(1, 16)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [Pong (misc)](../gymnax-pong-misc.md) | `gymnax/Pong-misc` | Discrete | `1` | `(1, 30, 40, 3)` | [gymnax](https://github.com/RobertTLange/gymnax) |

```{toctree}
:hidden:

../gymnax-asterix-minatar
../gymnax-breakout-minatar
../gymnax-freeway-minatar
../gymnax-spaceinvaders-minatar
../gymnax-catch-bsuite
../gymnax-deepsea-bsuite
../gymnax-memorychain-bsuite
../gymnax-umbrellachain-bsuite
../gymnax-discountingchain-bsuite
../gymnax-mnistbandit-bsuite
../gymnax-fourrooms-misc
../gymnax-metamaze-misc
../gymnax-bernoullibandit-misc
../gymnax-gaussianbandit-misc
../gymnax-reacher-misc
../gymnax-swimmer-misc
../gymnax-pong-misc
```
