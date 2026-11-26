# Environments

53 registered environments in 6 families. Every one is a lifetime: `init` once, `step` forever, with respawns as events and drift owned by the framework.

New here? Start with [ContinualGrid-v0](continualgrid-v0.md): it needs no extras and runs in seconds on a CPU.

| Family | Environments | Install |
|---|---:|---|
| [Native continual worlds](families/native.md) | 4 | `uv sync --locked` |
| [Multi-agent worlds](families/multi-agent.md) | 16 | `uv sync --locked`<br>`uv sync --locked --extra jaxmarl` |
| [Classic and MuJoCo control](families/control.md) | 7 | `uv sync --locked --extra interop`<br>`uv sync --locked --extra mjx` |
| [MinAtar, bsuite and misc (gymnax)](families/gymnax.md) | 17 | `uv sync --locked --extra interop` |
| [Continual benchmarks](families/benchmarks.md) | 7 | differs per environment; see the family page |
| [CPU bridge](families/cpu-bridge.md) | 2 | `uv sync --locked --extra cpu-bridge  # Linux` |

## All environments

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [Continual Grid](continualgrid-v0.md) | `ContinualGrid-v0` | Discrete | `1` | `(1, 4)` | — |
| [Door Grid](doorgrid-v0.md) | `DoorGrid-v0` | Discrete | `1` | `(1, 4)` | — |
| [Forager Grid](foragergrid-v0.md) | `ForagerGrid-v0` | Discrete | `1` | `(1, 4)` | — |
| [Bean World](beanworld-v0.md) | `BeanWorld-v0` | Discrete | `1` | `(1, 150)` | — |
| [Cooperative Grid](cooperativegrid-v0.md) | `CooperativeGrid-v0` | Discrete | `2` | `(2, 7)` | — |
| [Partner Grid](partnergrid-v0.md) | `PartnerGrid-v0` | Discrete | `2` (1 learner-controlled, 1 scripted) | `(2, 7)` | — |
| [Opponent Grid](opponentgrid-v0.md) | `OpponentGrid-v0` | Discrete | `2` (1 learner-controlled, 1 scripted) | `(2, 7)` | — |
| [Simple Spread](jaxmarl-mpe-simple-spread-v3.md) | `jaxmarl/MPE_simple_spread-v3` | Discrete | `3` | `(3, 18)` | [MPE2 Simple Spread](https://mpe2.farama.org/environments/simple_spread/) |
| [Simple](jaxmarl-mpe-simple-v3.md) | `jaxmarl/MPE_simple_v3` | Discrete | `1` | `(1, 4)` | [MPE2 Simple](https://mpe2.farama.org/environments/simple/) |
| [Simple Tag](jaxmarl-mpe-simple-tag-v3.md) | `jaxmarl/MPE_simple_tag_v3` | Discrete | `4` | `(4, 16)` | [MPE2 Simple Tag](https://mpe2.farama.org/environments/simple_tag/) |
| [Simple World Comm](jaxmarl-mpe-simple-world-comm-v3.md) | `jaxmarl/MPE_simple_world_comm_v3` | Continuous | `6` | `(6, 34)` | [MPE2 Simple World Comm](https://mpe2.farama.org/environments/simple_world_comm/) |
| [Simple Speaker Listener](jaxmarl-mpe-simple-speaker-listener-v4.md) | `jaxmarl/MPE_simple_speaker_listener_v4` | Discrete | `2` | `(2, 11)` | [MPE2 Simple Speaker Listener](https://mpe2.farama.org/environments/simple_speaker_listener/) |
| [Simple Push](jaxmarl-mpe-simple-push-v3.md) | `jaxmarl/MPE_simple_push_v3` | Discrete | `2` | `(2, 19)` | [MPE2 Simple Push](https://mpe2.farama.org/environments/simple_push/) |
| [Simple Adversary](jaxmarl-mpe-simple-adversary-v3.md) | `jaxmarl/MPE_simple_adversary_v3` | Discrete | `3` | `(3, 10)` | [MPE2 Simple Adversary](https://mpe2.farama.org/environments/simple_adversary/) |
| [Simple Reference](jaxmarl-mpe-simple-reference-v3.md) | `jaxmarl/MPE_simple_reference_v3` | Discrete | `2` | `(2, 21)` | [MPE2 Simple Reference](https://mpe2.farama.org/environments/simple_reference/) |
| [Simple FACMAC](jaxmarl-mpe-simple-facmac-v1.md) | `jaxmarl/MPE_simple_facmac_v1` | Continuous | `4` (3 learner-controlled, 1 scripted) | `(4, 16)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple FACMAC 3a](jaxmarl-mpe-simple-facmac-3a-v1.md) | `jaxmarl/MPE_simple_facmac_3a_v1` | Continuous | `4` (3 learner-controlled, 1 scripted) | `(4, 16)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple FACMAC 6a](jaxmarl-mpe-simple-facmac-6a-v1.md) | `jaxmarl/MPE_simple_facmac_6a_v1` | Continuous | `8` (6 learner-controlled, 2 scripted) | `(8, 28)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple FACMAC 9a](jaxmarl-mpe-simple-facmac-9a-v1.md) | `jaxmarl/MPE_simple_facmac_9a_v1` | Continuous | `12` (9 learner-controlled, 3 scripted) | `(12, 40)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple Crypto](jaxmarl-mpe-simple-crypto-v3.md) | `jaxmarl/MPE_simple_crypto_v3` | Discrete | `3` | `(3, 8)` | [MPE2 Simple Crypto](https://mpe2.farama.org/environments/simple_crypto/) |
| [CartPole](gymnax-cartpole-v1.md) | `gymnax/CartPole-v1` | Discrete | `1` | `(1, 4)` | [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |
| [Pendulum](gymnax-pendulum-v1.md) | `gymnax/Pendulum-v1` | Continuous | `1` | `(1, 3)` | [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) |
| [Acrobot](gymnax-acrobot-v1.md) | `gymnax/Acrobot-v1` | Discrete | `1` | `(1, 6)` | [Gymnasium Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) |
| [Mountain Car](gymnax-mountaincar-v0.md) | `gymnax/MountainCar-v0` | Discrete | `1` | `(1, 2)` | [Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) |
| [Mountain Car Continuous](gymnax-mountaincarcontinuous-v0.md) | `gymnax/MountainCarContinuous-v0` | Continuous | `1` | `(1, 2)` | [Gymnasium Continuous Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) |
| [MJX Pendulum](mjx-pendulum-v0.md) | `mjx/Pendulum-v0` | Continuous | `1` | `(1, 3)` | — |
| [MJX CartPole](mjx-cartpole-v0.md) | `mjx/CartPole-v0` | Continuous | `1` | `(1, 5)` | — |
| [Asterix (MinAtar)](gymnax-asterix-minatar.md) | `gymnax/Asterix-MinAtar` | Discrete | `1` | `(1, 10, 10, 4)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [Breakout (MinAtar)](gymnax-breakout-minatar.md) | `gymnax/Breakout-MinAtar` | Discrete | `1` | `(1, 10, 10, 4)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [Freeway (MinAtar)](gymnax-freeway-minatar.md) | `gymnax/Freeway-MinAtar` | Discrete | `1` | `(1, 10, 10, 7)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [SpaceInvaders (MinAtar)](gymnax-spaceinvaders-minatar.md) | `gymnax/SpaceInvaders-MinAtar` | Discrete | `1` | `(1, 10, 10, 6)` | [MinAtar](https://github.com/kenjyoung/MinAtar) |
| [Catch (bsuite)](gymnax-catch-bsuite.md) | `gymnax/Catch-bsuite` | Discrete | `1` | `(1, 10, 5)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [DeepSea (bsuite)](gymnax-deepsea-bsuite.md) | `gymnax/DeepSea-bsuite` | Discrete | `1` | `(1, 8, 8)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [MemoryChain (bsuite)](gymnax-memorychain-bsuite.md) | `gymnax/MemoryChain-bsuite` | Discrete | `1` | `(1, 3)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [UmbrellaChain (bsuite)](gymnax-umbrellachain-bsuite.md) | `gymnax/UmbrellaChain-bsuite` | Discrete | `1` | `(1, 3)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [DiscountingChain (bsuite)](gymnax-discountingchain-bsuite.md) | `gymnax/DiscountingChain-bsuite` | Discrete | `1` | `(1, 2)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [MNISTBandit (bsuite)](gymnax-mnistbandit-bsuite.md) | `gymnax/MNISTBandit-bsuite` | Discrete | `1` | `(1, 28, 28)` | [bsuite](https://github.com/google-deepmind/bsuite) |
| [FourRooms (misc)](gymnax-fourrooms-misc.md) | `gymnax/FourRooms-misc` | Discrete | `1` | `(1, 4)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [MetaMaze (misc)](gymnax-metamaze-misc.md) | `gymnax/MetaMaze-misc` | Discrete | `1` | `(1, 15)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [BernoulliBandit (misc)](gymnax-bernoullibandit-misc.md) | `gymnax/BernoulliBandit-misc` | Discrete | `1` | `(1, 4)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [GaussianBandit (misc)](gymnax-gaussianbandit-misc.md) | `gymnax/GaussianBandit-misc` | Discrete | `1` | `(1, 4)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [Reacher (misc)](gymnax-reacher-misc.md) | `gymnax/Reacher-misc` | Continuous | `1` | `(1, 8)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [Swimmer (misc)](gymnax-swimmer-misc.md) | `gymnax/Swimmer-misc` | Continuous | `1` | `(1, 16)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [Pong (misc)](gymnax-pong-misc.md) | `gymnax/Pong-misc` | Discrete | `1` | `(1, 30, 40, 3)` | [gymnax](https://github.com/RobertTLange/gymnax) |
| [MEAL Overcooked](meal-overcooked-v0.md) | `meal/overcooked-v0` | Discrete | `2` | `(2, 9, 9, 26)` | [MEAL](https://github.com/TTomilin/MEAL) |
| [Meta-World MT1](metaworld-mt1-v0.md) | `metaworld/MT1-v0` | Continuous | `1` | `(1, 39)` | [Meta-World](https://metaworld.farama.org/benchmark/benchmark_descriptions/) |
| [COOM](coom-coom-v0.md) | `coom/COOM-v0` | Discrete | `1` | `(1, 84, 84, 12)` | [COOM](https://github.com/TTomilin/COOM) |
| [MORPHEUS](morpheus-morpheus-v0.md) | `morpheus/Morpheus-v0` | Discrete | `1` | `(1, 13)` | [MORPHEUS (paper)](https://openreview.net/forum?id=31P1VAfLkJ) |
| [AgarCL (screen)](agarcl-agario-screen-v0.md) | `agarcl/agario-screen-v0` | Continuous | `1` | `(1, 84, 84, 3)` | [AgarCL](https://github.com/AgarCL/AgarCL) |
| [AgarCL (grid)](agarcl-agario-grid-v0.md) | `agarcl/agario-grid-v0` | Continuous | `1` | `(1, 128, 128, 8)` | [AgarCL](https://github.com/AgarCL/AgarCL) |
| [AgarCL Grand Arena](agarcl-grand-arena-v0.md) | `agarcl/grand-arena-v0` | Continuous | `1` | `(1, 128, 128, 3)` | [AgarCL](https://github.com/AgarCL/AgarCL) |
| [EnvPool CartPole](envpool-cartpole-v1.md) | `envpool/CartPole-v1` | Discrete | `32` | `(32, 4)` | [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |
| [EnvPool Atari](envpool-atari-v0.md) | `envpool/Atari-v0` | Discrete | `8` | `(8, 84, 84, 4)` | [the Arcade Learning Environment](https://ale.farama.org/environments/) |
