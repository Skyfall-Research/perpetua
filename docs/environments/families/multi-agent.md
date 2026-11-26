# Multi-agent worlds

Interacting worlds and multi-agent particle tasks. Every slot has its own reward and masks; partner and opponent drift change the other agents' policies, which a learner can only infer from their behaviour.

```bash
uv sync --locked --extra jaxmarl   # MPE tasks; the grids need nothing extra
```

Scripted partner and opponent slots are world participants with `active_mask=False`, so learners never train on them.

| Environment | Creation | Actions | Agents | Observation shape | Upstream |
|---|---|---|---|---|---|
| [Cooperative Grid](../cooperativegrid-v0.md) | `CooperativeGrid-v0` | Discrete | `2` | `(2, 7)` | — |
| [Partner Grid](../partnergrid-v0.md) | `PartnerGrid-v0` | Discrete | `2` (1 learner-controlled, 1 scripted) | `(2, 7)` | — |
| [Opponent Grid](../opponentgrid-v0.md) | `OpponentGrid-v0` | Discrete | `2` (1 learner-controlled, 1 scripted) | `(2, 7)` | — |
| [Simple Spread](../jaxmarl-mpe-simple-spread-v3.md) | `jaxmarl/MPE_simple_spread-v3` | Discrete | `3` | `(3, 18)` | [MPE2 Simple Spread](https://mpe2.farama.org/environments/simple_spread/) |
| [Simple](../jaxmarl-mpe-simple-v3.md) | `jaxmarl/MPE_simple_v3` | Discrete | `1` | `(1, 4)` | [MPE2 Simple](https://mpe2.farama.org/environments/simple/) |
| [Simple Tag](../jaxmarl-mpe-simple-tag-v3.md) | `jaxmarl/MPE_simple_tag_v3` | Discrete | `4` | `(4, 16)` | [MPE2 Simple Tag](https://mpe2.farama.org/environments/simple_tag/) |
| [Simple World Comm](../jaxmarl-mpe-simple-world-comm-v3.md) | `jaxmarl/MPE_simple_world_comm_v3` | Continuous | `6` | `(6, 34)` | [MPE2 Simple World Comm](https://mpe2.farama.org/environments/simple_world_comm/) |
| [Simple Speaker Listener](../jaxmarl-mpe-simple-speaker-listener-v4.md) | `jaxmarl/MPE_simple_speaker_listener_v4` | Discrete | `2` | `(2, 11)` | [MPE2 Simple Speaker Listener](https://mpe2.farama.org/environments/simple_speaker_listener/) |
| [Simple Push](../jaxmarl-mpe-simple-push-v3.md) | `jaxmarl/MPE_simple_push_v3` | Discrete | `2` | `(2, 19)` | [MPE2 Simple Push](https://mpe2.farama.org/environments/simple_push/) |
| [Simple Adversary](../jaxmarl-mpe-simple-adversary-v3.md) | `jaxmarl/MPE_simple_adversary_v3` | Discrete | `3` | `(3, 10)` | [MPE2 Simple Adversary](https://mpe2.farama.org/environments/simple_adversary/) |
| [Simple Reference](../jaxmarl-mpe-simple-reference-v3.md) | `jaxmarl/MPE_simple_reference_v3` | Discrete | `2` | `(2, 21)` | [MPE2 Simple Reference](https://mpe2.farama.org/environments/simple_reference/) |
| [Simple FACMAC](../jaxmarl-mpe-simple-facmac-v1.md) | `jaxmarl/MPE_simple_facmac_v1` | Continuous | `4` (3 learner-controlled, 1 scripted) | `(4, 16)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple FACMAC 3a](../jaxmarl-mpe-simple-facmac-3a-v1.md) | `jaxmarl/MPE_simple_facmac_3a_v1` | Continuous | `4` (3 learner-controlled, 1 scripted) | `(4, 16)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple FACMAC 6a](../jaxmarl-mpe-simple-facmac-6a-v1.md) | `jaxmarl/MPE_simple_facmac_6a_v1` | Continuous | `8` (6 learner-controlled, 2 scripted) | `(8, 28)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple FACMAC 9a](../jaxmarl-mpe-simple-facmac-9a-v1.md) | `jaxmarl/MPE_simple_facmac_9a_v1` | Continuous | `12` (9 learner-controlled, 3 scripted) | `(12, 40)` | [JaxMARL's MPE](https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe) |
| [Simple Crypto](../jaxmarl-mpe-simple-crypto-v3.md) | `jaxmarl/MPE_simple_crypto_v3` | Discrete | `3` | `(3, 8)` | [MPE2 Simple Crypto](https://mpe2.farama.org/environments/simple_crypto/) |

```{toctree}
:hidden:

../cooperativegrid-v0
../partnergrid-v0
../opponentgrid-v0
../jaxmarl-mpe-simple-spread-v3
../jaxmarl-mpe-simple-v3
../jaxmarl-mpe-simple-tag-v3
../jaxmarl-mpe-simple-world-comm-v3
../jaxmarl-mpe-simple-speaker-listener-v4
../jaxmarl-mpe-simple-push-v3
../jaxmarl-mpe-simple-adversary-v3
../jaxmarl-mpe-simple-reference-v3
../jaxmarl-mpe-simple-facmac-v1
../jaxmarl-mpe-simple-facmac-3a-v1
../jaxmarl-mpe-simple-facmac-6a-v1
../jaxmarl-mpe-simple-facmac-9a-v1
../jaxmarl-mpe-simple-crypto-v3
```
