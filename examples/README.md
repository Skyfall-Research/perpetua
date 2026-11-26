# Examples

Run any of these from the repository root, for example
`uv run --extra training python examples/train_ppo.py --updates 256`. Each
script describes itself with `--help`.

## Start here

| Script | What it shows |
|---|---|
| `continual_grid.py` | Batched continuing worlds stepped by scripted agents; no learner. |
| `train_ppo.py` | Continuing PPO on a small grid, with optional drift, checkpoint and resume. |
| `train_dqn.py` | The same for DQN and IQL. |
| `train_sac.py` | Continuing SAC on Pendulum. |
| `train_marl.py` | IPPO on interacting grids, JaxMARL MPE or a PettingZoo world. |
| `control_ppo.py` | One PPO learner on JAX, Gymnasium CPU or EnvPool CartPole. |
| `custom_env.py` | A minimal custom environment with hidden goal drift. |
| `custom_environment/` | The four steps of the custom-environment tutorial. |

The training scripts print progress lines to stderr and one JSON report to
stdout, so `python examples/train_ppo.py > report.json` keeps the report clean.

## Advanced

| Script | What it shows |
|---|---|
| `compute_ppo.py` | Compute-aware IPPO, where thinking time costs world time. |
| `compare_benchmarks.py` | One learner per action type across the continual benchmarks, scored with the same lifetime metrics. |

## Measurement harnesses

These produce the numbers on the measured-claims page (`docs/measurement.md`)
and are not tutorials: `measure_claims.py`, `measure_throughput.py` and
`measure_training_throughput.py`. `measure_switching.py` is a separate small
harness for switching throughput and pixel learning; the measured-claims page
does not use it.
