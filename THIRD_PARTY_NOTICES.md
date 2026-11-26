# Third-party source notices

## PureJaxRL PPO adaptation

- Source: https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo.py
- Copyright 2023 Chris Lu.
- License: Apache License, Version 2.0.
- Adapted implementation: `src/perpetua/agents/ppo.py`.
- Full upstream license: `src/perpetua/agents/LICENSE-PureJaxRL`.
- Modification notice: `src/perpetua/agents/NOTICE`.

The full license and notice are included in the Python package. The adaptation
removes episodic reset/done handling and finite-budget annealing, and adds
Perpetua masks, lifetime/learner state, configuration, health and checkpointing.

Other runtime libraries are dependencies, not vendored implementations; their
respective upstream licenses remain applicable.

## Optional EnvPool snapshot binding

- Source: EnvPool 1.2.7, revision `9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3`.
- Copyright 2021–2026 Garena Online Private Limited; Apache-2.0.
- Integration: `native/envpool_snapshot/`, including the explicitly marked
  `core.patch` for counter access and worker completion tracking.
- The standalone build reuses upstream CartPole physics and core headers,
  downloading hash-pinned sources rather than vendoring the full catalogue.
- Its native wheel includes EnvPool/Abseil licenses, ThreadPool's zlib license,
  concurrentqueue BSD/Boost and semaphore zlib notices, and pybind11's BSD license.
- Build/support details: `native/envpool_snapshot/README.md`.

## Optional continual-benchmark dependencies

- MEAL 0.3.1 (`meal-bench`), MIT; used as an external package, not vendored.
- Farama Meta-World 3.1.1, MIT, with MuJoCo 3.3.0, Apache-2.0; used as external
  packages. CW10/CW20 task order follows the cited Continual World paper; no
  code from the unlicensed original Continual World repository is included.
- COOM (MIT) with ViZDoom 1.3.2 (MIT; ZDoom components carry their own
  licences) runs in a separate runtime that `scripts/setup_coom.py` builds from
  COOM commit `7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e`. No COOM code is
  included in Perpetua; `coom_worker.py` is original code that imports it.

- MORPHEUS: the simulator is a hosted service; no simulator code is included.
  The live client (`perpetua.adapters.morpheus.HttpMorpheusClient`), the chaos
  payload, the failure severities and reward components, and the six-metric
  protocol (`perpetua.protocol`) are ported, with permission, from an earlier
  Skyfall evaluation harness for MORPHEUS, which is not public. So is the
  recorded-answer fixture `tests/fixtures/morpheus_metrics_oracle.json`. The six-regime
  schedule (`PAPER_REGIMES`) follows `Skyfall-Research/morpheus-evals` (MIT,
  commit `34053520ac39cdc3e40304132ccc812324b9c11f`).
- AgarCL (MIT), commit `d256e7b6a61bebfd3a23014f0343c2f9195ac9d4`, and GLFW
  3.4, built by `scripts/setup_agarcl.py` in a separate runtime. No AgarCL or
  GLFW source is included in Perpetua; `agarcl_worker.py` is original code.
  `agarcl/grand-arena-v0` uses world settings published in the AgarCL paper
  (arXiv:2505.18347, Appendix G.1); no configuration file is copied.
- MuJoCo MJX 3.3.0 (`mujoco-mjx`), Apache-2.0; used as an external package.
  The pendulum and cart-pole MJCF models in `perpetua.adapters.mjx` are
  original.
- `BeanWorld-v0` is an original JAX implementation in the style of Jelly Bean
  World (Platanios, Saparov and Mitchell, ICLR 2020); no Jelly Bean World code
  is included.

Perpetua's original code is released under the MIT license (see `LICENSE`).
Every third-party component above keeps its own license.

## Upstream license audit

Checked on 2026-11-24 from each project's GitHub license file or, where GitHub
reports none, from its README (ViZDoom) or its PyPI license field (the `meal-bench`
package). Jelly Bean World (Apache-2.0) is only acknowledged: no code is used.

| Project | License | How Perpetua uses it |
|---|---|---|
| PureJaxRL | Apache-2.0 | PPO adapted in `agents/ppo.py` |
| EnvPool | Apache-2.0 | optional `cpu-bridge` extra; `native/envpool_snapshot` |
| gymnax | Apache-2.0 | optional `interop` extra |
| JaxMARL | Apache-2.0 | optional `jaxmarl` extra |
| Gymnasium | MIT | core dependency |
| PettingZoo, MPE2 | MIT | optional `marl` extra |
| Meta-World | MIT | optional `metaworld` extra |
| MuJoCo, MJX | Apache-2.0 | optional `mjx` extra |
| MEAL (`meal-bench`) | MIT | optional `meal` extra |
| distrax, TensorFlow Probability | Apache-2.0 | optional `meal` extra |
| COOM | MIT | separate runtime built by `scripts/setup_coom.py` |
| ViZDoom | MIT for its own code; ZDoom components vary | separate runtime, with COOM |
| AgarCL | MIT | separate runtime built by `scripts/setup_agarcl.py` |
| morpheus-evals | MIT | regime schedule followed, no code copied |
| JAX, flax, optax, orbax | Apache-2.0 | core and `training` dependencies |
| httpx | BSD-3-Clause | optional `morpheus` extra |
| PyTorch | BSD-style | optional `torch` extra |

Two GPL projects are not dependencies. MinAtar (GPL-3.0) and the Arcade Learning
Environment (GPL-2.0) are linked from the docs only: Perpetua does not import
or install either, and gymnax provides its own Apache-2.0 MinAtar-style
environments. EnvPool's Atari environments are built on ALE, so someone who
runs EnvPool Atari uses ALE under its own license.
