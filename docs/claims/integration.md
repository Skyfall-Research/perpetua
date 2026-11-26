# Claim 5: integration cost

**Hypothesis:** integration cost drops, measured as the glue code needed to run one random agent on every benchmark, against each benchmark's own example. Part of the [measured claims](../measurement.md), which has the instruments and the limits that apply to every number.

## Integration cost (claim 5)

**Verdict: supported.** One random agent runs on all five benchmarks in 5
lines against 51 upstream (5–17 per benchmark); per benchmark it is 4 lines
against 5–17. Lines of code are a weak proxy, and the adapter cost is
reported separately.

`uv run python scripts/measure_integration_cost.py --execute` clones each
pinned upstream commit to a temporary directory, extracts a documented
construct-and-step example, and counts executable lines (comments and
docstrings excluded). The upstream examples step with
`env.action_space.sample()`. The Perpetua side is therefore the like-for-like
snippet: construct, then step 1,000 ticks with uniformly random actions
(`pp.policies.uniform`). `--execute` runs the all-benchmarks snippet to prove
the counted code works. Results on 2026-11-23 (the adapter column recounted on 2026-11-25):

| Benchmark | Upstream example LOC | Perpetua LOC | Adapter/setup LOC |
|---|---:|---:|---:|
| MEAL | 5 (`README.md`) | 4 | 154 |
| Meta-World / Continual World | 8 (`README.md`) | 4 | 81 |
| COOM | 11 (`README.md`) | 4 | 489 |
| AgarCL | 17 (`README.md`) | 4 | 989 |
| MORPHEUS | 10 (`README.md`) | 4 | 1,026 |
| **One agent on all five** (the claim as worded) | **51** (sum; each needs its own API) | **5** | — |

As worded, the claim is supported: one 5-line loop runs a random agent on all
five benchmarks, against 51 lines of separate upstream examples. Per
benchmark, Perpetua is also shorter everywhere, but only by one line against
MEAL.

Limits:
- **Upstream examples** are not necessarily minimal, and they include loop and
  reset bookkeeping that Perpetua's continuing lifetimes do not need. That
  bookkeeping is part of the integration difference, but LOC remains a weak
  proxy.
- **Adapter/setup LOC** is the framework's own maintenance cost. It includes
  the shared worker protocol for COOM and AgarCL, and it is not counted as
  user glue. It is the executable lines (comments and docstrings excluded) of
  the files in the `ADAPTERS` list of `scripts/measure_integration_cost.py`
  and, for Continual World, the `MetaWorldLifetime` class; it grows with the
  adapters, so recount it with that script. This column was last recounted on
  2026-11-24.

**Learner by benchmark grid (claim 5).** PPO, IPPO, DQN, IQL and SAC on MEAL,
Continual World, COOM, Atari switching, AgarCL and MORPHEUS (mock), each
through its runner for 32 ticks across one task switch, on a Linux CPU. All
18 pairs whose action type fits ran with a finite state; the other 12 are
not applicable (SAC on discrete actions; the discrete learners on Continual
World and AgarCL). This shows that each pair runs, not that it learns.
