# MORPHEUS setup and the live service

MORPHEUS is a continual-RL benchmark built on a persistent enterprise
simulator, introduced in [MORPHEUS: A Persistent Enterprise Benchmark for Continual RL in the Big World](https://openreview.net/forum?id=31P1VAfLkJ); "the paper" on this page means
that one. The world keeps running between
decisions, and its chaos engine raises incident **tickets**, each produced by a
failure type such as `rate_limit` or `missing_data`. The agent remediates
tickets while operational work (orders, inventory, shipments) moves the
world's financial position, and a verifier scores each fix. Perpetua exposes
this as one continuing host lifetime, `morpheus/Morpheus-v0`, with a
deterministic mock and a live HTTP client behind the same state contract.

## Use the mock (default)

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("morpheus/Morpheus-v0")  # seeded mock, no service required
state, obs, info = env.init(jax.random.key(0))
state, obs, reward, info = env.step(state, jnp.asarray([1]), jax.random.key(1))
```

The mock advances one simulated hour per decision. Each hour it does the
following:
- **Incidents:** one arrives with the active regime's probability, with a
  failure type and priority drawn from the regime's mixes.
- **Costs:** open incidents overrun planned cost by their severity.
- **Orders:** orders complete while inventory lasts.
- **Delays:** restocks and deliveries land two hours after the action.

The mock is seeded by the lifetime's key, so equal seeds reproduce exactly.
It is what CI, conformance and the learner tests use.

## Regimes are declared drift

The `regime` parameter indexes a tuple of `Regime` configurations. Each is an
arrival rate plus failure-type and priority mixes. A switch is an ordinary
declared drift switch:
- it is pushed to the world before the next transition (`PUT
  /world/{id}/chaos` live; the mock's generator otherwise);
- recovery tracking sees it;
- the regime is exposed only through the Oracle (`regime_index`, `task_id`,
  `chaos_active`).

The Oracle labels follow the declared switch, like `task_id` everywhere else:
on the tick that declares a switch, `regime_index` and `task_id` already name
the new regime and `boundary` is true. The world itself takes the new regime
before the next transition, so the reward on the switch tick was earned under
the previous one. `chaos_active` is false for the first regime and for a
revisit of it, and true for every other regime.

`pp.presets.morpheus_regimes(period=500)` runs the six regimes of the
MORPHEUS evaluation suite
([Skyfall-Research/morpheus-evals](https://github.com/Skyfall-Research/morpheus-evals),
MIT):
`baseline`, `edi_spike`, `infra_degradation`, `financial_pressure`,
`workforce_constrained` and `baseline_revisit`, in that order, `period`
decisions each. It runs them once: after the five switches the world stays in
the baseline revisit and does not wrap into another baseline block. The revisit
is labelled as the baseline it repeats, so forgetting can compare the two
encounters.

```python
from perpetua.adapters.morpheus import PAPER_REGIMES

env = pp.presets.morpheus_regimes(period=500)
custom = pp.make(
    "morpheus/Morpheus-v0",
    regimes=PAPER_REGIMES[:2],
    drift={"regime": pp.drift.Cyclic([0.0, 1.0], period=200)},
)
```

## Observations, actions, rewards

**Observations** have shape `(1, 13)`:
- open tickets;
- open tickets by priority (low, medium, high, critical);
- open tickets by status (new, open, in_progress, on_hold);
- the verifier pass fraction of this tick's resolve (0 on a tick without
  one, so it does not carry over);
- in-transit shipments;
- units processed since the lifetime began;
- elapsed simulated hours.

The last value is recorded only; it is not used for discounting.

**Actions** (`Discrete(5)`):

| Index | Action |
|---|---|
| 0 | No-op (the world keeps running) |
| 1 | Triage: move the highest-priority untriaged ticket to `in_progress` |
| 2 | Resolve: the highest-priority in-progress ticket (else the highest-priority open one) to `resolved`, then verify it |
| 3 | Receive 25 units into inventory |
| 4 | Mark the oldest in-transit shipment delivered |

Actions without a target are no-ops. Targets are chosen by priority, then
age. The mock passes both verifier checks for a ticket triaged before it is
resolved, and one otherwise.

**Rewards** (`reward=`):
- **`"paper"` (default):** the paper's composite (Appendix A),
  `0.5·r_failure + 0.25·r_ledger + 0.25·r_throughput`:
  - `r_failure` is minus the summed severity of the incidents raised this tick,
    with severities in the paper's bands, e.g. 1.0 for `permission_denied` and
    0.1 for `timing_issue`;
  - `r_ledger = clip(1 − Δactual/Δplanned, −1, 1)`;
  - `r_throughput = clip(Δunits / 1, 0, 1)`.

  The paper scores ledger and throughput once at episode end. A lifetime has
  no episode end, so here they are computed per tick on that tick's movements.
  This is an adaptation, not the paper's exact schedule.
- **`"failure"`, `"verification"`, `"capital"`:** respectively, the failure
  term alone, the non-negative improvement in the verifier pass fraction since
  the previous tick, and the change in net balance.
- **A callable** `callable(before, after) -> float`.

The research code contains three reward stacks that disagree, and which one
produced the paper's figures is not confirmed. `"paper"` follows the
appendix's stated formula.

## The six-metric protocol

`pp.protocol` implements the paper's metrics on a labelled reward trace:
- per-configuration reward;
- adaptation speed;
- forgetting, as `within_trace_forgetting` (first minus second encounter within
  one trace; positive means worse);
- recovery time;
- stability;
- performance gap.

It is ported from an earlier Skyfall evaluation harness. The port reproduces all
140 answers that harness recorded from the reference evaluation code
(`tests/test_protocol.py`). `morpheus.regime_labels` turns Oracle regime
indices into labels, and `examples/compare_benchmarks.py` reports the protocol
for every sequence benchmark.

## Live deployment

```bash
uv sync --locked --extra morpheus   # httpx; no Git-URL dependency
export MORPHEUS_URL=https://your-instance.example
```

```python
env = pp.make("morpheus/Morpheus-v0", mock=False, headers={"Authorization": "..."})
```

`mock=False` is what selects the live service. `base_url=`, `headers=` or a
live `client=` without it raise a `ValueError` naming `mock=False`, rather
than running the mock and ignoring them. `MORPHEUS_URL` alone does not select
the live service either: with it set and `mock` left at its default, the mock
runs and a warning says so.

The service address is `base_url=` if given, otherwise `MORPHEUS_URL`, otherwise
`http://localhost:8282`. A connection failure names the address it tried.
`headers=` (your access token, for example) are sent with every request,
reads and writes alike, so reach a remote service over `https`. The plain
`http` default is meant for a service on your own machine: a credential
together with an `http` address that is not the local machine (`localhost`, a
`.localhost` name or a loopback IP) raises a warning when the client is built.
The check covers exactly these:

- a header whose name, compared without case, contains `auth`, `token`, `key`,
  `secret` or `cookie` (`Authorization`, `Proxy-Authorization`, `Cookie`,
  `X-Auth`, `X-API-Key`, `apikey` and the like), whether passed as `headers=`
  or set on a preconfigured client;
- `auth=` set on a preconfigured `httpx.Client`;
- user-info in the address (`http://user:password@host`).

The warning names the header or source, never its value. A credential under
any other header name, or one added per request by an event hook or a custom
transport, is not detected. The
other client options (`timeout`, `retries`, `retry_backoff`, `delete_on_close`)
belong to `HttpMorpheusClient`, which `pp.make` takes through `client=`:

```python
from perpetua.adapters.morpheus import HttpMorpheusClient

client = HttpMorpheusClient(
    "https://your-instance.example",
    headers={"Authorization": "..."},
    delete_on_close=False,  # keep the world after env.close()
)
env = pp.make("morpheus/Morpheus-v0", mock=False, client=client)
```

`client=` on `HttpMorpheusClient` itself takes a preconfigured `httpx.Client`;
`headers=` is then not applied, so set the headers on that client.

The live client speaks the deployment's REST API directly with `httpx`. That
means one client for any number of worlds and nothing that blocks a PyPI
release. The endpoints, camelCase field names and the chaos payload were
verified against a live deployment by that harness, and the
surface here is ported from it:
- **Worlds:** create, `GET`, delete.
- **Chaos and actions:** `PUT /world/{id}/chaos`, and `POST /world/act` for
  every action.
- **Reading state:**
  - tickets per open status, with cursor pagination;
  - in-transit shipments;
  - inventory;
  - the finance summary;
  - the order, invoice and delivered-shipment records behind the ledger and
    throughput proxies, read through every page on each tick.
- **Verifier:** `POST /{id}/verification/verify-ticket`.

A test drives a whole lifetime through a recorded fake deployment
(`httpx.MockTransport`) and pins every path and body. Another serves the
finance records over several pages, by cursor and by offset, and checks that
all of them are counted; the page size of a real deployment is not known. Only
`GET` requests are retried (twice, with exponential backoff) on transport errors
and HTTP 408, 425, 429, 500, 502, 503 or 504. Actions, world creation/deletion
and the chaos push are **not** retried, because a lost reply does not prove the
write did not happen. An action the service rejects with `success: false` is an invalid
action outcome for the Oracle, not a transport failure. Without a live instance
here, the client has **not** been run against a real world by Perpetua; a
live smoke test runs only when `MORPHEUS_URL` is set. Self-hosting means
duplicating the public Hugging Face Space and adding an `OPENAI_API_KEY`. The
world is deleted on close unless the client is built with
`delete_on_close=False`, as above.

## Limits

- **No checkpoints:** a live world is not reproducible or checkpointable,
  because its clock and state live on the service. The mock is reproducible
  but also not checkpointable, since it stands in for the service.
- **No resets or respawns:** construct a new lifetime for a new world.
- **A small action menu:** a fixed discrete menu suits Perpetua's numeric
  learners. LLM agents that emit free-form tool calls are better served by a
  tool-call action space.
- **Proxy finances:** the ledger and throughput figures are reconstructed from
  what the deployment records (invoices, freight, completed orders and
  deliveries). It records no first-class planned cost or capacity. Each tick
  reads all of those records, so the number of requests per tick grows with the
  world's history, one request per page.
- **Live costs:** the live service may need an OpenAI key and incurs its own
  costs.
- **New failures over a long lifetime:** a ticket is a new failure the first
  time it is listed as open. The live client remembers the open tickets and the
  last 10,000 that left the open list, so its memory stays bounded; a ticket
  that is listed as open again after 10,000 others have left counts as new
  again.
