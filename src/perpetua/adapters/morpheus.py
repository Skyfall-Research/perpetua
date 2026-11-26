"""MORPHEUS persistent enterprise worlds: a live HTTP client and a deterministic mock.

A MORPHEUS world keeps running between decisions. Its chaos engine raises incident **tickets**
(each produced by a failure type such as ``rate_limit`` or ``missing_data``); the agent remediates
them, a verifier scores each fix, and operational work (orders, inventory, shipments) moves the
world's financial position. This adapter exposes that as a continuing host lifetime:

* **Regimes are declared drift.** The ``regime`` parameter indexes a tuple of :class:`Regime`
  configurations (arrival rate and failure-type mix). A switch pushes the new chaos
  configuration to the world (``PUT /world/{id}/chaos`` live; the mock's generator otherwise), so
  recovery tracking and the six-metric protocol see real, declared changes.
  ``pp.presets.morpheus_regimes`` runs the paper's six regimes once, ``period`` decisions each,
  then stays in the baseline revisit.
* **Observations** ``(1, 13)``: open tickets; open tickets by priority (low, medium, high,
  critical) and by status (new, open, in_progress, on_hold); the verifier pass fraction of this
  tick's resolve (0 on a tick without one); in-transit shipments; units processed since the
  lifetime began; elapsed simulated hours (recorded, not used for discounting). The regime
  itself is Oracle-only.
* **Actions** (``Discrete(5)``): no-op, triage (highest-priority untriaged ticket to
  ``in_progress``), resolve (highest-priority in-progress ticket, else the highest-priority open
  one, to ``resolved``, then verify it), restock inventory, deliver the oldest in-transit
  shipment. Actions without a target are no-ops.
* **Rewards** (``reward=``): ``"paper"`` (default) is the MORPHEUS paper's composite
  ``0.5*failure + 0.25*ledger + 0.25*throughput`` (Appendix A), applied per tick to the tick's
  movements because a lifetime has no episode end; ``"failure"``, ``"verification"`` (per-tick
  verifier progress) and ``"capital"`` (change in net balance) are alternatives, or pass
  ``callable(before, after) -> float``.

The live client speaks the deployment's REST API with ``httpx`` -- the endpoints, wire field
names and chaos payload were verified against a live deployment by an earlier Skyfall
evaluation harness, from which this surface is ported with permission. It is **not** validated
here without a deployment. The mock is deterministic for a seed and implements the same state
contract, so CI validates the lifetime, Oracle, regimes and learners without a server.
"""

# httpx is an optional extra; its response JSON is untyped by nature.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import ipaddress
import math
import os
import random
import time
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple, Protocol, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from perpetua.adapters.host import BackendResult, CPULifetime, CPUState, CPUWorld, HostParams
from perpetua.adapters.snapshots import BackendSnapshot
from perpetua.core import EnvInfo, JSONValue
from perpetua.drift import Drift

__all__ = [
    "FEATURES",
    "State",
    "RewardFunction",
    "severity",
    "Regime",
    "PAPER_REGIMES",
    "chaos_payload",
    "failure_reward",
    "ledger_reward",
    "throughput_reward",
    "paper_reward",
    "verification_reward",
    "capital_reward",
    "MorpheusClient",
    "MockMorpheusClient",
    "HttpMorpheusClient",
    "MorpheusParams",
    "MorpheusLifetime",
    "regime_labels",
]

DEFAULT_LAYOUT = "process-outbound"
ACTION_NAMES = ("noop", "triage_ticket", "resolve_ticket", "restock", "deliver_oldest")
PRIORITIES = ("low", "medium", "high", "critical")
OPEN_STATUSES = ("new", "open", "in_progress", "on_hold")
FEATURES = (
    "open_tickets",
    *(f"priority_{name}" for name in PRIORITIES),
    *(f"status_{name}" for name in OPEN_STATUSES),
    "verifier_pass_fraction",
    "in_transit_shipments",
    "units_processed",
    "elapsed_sim_hours",
)
State = dict[str, Any]
RewardFunction = Callable[[Mapping[str, Any], Mapping[str, Any]], float]

#: Failure-type severities (MORPHEUS paper Appendix A bands, as ported from the earlier harness).
SEVERITY_WEIGHTS: dict[str, float] = {
    "permission_denied": 1.0,
    "dependency_failure": 1.0,
    "data_corruption": 0.8,
    "missing_data": 0.7,
    "invalid_state": 0.6,
    "rate_limit": 0.4,
    "format_change": 0.35,
    "partial_data": 0.3,
    "stale_data": 0.2,
    "duplicate_data": 0.15,
    "timing_issue": 0.1,
}
DEFAULT_SEVERITY = 0.5
_PRIORITY_RANK = {name: rank for rank, name in enumerate(PRIORITIES)}


def severity(failure_type: str | None) -> float:
    """Severity of a failure type; unknown or missing types are mid-band (0.5)."""
    if not failure_type:
        return DEFAULT_SEVERITY
    return SEVERITY_WEIGHTS.get(failure_type.lower(), DEFAULT_SEVERITY)


@dataclass(frozen=True)
class Regime:
    """One chaos configuration: how often incidents arrive and which kinds.

    Only ``arrival_rate`` and ``chaos_types`` are pushed to a live deployment; the priority mix
    is used by the mock. ``origin`` names the configuration a revisit repeats.
    """

    regime_id: str
    arrival_rate: float
    chaos_types: Mapping[str, float]
    priorities: Mapping[str, float] = field(
        default_factory=lambda: {"low": 0.4, "medium": 0.4, "high": 0.15, "critical": 0.05}
    )
    origin: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.arrival_rate <= 1.0:
            raise ValueError(
                f"Regime {self.regime_id!r}: arrival_rate = {self.arrival_rate!r} "
                "must lie in [0, 1]."
            )
        unknown = set(self.chaos_types) - set(SEVERITY_WEIGHTS)
        if unknown:
            raise ValueError(
                f"Unknown MORPHEUS failure types {sorted(unknown)} in regime {self.regime_id!r}."
            )
        unknown = set(self.priorities) - set(PRIORITIES)
        if unknown:
            raise ValueError(
                f"Unknown MORPHEUS priorities {sorted(unknown)} in regime {self.regime_id!r}; "
                f"choose from {list(PRIORITIES)}."
            )
        self._check_mix("chaos_types", self.chaos_types)
        self._check_mix("priorities", self.priorities)

    def _check_mix(self, name: str, mix: Mapping[str, float]) -> None:
        for key, weight in mix.items():
            if not isinstance(weight, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"Regime {self.regime_id!r}: {name}[{key!r}] = {weight!r} must be a number."
                )
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"Regime {self.regime_id!r}: {name}[{key!r}] = {weight!r} "
                    "must be finite and non-negative."
                )
        if sum(mix.values()) <= 0:
            raise ValueError(
                f"Regime {self.regime_id!r}: {name} needs at least one positive weight, "
                f"got {dict(mix)!r}."
            )


_BASELINE = Regime(
    "baseline",
    0.2,
    {
        "missing_data": 0.25,
        "stale_data": 0.20,
        "format_change": 0.15,
        "rate_limit": 0.10,
        "dependency_failure": 0.10,
        "invalid_state": 0.10,
        "partial_data": 0.10,
    },
)
#: The six-regime schedule of the MORPHEUS evaluation suite (Skyfall-Research/morpheus-evals,
#: MIT), restricted to the axes a world or this mock acts on. The last regime revisits the first.
PAPER_REGIMES: tuple[Regime, ...] = (
    _BASELINE,
    Regime(
        "edi_spike",
        0.6,
        {
            "format_change": 0.35,
            "missing_data": 0.25,
            "duplicate_data": 0.15,
            "invalid_state": 0.15,
            "dependency_failure": 0.10,
        },
        {"low": 0.10, "medium": 0.30, "high": 0.40, "critical": 0.20},
    ),
    Regime(
        "infra_degradation",
        0.5,
        {
            "rate_limit": 0.30,
            "dependency_failure": 0.30,
            "timing_issue": 0.20,
            "partial_data": 0.10,
            "stale_data": 0.10,
        },
    ),
    Regime(
        "financial_pressure",
        0.35,
        {
            "missing_data": 0.2,
            "format_change": 0.2,
            "stale_data": 0.2,
            "invalid_state": 0.2,
            "partial_data": 0.2,
        },
        {"low": 0.05, "medium": 0.25, "high": 0.45, "critical": 0.25},
    ),
    Regime(
        "workforce_constrained",
        0.4,
        {
            "missing_data": 0.20,
            "rate_limit": 0.25,
            "dependency_failure": 0.20,
            "timing_issue": 0.20,
            "partial_data": 0.15,
        },
        {"low": 0.20, "medium": 0.35, "high": 0.30, "critical": 0.15},
    ),
    Regime(
        "baseline_revisit",
        _BASELINE.arrival_rate,
        _BASELINE.chaos_types,
        _BASELINE.priorities,
        origin="baseline",
    ),
)


def chaos_payload(regime: Regime, *, seed: int) -> dict[str, Any]:
    """A regime in the form ``PUT /world/{id}/chaos`` accepts.

    The arrival rate is sent twice on purpose: the running engine reads ``probability`` and the
    world record keeps ``chaosProbability`` (verified live by the earlier harness).
    """
    return {
        "processChaosEnabled": True,
        "infraChaosEnabled": True,
        "enabled": True,
        "probability": regime.arrival_rate,
        "chaosProbability": regime.arrival_rate,
        "scenarios": [
            {
                "type": name,
                "weight": float(weight),
                "description": f"{regime.regime_id}: {name}",
                "config": {},
            }
            for name, weight in regime.chaos_types.items()
        ],
        "seed": seed,
    }


# --- rewards --------------------------------------------------------------------------------


def _delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> float:
    return float(after["finance"][key]) - float(before["finance"][key])


def failure_reward(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """``-sum(severity)`` of the incidents raised this tick (paper ``r_f``)."""
    del before
    return -sum(severity(kind) for kind in after["new_failures"])


def ledger_reward(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """``clip(1 - d_actual / d_planned, -1, 1)`` over the tick; 0 with nothing planned."""
    planned = _delta(before, after, "planned_cost")
    if planned <= 0.0:
        return 0.0
    return float(np.clip(1.0 - _delta(before, after, "actual_cost") / planned, -1.0, 1.0))


def throughput_reward(
    before: Mapping[str, Any], after: Mapping[str, Any], *, capacity: float = 1.0
) -> float:
    """``clip(d_units / capacity, 0, 1)`` over the tick (paper ``r_p``, per tick)."""
    if capacity <= 0.0:
        return 0.0
    return float(np.clip(_delta(before, after, "units") / capacity, 0.0, 1.0))


def paper_reward(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """``0.5*r_f + 0.25*r_l + 0.25*r_p`` (paper Appendix A weights), per tick."""
    return (
        0.5 * failure_reward(before, after)
        + 0.25 * ledger_reward(before, after)
        + 0.25 * throughput_reward(before, after)
    )


def verification_reward(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """Non-negative improvement in the verifier pass fraction since the previous tick."""
    return max(0.0, float(after["pass_fraction"]) - float(before["pass_fraction"]))


def capital_reward(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """Change in the world's net balance."""
    return float(after["capital"]) - float(before["capital"])


REWARDS: dict[str, RewardFunction] = {
    "paper": paper_reward,
    "failure": failure_reward,
    "verification": verification_reward,
    "capital": capital_reward,
}


# --- clients --------------------------------------------------------------------------------


class MorpheusClient(Protocol):
    """The state contract shared by the mock and the live client.

    ``create``/``step`` return a snapshot with keys ``tickets`` (open tickets as dicts with
    ``id``, ``status``, ``priority``, ``chaos_type``, ``created``), ``shipments`` (in-transit
    ids, oldest first), ``warehouse_id``, ``product_id``, ``finance`` (``planned_cost``,
    ``actual_cost``, ``units``, ``revenue``, ``costs``), ``capital``, ``sim_time`` (hours),
    ``verifications`` (``(passed, total)`` checks from this tick's action), ``new_failures``
    (the failure types of incidents raised this tick, which the default ``"paper"`` reward
    reads) and ``invalid`` (whether the action was rejected).

    Reward callables receive ``(before, after)`` snapshots that also carry the derived
    ``pass_fraction`` (passed over total verification checks this tick, 0 with none) and
    ``units_since_start`` (units completed since ``create``).
    """

    def create(self, seed: int, regime: Regime) -> State: ...

    def set_regime(self, regime: Regime) -> None: ...

    def step(self, action: int, state: State) -> State: ...

    def close(self) -> None: ...


def _rank(ticket: Mapping[str, Any]) -> tuple[int, str]:
    return (-_PRIORITY_RANK.get(str(ticket["priority"]), 0), str(ticket["created"]))


def target_ticket(tickets: Sequence[Mapping[str, Any]], action: int) -> Mapping[str, Any] | None:
    """The ticket an action addresses: highest priority first, then oldest."""
    if action == 1:
        pool = [t for t in tickets if t["status"] in ("new", "open", "on_hold")]
    elif action == 2:
        pool = [t for t in tickets if t["status"] == "in_progress"] or list(tickets)
    else:
        return None
    return min(pool, key=_rank) if pool else None


def _sample(rng: random.Random, weights: Mapping[str, float]) -> str:
    names = sorted(weights)
    return rng.choices(names, weights=[float(weights[name]) for name in names])[0]


class MockMorpheusClient:
    """A seeded, ticket-centric world with one-hour decisions and the regime's chaos.

    Each simulated hour an incident arrives with the regime's ``arrival_rate``, with a failure
    type and priority drawn from its mixes. Open incidents overrun cost by their severity (the
    ledger signal); orders complete while inventory lasts (throughput); restocks and deliveries
    land two hours after the action. Resolving a ticket passes both verifier checks if it was
    triaged first, one otherwise.
    """

    def __init__(self) -> None:
        self.closed = False
        self.regime = _BASELINE
        self.pushed: list[Regime] = []

    def create(self, seed: int, regime: Regime) -> State:
        self.rng = random.Random(seed)
        self.regime = regime
        self.world: State = {
            "tickets": [],
            "next_ticket": 0,
            "inventory": 40.0,
            "shipments": ["shipment-0", "shipment-1"],
            "next_shipment": 2,
            "finance": {
                "planned_cost": 0.0,
                "actual_cost": 0.0,
                "units": 0.0,
                "revenue": 0.0,
                "costs": 0.0,
            },
            "sim_time": 0.0,
            "scheduled": [],
        }
        for _ in range(3):
            self._raise_ticket()
        return self._snapshot([], [], False)

    def set_regime(self, regime: Regime) -> None:
        self.regime = regime
        self.pushed.append(regime)

    def _raise_ticket(self) -> str:
        world = self.world
        kind = _sample(self.rng, self.regime.chaos_types)
        world["tickets"].append(
            {
                "id": f"ticket-{world['next_ticket']}",
                "status": "new",
                "priority": _sample(self.rng, self.regime.priorities),
                "chaos_type": kind,
                "created": f"{world['sim_time']:012.3f}-{world['next_ticket']:06d}",
                "triaged": False,
            }
        )
        world["next_ticket"] += 1
        return kind

    def _snapshot(
        self, new_failures: list[str], verifications: list[tuple[int, int]], invalid: bool
    ) -> State:
        world = self.world
        finance = dict(world["finance"])
        return {
            "tickets": [
                {key: ticket[key] for key in ("id", "status", "priority", "chaos_type", "created")}
                for ticket in world["tickets"]
            ],
            "shipments": list(world["shipments"]),
            "warehouse_id": "warehouse-0",
            "product_id": "sku-0",
            "finance": finance,
            "capital": finance["revenue"] - finance["costs"],
            "sim_time": world["sim_time"],
            "new_failures": new_failures,
            "verifications": verifications,
            "invalid": invalid,
        }

    def step(self, action: int, state: State) -> State:
        del state
        world, finance = self.world, self.world["finance"]
        verifications: list[tuple[int, int]] = []
        target = target_ticket(world["tickets"], action)
        later = world["sim_time"] + 2.0
        if action == 1 and target is not None:
            ticket = next(t for t in world["tickets"] if t["id"] == target["id"])
            ticket["status"], ticket["triaged"] = "in_progress", True
        elif action == 2 and target is not None:
            ticket = next(t for t in world["tickets"] if t["id"] == target["id"])
            world["tickets"].remove(ticket)
            verifications.append((2 if ticket["triaged"] else 1, 2))
        elif action == 3:
            world["scheduled"].append((later, "restock"))
            finance["planned_cost"] += 25.0
        elif action == 4 and world["shipments"]:
            world["shipments"].pop(0)
            world["scheduled"].append((later, "deliver"))
            finance["planned_cost"] += 20.0
        world["sim_time"] += 1.0
        due = [event for event in world["scheduled"] if event[0] <= world["sim_time"]]
        world["scheduled"] = [e for e in world["scheduled"] if e[0] > world["sim_time"]]
        for _, kind in due:
            if kind == "restock":
                world["inventory"] += 25.0
                finance["actual_cost"] += 25.0
                finance["costs"] += 25.0
            else:
                finance["units"] += 1.0
                finance["actual_cost"] += 20.0
                finance["costs"] += 20.0
        # Planned operating spend accrues every hour; open incidents overrun it by severity.
        finance["planned_cost"] += 10.0
        overrun = sum(4.0 * severity(t["chaos_type"]) for t in world["tickets"])
        finance["actual_cost"] += 10.0 + overrun
        finance["costs"] += 10.0 + overrun
        if world["inventory"] >= 5.0:
            world["inventory"] -= 5.0
            finance["units"] += 1.0
            finance["revenue"] += 50.0
        new_failures = (
            [self._raise_ticket()] if self.rng.random() < self.regime.arrival_rate else []
        )
        if world["sim_time"] % 3 == 0 and len(world["shipments"]) < 3:
            world["shipments"].append(f"shipment-{world['next_shipment']}")
            world["next_shipment"] += 1
        return self._snapshot(new_failures, verifications, False)

    def close(self) -> None:
        self.closed = True


def _records(payload: Any) -> list[Mapping[str, Any]]:
    """The list of records inside whatever envelope a response used."""
    for key in ("data", "items", "tickets"):
        if isinstance(payload, Mapping) and key in payload:
            payload = payload[key]
    if isinstance(payload, Mapping):
        payload = payload.get("items", [])
    return [r for r in payload if isinstance(r, Mapping)] if isinstance(payload, list) else []


def _number(record: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """``record[key]`` as a float; ``default`` when the key is absent or null.

    A value that is present but not a number raises: these numbers feed the
    reward, and reading a malformed one as the default would hide the error.
    """
    value = record.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"MORPHEUS field {key!r} is not a number: {value!r}") from None


_FAILURE_KEYS = ("chaos_type", "chaosType", "failure_type", "failureType")


def _failure_type(ticket: Mapping[str, Any]) -> str | None:
    for source in (ticket, ticket.get("metadata")):
        if isinstance(source, Mapping):
            for key in _FAILURE_KEYS:
                if source.get(key):
                    return str(source[key]).lower()
    return None


class HttpMorpheusClient:
    """A live MORPHEUS deployment over its REST API (``httpx``; the ``morpheus`` extra).

    The world id is a path segment on every call, so the client holds no per-world binding.
    ``client=`` accepts a preconfigured ``httpx.Client`` (for example with a mock transport).
    Wire field names are camelCase where the deployment expects them; a wrong key is ignored by
    the server, so request bodies are pinned by tests.

    ``headers`` (an access token, for example) are sent with every request. They are set on
    the client this class builds, so with ``client=`` put them on that client instead. Reach
    a remote service over ``https``: a credential and a plain ``http`` address that is not the
    local machine (``localhost``, a ``.localhost`` name or a loopback IP) warn once here,
    because every request would carry the credential unencrypted. What counts as a
    credential: a header whose name, compared without case, contains ``auth``, ``token``,
    ``key``, ``secret`` or ``cookie`` (``Authorization``, ``Proxy-Authorization``, ``Cookie``,
    ``X-Auth``, ``X-API-Key``, ``apikey`` and the like); ``auth=`` set on a preconfigured
    ``httpx`` client; and user-info in the address (``http://user:password@host``). The
    warning names the header or source, never its value. A credential under another header
    name, or one added per request by an event hook or a custom transport, is not seen.

    A ticket counts as a new failure the first time it is listed as open. To tell, the client
    remembers the id of every ticket open at the last read and of the last 10,000 that left
    the open list, so its memory is bounded by the open tickets plus 10,000 ids. A ticket
    listed as open again after 10,000 others have left the list since it did counts as new again.
    """

    _MAX_PAGES = 1000
    _MAX_CLOSED_SEEN: int = 10_000
    _CREDENTIAL_WORDS = ("auth", "token", "key", "secret", "cookie")
    _RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        base_url: str | None = None,
        *,
        layout: str = DEFAULT_LAYOUT,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        retries: int = 2,
        retry_backoff: float = 0.1,
        delete_on_close: bool = True,
        client: Any = None,
    ) -> None:
        import httpx

        if type(retries) is not int or retries < 0:
            raise ValueError("MORPHEUS retries must be a non-negative integer.")
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ValueError("MORPHEUS retry_backoff must be finite and non-negative.")
        self.base_url = base_url or os.environ.get("MORPHEUS_URL", "http://localhost:8282")
        self.layout = layout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.delete_on_close = delete_on_close
        self.http: Any = client or httpx.Client(
            base_url=self.base_url, timeout=timeout, headers=dict(headers or {})
        )
        self.world_id = ""
        self.seed = 0
        # Insertion-ordered: the ids open at the last read are the newest entries.
        self.seen: dict[str, None] = {}
        self._warn_if_credentials_travel_unencrypted()

    def _warn_if_credentials_travel_unencrypted(self) -> None:
        import httpx

        url = httpx.URL(str(getattr(self.http, "base_url", "")) or self.base_url)
        host = url.host
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost" or host.endswith(".localhost")
        names = {str(name).lower() for name in getattr(self.http, "headers", {})}
        credentials = sorted(
            name for name in names if any(word in name for word in self._CREDENTIAL_WORDS)
        )
        if getattr(self.http, "auth", None) is not None:
            credentials.append("the client's auth")
        if url.userinfo:
            credentials.append("the address's user-info")
        if url.scheme == "http" and not local and credentials:
            warnings.warn(
                f"MORPHEUS client sends {', '.join(credentials)} to {url.scheme}://{host} "
                "unencrypted on every request; use an https address for a remote service.",
                stacklevel=3,
            )

    def _remember(self, open_ids: Iterable[str]) -> None:
        """Mark ``open_ids`` as seen, keeping at most ``_MAX_CLOSED_SEEN`` ids beyond them."""
        count = 0
        for ticket_id in open_ids:
            self.seen.pop(ticket_id, None)
            self.seen[ticket_id] = None
            count += 1
        excess = len(self.seen) - count - self._MAX_CLOSED_SEEN
        for stale in list(islice(self.seen, max(excess, 0))):
            del self.seen[stale]

    # --- transport ------------------------------------------------------------------------

    def _request(self, method: str, url: str, *, what: str, **kwargs: Any) -> Any:
        import httpx

        # Only reads are retried. A write may already have taken effect even
        # when its reply was lost, and retrying it could resolve a ticket twice
        # or receive inventory twice.
        retryable = method == "GET"
        attempt = 0
        while True:
            try:
                response = self.http.request(method, url, **kwargs)
            except httpx.TransportError as error:
                if retryable and attempt < self.retries:
                    time.sleep(self.retry_backoff * (2**attempt))
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"MORPHEUS {what} failed: {error} (base URL {self.base_url})"
                ) from error
            except httpx.HTTPError as error:
                raise RuntimeError(
                    f"MORPHEUS {what} failed: {error} (base URL {self.base_url})"
                ) from error
            if (
                retryable
                and response.status_code in self._RETRY_STATUSES
                and attempt < self.retries
            ):
                response.close()
                time.sleep(self.retry_backoff * (2**attempt))
                attempt += 1
                continue
            break
        try:
            payload = response.json()
        except ValueError:
            if response.is_success and response.content.strip():
                # A proxy or login page answering 200 would otherwise read as an
                # empty, valid-looking world.
                raise RuntimeError(
                    f"MORPHEUS {what} returned a body that is not JSON "
                    f"({response.headers.get('content-type', 'no content type')})."
                ) from None
            payload = None
        if not response.is_success:
            detail = payload.get("error") if isinstance(payload, Mapping) else None
            raise RuntimeError(
                f"MORPHEUS {what} failed: {detail or f'HTTP {response.status_code}'}"
            )
        return payload

    def _data(self, payload: Any) -> Mapping[str, Any]:
        data = payload.get("data", payload) if isinstance(payload, Mapping) else None
        return data if isinstance(data, Mapping) else {}

    # --- lifecycle ------------------------------------------------------------------------

    def create(self, seed: int, regime: Regime) -> State:
        self.seed = seed
        payload = self._request(
            "POST",
            "/world",
            what="create_world",
            json={
                "name": f"perpetua-{uuid.uuid4().hex[:8]}",
                "layout": self.layout,
                "chaos": {"processChaosEnabled": True, "infraChaosEnabled": True},
            },
        )
        try:
            self.world_id = str(self._data(payload)["world"]["_id"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"MORPHEUS create_world returned no world id: {payload!r}"
            ) from error
        self.set_regime(regime)
        world = self._data(self._request("GET", f"/world/{self.world_id}", what="get_world"))
        self.real_hours_per_sim_day = _number(world, "realHoursPerSimDay", 2.0) or 2.0
        self.started = time.monotonic()
        state = self.snapshot([], False)
        self.seen = dict.fromkeys(str(ticket["id"]) for ticket in state["tickets"])
        return state

    def set_regime(self, regime: Regime) -> None:
        self._request(
            "PUT",
            f"/world/{self.world_id}/chaos",
            what="set_chaos",
            json=chaos_payload(regime, seed=self.seed),
        )

    def close(self) -> None:
        try:
            if self.delete_on_close and self.world_id:
                self._request("DELETE", f"/world/{self.world_id}", what="delete_world")
                self.world_id = ""
        finally:
            self.http.close()

    # --- observation ----------------------------------------------------------------------

    def _cursor_pages(self, path: str, **params: Any) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = []
        cursor: str | None = None
        for _ in range(self._MAX_PAGES):
            query = {**params, "limit": 100, **({"cursor": cursor} if cursor else {})}
            payload = self._request("GET", path, what=f"GET {path}", params=query)
            items.extend(_records(payload))
            data = self._data(payload)
            page = payload.get("pagination", data) if isinstance(payload, Mapping) else {}
            page = page if isinstance(page, Mapping) else {}
            cursor = page.get("nextCursor") if page.get("hasMore") else None
            if not cursor:
                return items
        raise RuntimeError(f"MORPHEUS listing {path} exceeded {self._MAX_PAGES} pages.")

    def _offset_pages(self, path: str) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = []
        for _ in range(self._MAX_PAGES):
            payload = self._request(
                "GET", path, what=f"GET {path}", params={"limit": 100, "offset": len(items)}
            )
            page = _records(payload)
            items.extend(page)
            pagination = self._data(payload).get("pagination")
            if not page or not (isinstance(pagination, Mapping) and pagination.get("hasMore")):
                return items
        raise RuntimeError(f"MORPHEUS listing {path} exceeded {self._MAX_PAGES} pages.")

    def _all_pages(
        self, path: str, *, page_size: int = 100, **params: Any
    ) -> list[Mapping[str, Any]]:
        """Every record of a list that pages by cursor or by offset, whichever the reply offers."""
        items: list[Mapping[str, Any]] = []
        cursor: str | None = None
        for _ in range(self._MAX_PAGES):
            query: dict[str, Any] = {**params, "limit": page_size}
            if cursor:
                query["cursor"] = cursor
            elif items:
                query["offset"] = len(items)
            payload = self._request("GET", path, what=f"GET {path}", params=query)
            page = _records(payload)
            items.extend(page)
            data = self._data(payload)
            pagination = payload.get("pagination") if isinstance(payload, Mapping) else None
            pagination = pagination if isinstance(pagination, Mapping) else data.get("pagination")
            pagination = pagination if isinstance(pagination, Mapping) else data
            cursor = pagination.get("nextCursor")
            if not page or not pagination.get("hasMore"):
                return items
        raise RuntimeError(f"MORPHEUS listing {path} exceeded {self._MAX_PAGES} pages.")

    def _finance(self) -> dict[str, float]:
        """Ledger and throughput proxies, reconstructed as the earlier harness's projector does."""
        world, revenue, units = self.world_id, 0.0, 0.0
        issued = uncollected = base_freight = overrun = freight = 0.0
        for order in self._all_pages(f"/{world}/erp/orders"):
            if order.get("status") == "COMPLETED":
                revenue += _number(order, "totalAmount")
                units += 1.0
        for invoice in self._all_pages(f"/{world}/erp/invoices", page_size=1000):
            amount = _number(invoice, "totalAmount")
            issued += amount
            if invoice.get("status") == "PAID":
                revenue += amount
            else:
                uncollected += amount
        for shipment in self._all_pages(f"/{world}/tms/shipments", status="DELIVERED"):
            costs = shipment.get("costs")
            costs = costs if isinstance(costs, Mapping) else {}
            total = _number(costs, "totalCost")
            base = _number(costs, "baseCost", total)
            base_freight += base
            overrun += max(0.0, total - base)
            freight += total
            units += 1.0
        return {
            "planned_cost": issued + base_freight,
            "actual_cost": uncollected + overrun,
            "units": units,
            "revenue": revenue,
            "costs": freight,
        }

    def snapshot(self, verifications: list[tuple[int, int]], invalid: bool) -> State:
        world = self.world_id
        tickets: dict[str, dict[str, Any]] = {}
        for status in OPEN_STATUSES:
            for record in self._cursor_pages(f"/{world}/tickets", status=status):
                ticket_id = str(record.get("_id") or record.get("ticket_id") or "")
                if ticket_id:
                    priority = str(record.get("priority", "low")).lower()
                    tickets.setdefault(
                        ticket_id,
                        {
                            "id": ticket_id,
                            "status": status,
                            "priority": priority if priority in PRIORITIES else "low",
                            "chaos_type": _failure_type(record),
                            "created": str(record.get("createdAt") or ""),
                        },
                    )
        shipments = self._cursor_pages(f"/{world}/tms/shipments/in-transit")
        shipments.sort(key=lambda shipment: str(shipment.get("createdAt") or ""))
        inventory = self._offset_pages(f"/{world}/wms/inventory")
        summary = self._data(self._request("GET", f"/{world}/finance/summary", what="finance"))
        new = [t for t in tickets.values() if t["id"] not in self.seen]
        self._remember(tickets)
        elapsed = (time.monotonic() - self.started) * 24.0 / self.real_hours_per_sim_day / 3600.0
        return {
            "tickets": sorted(tickets.values(), key=lambda ticket: ticket["created"]),
            # A shipment without an id cannot be acted on; never address it as "None".
            "shipments": [
                str(shipment_id)
                for s in shipments
                if (shipment_id := s.get("shipmentId") or s.get("_id"))
            ],
            "warehouse_id": str(inventory[0].get("warehouseId", "")) if inventory else "",
            "product_id": str(inventory[0].get("sku", "")) if inventory else "",
            "finance": self._finance(),
            "capital": _number(summary, "netBalance"),
            "sim_time": elapsed,
            "new_failures": [ticket["chaos_type"] for ticket in new],
            "verifications": verifications,
            "invalid": invalid,
        }

    # --- action ---------------------------------------------------------------------------

    def tool_call(self, action: int, state: State) -> tuple[str, str, dict[str, Any]] | None:
        """The ``(method, path, body)`` an action maps to, or ``None`` without a target."""
        world, target = self.world_id, target_ticket(state["tickets"], action)
        if action == 1 and target is not None:
            return "PUT", f"/{world}/tickets/{target['id']}/status", {"status": "in_progress"}
        if action == 2 and target is not None:
            return "PUT", f"/{world}/tickets/{target['id']}/status", {"status": "resolved"}
        if action == 3 and state["warehouse_id"] and state["product_id"]:
            body = {
                "warehouseId": state["warehouse_id"],
                "transactionType": "RECEIVE",
                "productId": state["product_id"],
                "quantity": 25,
            }
            return "POST", f"/{world}/wms/inventory-transactions", body
        if action == 4 and state["shipments"]:
            body = {"status": "DELIVERED", "source": "perpetua"}
            return "PUT", f"/{world}/tms/shipments/{state['shipments'][0]}/status", body
        return None

    def step(self, action: int, state: State) -> State:
        call = self.tool_call(action, state)
        verifications: list[tuple[int, int]] = []
        invalid = False
        if call is not None:
            method, path, body = call
            # A rejected call is an outcome (reported as invalid), not a transport failure.
            result = self._request(
                "POST",
                "/world/act",
                what="act",
                json={"path": path, "method": method, "body": body},
            )
            invalid = isinstance(result, Mapping) and result.get("success") is False
            if action == 2 and not invalid:
                ticket_id = path.split("/")[-2]
                verdict = self._data(
                    self._request(
                        "POST",
                        f"/{self.world_id}/verification/verify-ticket",
                        what="verify",
                        json={"ticketId": ticket_id},
                    )
                )
                total = max(int(_number(verdict, "totalChecks")), 0)
                passed = min(max(int(_number(verdict, "passedChecks")), 0), total)
                verifications.append((passed, total))
        return self.snapshot(verifications, invalid)


# --- the lifetime ---------------------------------------------------------------------------


class _MorpheusBackend:
    num_slots = 1
    uses_reset_seed = False
    checkpointable = False

    def __init__(
        self,
        client: MorpheusClient,
        *,
        regimes: tuple[Regime, ...],
        reward: str | RewardFunction,
        configuration: dict[str, JSONValue],
    ) -> None:
        if isinstance(reward, str) and reward not in REWARDS:
            raise ValueError(f"Unknown MORPHEUS reward {reward!r}; choose from {sorted(REWARDS)}.")
        self.client = client
        self.regimes = regimes
        self.regime = 0
        self.declared = 0
        self.reward = REWARDS[reward] if isinstance(reward, str) else reward
        self.state: State = {}
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(0.0, np.inf, (len(FEATURES),), dtype=np.float32)
        self.configuration = configuration

    def _derived(self, state: State) -> State:
        checks = state["verifications"]
        total = sum(t for _, t in checks)
        state["pass_fraction"] = sum(p for p, _ in checks) / total if total else 0.0
        state["units_since_start"] = float(state["finance"]["units"]) - self.origin_units
        return state

    def _observation(self, state: State) -> NDArray[np.float32]:
        tickets = state["tickets"]
        return np.asarray(
            [
                len(tickets),
                *(sum(t["priority"] == p for t in tickets) for p in PRIORITIES),
                *(sum(t["status"] == s for t in tickets) for s in OPEN_STATUSES),
                state["pass_fraction"],
                len(state["shipments"]),
                state["units_since_start"],
                state["sim_time"],
            ],
            dtype=np.float32,
        )[None]

    def init(self, seed: int) -> NDArray[np.float32]:
        state = self.client.create(seed, self.regimes[self.regime])
        self.origin_units = float(state["finance"]["units"])
        self.state = self._derived(state)
        return self._observation(self.state)

    def apply_regime(self, index: int) -> None:
        if index != self.regime:
            self.regime = index
            self.client.set_regime(self.regimes[index])

    def step(self, action: NDArray[Any], reset_seed: int) -> BackendResult:
        del reset_seed
        previous = self.state
        self.state = self._derived(self.client.step(int(action[0]), previous))
        return BackendResult(
            self._observation(self.state),
            np.asarray([self.reward(previous, self.state)], dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            np.asarray([False], dtype=np.bool_),
        )

    def snapshot(self) -> BackendSnapshot:
        raise ValueError("MORPHEUS world state is not snapshot-capable.")

    def restore(self, snapshot: BackendSnapshot) -> None:
        del snapshot
        raise ValueError("MORPHEUS world state is not snapshot-capable.")

    def close(self) -> None:
        self.client.close()


class MorpheusParams(NamedTuple):
    """Reward multiplier, float32 (1,), and the active regime index, float32 scalar."""

    reward_multiplier: jax.Array
    regime: jax.Array


class MorpheusWorld(CPUWorld):
    """A CPU world whose ``regime`` parameter reconfigures the MORPHEUS chaos engine."""

    default_drift_field = "regime"
    oracle_fields = ("chaos_active", "regime_index", "task_id", "invalid_action")

    @property
    def default_params(self) -> MorpheusParams:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MorpheusParams(jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray(0.0))

    def derive(self, params: MorpheusParams) -> MorpheusParams:  # pyright: ignore[reportIncompatibleMethodOverride]
        count = len(self._backend.regimes)
        return MorpheusParams(
            params.reward_multiplier, jnp.clip(jnp.round(params.regime), 0, count - 1)
        )

    @property
    def _backend(self) -> _MorpheusBackend:
        return cast(_MorpheusBackend, self.backend)

    def init(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: jax.Array, params: MorpheusParams
    ) -> tuple[CPUState, jax.Array, EnvInfo]:
        backend = self._backend
        previous = backend.regime, backend.declared
        # The backend creates the world in this regime, so it is set first and
        # put back if the init is refused.
        backend.regime = backend.declared = int(params.regime)
        try:
            return super().init(key, HostParams(params.reward_multiplier))
        except Exception:
            backend.regime, backend.declared = previous
            raise

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: CPUState, action: Any, key: jax.Array, params: MorpheusParams
    ) -> tuple[CPUState, jax.Array]:
        self._validate(state)
        self._actions(action)
        self._backend.apply_regime(int(params.regime))
        return super().step(state, action, key, HostParams(params.reward_multiplier))

    def after_drift(
        self, state: CPUState, previous: MorpheusParams, current: MorpheusParams, key: jax.Array
    ) -> CPUState:
        """Label the declared regime now; the world takes it before the next transition."""
        del previous, key
        self._backend.declared = int(current.regime)
        return state

    def observe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: CPUState, params: MorpheusParams
    ) -> tuple[jax.Array, EnvInfo]:
        return super().observe(state, HostParams(params.reward_multiplier))

    def privileged(self, state: CPUState) -> dict[str, jax.Array]:
        self._validate(state)
        backend = self._backend
        regime, first = backend.regimes[backend.declared], backend.regimes[0]
        return {
            "chaos_active": jnp.asarray(
                (regime.origin or regime.regime_id) != (first.origin or first.regime_id)
            ),
            "regime_index": jnp.asarray(backend.declared, dtype=jnp.int32),
            "task_id": jnp.asarray(backend.declared, dtype=jnp.int32),
            "invalid_action": jnp.asarray(bool(backend.state.get("invalid", False))),
        }


class MorpheusLifetime(CPULifetime):
    """One persistent MORPHEUS world; ``mock=True`` (default) needs no service.

    The live service needs ``mock=False``. ``base_url=``, ``headers=`` or a
    live ``client=`` together with ``mock=True`` raise ``ValueError`` instead
    of running the mock (a :class:`MockMorpheusClient` passed as ``client=``
    is the mock and is accepted). With ``mock=True`` and ``MORPHEUS_URL`` set
    in the environment, the mock runs and a warning says so.
    """

    world_class = MorpheusWorld

    def __init__(
        self,
        *,
        mock: bool = True,
        base_url: str | None = None,
        layout: str = DEFAULT_LAYOUT,
        headers: Mapping[str, str] | None = None,
        regimes: Sequence[Regime] = PAPER_REGIMES,
        reward: str | RewardFunction = "paper",
        client: MorpheusClient | None = None,
        drift: Drift | None = None,
    ) -> None:
        regimes = tuple(regimes)
        if not regimes:
            raise ValueError("MORPHEUS needs at least one regime.")
        if mock:
            live = [
                name
                for name, given in (
                    ("base_url", base_url is not None),
                    ("headers", headers is not None),
                    ("client", client is not None and not isinstance(client, MockMorpheusClient)),
                )
                if given
            ]
            if live:
                raise ValueError(
                    f"morpheus/Morpheus-v0 got {', '.join(f'{name}=' for name in live)} with "
                    "mock=True (the default), which would run the seeded mock and ignore the "
                    "live service; pass mock=False to use it."
                )
            if client is None and os.environ.get("MORPHEUS_URL"):
                warnings.warn(
                    "MORPHEUS_URL is set but morpheus/Morpheus-v0 runs the seeded mock "
                    "(mock=True is the default); pass mock=False to use the live service.",
                    stacklevel=2,
                )
        if client is None:
            client = (
                MockMorpheusClient()
                if mock
                else HttpMorpheusClient(base_url, layout=layout, headers=headers)
            )
        configuration: dict[str, JSONValue] = {
            "backend": "morpheus-mock"
            if mock and isinstance(client, MockMorpheusClient)
            else "morpheus-http",
            "layout": layout,
            "actions": list(ACTION_NAMES),
            "features": list(FEATURES),
            "regimes": [regime.regime_id for regime in regimes],
            "reward": reward if isinstance(reward, str) else getattr(reward, "__name__", "custom"),
            "clock": "elapsed simulated hours in observation; not used for discounting",
            "respawn": "never",
        }
        super().__init__(
            _MorpheusBackend(client, regimes=regimes, reward=reward, configuration=configuration),
            env_id="morpheus/Morpheus-v0",
            drift=drift,
        )


def regime_labels(
    regimes: Sequence[Regime], indices: Sequence[int]
) -> tuple[list[str], list[str | None]]:
    """Per-tick labels and revisit origins for :class:`perpetua.protocol.ProtocolTrace`."""
    labels = [regimes[index].regime_id for index in indices]
    origins = [regimes[index].origin for index in indices]
    return labels, origins
