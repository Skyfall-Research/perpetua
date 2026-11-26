"""MORPHEUS: mock world, declared regimes, rewards, the HTTP client and host learning."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownLambdaType=false

import json
import os
from collections import Counter
from typing import Any, cast

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters import morpheus
from perpetua.adapters.morpheus import (
    FEATURES,
    PAPER_REGIMES,
    HttpMorpheusClient,
    MockMorpheusClient,
    MorpheusLifetime,
    Regime,
)


def step(env: Any, state: Any, action: int, tick: int) -> tuple[Any, jax.Array, jax.Array, Any]:
    return env.step(state, jnp.asarray([action], dtype=jnp.int32), jax.random.key(tick))


def feature(obs: jax.Array, name: str) -> float:
    return float(obs[0, FEATURES.index(name)])


def client_of(env: Any) -> Any:
    return env.env.backend.client


def test_mock_is_seeded_and_reproducible() -> None:
    first, second, other = (pp.make("morpheus/Morpheus-v0") for _ in range(3))
    try:
        a, obs_a, _ = first.init(jax.random.key(0))
        b, obs_b, _ = second.init(jax.random.key(0))
        other.init(jax.random.key(5))
        assert obs_a.shape == (1, len(FEATURES))
        np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
        rewards = []
        for tick in range(20):
            a, obs_a, reward_a, _ = step(first, a, tick % 5, tick)
            b, obs_b, reward_b, _ = step(second, b, tick % 5, tick)
            np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
            rewards.append((float(reward_a[0]), float(reward_b[0])))
        assert all(x == y for x, y in rewards)
        assert client_of(first).world["tickets"] != client_of(other).world["tickets"]
    finally:
        for env in (first, second, other):
            cast(Any, env).close()


def test_triage_then_resolve_passes_more_verifier_checks() -> None:
    env = pp.make("morpheus/Morpheus-v0", reward="verification")
    try:
        state, obs, _ = env.init(jax.random.key(0))
        open_before = feature(obs, "open_tickets")
        state, obs, _, _ = step(env, state, 1, 0)  # triage
        assert feature(obs, "status_in_progress") == 1.0
        state, obs, reward, _ = step(env, state, 2, 1)  # resolve the triaged ticket
        assert feature(obs, "verifier_pass_fraction") == 1.0
        assert float(reward[0]) == 1.0
        assert feature(obs, "open_tickets") <= open_before
        state, obs, reward, _ = step(env, state, 2, 2)  # resolve untriaged: 1 of 2 checks
        assert feature(obs, "verifier_pass_fraction") == 0.5
        assert float(reward[0]) == 0.0
    finally:
        cast(Any, env).close()


def test_restock_and_delivery_take_effect_two_hours_later() -> None:
    env = pp.make("morpheus/Morpheus-v0")
    try:
        state, obs, _ = env.init(jax.random.key(0))
        state, obs, _, _ = step(env, state, 4, 0)  # deliver: shipment leaves transit now
        assert feature(obs, "in_transit_shipments") == 1.0
        units = feature(obs, "units_processed")
        state, obs, _, _ = step(env, state, 0, 1)
        state, obs, _, _ = step(env, state, 0, 2)
        # Ordinary order completions add one unit per hour; the delivery adds one more.
        assert feature(obs, "units_processed") == units + 3.0
        assert feature(obs, "elapsed_sim_hours") == 3.0
    finally:
        cast(Any, env).close()


def test_paper_reward_components() -> None:
    before = {"finance": {"planned_cost": 0.0, "actual_cost": 0.0, "units": 0.0}}
    after = {
        "finance": {"planned_cost": 10.0, "actual_cost": 12.0, "units": 3.0},
        "new_failures": ["rate_limit", "permission_denied"],
    }
    assert morpheus.failure_reward(before, after) == pytest.approx(-1.4)
    assert morpheus.ledger_reward(before, after) == pytest.approx(-0.2)
    assert morpheus.throughput_reward(before, after) == 1.0
    assert morpheus.paper_reward(before, after) == pytest.approx(0.5 * -1.4 + 0.25 * -0.2 + 0.25)
    assert morpheus.severity("unknown_kind") == 0.5


def test_regimes_are_declared_switches_pushed_to_the_world() -> None:
    env = pp.make("morpheus/Morpheus-v0", drift={"regime": pp.drift.Cyclic([0.0, 1.0], period=4)})
    try:
        state, _, info = env.init(jax.random.key(0))
        assert not {"regime_index", "chaos_active"} & set(info)
        changed = []
        for tick in range(9):
            state, _, _, info = step(env, state, 0, tick)
            changed.append(bool(env.privileged(state)["drift_changed"]))
        assert changed == [False, False, False, True, False, False, False, True, False]
        pushed = [regime.regime_id for regime in client_of(env).pushed]
        assert pushed == ["edi_spike", "baseline"]
        privileged = env.privileged(state)
        assert int(privileged["regime_index"]) == 0
        assert not bool(privileged["chaos_active"])
        oracle = pp.Oracle(env, expose=["regime_index", "chaos_active"])
        _, exposed = oracle.observe(state)
        assert set(cast("dict[str, Any]", exposed.get("oracle"))) == {
            "regime_index",
            "chaos_active",
        }
    finally:
        cast(Any, env).close()


def test_regime_changes_the_mock_incident_mix() -> None:
    counts = {}
    for regime in (PAPER_REGIMES[0], PAPER_REGIMES[1]):
        client = MockMorpheusClient()
        client.create(3, regime)
        kinds = Counter()
        for _ in range(400):
            kinds.update(client.step(0, {})["new_failures"])
        counts[regime.regime_id] = kinds
    assert sum(counts["edi_spike"].values()) > 2 * sum(counts["baseline"].values())
    assert counts["edi_spike"]["format_change"] > counts["baseline"]["format_change"]
    assert set(counts["edi_spike"]) <= set(PAPER_REGIMES[1].chaos_types)


def test_regime_validation_and_paper_schedule() -> None:
    assert [r.regime_id for r in PAPER_REGIMES][-1] == "baseline_revisit"
    assert PAPER_REGIMES[-1].origin == "baseline"
    with pytest.raises(ValueError, match="Unknown MORPHEUS failure types"):
        Regime("bad", 0.1, {"meteor": 1.0})
    payload = morpheus.chaos_payload(PAPER_REGIMES[1], seed=7)
    assert payload["probability"] == payload["chaosProbability"] == 0.6
    assert {s["type"] for s in payload["scenarios"]} == set(PAPER_REGIMES[1].chaos_types)


def test_protocol_metrics_score_a_regime_schedule() -> None:
    env = pp.presets.morpheus_regimes(period=8)
    try:
        state, _, _ = env.init(jax.random.key(1))
        rewards, indices = [], []
        for tick in range(48):
            state, _, reward, _ = step(env, state, tick % 3, tick)
            rewards.append(float(reward[0]))
            privileged = env.privileged(state)
            indices.append(int(privileged["regime_index"]))
            # The generic Oracle name reports the same regime, not the schedule phase.
            assert int(privileged["task_id"]) == indices[-1]
        labels, origins = morpheus.regime_labels(PAPER_REGIMES, indices)
        trace = pp.protocol.ProtocolTrace.from_arrays(rewards, labels, origins=origins)
        result = pp.protocol.report(trace)
        assert [shift["to"] for shift in result["shifts"]] == [
            r.regime_id for r in PAPER_REGIMES[1:]
        ]
        # The revisit is scored as the baseline's second encounter.
        assert result["per_regime"]["baseline"]["within_trace_forgetting"] is not None
    finally:
        env.close()


def test_morpheus_trains_through_host_ppo() -> None:
    from perpetua.agents import PPO, HostPPORunner, PPOConfig

    env = pp.make("morpheus/Morpheus-v0")
    try:
        config = PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
        trainer = HostPPORunner(env, PPO(action_size=5, config=config))
        state = trainer(trainer.init(seed=0))
        assert bool(state.healthy)
        assert state.agent.gradient_steps.to_int() == 1
    finally:
        cast(Any, env).close()


class FakeDeployment:
    """An in-memory deployment answering the client's HTTP calls, recording each one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.finance_failures = 0
        self.act_failure: int | None = None
        self.reject_act = False
        self.act_calls = 0
        self.orders: list[dict[str, Any]] | None = None
        self.invoices: list[dict[str, Any]] | None = None
        self.delivered: list[dict[str, Any]] | None = None
        self.tickets = {
            "t-high": {
                "_id": "t-high",
                "status": "new",
                "priority": "high",
                "createdAt": "2026-01-01T02:00:00Z",
                "metadata": {"chaosType": "RATE_LIMIT"},
            },
            "t-low": {
                "_id": "t-low",
                "status": "new",
                "priority": "low",
                "createdAt": "2026-01-01T01:00:00Z",
                "chaos_type": "stale_data",
            },
        }

    @staticmethod
    def page(
        records: list[dict[str, Any]], params: dict[str, str], cursor: bool = False
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """One page of at most 100 records, whatever limit was asked for, and where it ends."""
        start = int(params.get("cursor" if cursor else "offset", 0))
        end = start + min(int(params.get("limit", 100)), 100)
        return records[start:end], end, end < len(records)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body: Any = json.loads(request.content) if request.content else None
        params = dict(request.url.params)
        path, method = request.url.path, request.method
        self.calls.append((method, path, params, body))
        ok = {"success": True, "status": 200}
        if method == "POST" and path == "/world":
            return httpx.Response(201, json={"world": {"_id": "w1"}, "mainCompany": {}})
        if method == "GET" and path == "/world/w1":
            return httpx.Response(200, json={**ok, "data": {"realHoursPerSimDay": 2.0}})
        if path == "/w1/tickets":
            matching = [t for t in self.tickets.values() if t["status"] == params["status"]]
            # Two pages: the first ticket, then the rest, through the cursor.
            if "cursor" not in params and len(matching) > 1:
                page = {"items": matching[:1], "hasMore": True, "nextCursor": "page-2"}
            else:
                page = {"items": matching[1:] if "cursor" in params else matching, "hasMore": False}
            return httpx.Response(200, json={**ok, "data": page})
        if path == "/w1/tms/shipments/in-transit":
            shipments = [{"shipmentId": "S1", "createdAt": "2026-01-01T00:00:00Z"}]
            return httpx.Response(
                200, json={**ok, "data": shipments, "pagination": {"hasMore": False}}
            )
        if path == "/w1/wms/inventory":
            items = [{"warehouseId": "WH-1", "sku": "SKU-1", "quantityAvailable": 4}]
            return httpx.Response(
                200, json={**ok, "data": {"items": items, "pagination": {"hasMore": False}}}
            )
        if path == "/w1/finance/summary":
            if self.finance_failures:
                self.finance_failures -= 1
                return httpx.Response(503, json={"success": False, "error": "busy"})
            return httpx.Response(200, json={**ok, "data": {"netBalance": 120.0}})
        if path == "/w1/erp/orders" and self.orders is not None:
            # Offset paging with the pagination block beside the records.
            records, _, more = self.page(self.orders, params)
            return httpx.Response(
                200, json={**ok, "data": records, "pagination": {"hasMore": more}}
            )
        if path == "/w1/erp/invoices" and self.invoices is not None:
            # Cursor paging, the cursor being the next offset.
            records, end, more = self.page(self.invoices, params, cursor=True)
            page = {"items": records, "hasMore": more, "nextCursor": str(end)}
            return httpx.Response(200, json={**ok, "data": page})
        if path == "/w1/tms/shipments" and self.delivered is not None:
            assert params["status"] == "DELIVERED"
            records, _, more = self.page(self.delivered, params)
            page = {"items": records, "pagination": {"hasMore": more}}
            return httpx.Response(200, json={**ok, "data": page})
        if path == "/w1/erp/orders":
            return httpx.Response(
                200, json={**ok, "data": [{"status": "COMPLETED", "totalAmount": 9}]}
            )
        if path in ("/w1/erp/invoices", "/w1/tms/shipments"):
            return httpx.Response(200, json={**ok, "data": []})
        if method == "POST" and path == "/world/act":
            self.act_calls += 1
            if self.act_failure is not None:
                return httpx.Response(
                    self.act_failure, json={"success": False, "error": "write failed"}
                )
            if self.reject_act:
                return httpx.Response(200, json={"success": False, "error": "rejected"})
            ticket = body["path"].split("/")[3] if "/tickets/" in body["path"] else None
            if ticket in self.tickets:
                self.tickets[ticket]["status"] = body["body"]["status"]
            return httpx.Response(200, json={**ok, "data": {}})
        if path == "/w1/verification/verify-ticket":
            return httpx.Response(
                200, json={**ok, "data": {"passed": True, "totalChecks": 3, "passedChecks": 2}}
            )
        if method in ("PUT", "DELETE") and path in ("/world/w1/chaos", "/world/w1"):
            return httpx.Response(200, json=ok)
        return httpx.Response(404, json={"success": False, "error": f"no route {method} {path}"})


def test_http_client_speaks_the_verified_wire_format() -> None:
    deployment = FakeDeployment()
    transport = httpx.MockTransport(deployment.handle)
    client = HttpMorpheusClient(
        "http://morpheus.test",
        client=httpx.Client(base_url="http://morpheus.test", transport=transport),
    )
    env = pp.make(
        "morpheus/Morpheus-v0",
        mock=False,
        client=client,
        drift={"regime": pp.drift.Cyclic([0.0, 2.0], period=2)},
    )
    try:
        assert cast(Any, env.metadata)["configuration"]["backend"]["backend"] == "morpheus-http"
        state, obs, _ = env.init(jax.random.key(0))
        create = next(call for call in deployment.calls if call[:2] == ("POST", "/world"))
        assert create[3]["layout"] == "process-outbound"
        assert create[3]["chaos"] == {"processChaosEnabled": True, "infraChaosEnabled": True}
        assert feature(obs, "open_tickets") == 2.0
        assert feature(obs, "priority_high") == 1.0
        assert feature(obs, "in_transit_shipments") == 1.0
        ticket_pages = [
            c for c in deployment.calls if c[1] == "/w1/tickets" and c[2]["status"] == "new"
        ]
        assert [page[2].get("cursor") for page in ticket_pages] == [None, "page-2"]

        state, obs, _, _ = step(env, state, 1, 0)  # triage the high-priority ticket
        act = [c for c in deployment.calls if c[1] == "/world/act"][-1][3]
        assert act == {
            "path": "/w1/tickets/t-high/status",
            "method": "PUT",
            "body": {"status": "in_progress"},
        }
        state, obs, _, _ = step(env, state, 2, 1)  # resolve it, then verify
        verify = [c for c in deployment.calls if c[1] == "/w1/verification/verify-ticket"][-1]
        assert verify[3] == {"ticketId": "t-high"}
        assert feature(obs, "verifier_pass_fraction") == pytest.approx(2 / 3)
        step(env, state, 3, 2)  # restock with camelCase inventory fields
        restock = [c for c in deployment.calls if c[1] == "/world/act"][-1][3]
        # The switch flagged at t=2 is pushed before the next transition runs under it.
        chaos = [c[3] for c in deployment.calls if c[:2] == ("PUT", "/world/w1/chaos")]
        assert [payload["probability"] for payload in chaos] == [0.2, 0.5]
        assert all(payload["probability"] == payload["chaosProbability"] for payload in chaos)
        assert restock["body"] == {
            "warehouseId": "WH-1",
            "transactionType": "RECEIVE",
            "productId": "SKU-1",
            "quantity": 25,
        }
    finally:
        cast(Any, env).close()
    assert ("DELETE", "/world/w1") in [call[:2] for call in deployment.calls]


def test_http_errors_name_the_failing_call() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json={"error": "boom"}))
    client = HttpMorpheusClient(
        "http://morpheus.test",
        client=httpx.Client(base_url="http://morpheus.test", transport=transport),
    )
    with pytest.raises(RuntimeError, match="create_world failed: boom"):
        client.create(0, PAPER_REGIMES[0])


def test_transient_reads_are_retried_but_writes_are_not() -> None:
    deployment = FakeDeployment()
    transport = httpx.MockTransport(deployment.handle)
    client = HttpMorpheusClient(
        "http://morpheus.test",
        retry_backoff=0.0,
        client=httpx.Client(base_url="http://morpheus.test", transport=transport),
    )
    try:
        deployment.finance_failures = 2
        client.create(0, PAPER_REGIMES[0])
        summary_calls = [call for call in deployment.calls if call[1] == "/w1/finance/summary"]
        assert len(summary_calls) == 3
        assert deployment.finance_failures == 0

        deployment.act_failure = 503
        with pytest.raises(RuntimeError, match="MORPHEUS act failed: write failed"):
            client.step(1, client.snapshot([], False))
        assert deployment.act_calls == 1  # a write is attempted once, even on a retryable status
    finally:
        client.close()


def test_rejected_action_is_an_outcome_not_a_transport_error() -> None:
    deployment = FakeDeployment()
    transport = httpx.MockTransport(deployment.handle)
    client = HttpMorpheusClient(
        "http://morpheus.test",
        client=httpx.Client(base_url="http://morpheus.test", transport=transport),
    )
    env = pp.make("morpheus/Morpheus-v0", mock=False, client=client)
    try:
        deployment.reject_act = True
        state, _, _ = env.init(jax.random.key(0))
        state, _, _, _ = step(env, state, 1, 0)
        assert bool(env.privileged(state)["invalid_action"])
        assert deployment.act_calls == 1
    finally:
        cast(Any, env).close()


@pytest.mark.skipif(not os.environ.get("MORPHEUS_URL"), reason="No live MORPHEUS instance.")
def test_live_morpheus_smoke() -> None:
    env = pp.make("morpheus/Morpheus-v0", mock=False)
    try:
        state, obs, _ = env.init(jax.random.key(0))
        state, obs, reward, info = step(env, state, 0, 0)
        assert obs.shape == (1, len(FEATURES))
        assert reward.shape == (1,)
        assert info["t"].to_int() == 1
    finally:
        cast(Any, env).close()


def test_regime_preset_holds_the_baseline_revisit_after_its_horizon() -> None:
    # Over its horizon the preset matches a plain cycle of the six regimes;
    # afterwards it stays in the baseline revisit instead of wrapping into
    # another baseline block, with no further declared switches.
    from perpetua.adapters.morpheus import MorpheusLifetime

    def trace(env: Any, ticks: int) -> tuple[list[int], int]:
        state, _, _ = env.init(jax.random.key(0))
        indices, switches = [], 0
        for tick in range(ticks):
            state, _, _, _ = step(env, state, tick % 3, tick)
            indices.append(int(env.privileged(state)["regime_index"]))
            switches += int(bool(env.privileged(state)["drift_changed"]))
        return indices, switches

    period, horizon = 4, 24
    preset = pp.presets.morpheus_regimes(period=period)
    cycled = MorpheusLifetime(
        regimes=PAPER_REGIMES,
        drift=cast(Any, {"regime": pp.drift.Cyclic([float(i) for i in range(6)], period)}),
    )
    try:
        held, held_switches = trace(preset, horizon + 20)
        # The plain cycle wraps on the horizon's last tick; the preset holds instead.
        cycle, _ = trace(cycled, horizon - 1)
    finally:
        preset.close()
        cycled.close()
    assert held[: horizon - 1] == cycle
    assert held[horizon - period - 1 :] == [5] * (len(held) - horizon + period + 1)
    assert held_switches == 5
    assert preset.metadata["configuration"]["preset"]["horizon_ticks"] == horizon


def test_stationary_can_hold_a_given_value() -> None:
    held = pp.drift.Stationary(np.asarray([2.0, 3.0]))
    state = held.init(jnp.zeros(2), jax.random.key(0))
    np.testing.assert_array_equal(np.asarray(state.value), [2.0, 3.0])
    assert held.configuration == {"type": "stationary", "value": [2.0, 3.0]}
    assert pp.drift.Stationary().configuration == {"type": "stationary"}
    with pytest.raises(ValueError, match="shape"):
        held.init(jnp.zeros(3), jax.random.key(0))


def _client_with(override: Any) -> tuple[HttpMorpheusClient, FakeDeployment]:
    """A client on the fake deployment, with ``override(request)`` answering first."""
    deployment = FakeDeployment()

    def handle(request: httpx.Request) -> httpx.Response:
        response = override(request)
        return response if response is not None else deployment.handle(request)

    client = HttpMorpheusClient(
        "http://morpheus.test",
        client=httpx.Client(base_url="http://morpheus.test", transport=httpx.MockTransport(handle)),
    )
    return client, deployment


def _env(client: HttpMorpheusClient) -> Any:
    return pp.make("morpheus/Morpheus-v0", mock=False, client=client)


def test_a_successful_reply_that_is_not_json_is_an_error() -> None:
    def proxy_page(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/w1/finance/summary":
            return httpx.Response(
                200, text="<html>login</html>", headers={"content-type": "text/html"}
            )
        return None

    client, _ = _client_with(proxy_page)
    env = _env(client)
    try:
        with pytest.raises(
            RuntimeError, match=r"finance returned a body that is not JSON \(text/html\)"
        ):
            env.init(jax.random.key(0))
    finally:
        env.close()


@pytest.mark.parametrize(("balance", "error"), [("n/a", True), (None, False)])
def test_a_malformed_number_is_an_error_and_a_null_one_is_the_default(
    balance: Any, error: bool
) -> None:
    def summary(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/w1/finance/summary":
            return httpx.Response(200, json={"success": True, "data": {"netBalance": balance}})
        return None

    client, _ = _client_with(summary)
    env = _env(client)
    try:
        if error:
            with pytest.raises(RuntimeError, match=r"field 'netBalance' is not a number: 'n/a'"):
                env.init(jax.random.key(0))
        else:
            env.init(jax.random.key(0))
    finally:
        env.close()


def test_a_shipment_without_an_id_is_never_addressed_as_none() -> None:
    def shipments(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/w1/tms/shipments/in-transit":
            records = [
                {"createdAt": "2026-01-01T00:00:00Z"},  # no shipmentId and no _id
                {"shipmentId": "S1", "createdAt": "2026-01-01T01:00:00Z"},
            ]
            return httpx.Response(
                200, json={"success": True, "data": records, "pagination": {"hasMore": False}}
            )
        return None

    client, deployment = _client_with(shipments)
    env = _env(client)
    try:
        state, obs, _ = env.init(jax.random.key(0))
        assert feature(obs, "in_transit_shipments") == 1.0
        step(env, state, 4, 0)  # deliver the oldest shipment
        act = [c for c in deployment.calls if c[1] == "/world/act"][-1][3]
        assert act["path"] == "/w1/tms/shipments/S1/status"
    finally:
        env.close()


def test_finance_reads_follow_every_page_of_orders_invoices_and_shipments() -> None:
    client, deployment = _client_with(lambda request: None)
    deployment.orders = [
        {"status": "COMPLETED" if i < 230 else "PENDING", "totalAmount": 2} for i in range(250)
    ]
    deployment.invoices = [
        {"status": "PAID" if i < 120 else "OPEN", "totalAmount": 3} for i in range(150)
    ]
    deployment.delivered = [{"costs": {"totalCost": 5, "baseCost": 4}} for _ in range(130)]
    try:
        finance = client.create(0, PAPER_REGIMES[0])["finance"]
        assert finance == {
            "units": 230 + 130,
            "revenue": 230 * 2 + 120 * 3,
            "costs": 130 * 5,
            "planned_cost": 150 * 3 + 130 * 4,
            "actual_cost": 30 * 3 + 130,
        }

        def asked(path: str, name: str) -> list[str | None]:
            return [c[2].get(name) for c in deployment.calls if c[1] == path]

        assert asked("/w1/erp/orders", "offset") == [None, "100", "200"]
        assert asked("/w1/erp/invoices", "cursor") == [None, "100"]
        assert asked("/w1/tms/shipments", "offset") == [None, "100"]
    finally:
        client.close()


def test_a_connection_failure_names_the_base_url() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    client = HttpMorpheusClient(
        "http://morpheus.test",
        retries=0,
        client=httpx.Client(base_url="http://morpheus.test", transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(
        RuntimeError, match=r"create_world failed: .*\(base URL http://morpheus.test\)"
    ):
        client.create(0, PAPER_REGIMES[0])


def test_the_default_base_url_is_localhost_without_morpheus_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MORPHEUS_URL", raising=False)
    client = HttpMorpheusClient()
    try:
        assert client.base_url == "http://localhost:8282"
    finally:
        client.http.close()


@pytest.mark.parametrize(
    ("chaos_types", "priorities", "message"),
    [
        (
            {"missing_data": 0.0},
            None,
            r"Regime 'r': chaos_types needs at least one positive weight, "
            r"got \{'missing_data': 0.0\}",
        ),
        (
            {"missing_data": float("nan")},
            None,
            r"Regime 'r': chaos_types\['missing_data'\] = nan must be finite",
        ),
        (
            {"missing_data": float("inf")},
            None,
            r"Regime 'r': chaos_types\['missing_data'\] = inf must be finite",
        ),
        (
            {"missing_data": 1.0, "stale_data": -0.5},
            None,
            r"Regime 'r': chaos_types\['stale_data'\] = -0.5 must be finite and non-negative",
        ),
        (
            {"missing_data": 1.0},
            {"urgent": 1.0},
            r"Unknown MORPHEUS priorities \['urgent'\] in regime 'r'",
        ),
        (
            {"missing_data": 1.0},
            {"low": 0.0},
            r"Regime 'r': priorities needs at least one positive weight",
        ),
        (
            {"missing_data": 1.0},
            {"low": float("nan")},
            r"Regime 'r': priorities\['low'\] = nan must be finite",
        ),
    ],
)
def test_regime_validation_names_the_regime_and_the_bad_value(
    chaos_types: dict[str, float], priorities: dict[str, float] | None, message: str
) -> None:
    extra: dict[str, Any] = {} if priorities is None else {"priorities": priorities}
    with pytest.raises(ValueError, match=message):
        Regime("r", 0.5, chaos_types, **extra)


@pytest.mark.parametrize("rate", [float("nan"), 1.5, -0.1])
def test_regime_arrival_rate_error_names_the_regime_and_the_value(rate: float) -> None:
    with pytest.raises(ValueError, match=r"Regime 'r': arrival_rate = .* must lie in \[0, 1\]"):
        Regime("r", rate, {"missing_data": 1.0})


def test_regime_weight_that_is_not_a_number_names_the_regime_and_the_value() -> None:
    with pytest.raises(TypeError, match=r"Regime 'r': chaos_types\['missing_data'\] = '1' must be"):
        Regime("r", 0.5, {"missing_data": "1"})  # pyright: ignore[reportArgumentType]


def test_the_revisit_is_not_chaos_and_the_labels_follow_the_declared_switch() -> None:
    env = pp.presets.morpheus_regimes(period=3)
    try:
        state, _, _ = env.init(jax.random.key(0))
        rows = []
        for tick in range(20):
            state, _, _, _ = step(env, state, 0, tick)
            privileged = env.privileged(state)
            rows.append(
                (
                    int(privileged["regime_index"]),
                    int(privileged["task_id"]),
                    int(round(float(privileged["regime_parameters"]))),
                    bool(privileged["chaos_active"]),
                    bool(privileged["boundary"]),
                    len(client_of(env).pushed),
                )
            )
        # The labels agree with the declared parameter on every tick, switch ticks included.
        assert all(index == task == declared for index, task, declared, *_ in rows)
        assert [row[0] for row in rows[:6]] == [0, 0, 1, 1, 1, 2]
        assert [row[4] for row in rows if row[4]] == [True] * 5
        # Only the four perturbed regimes are chaos; the baseline revisit is the baseline.
        assert [row[3] for row in rows] == [row[0] not in (0, 5) for row in rows]
        assert rows[-1][0] == 5
        assert not rows[-1][3]
        # The world still takes a switch before the next transition, not on the declared tick.
        assert rows[2][5] == 0
        assert rows[3][5] == 1
    finally:
        env.close()


def test_the_verifier_pass_fraction_is_this_ticks_and_zero_without_a_resolve() -> None:
    env = pp.make("morpheus/Morpheus-v0")
    try:
        state, _, _ = env.init(jax.random.key(0))
        state, obs, _, _ = step(env, state, 1, 0)
        state, obs, _, _ = step(env, state, 2, 1)
        assert feature(obs, "verifier_pass_fraction") == 1.0
        state, obs, _, _ = step(env, state, 0, 2)
        assert feature(obs, "verifier_pass_fraction") == 0.0
    finally:
        cast(Any, env).close()


def test_a_refused_init_or_action_leaves_the_regime_as_the_server_has_it() -> None:
    from perpetua.adapters.morpheus import MorpheusParams

    env: Any = pp.make("morpheus/Morpheus-v0")
    try:
        world = env.env
        backend = world.backend
        multiplier = jnp.asarray([1.0], dtype=jnp.float32)
        baseline = MorpheusParams(multiplier, jnp.asarray(0.0))
        spike = MorpheusParams(multiplier, jnp.asarray(1.0))
        state, _, _ = world.init(jax.random.key(0), baseline)
        with pytest.raises(RuntimeError, match="new CPU backend"):
            world.init(jax.random.key(1), spike)
        assert (backend.regime, backend.declared) == (0, 0)
        with pytest.raises(ValueError, match="outside the backend action space"):
            world.step(state, jnp.asarray([99], dtype=jnp.int32), jax.random.key(2), spike)
        assert backend.regime == 0
        assert client_of(env).pushed == []
        world.step(state, jnp.asarray([0], dtype=jnp.int32), jax.random.key(2), spike)
        assert [regime.regime_id for regime in client_of(env).pushed] == ["edi_spike"]
    finally:
        env.close()


class _Churn(FakeDeployment):
    """A deployment whose tickets the test opens, hides and closes between reads."""

    def __init__(self) -> None:
        super().__init__()
        self.tickets = {}
        self.authorization: list[str | None] = []

    def open(self, ticket_id: str) -> None:
        self.tickets[ticket_id] = {"_id": ticket_id, "status": "new", "priority": "low"}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.authorization.append(request.headers.get("authorization"))
        return super().handle(request)


def _churn_client(cap: int, **headers: str) -> tuple[HttpMorpheusClient, _Churn]:
    class Capped(HttpMorpheusClient):
        _MAX_CLOSED_SEEN = cap

    deployment = _Churn()
    client = Capped(
        "https://morpheus.test",
        client=httpx.Client(
            base_url="https://morpheus.test",
            headers=headers,
            transport=httpx.MockTransport(deployment.handle),
        ),
    )
    client.create(0, PAPER_REGIMES[0])
    return client, deployment


def test_the_seen_ticket_record_is_bounded_over_a_long_lifetime() -> None:
    cap = 8
    client, deployment = _churn_client(cap)
    reported: list[str] = []
    for tick in range(300):
        # Three tickets open per read and stay open for two reads.
        for slot in range(3):
            deployment.open(f"t{tick}-{slot}")
        for slot in range(3):
            deployment.tickets.pop(f"t{tick - 2}-{slot}", None)
        state = client.snapshot([], False)
        reported += state["new_failures"]
        open_now = {ticket["id"] for ticket in state["tickets"]}
        assert open_now <= set(client.seen)
        assert len(client.seen) <= len(open_now) + cap
    # 900 tickets, each a new failure exactly once: none lost, none repeated.
    assert len(reported) == 900
    assert len(client.seen) == 6 + cap


def test_a_ticket_that_returns_within_the_bound_is_not_a_new_failure_again() -> None:
    client, deployment = _churn_client(cap=4)
    deployment.open("flaky")
    assert len(client.snapshot([], False)["new_failures"]) == 1
    hidden = deployment.tickets.pop("flaky")
    for index in range(4):  # four other tickets leave the open list meanwhile
        deployment.open(f"other-{index}")
        assert len(client.snapshot([], False)["new_failures"]) == 1
        del deployment.tickets[f"other-{index}"]
    deployment.tickets["flaky"] = hidden
    assert client.snapshot([], False)["new_failures"] == []
    # Past the bound it is forgotten, which the class docstring states.
    del deployment.tickets["flaky"]
    for index in range(5):
        deployment.open(f"later-{index}")
        client.snapshot([], False)
        del deployment.tickets[f"later-{index}"]
    deployment.tickets["flaky"] = hidden
    assert len(client.snapshot([], False)["new_failures"]) == 1


def test_the_token_header_is_sent_with_every_request() -> None:
    client, deployment = _churn_client(8, Authorization="Bearer secret")
    deployment.open("t1")
    state = client.snapshot([], False)
    client.step(2, state)
    client.close()
    assert len(deployment.authorization) > 20
    assert set(deployment.authorization) == {"Bearer secret"}
    assert {"POST", "PUT", "GET", "DELETE"} == {call[0] for call in deployment.calls}


@pytest.mark.parametrize(
    ("url", "headers", "warns"),
    [
        ("http://morpheus.example", {"Authorization": "Bearer secret"}, True),
        ("http://10.0.0.5:8282", {"X-API-Key": "secret"}, True),
        ("https://morpheus.example", {"Authorization": "Bearer secret"}, False),
        ("http://localhost:8282", {"Authorization": "Bearer secret"}, False),
        ("http://127.0.0.1:8282", {"Authorization": "Bearer secret"}, False),
        ("http://[::1]:8282", {"Authorization": "Bearer secret"}, False),
        ("http://morpheus.example", {}, False),
        ("http://morpheus.example", {"X-Auth": "secret"}, True),
        ("http://morpheus.example", {"APIKEY": "secret"}, True),
        ("http://morpheus.example", {"x-client-SECRET": "hunter2"}, True),
        ("http://morpheus.example", {"Session-Cookie": "secret"}, True),
        ("http://morpheus.example", {"aUtHoRiZaTiOn": "Bearer secret"}, True),
        ("http://morpheus.example", {"X-Request-Id": "1"}, False),
        ("http://user:secret@morpheus.example", {}, True),
        ("https://user:secret@morpheus.example", {}, False),
        ("http://user:secret@localhost:8282", {}, False),
    ],
)
def test_a_credential_over_plain_http_to_a_remote_host_warns_once(
    url: str, headers: dict[str, str], warns: bool
) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client = HttpMorpheusClient(url, headers=headers)
        client.http.close()
        # The same check reads a preconfigured client's own address and headers.
        given = HttpMorpheusClient(client=httpx.Client(base_url=url, headers=headers))
        given.http.close()
    messages = [str(w.message) for w in caught if "MORPHEUS client sends" in str(w.message)]
    assert len(messages) == (2 if warns else 0)
    if warns:
        assert "use an https address" in messages[0]
        assert all(
            value not in messages[0]
            for value in ("Bearer", "hunter2", ":secret", *headers.values())
        )


@pytest.mark.parametrize(
    ("url", "warns"),
    [("http://morpheus.example", True), ("https://morpheus.example", False)],
)
def test_client_auth_over_plain_http_to_a_remote_host_warns(url: str, warns: bool) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        given = HttpMorpheusClient(client=httpx.Client(base_url=url, auth=("user", "secret")))
        given.http.close()
    messages = [str(w.message) for w in caught if "MORPHEUS client sends" in str(w.message)]
    assert len(messages) == int(warns)
    if warns:
        assert "the client's auth" in messages[0]
        assert "secret" not in messages[0]


def test_user_info_in_the_address_is_named_without_its_value() -> None:
    with pytest.warns(UserWarning, match="the address's user-info") as caught:
        client = HttpMorpheusClient("http://user:hunter2@morpheus.example")
    client.http.close()
    assert all("hunter2" not in str(w.message) for w in caught)


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"base_url": "https://morpheus.example"}, "base_url="),
        ({"headers": {"Authorization": "Bearer secret"}}, "headers="),
        (
            {"base_url": "https://morpheus.example", "headers": {"Authorization": "x"}},
            "base_url=, headers=",
        ),
        ({"client": "http"}, "client="),
    ],
)
def test_live_arguments_with_the_default_mock_are_refused(
    kwargs: dict[str, Any], named: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MORPHEUS_URL", raising=False)
    if kwargs.get("client") == "http":
        transport = httpx.MockTransport(FakeDeployment().handle)
        kwargs = {
            "client": HttpMorpheusClient(
                client=httpx.Client(base_url="http://morpheus.test", transport=transport)
            )
        }
    with pytest.raises(ValueError, match=rf"got {named} with mock=True.*pass mock=False"):
        pp.make("morpheus/Morpheus-v0", **kwargs)
    with pytest.raises(ValueError, match="pass mock=False"):
        MorpheusLifetime(mock=True, **kwargs)
    # The mock's own client is not a live argument.
    env = pp.make("morpheus/Morpheus-v0", client=MockMorpheusClient())
    assert cast(Any, env.metadata)["configuration"]["backend"]["backend"] == "morpheus-mock"
    cast(Any, env).close()


def test_the_mock_warns_once_when_morpheus_url_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import warnings

    monkeypatch.setenv("MORPHEUS_URL", "https://morpheus.example")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = pp.make("morpheus/Morpheus-v0")
    messages = [str(w.message) for w in caught if "MORPHEUS_URL" in str(w.message)]
    assert len(messages) == 1
    assert "pass mock=False" in messages[0]
    assert cast(Any, env.metadata)["configuration"]["backend"]["backend"] == "morpheus-mock"
    cast(Any, env).close()
    monkeypatch.delenv("MORPHEUS_URL")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cast(Any, pp.make("morpheus/Morpheus-v0")).close()
    assert not [w for w in caught if "MORPHEUS_URL" in str(w.message)]
