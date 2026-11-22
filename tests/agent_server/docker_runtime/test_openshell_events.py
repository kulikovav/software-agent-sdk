import json
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import grpc
import httpx
import pytest

from openhands.agent_server.docker_runtime.openshell import events as events_module
from openhands.agent_server.docker_runtime.openshell.events import (
    DENIAL_EVENT_WINDOW_SECONDS,
    GOVERNANCE_CONTEXT,
    MAX_DENIAL_EVENTS_PER_WINDOW,
    NOTICES_PATH,
    PROPOSAL_TTL_SECONDS,
    PolicyDenial,
    PolicyEventWatcher,
    PolicyLedger,
    PolicyProposalRecord,
    governed_hook_config,
    governed_launch_additions,
    parse_denial,
    proposal_from_chunk,
)
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.hooks import HookConfig


def _chunk(**overrides: Any) -> Any:
    """A duck-typed gateway policy chunk."""
    base: dict[str, Any] = {
        "id": "chunk-1",
        "status": "pending",
        "rule_name": None,
        "proposed_rule": SimpleNamespace(host="", ports=[], path=""),
        "rationale": "repeated egress attempts",
        "rejection_reason": None,
        "validation_result": None,
        "review_token": "token-1",
        "created_time": None,
        "decided_time": None,
    }
    proposed = overrides.pop("proposed_rule", None)
    base.update(overrides)
    if proposed is not None:
        base["proposed_rule"] = SimpleNamespace(**proposed)
    return type("Chunk", (), base)()


def _proposal(**overrides: Any) -> PolicyProposalRecord:
    """A typed proposal record from a duck-typed gateway chunk."""
    record = proposal_from_chunk(_chunk(**overrides))
    assert record is not None
    return record


def _denial(destination: str = "example.com", layer: str = "network") -> PolicyDenial:
    return PolicyDenial(
        layer=layer, destination=destination, reason="policy_denied", rule=None
    )


class TestParseDenial:
    def test_parse_network_refusal(self):
        denial = parse_denial(
            "NET:REFUSE DENIED example.com [reason:policy_dns_ineligible]"
        )

        assert denial == PolicyDenial(
            layer="network",
            destination="example.com",
            reason="policy_dns_ineligible",
        )

    def test_parse_network_connect_denial(self):
        denial = parse_denial(
            "NET:OPEN  DENIED /usr/bin/curl(0) -> example.com:443 "
            "[reason:transparent_tcp_policy_denied]"
        )

        assert denial == PolicyDenial(
            layer="network",
            destination="example.com:443",
            reason="transparent_tcp_policy_denied",
        )

    def test_parse_http_denial_with_policy_and_rule(self):
        denial = parse_denial(
            "HTTP:POST DENIED  POST https://litellm.example/v1/other "
            '[policy:_provider_litellm engine:l7 rule "POST /other"]'
        )

        assert denial == PolicyDenial(
            layer="http",
            destination="https://litellm.example/v1/other",
            reason="_provider_litellm",
            rule="POST /other",
        )

    def test_parse_prefers_structured_fields(self):
        denial = parse_denial(
            "NET:REFUSE DENIED example.com [reason:transparent_tcp_policy_denied]",
            fields={"reason": "allow_deny_overlap", "tool": "terminal"},
        )

        assert denial is not None
        assert denial.reason == "allow_deny_overlap"
        assert denial.tool == "terminal"

    def test_parse_ignores_an_allowed_line(self):
        assert parse_denial("HTTP:POST ALLOWED POST https://example/v1/x") is None

    def test_parse_request_target_encoded_slash_rejection(self):
        denial = parse_denial(
            "HTTP request-target rejected: request-target contains an encoded "
            "'/' (%2F) which is not allowed on this endpoint"
        )

        assert denial == PolicyDenial(
            layer="http",
            destination="request-target",
            reason="encoded_slash",
        )

    def test_parse_request_target_rejection_with_another_cause(self):
        denial = parse_denial(
            "HTTP request-target rejected: request-target contains a fragment",
        )

        assert denial == PolicyDenial(
            layer="http",
            destination="request-target",
            reason="request_target_rejected",
        )

    def test_parse_request_target_rejection_prefers_structured_fields(self):
        denial = parse_denial(
            "HTTP request-target rejected: request-target contains a fragment",
            fields={
                "reason": "allow_deny_overlap",
                "rule": "GET /x",
                "tool": "terminal",
            },
        )

        assert denial == PolicyDenial(
            layer="http",
            destination="request-target",
            reason="allow_deny_overlap",
            rule="GET /x",
            tool="terminal",
        )


class TestGovernanceInjection:
    def test_governance_context_states_default_deny_and_one_retry(self):
        assert "denied by default" in GOVERNANCE_CONTEXT
        assert "one retry" in GOVERNANCE_CONTEXT

    def test_governance_context_names_policy_dns_and_encoded_slash_symptoms(self):
        assert "198.18.0.0/15" in GOVERNANCE_CONTEXT
        assert "stale policy-DNS" in GOVERNANCE_CONTEXT
        assert "%2F" in GOVERNANCE_CONTEXT
        assert "TLS EOF" in GOVERNANCE_CONTEXT
        assert "numeric identifier" in GOVERNANCE_CONTEXT

    def test_hook_config_reads_and_clears_the_notices_file(self):
        config = governed_hook_config(None)

        command = config.user_prompt_submit[0].hooks[0].command
        assert NOTICES_PATH in command
        assert "json.dumps" in command
        # The file is cleared after it is read, so a notice injects once.
        assert '"w"' in command

    def test_governed_hook_config_merges_agent_hooks(self):
        existing = HookConfig.model_validate(
            {
                "user_prompt_submit": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": "echo"}]}
                ]
            }
        )

        merged = governed_hook_config(existing)

        names = [
            hook.name for matcher in merged.user_prompt_submit for hook in matcher.hooks
        ]
        assert "governance-notices" in names

    def test_governed_launch_additions_keep_the_caller_suffix(self):
        additions = AgentLaunchAdditions(system_message_suffix_append="Caller rules")

        additions = governed_launch_additions(additions)

        assert additions.system_message_suffix_append is not None
        assert additions.system_message_suffix_append.startswith("Caller rules")
        assert "denied by default" in (additions.system_message_suffix_append or "")

    def test_governed_launch_additions_without_a_caller_suffix(self):
        additions = governed_launch_additions(None)

        assert additions.system_message_suffix_append == GOVERNANCE_CONTEXT


class TestPolicyLedgerDenials:
    def test_one_denial_event_names_its_proposal_provenance(self):
        ledger = PolicyLedger()
        proposal = _proposal(
            id="chunk-9",
            proposed_rule={"host": "example.com", "ports": [443], "path": ""},
            validation_result="within_boundary",
        )
        ledger.link_proposal(proposal)

        event = ledger.process_denial(_denial("example.com"), now=1.0)

        assert event is not None
        assert event.status == "denied"
        assert event.provenance == {
            "proposal": "chunk-9",
            "scope": "example.com:443",
            "prover": "within_boundary",
        }

    def test_a_denial_of_a_known_destination_is_deduped(self):
        ledger = PolicyLedger()

        assert ledger.process_denial(_denial(), now=1.0) is not None
        assert ledger.process_denial(_denial(), now=2.0) is None
        assert ledger.process_denial(_denial("other.com"), now=2.0) is not None
        # After the dedup window the same destination is named again.
        assert (
            ledger.process_denial(_denial(), now=1.0 + DENIAL_EVENT_WINDOW_SECONDS)
            is not None
        )

    def test_a_prompt_injected_flood_hits_the_cap(self):
        ledger = PolicyLedger()

        events = [
            ledger.process_denial(_denial(f"host-{i}.com"), now=1.0)
            for i in range(MAX_DENIAL_EVENTS_PER_WINDOW + 10)
        ]

        assert (
            sum(event is not None for event in events) == MAX_DENIAL_EVENTS_PER_WINDOW
        )
        assert events[-1] is None


class TestPolicyLedgerProposals:
    def test_a_pending_proposal_without_a_denial_posts_one_event(self):
        ledger = PolicyLedger()
        record = _proposal()

        outcome = ledger.process_proposal(record, now=1.0)

        assert outcome.event is not None
        assert outcome.event.status == "denied"
        assert outcome.event.provenance["proposal"] == "chunk-1"

    def test_an_unchanged_status_posts_nothing_twice(self):
        record = _proposal(status="approved")
        ledger = PolicyLedger()

        first = ledger.process_proposal(record, now=1.0)
        second = ledger.process_proposal(record, now=2.0)

        assert first.event is not None and first.widen_pending
        assert second.event is None
        # An uncommitted widening keeps being offered at later boundaries.
        assert second.widen_pending is True

    def test_an_approval_widens_exactly_once(self):
        record = _proposal(status="approved")
        ledger = PolicyLedger()

        outcome = ledger.process_proposal(record, now=1.0)

        assert outcome.event is not None
        assert outcome.event.status == "approved"
        assert outcome.widen_pending is True
        assert ledger.is_applied("chunk-1") is False

        ledger.widen_decided("chunk-1")
        assert ledger.is_applied("chunk-1")
        again = ledger.process_proposal(record, now=2.0)
        assert not again

    def test_an_auto_approval_widens_without_a_review_token(self):
        record = _proposal(status="approved", review_token="")
        ledger = PolicyLedger()

        outcome = ledger.process_proposal(record, now=1.0)

        assert outcome.event is not None
        assert outcome.event.status == "approved"
        assert outcome.event.reason == "approved automatically"
        assert outcome.widen_pending is True

        # An uncommitted auto approval keeps being offered like a human one.
        assert ledger.process_proposal(record, now=2.0).widen_pending is True

        ledger.widen_decided("chunk-1")
        assert not ledger.process_proposal(record, now=3.0)

    def test_a_rejection_reason_reaches_the_notice_for_the_next_turn(self):
        record = _proposal(
            status="rejected",
            rejection_reason="outside project scope",
            proposed_rule={"host": "example.com", "ports": [], "path": ""},
        )
        ledger = PolicyLedger()

        outcome = ledger.process_proposal(record, now=1.0)

        assert outcome.event is not None
        assert outcome.event.status == "rejected"
        assert outcome.event.reason == "outside project scope"
        assert outcome.notice is not None
        assert "outside project scope" in outcome.notice
        assert "example.com" in outcome.notice

    def test_a_pending_request_past_its_ttl_expires_as_denied(self):
        stale = _proposal(
            created_time=_fake_timestamp(time.time() - 700.0),
        )
        fresh = _proposal()

        ledger = PolicyLedger()
        fresh_outcome = ledger.process_proposal(fresh, now=time.time())
        expired = PolicyLedger().process_proposal(stale, now=time.time())

        assert fresh_outcome.event is not None
        assert fresh_outcome.event.status == "denied"
        assert expired.event is not None and expired.event.status == "expired"
        assert expired.notice is not None
        assert "stays denied" in expired.notice

    def test_a_pending_proposal_watched_across_its_ttl_expires_later(self):
        record = _proposal(created_time=_fake_timestamp(0.0))
        ledger = PolicyLedger()

        first = ledger.process_proposal(record, now=1.0)
        later = ledger.process_proposal(record, now=PROPOSAL_TTL_SECONDS + 2.0)

        assert first.event is not None and first.event.status == "denied"
        assert later.event is not None and later.event.status == "expired"
        assert later.notice is not None

    def test_an_expired_proposal_is_announced_once(self):
        record = _proposal(created_time=_fake_timestamp(0.0))
        ledger = PolicyLedger()

        ledger.process_proposal(record, now=1.0)
        expired = ledger.process_proposal(record, now=PROPOSAL_TTL_SECONDS + 2.0)
        again = ledger.process_proposal(record, now=PROPOSAL_TTL_SECONDS + 7.0)

        assert expired.event is not None and expired.event.status == "expired"
        # The gateway keeps a stale request pending; the watch must not re-post
        # the same expiry on every cycle.
        assert again.event is None
        assert again.notice is None

    def test_prime_silently_records_preexisting_decisions(self):
        decided_before = _proposal(
            status="approved", decided_time=_fake_timestamp(50.0)
        )
        decided_lately = _proposal(
            id="chunk-2", status="approved", decided_time=_fake_timestamp(200.0)
        )
        pending = _proposal(id="chunk-3", status="pending", decided_time=None)
        ledger = PolicyLedger()

        # A cutoff inside the watch's life: only older decisions are primed,
        # and a pending proposal keeps its owed denial event.
        ledger.prime([decided_before, decided_lately, pending], cutoff=100.0)

        assert ledger.process_proposal(decided_before, now=100.0).event is None
        assert ledger.process_proposal(pending, now=100.0).event is not None
        assert ledger.process_proposal(decided_lately, now=100.0).event is not None

    def test_state_roundtrip_keeps_decisions_unposted(self):
        ledger = PolicyLedger()
        record = _proposal(status="approved")
        ledger.process_proposal(record, now=1.0)
        ledger.process_denial(_denial(), now=1.0)
        ledger.widen_decided("chunk-1")

        restored = PolicyLedger(ledger.to_state())

        assert not restored.process_proposal(record, now=2.0)
        # Within the dedup window the same denial stays suppressed.
        assert restored.process_denial(_denial(), now=1.1) is None

    def test_the_review_token_never_reaches_state_or_events(self):
        record = _proposal(status="approved")
        ledger = PolicyLedger()

        outcome = ledger.process_proposal(record, now=1.0)

        assert outcome.event is not None
        assert "token-1" not in outcome.event.model_dump_json()
        assert "token-1" not in str(ledger.to_state())

    def test_auto_approved_is_inferred_from_the_review_token(self):
        assert _proposal(status="approved").auto_approved is False
        assert _proposal(status="approved", review_token="").auto_approved is True


def _fake_timestamp(epoch: float) -> Any:
    class FakeTimestamp:
        def ToSeconds(self) -> float:
            return epoch

    return FakeTimestamp()


class _Clock:
    """A fake ``time.time`` the watcher tests drive deterministically."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        now = self.now
        self.now += 0.001
        return now


def _watcher(
    registry: "_FakeRegistry",
    posts: list[dict[str, Any]],
    execution_status: str = "idle",
) -> PolicyEventWatcher:
    watcher = PolicyEventWatcher(
        registry.gateway, registry, interval=5.0, sync_retry_interval=0.0
    )
    transport = _transport(posts, execution_status)
    watcher._http_client = lambda container: httpx.AsyncClient(  # type: ignore[method-assign]
        transport=transport
    )
    return watcher


def _transport(
    posts: list[dict[str, Any]],
    execution_status: str = "idle",
    fail_posts: bool = False,
    updated_at: str = "2026-01-01T00:00:00Z",
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if fail_posts:
                return httpx.Response(500, json={"detail": "sandbox down"})
            posts.append(
                {
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "json": json.loads(request.content),
                }
            )
            return httpx.Response(200, json={"success": True})
        return httpx.Response(
            200,
            json={"execution_status": execution_status, "updated_at": updated_at},
        )

    return httpx.MockTransport(handler)


def _failing_watcher(registry: "_FakeRegistry") -> PolicyEventWatcher:
    watcher = PolicyEventWatcher(
        registry.gateway, registry, interval=5.0, sync_retry_interval=0.0
    )
    watcher._http_client = lambda container: httpx.AsyncClient(  # type: ignore[method-assign]
        transport=_transport([], fail_posts=True)
    )
    return watcher


class _RpcError(grpc.RpcError):
    """A gRPC error with only the ``code()`` surface the watcher reads."""

    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


class _FakeGateway:
    def __init__(self) -> None:
        self.log_lines: list[Any] = []
        self.chunks: list[Any] = []
        self.committed: list[tuple[str, str]] = []
        self.notices: list[str] = []
        self.commit_error: grpc.StatusCode | None = None

    def logs(self, name: str, since_epoch: float | None = None) -> list[Any]:
        return self.log_lines

    def draft_chunks(self, name: str) -> list[Any]:
        return self.chunks

    def commit_approved_chunk(self, name: str, chunk_id: str, review_token: str) -> int:
        self.committed.append((chunk_id, review_token))
        if self.commit_error is not None:
            raise _RpcError(self.commit_error)
        return 2

    def append_notices(self, name: str, text: str) -> None:
        self.notices.append(text)


class _FakeContainer:
    def __init__(self) -> None:
        self.host = "https://sandbox:17670"
        self.api_key = "inner-key"
        self.ca_bundle = None
        self._running = True

    def is_running(self) -> bool:
        return self._running


class _FakeRegistry:
    def __init__(self) -> None:
        self.gateway = _FakeGateway()
        self.container = _FakeContainer()
        self.conversation_id = uuid4()
        self.leases = 0
        self._state: dict[str, Any] | None = None
        self.saved: list[dict[str, Any]] = []
        self.synced: list[Any] = []
        self.fail_syncs = 0

    def sandbox_name(self, conversation_id: Any) -> str:
        return "oh-sandbox"

    def live_governed_conversations(self) -> list[tuple[Any, _FakeContainer]]:
        if self.container._running:
            return [(self.conversation_id, self.container)]
        return []

    async def sync_conversation_record(self, conversation_id: Any) -> bool:
        self.synced.append(conversation_id)
        if self.fail_syncs:
            self.fail_syncs -= 1
            return False
        return True

    def acquire_lease(self, conversation_id: Any):
        self.leases += 1
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self.leases -= 1

        return release

    def load_policy_state(self, conversation_id: Any) -> dict[str, Any] | None:
        return self._state

    def save_policy_state(self, conversation_id: Any, state: dict[str, Any]) -> None:
        self._state = state
        self.saved.append(state)


class TestPolicyEventWatcher:
    @pytest.mark.asyncio
    async def test_a_denial_posts_one_policy_event(self):
        registry = _FakeRegistry()
        registry.gateway.log_lines = [
            SimpleNamespace(
                message="NET:REFUSE DENIED example.com:443 "
                "[reason:transparent_tcp_policy_denied]",
                fields={},
                event_time=_fake_timestamp(10.0),
            )
        ]
        registry.gateway.chunks = [
            _chunk(proposed_rule={"host": "example.com:443", "ports": [], "path": ""})
        ]
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts)

        await watcher.cycle()

        assert len(posts) == 1
        delivered = posts[0]["json"]
        assert delivered["kind"] == "PolicyEvent"
        assert delivered["status"] == "denied"
        assert delivered["reason"] == "transparent_tcp_policy_denied"
        assert delivered["provenance"]["proposal"] == "chunk-1"
        assert posts[0]["headers"]["x-session-api-key"] == "inner-key"
        assert posts[0]["url"] == (
            f"https://sandbox:17670/api/conversations/"
            f"{registry.conversation_id.hex}/events/_policy"
        )
        assert registry.leases == 0
        assert registry.saved and registry.saved[0]["log_cursor"] == 10.0

    @pytest.mark.asyncio
    async def test_an_approval_commits_only_at_a_turn_boundary(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="approved",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts, execution_status="running")
        clock.now += 0.001
        await watcher.cycle()

        assert registry.gateway.committed == []
        assert [event["json"]["status"] for event in posts] == ["approved"]

        # A later process (a restart) reloads the saved state; it re-offers
        # the still-uncommitted widening without a second approval event.
        clock.now += 60.0
        watcher2 = _watcher(registry, posts, execution_status="idle")
        await watcher2.cycle()

        assert registry.gateway.committed == [("chunk-1", "token-1")]
        assert [event["json"]["status"] for event in posts] == ["approved"]

    @pytest.mark.asyncio
    async def test_an_auto_approval_commits_while_the_turn_is_running(
        self, monkeypatch
    ):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="approved",
                review_token="",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts, execution_status="running")

        await watcher.cycle()

        # No human retry waits for the boundary, so an auto approval is
        # committed in-turn with the empty token the proto treats as unset.
        assert registry.gateway.committed == [("chunk-1", "")]
        assert posts[0]["json"]["reason"] == "approved automatically"

        await watcher.cycle()

        assert len(posts) == 1
        assert registry.gateway.committed == [("chunk-1", "")]

    @pytest.mark.asyncio
    async def test_a_rejection_writes_its_notice_for_the_next_turn(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="rejected",
                rejection_reason="no reason given",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts)

        await watcher.cycle()

        assert [event["json"]["status"] for event in posts] == ["rejected"]
        assert len(registry.gateway.notices) == 1
        assert "no reason given" in registry.gateway.notices[0]
        # The approved-retry permit only ever covers one human-approved chunk,
        # so a rejection never reaches the widening path.
        assert registry.gateway.committed == []

    @pytest.mark.asyncio
    async def test_a_second_cycle_does_not_repeat_the_widening(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="approved",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts)

        await watcher.cycle()
        await watcher.cycle()

        assert registry.gateway.committed == [("chunk-1", "token-1")]
        assert len(posts) == 1

    @pytest.mark.asyncio
    async def test_a_failed_delivery_retries_on_the_next_cycle(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="approved",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]

        await _failing_watcher(registry).cycle()

        # Nothing was committed, and the state records no delivered decision.
        assert registry.gateway.committed == []
        assert registry.saved[-1]["chunk_status"] == {}

        posts: list[dict[str, Any]] = []
        await _watcher(registry, posts).cycle()

        assert [event["json"]["status"] for event in posts] == ["approved"]
        assert registry.gateway.committed == [("chunk-1", "token-1")]

    @pytest.mark.asyncio
    async def test_a_permanent_commit_failure_records_the_chunk_failed(
        self, monkeypatch
    ):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="approved",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]
        registry.gateway.commit_error = grpc.StatusCode.INVALID_ARGUMENT
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts)

        await watcher.cycle()
        await watcher.cycle()

        assert registry.gateway.committed == [("chunk-1", "token-1")]
        assert registry.saved[-1]["failed"] == ["chunk-1"]
        assert len(registry.gateway.notices) == 1
        assert "stays denied" in registry.gateway.notices[0]
        assert "INVALID_ARGUMENT" in registry.gateway.notices[0]
        assert [event["json"]["status"] for event in posts] == ["approved"]

        restored = PolicyLedger(registry.saved[-1])
        assert restored.is_failed("chunk-1")
        assert not restored.process_proposal(_proposal(status="approved"), now=2.0)

    @pytest.mark.asyncio
    async def test_a_transient_commit_failure_retries_on_the_next_cycle(
        self, monkeypatch
    ):
        clock = _Clock()
        monkeypatch.setattr(events_module, "time", SimpleNamespace(time=clock))
        registry = _FakeRegistry()
        registry.gateway.chunks = [
            _chunk(
                id="chunk-1",
                status="approved",
                decided_time=_fake_timestamp(clock.now + 1.0),
            )
        ]
        registry.gateway.commit_error = grpc.StatusCode.UNAVAILABLE
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts)

        await watcher.cycle()
        await watcher.cycle()

        assert registry.gateway.committed == [("chunk-1", "token-1")] * 2
        assert registry.saved[-1]["failed"] == []
        assert registry.gateway.notices == []

    @pytest.mark.asyncio
    async def test_a_stopped_sandbox_is_not_watched(self):
        registry = _FakeRegistry()
        registry.container._running = False
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts)

        await watcher.cycle()

        assert posts == []
        assert registry.saved == []
        # No lease is held against an idle runtime.
        assert registry.leases == 0

    @pytest.mark.asyncio
    async def test_a_terminal_run_refreshes_the_catalog_record_once(self):
        registry = _FakeRegistry()
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts, execution_status="finished")

        await watcher.cycle()
        await watcher.cycle()

        # One refresh per record identity: a finished conversation must not
        # rewrite its catalog row on every watch cycle.
        assert registry.synced == [registry.conversation_id]

    @pytest.mark.asyncio
    async def test_a_running_conversation_keeps_its_catalog_record(self):
        registry = _FakeRegistry()
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts, execution_status="running")

        await watcher.cycle()

        assert registry.synced == []

    @pytest.mark.asyncio
    async def test_each_finished_turn_refreshes_the_catalog_record(self):
        registry = _FakeRegistry()
        snapshots = iter(
            [
                ("running", "2026-01-01T00:00:00Z"),
                ("finished", "2026-01-01T00:00:01Z"),
                ("finished", "2026-01-01T00:00:01Z"),
                ("running", "2026-01-01T00:00:02Z"),
                ("finished", "2026-01-01T00:00:03Z"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            status, updated_at = next(snapshots)
            return httpx.Response(
                200, json={"execution_status": status, "updated_at": updated_at}
            )

        watcher = PolicyEventWatcher(
            registry.gateway, registry, interval=5.0, sync_retry_interval=0.0
        )
        watcher._http_client = lambda container: httpx.AsyncClient(  # type: ignore[method-assign]
            transport=httpx.MockTransport(handler)
        )

        for _ in range(5):
            await watcher.cycle()

        # The first and the second turn each reach a terminal state; the two
        # cycles that observe the same finished record refresh nothing.
        assert registry.synced == [registry.conversation_id] * 2

    @pytest.mark.asyncio
    async def test_a_quick_second_turn_is_synced_between_polls(self):
        registry = _FakeRegistry()
        snapshots = iter(
            [
                ("finished", "2026-01-01T00:00:01Z"),
                ("finished", "2026-01-01T00:00:02Z"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            status, updated_at = next(snapshots)
            return httpx.Response(
                200, json={"execution_status": status, "updated_at": updated_at}
            )

        watcher = PolicyEventWatcher(
            registry.gateway, registry, interval=5.0, sync_retry_interval=0.0
        )
        watcher._http_client = lambda container: httpx.AsyncClient(  # type: ignore[method-assign]
            transport=httpx.MockTransport(handler)
        )

        await watcher.cycle()
        await watcher.cycle()

        # Both observations read finished, but the second turn produced a new
        # record identity, so its terminal state still reaches the catalog.
        assert registry.synced == [registry.conversation_id] * 2

    @pytest.mark.asyncio
    async def test_a_failed_catalog_refresh_is_retried(self):
        registry = _FakeRegistry()
        registry.fail_syncs = 1
        posts: list[dict[str, Any]] = []
        watcher = _watcher(registry, posts, execution_status="finished")

        await watcher.cycle()

        assert registry.synced == [registry.conversation_id]

        await watcher.cycle()

        assert registry.synced == [registry.conversation_id] * 2

        await watcher.cycle()

        # The confirmed identity deduplicates later cycles.
        assert registry.synced == [registry.conversation_id] * 2

    @pytest.mark.asyncio
    async def test_a_payload_without_updated_at_falls_back_to_status(self):
        registry = _FakeRegistry()
        statuses = iter(["finished", "finished"])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"execution_status": next(statuses)})

        watcher = PolicyEventWatcher(
            registry.gateway, registry, interval=5.0, sync_retry_interval=0.0
        )
        watcher._http_client = lambda container: httpx.AsyncClient(  # type: ignore[method-assign]
            transport=httpx.MockTransport(handler)
        )

        await watcher.cycle()
        await watcher.cycle()

        # Without updated_at the identity degrades to the status, so a second
        # turn that completes between polls is not detected.
        assert registry.synced == [registry.conversation_id]
