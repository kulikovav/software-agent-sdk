"""Governance context, policy decisions, and the policy-event watch.

The governed runtime driver holds the gateway identity, so it also drives the
governance loop: it states the standing rules when it provisions a sandbox,
watches the gateway's denial and proposal records, delivers each decision to
the conversation as a :class:`PolicyEvent`, and commits a human-approved rule
widening only at a turn boundary. Narrowing and revocation are gateway safety
actions and pass through immediately.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import grpc

from openhands.agent_server.docker_runtime.registry import ConversationContainer
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import PolicyEvent
from openhands.sdk.hooks import HookConfig
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

GOVERNANCE_CONTEXT = (
    "This conversation runs inside a governed sandbox: egress is denied by "
    "default and only the model endpoint is allowed; the workspace is "
    "ephemeral; credentials resolve outside the sandbox, so only placeholders "
    "reach the tools. A denied request is named in the tool outcome and in a "
    "policy event. Ask the operator for an approval when a task needs another "
    "destination; after an approval, exactly one retry is allowed on the next "
    "turn."
)

# Proposals expire unanswered after this long; the next turn learns of it.
PROPOSAL_TTL_SECONDS = 600.0
# A prompt-injected flood hits this cap instead of spamming the transcript.
# Human decisions (approvals, rejections, expiries) are never capped.
MAX_DENIAL_EVENTS_PER_WINDOW = 50
DENIAL_EVENT_WINDOW_SECONDS = 3600.0
# A destination denied again within this window updates nothing: the denial
# of the same destination is already on the transcript.
DENIAL_DEDUP_SECONDS = 300.0

# The sandbox working directory; the governed registry sets it as HOME.
NOTICES_PATH = "/sandbox/.policy/notices"
NOTICES_DIR = "/sandbox/.policy"

_NOTICES_HOOK_COMMAND = (
    "python3 -c 'import json,os;"
    f'p="{NOTICES_PATH}";'
    't=(open(p).read(8000) if os.path.isfile(p) else "");'
    'print(json.dumps({"additionalContext": t}) if t.strip() else "");'
    '(open(p, "w").close() if t else None)\''
)


def governance_hook_config() -> dict[str, Any]:
    """The hook config that carries denial notices into the next turn.

    The command hook reads the notices file the watcher writes and clears it,
    so each notice is delivered once, as context on the user's next turn.
    """
    return {
        "user_prompt_submit": [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "name": "governance-notices",
                        "command": _NOTICES_HOOK_COMMAND,
                        "timeout": 10,
                    }
                ],
            }
        ]
    }


def governed_hook_config(existing: HookConfig | None) -> HookConfig:
    """Merge the governance hooks into an agent's own hook config."""
    governance = HookConfig.model_validate(governance_hook_config())
    merged = HookConfig.merge([existing, governance]) if existing else governance
    return governance if merged is None else merged


def governed_launch_additions(
    additions: AgentLaunchAdditions | None,
) -> AgentLaunchAdditions:
    """Append the governance context to the launch-time system suffix."""
    standing = (additions.system_message_suffix_append or "") if additions else ""
    suffix = (
        GOVERNANCE_CONTEXT if not standing else f"{standing}\n\n{GOVERNANCE_CONTEXT}"
    )
    if additions is None:
        return AgentLaunchAdditions(system_message_suffix_append=suffix)
    return additions.model_copy(update={"system_message_suffix_append": suffix})


@dataclass(frozen=True)
class PolicyDenial:
    """One denial record parsed from the gateway's sandbox log."""

    layer: str
    destination: str
    reason: str
    rule: str | None = None
    tool: str | None = None

    @property
    def summary(self) -> str:
        rule = f" (rule {self.rule})" if self.rule else ""
        return (
            f"{self.layer} access to {self.destination} was denied: {self.reason}{rule}"
        )


def parse_denial(
    line: str, fields: dict[str, str] | None = None
) -> PolicyDenial | None:
    """Parse one sandbox log line, or return ``None`` for an unrelated line.

    ``fields`` names the structured sandbox-log fields; a structured value wins
    over one parsed out of the message text.
    """
    text = line.strip()
    if " DENIED " not in text:
        return None
    prefix, rest = text.split(" DENIED ", 1)
    rest = rest.strip()
    bracket = rest.find("[")
    detail = rest[bracket:] if bracket >= 0 else ""
    body = rest[:bracket].strip() if bracket >= 0 else rest

    if prefix.startswith("NET:REFUSE"):
        layer = "network"
        destination = body
    elif prefix.startswith("NET:OPEN"):
        layer = "network"
        destination = body.split(" -> ", 1)[1].strip() if " -> " in body else body
    elif prefix.startswith("HTTP:"):
        layer = "http"
        parts = body.split()
        destination = parts[1] if len(parts) > 1 else body
    else:
        return None

    parsed = _scan_fields(detail)
    if fields:
        parsed.update(fields)
    reason = parsed.get("reason") or parsed.get("policy") or "policy_denied"
    return PolicyDenial(
        layer=layer,
        destination=destination,
        reason=reason,
        rule=parsed.get("rule"),
        tool=parsed.get("tool"),
    )


def _scan_fields(detail: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in re.findall(r"([a-z_]+):([^\]\s]+)", detail):
        fields.setdefault(key, value)
    # The gateway quotes some values, such as rule "POST /v1"; those values
    # can contain spaces and colons the unquoted scan would misread.
    for key, value in re.findall(r'([a-z_]+) "([^"]+)"', detail):
        fields[key] = value
    return fields


@dataclass(frozen=True)
class PolicyProposalRecord:
    """One rule proposal the gateway drafted from a denial."""

    chunk_id: str
    status: str
    destination: str
    rule_name: str | None
    rationale: str
    rejection_reason: str | None
    prover_verdict: str | None
    review_token: str
    created_epoch: float | None
    decided_epoch: float | None
    requested_scope: str

    def provenance(self) -> dict[str, str]:
        """Provenance labels safe for the event stream and the state file.

        The review token is a commit credential; it never joins provenance.
        """
        return {
            "proposal": self.chunk_id,
            "scope": self.requested_scope,
            "prover": self.prover_verdict or "unverified",
        }


def proposal_from_chunk(chunk: Any) -> PolicyProposalRecord | None:
    """Turn a gateway policy chunk into a typed record.

    The chunk is duck-typed so tests can pass plain stand-ins.
    """
    chunk_id = getattr(chunk, "id", "") or ""
    if not chunk_id:
        return None
    rule = getattr(chunk, "proposed_rule", None)
    destination = (
        str(getattr(rule, "host", "") or "") or getattr(chunk, "rule_name", "") or ""
    )
    if not destination:
        destination = "<unknown>"
    ports = list(getattr(rule, "ports", []) or [])
    path = str(getattr(rule, "path", "") or "")
    scope = destination
    if ports:
        scope = f"{destination}:{'+'.join(str(port) for port in ports)}"
    if path:
        scope = f"{scope}{path}"
    created = getattr(chunk, "created_time", None)
    created_epoch = (
        float(created.ToSeconds())
        if created is not None and hasattr(created, "ToSeconds")
        else None
    )
    verdict = getattr(chunk, "validation_result", None)
    decided = getattr(chunk, "decided_time", None)
    return PolicyProposalRecord(
        chunk_id=str(chunk_id),
        status=str(getattr(chunk, "status", "") or "pending"),
        destination=destination,
        rule_name=getattr(chunk, "rule_name", None) or None,
        rationale=str(getattr(chunk, "rationale", "") or ""),
        rejection_reason=getattr(chunk, "rejection_reason", None) or None,
        prover_verdict=str(verdict) if verdict not in (None, "") else None,
        review_token=str(getattr(chunk, "review_token", "") or ""),
        created_epoch=created_epoch,
        decided_epoch=(
            float(decided.ToSeconds())
            if decided is not None and hasattr(decided, "ToSeconds")
            else None
        ),
        requested_scope=scope,
    )


@dataclass
class ProposalOutcome:
    """What one proposal record changed in the conversation."""

    event: PolicyEvent | None = None
    notice: str | None = None
    widen_pending: bool = False

    def __bool__(self) -> bool:
        return self.event is not None or self.notice is not None or self.widen_pending


class PolicyLedger:
    """Per-conversation state for the policy-event watch.

    Pure decision logic: the watcher feeds it parsed records and posts or
    writes whatever the ledger returns. The last posted state survives a
    restart, so decisions are never posted twice.
    """

    MAX_TRACKED_CHUNKS = 200

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        self.log_cursor: float | None = state.get("log_cursor")
        self.chunk_status: dict[str, str] = dict(state.get("chunk_status") or {})
        self.applied: set[str] = set(state.get("applied") or [])
        self.denial_posted: dict[str, float] = dict(state.get("denial_posted") or {})
        self.destinations: dict[str, str] = dict(state.get("destinations") or {})
        self.proposal_meta: dict[str, dict[str, str]] = dict(
            state.get("proposal_meta") or {}
        )
        self._window_start: float = state.get("window_start") or 0.0
        self._window_denials: int = state.get("window_denials") or 0

    def to_state(self) -> dict[str, Any]:
        status = self.chunk_status
        if len(status) > self.MAX_TRACKED_CHUNKS:
            status = dict(list(status.items())[-self.MAX_TRACKED_CHUNKS :])
        return {
            "log_cursor": self.log_cursor,
            "chunk_status": status,
            "applied": sorted(self.applied),
            "denial_posted": dict(list(self.denial_posted.items())[-500:]),
            "destinations": self.destinations,
            "proposal_meta": self.proposal_meta,
            "window_start": self._window_start,
            "window_denials": self._window_denials,
        }

    def prime(self, records: Iterable[PolicyProposalRecord], cutoff: float) -> None:
        """Record decisions that predate this watch without posting them.

        A decision the gateway made while this watch was running is delivered
        by the normal path; only decisions older than the cutoff are primed.
        A pending proposal is never primed: its denial event is still owed.
        """
        for record in records:
            if record.status == "pending":
                continue
            decided = record.decided_epoch
            if decided is not None and decided >= cutoff:
                continue
            self.chunk_status.setdefault(record.chunk_id, record.status)

    def link_proposal(self, record: PolicyProposalRecord) -> None:
        """Remember a destination's proposal so denials name its provenance."""
        self.destinations.setdefault(record.destination, record.chunk_id)
        self.proposal_meta.setdefault(record.chunk_id, record.provenance())

    def provenance_for(self, destination: str) -> dict[str, str]:
        chunk_id = self.destinations.get(destination)
        if chunk_id is None:
            return {}
        return dict(self.proposal_meta.get(chunk_id, {"proposal": chunk_id}))

    def _begin_window(self, now: float) -> None:
        if now - self._window_start >= DENIAL_EVENT_WINDOW_SECONDS:
            self._window_start = now
            self._window_denials = 0

    def _denial_capacity(self, now: float) -> bool:
        self._begin_window(now)
        return self._window_denials < MAX_DENIAL_EVENTS_PER_WINDOW

    def process_denial(self, denial: PolicyDenial, now: float) -> PolicyEvent | None:
        self._begin_window(now)
        if self._window_denials >= MAX_DENIAL_EVENTS_PER_WINDOW:
            return None
        key = f"{denial.layer}:{denial.destination}"
        last = self.denial_posted.get(key)
        if last is not None and now - last < DENIAL_DEDUP_SECONDS:
            return None
        self._window_denials += 1
        self.denial_posted[key] = now
        return PolicyEvent(
            status="denied",
            destination=denial.destination,
            reason=denial.reason,
            rule=denial.rule,
            provenance=self.provenance_for(denial.destination),
        )

    def process_proposal(
        self, record: PolicyProposalRecord, now: float
    ) -> ProposalOutcome:
        """Turn one proposal record into the conversation-visible outcome."""
        status = record.status
        if status == "pending" and self._proposal_stale(record, now):
            status = "expired"
        previous = self.chunk_status.get(record.chunk_id)
        if previous == status:
            # A committed widening needs no re-offer; one the commit still
            # holds at the boundary does, across cycles and restarts alike.
            if (
                status == "approved"
                and record.review_token
                and not self.is_applied(record.chunk_id)
            ):
                return ProposalOutcome(widen_pending=True)
            return ProposalOutcome()
        self.chunk_status[record.chunk_id] = status
        self.link_proposal(record)

        if status == "pending":
            outcome = ProposalOutcome()
            if not self._denial_seen(record.destination) and self._denial_capacity(now):
                outcome.event = PolicyEvent(
                    status="denied",
                    destination=record.destination,
                    reason=record.rationale or "policy_denied",
                    rule=record.rule_name,
                    provenance=record.provenance(),
                )
                self.denial_posted[f"pending:{record.destination}"] = now
                self._window_denials += 1
            return outcome

        if status == "expired":
            return self._expired_outcome(record)
        if status == "approved":
            return ProposalOutcome(
                event=PolicyEvent(
                    status="approved",
                    destination=record.destination,
                    reason="approved by operator",
                    rule=record.rule_name,
                    provenance=record.provenance(),
                ),
                widen_pending=bool(record.review_token),
            )
        if status == "rejected":
            reason = record.rejection_reason or "rejected by operator"
            return ProposalOutcome(
                event=PolicyEvent(
                    status="rejected",
                    destination=record.destination,
                    reason=reason,
                    rule=record.rule_name,
                    provenance=record.provenance(),
                ),
                notice=(
                    f"Governance: the approval request for {record.destination} "
                    f"was rejected ({reason}). Narrow the next request to the "
                    "allowed destinations."
                ),
            )
        return ProposalOutcome()

    def _proposal_stale(self, record: PolicyProposalRecord, now: float) -> bool:
        return (
            record.created_epoch is not None
            and now - record.created_epoch >= PROPOSAL_TTL_SECONDS
        )

    def _expired_outcome(self, record: PolicyProposalRecord) -> ProposalOutcome:
        """A pending request past its TTL expires as denied (R8)."""
        self.chunk_status[record.chunk_id] = "expired"
        return ProposalOutcome(
            event=PolicyEvent(
                status="expired",
                destination=record.destination,
                reason="proposal_expired",
                rule=record.rule_name,
                provenance=record.provenance(),
            ),
            notice=(
                f"Governance: the approval request for {record.destination} "
                "was not answered in time, so the destination stays denied."
            ),
        )

    def widen_decided(self, chunk_id: str) -> None:
        self.applied.add(chunk_id)

    def is_applied(self, chunk_id: str) -> bool:
        return chunk_id in self.applied

    def revert_denial(self, denial: PolicyDenial) -> None:
        """Return an undelivered denial's marks to the pool."""
        key = f"{denial.layer}:{denial.destination}"
        if self.denial_posted.pop(key, None) is not None:
            self._window_denials = max(0, self._window_denials - 1)

    def revert_status(self, chunk_id: str, previous: str | None) -> None:
        """Return an undelivered decision to the unannounced state."""
        if previous is None:
            self.chunk_status.pop(chunk_id, None)
        else:
            self.chunk_status[chunk_id] = previous

    def _denial_seen(self, destination: str) -> bool:
        suffix = f":{destination}"
        return any(key.endswith(suffix) for key in self.denial_posted)


def policy_event_url(host: str, conversation_id: UUID) -> str:
    """The append route the conversation's own server serves.

    The id travels in its undashed form, as the sandbox names its conversation
    directory. A server without this route answers 405: ``GET
    /events/{event_id}`` matches the path as a partial route, so a missing
    append route does not surface as a 404.
    """
    return f"{host}/api/conversations/{conversation_id.hex}/events/_policy"


class PolicyEventWatcher:
    """Watch the gateway records of one governed workspace.

    Each cycle reads the sandbox log (new lines since the cursor) and the
    gateway's proposal records for every live governed conversation. The
    ledger decides what to produce; the watcher performs the I/O: policy
    events go to the conversation's own agent server, notices go into the
    sandbox file the governance hook reads, and widening commits wait for a
    turn boundary. The same cycle also pulls each conversation's record into
    the outer catalog when a turn reaches a terminal state, so the catalog
    does not stay ``running`` after a run.
    """

    def __init__(
        self,
        gateway: Any,
        registry: Any,
        *,
        interval: float = 5.0,
        sync_retry_interval: float = 60.0,
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.interval = interval
        self.sync_retry_interval = sync_retry_interval
        self._ledgers: dict[UUID, PolicyLedger] = {}
        self._primed: set[UUID] = set()
        # Record identity (status plus updated_at) of the last confirmed sync.
        self._last_synced_identity: dict[UUID, str] = {}
        # Last failed sync attempt, so a permanently incomplete copy backs off.
        self._sync_attempts: dict[UUID, float] = {}
        self._started = time.time()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Policy watch cycle failed")

    def _ledger_for(self, conversation_id: UUID) -> PolicyLedger:
        ledger = self._ledgers.get(conversation_id)
        if ledger is None:
            ledger = PolicyLedger(self.registry.load_policy_state(conversation_id))
            self._ledgers[conversation_id] = ledger
        return ledger

    async def cycle(self) -> None:
        now = time.time()
        live = self.registry.live_governed_conversations()
        for conversation_id, container in live:
            try:
                await self._watch_one(conversation_id, container, now)
            except Exception:
                logger.exception("Policy watch failed for %s", conversation_id)
        watched = {conversation_id for conversation_id, _ in live}
        for conversation_id in (
            set(self._last_synced_identity)
            | set(self._ledgers)
            | self._primed
            | set(self._sync_attempts)
        ) - watched:
            self._last_synced_identity.pop(conversation_id, None)
            self._ledgers.pop(conversation_id, None)
            self._primed.discard(conversation_id)
            self._sync_attempts.pop(conversation_id, None)

    async def _watch_one(
        self,
        conversation_id: UUID,
        container: ConversationContainer,
        now: float,
    ) -> None:
        ledger = self._ledger_for(conversation_id)
        sandbox = self.registry.sandbox_name(conversation_id)
        if not await asyncio.to_thread(container.is_running):
            return

        snapshot = await self._conversation_snapshot(conversation_id, container)
        status = snapshot[0] if snapshot is not None else None
        if snapshot is not None:
            await self._sync_terminal_record(
                conversation_id, snapshot[0], snapshot[1], now
            )

        chunks = await asyncio.to_thread(self.gateway.draft_chunks, sandbox) or []
        records = [
            record
            for record in (proposal_from_chunk(chunk) for chunk in chunks)
            if record is not None
        ]
        if conversation_id not in self._primed:
            # Decisions the gateway made before this watch started were
            # already delivered in the process that saw them.
            ledger.prime(records, cutoff=self._started)
            self._primed.add(conversation_id)
        for record in records:
            if record.status == "pending":
                ledger.link_proposal(record)

        log_lines = (
            await asyncio.to_thread(self.gateway.logs, sandbox, ledger.log_cursor) or []
        )
        widening: list[PolicyProposalRecord] = []
        delivered_notices: list[str] = []
        for line in log_lines:
            denial = parse_denial(str(line.message), dict(line.fields))
            if denial is None:
                continue
            event = ledger.process_denial(denial, now)
            if event is None:
                continue
            posted = await self._post_event(conversation_id, container, event)
            if not posted:
                # A denied delivery keeps its marks unspent for the next cycle.
                ledger.revert_denial(denial)
        for record in records:
            previous = ledger.chunk_status.get(record.chunk_id)
            outcome = ledger.process_proposal(record, now)
            if outcome.event is not None:
                posted = await self._post_event(
                    conversation_id, container, outcome.event
                )
                if not posted:
                    ledger.revert_status(record.chunk_id, previous)
                    continue
            if outcome.notice is not None:
                delivered_notices.append(outcome.notice)
            if (
                outcome.widen_pending
                and not ledger.is_applied(record.chunk_id)
                and record not in widening
            ):
                widening.append(record)
        if delivered_notices:
            await asyncio.to_thread(
                self.gateway.append_notices,
                sandbox,
                "\n".join(delivered_notices) + "\n",
            )
        for record in widening:
            if await self._widen_at_boundary(
                conversation_id, container, record, status
            ):
                ledger.widen_decided(record.chunk_id)

        if log_lines:
            newest = max(
                (
                    float(line.event_time.ToSeconds())
                    for line in log_lines
                    if hasattr(line.event_time, "ToSeconds")
                ),
                default=0.0,
            )
            if ledger.log_cursor is None or newest > ledger.log_cursor:
                ledger.log_cursor = newest
        self.registry.save_policy_state(conversation_id, ledger.to_state())

    async def _post_event(
        self,
        conversation_id: UUID,
        container: ConversationContainer,
        event: PolicyEvent,
    ) -> bool:
        """Deliver one event; ``False`` leaves the decision undelivered."""
        release = self.registry.acquire_lease(conversation_id)
        try:
            async with self._http_client(container) as client:
                response = await client.post(
                    policy_event_url(container.host, conversation_id),
                    headers={"X-Session-API-Key": container.api_key},
                    json=event.model_dump(mode="json"),
                )
        except Exception:
            logger.warning("Could not deliver a policy event to %s", conversation_id)
            return False
        finally:
            release()
        if response.is_error:
            logger.warning(
                "Policy event for %s was not accepted: %s",
                conversation_id,
                response.status_code,
            )
            return False
        return True

    def _http_client(self, container: ConversationContainer) -> Any:
        """One HTTP client for the conversation's own agent server."""
        import httpx

        return httpx.AsyncClient(verify=container.ca_bundle or True)

    async def _widen_at_boundary(
        self,
        conversation_id: UUID,
        container: ConversationContainer,
        record: PolicyProposalRecord,
        status: ConversationExecutionStatus | None,
    ) -> bool:
        """Commit the widening only while no agent run holds the turn."""
        if status is None:
            logger.warning("Could not read the conversation state; not widening")
            return False
        if status == ConversationExecutionStatus.RUNNING:
            return False
        # The cycle's snapshot can predate a turn that started while the
        # policy records were processed; re-read immediately before the
        # commit, which is the turn-boundary gate.
        if await self._conversation_is_running(conversation_id, container):
            return False
        try:
            await asyncio.to_thread(
                self.gateway.commit_approved_chunk,
                self.registry.sandbox_name(conversation_id),
                record.chunk_id,
                record.review_token,
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.ALREADY_EXISTS:
                return True  # The gateway already committed this chunk.
            logger.warning("Widening commit for chunk %s failed", record.chunk_id)
            return False
        except Exception:
            logger.warning("Widening commit for chunk %s failed", record.chunk_id)
            return False
        return True

    async def _conversation_snapshot(
        self,
        conversation_id: UUID,
        container: ConversationContainer,
    ) -> tuple[ConversationExecutionStatus, str | None] | None:
        """Read the conversation's live status and record identity.

        ``updated_at`` changes with conversation events, so it identifies the
        turn that produced the current record. It is None when the payload
        does not carry the field.
        """
        try:
            async with self._http_client(container) as client:
                response = await client.get(
                    f"{container.host}/api/conversations/{conversation_id.hex}",
                    headers={"X-Session-API-Key": container.api_key},
                )
        except Exception:
            logger.debug(
                "Could not read the conversation state of %s",
                conversation_id,
                exc_info=True,
            )
            return None
        if response.is_error:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            status = ConversationExecutionStatus(payload.get("execution_status"))
        except ValueError:
            return None
        updated_at = payload.get("updated_at")
        return status, None if updated_at is None else str(updated_at)

    async def _sync_terminal_record(
        self,
        conversation_id: UUID,
        status: ConversationExecutionStatus,
        updated_at: str | None,
        now: float,
    ) -> None:
        """Pull the runtime record once a turn has ended.

        The outer catalog copies the sandbox record only at request
        boundaries. A client that keeps its event stream open crosses none at
        the end of a turn, so the row would stay ``running``. Deduplicate on
        the record identity, not on the status: a second turn can start and
        finish between two polls, so both observations read ``finished``. When
        the payload has no ``updated_at``, the identity falls back to the
        status alone, so such a payload can still miss a quick second turn.
        Advance the identity only after a confirmed copy, so a failed or
        partial copy is retried after ``sync_retry_interval``.
        """
        if not status.is_terminal():
            return
        identity = (
            f"{status.value}:{updated_at}" if updated_at is not None else status.value
        )
        if self._last_synced_identity.get(conversation_id) == identity:
            return
        last_attempt = self._sync_attempts.get(conversation_id, 0.0)
        if now - last_attempt < self.sync_retry_interval:
            return
        self._sync_attempts[conversation_id] = now
        try:
            synced = await self.registry.sync_conversation_record(conversation_id)
        except Exception:
            # A catalog refresh must not stop the policy watch; the next
            # attempt waits for the retry interval.
            logger.warning(
                "Could not refresh the catalog record of %s",
                conversation_id,
                exc_info=True,
            )
            return
        if synced:
            self._last_synced_identity[conversation_id] = identity
            self._sync_attempts.pop(conversation_id, None)

    async def _conversation_is_running(
        self,
        conversation_id: UUID,
        container: ConversationContainer,
    ) -> bool:
        snapshot = await self._conversation_snapshot(conversation_id, container)
        if snapshot is None:
            logger.warning(
                "Could not read the conversation state; treating it as active"
            )
            return True  # Defensive fallback: an unknown state must never widen.
        return snapshot[0] == ConversationExecutionStatus.RUNNING
