from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

from openhands.sdk.conversation.types import TraceMetadataValue
from openhands.sdk.logger import get_logger
from openhands.sdk.observability.laminar import (
    default_observability_span_name_from_env,
    observability_metadata_from_env,
    observability_parent_span_context_from_env,
    observability_tags_from_env,
)
from openhands.sdk.workspace.bridge_tags import (
    automation_bridge_tags,
    bridge_repo_tags,
)


logger = get_logger(__name__)

_AUTOMATION_ID_KEY = "automation.id"
_AUTOMATION_NAME_KEY = "automation.name"
_AUTOMATION_RUN_ID_KEY = "automation.run_id"
_AUTOMATION_TRIGGER_SOURCE_KEY = "automation.trigger_source"
_AUTOMATION_REPO_KEY = "automation.repository"
_AUTOMATION_REF_KEY = "automation.ref"
_AUTOMATION_PROVIDER_KEY = "automation.provider"
_DEFAULT_AUTOMATION_SPAN_NAME = "automation.conversation"

# The conversation tags an automation launch sets; the runtime reads them back.
AUTOMATION_TRIGGER_TAG = "automationtrigger"
AUTOMATION_ID_TAG = "automationid"
AUTOMATION_RUN_ID_TAG = "automationrunid"
AUTOMATION_TAG_KEYS = (
    AUTOMATION_TRIGGER_TAG,
    AUTOMATION_ID_TAG,
    AUTOMATION_RUN_ID_TAG,
)


class AutomationObservabilityContext(TypedDict):
    observability_metadata: NotRequired[dict[str, TraceMetadataValue]]
    observability_tags: NotRequired[list[str]]
    observability_span_name: NotRequired[str]
    observability_parent_span_context: NotRequired[str]


class AutomationConversationKwargs(AutomationObservabilityContext, total=False):
    tags: dict[str, str]


def _merge_metadata(
    metadata: Mapping[str, TraceMetadataValue] | None,
) -> dict[str, TraceMetadataValue]:
    merged = observability_metadata_from_env()
    if metadata:
        merged.update(dict(metadata))
    return merged


def _merge_tags(tags: Sequence[str] | None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for tag in [*observability_tags_from_env(), *(tags or [])]:
        tag = tag.strip()
        if tag and tag not in seen:
            seen.add(tag)
            merged.append(tag)
    return merged


def _metadata_str(metadata: Mapping[str, TraceMetadataValue], key: str) -> str | None:
    value = metadata.get(key)
    if value is None or isinstance(value, bool | list):
        return None
    return str(value)


def automation_observability_context(
    *,
    metadata: Mapping[str, TraceMetadataValue] | None = None,
    tags: Sequence[str] | None = None,
    span_name: str | None = None,
    parent_span_context: str | None = None,
) -> AutomationObservabilityContext:
    """Collect generic automation observability fields for SDK conversations.

    The automation service owns automation-specific values and supplies them via
    generic ``OPENHANDS_OBSERVABILITY_*`` environment variables. This helper
    collects those generic values and merges caller-provided additions, so custom
    automation scripts do not need to duplicate the boilerplate for every
    ``Conversation(...)`` call.
    """
    context: AutomationObservabilityContext = {}

    merged_metadata = _merge_metadata(metadata)
    if merged_metadata:
        context["observability_metadata"] = merged_metadata

    merged_tags = _merge_tags(tags)
    if merged_tags:
        context["observability_tags"] = merged_tags

    resolved_span_name = span_name or default_observability_span_name_from_env()
    if resolved_span_name:
        context["observability_span_name"] = resolved_span_name

    resolved_parent_context = (
        parent_span_context or observability_parent_span_context_from_env()
    )
    if resolved_parent_context:
        context["observability_parent_span_context"] = resolved_parent_context

    return context


def automation_conversation_tags(
    metadata: Mapping[str, TraceMetadataValue] | None = None,
) -> dict[str, str]:
    """Build product conversation tags from automation observability metadata.

    The bridge family rides the result: repository provenance from the
    metadata (or from ``AUTOMATION_EVENT_PAYLOAD`` when the metadata leaves
    it open) and the ownership/archive marks beside the run id.
    """
    merged_metadata = _merge_metadata(metadata)
    tags: dict[str, str] = {}

    automation_id = _metadata_str(merged_metadata, _AUTOMATION_ID_KEY)
    automation_name = _metadata_str(merged_metadata, _AUTOMATION_NAME_KEY)
    automation_run_id = _metadata_str(merged_metadata, _AUTOMATION_RUN_ID_KEY)
    trigger_source = _metadata_str(merged_metadata, _AUTOMATION_TRIGGER_SOURCE_KEY)

    if automation_id or automation_name or automation_run_id or trigger_source:
        tags["automationtrigger"] = "automation"
    if automation_id:
        tags["automationid"] = automation_id
    if automation_name:
        tags["automationname"] = automation_name
    if automation_run_id:
        tags["automationrunid"] = automation_run_id

    # The event payload backfills provenance the metadata leaves open.
    repository = _metadata_str(merged_metadata, _AUTOMATION_REPO_KEY)
    if repository:
        tags.update(
            bridge_repo_tags(
                repository,
                _metadata_str(merged_metadata, _AUTOMATION_REF_KEY),
                _metadata_str(merged_metadata, _AUTOMATION_PROVIDER_KEY),
            )
        )
    return automation_bridge_tags(tags)


def automation_conversation_kwargs(
    *,
    metadata: Mapping[str, TraceMetadataValue] | None = None,
    observability_tags: Sequence[str] | None = None,
    conversation_tags: Mapping[str, str] | None = None,
    span_name: str | None = None,
    parent_span_context: str | None = None,
    include_conversation_tags: bool = True,
    repository: str | None = None,
    ref: str | None = None,
    provider: str | None = None,
) -> AutomationConversationKwargs:
    """Return kwargs to pass into ``Conversation(...)`` from an automation script.

    Example:
        ``Conversation(..., **automation_conversation_kwargs())``

    A run that works with a repository can pass its repository (and the ref it
    branches from) explicitly, so the conversation carries the bridge
    ``bridgerepo``/``bridgebase`` tags without payload plumbing. Explicit
    arguments override tags derived from metadata, and caller
    ``conversation_tags`` win last.
    """
    kwargs: AutomationConversationKwargs = {}
    resolved_span_name = (
        span_name
        or default_observability_span_name_from_env()
        or _DEFAULT_AUTOMATION_SPAN_NAME
    )
    context = automation_observability_context(
        metadata=metadata,
        tags=observability_tags,
        span_name=resolved_span_name,
        parent_span_context=parent_span_context,
    )
    if context_metadata := context.get("observability_metadata"):
        kwargs["observability_metadata"] = context_metadata
    if context_tags := context.get("observability_tags"):
        kwargs["observability_tags"] = context_tags
    if context_span_name := context.get("observability_span_name"):
        kwargs["observability_span_name"] = context_span_name
    if context_parent := context.get("observability_parent_span_context"):
        kwargs["observability_parent_span_context"] = context_parent

    if include_conversation_tags:
        merged_conversation_tags = automation_conversation_tags(metadata)
        if repository:
            merged_conversation_tags.update(bridge_repo_tags(repository, ref, provider))
        if conversation_tags:
            merged_conversation_tags.update(dict(conversation_tags))
        if merged_conversation_tags:
            kwargs["tags"] = merged_conversation_tags

    return kwargs


def automation_observability_headers(
    *,
    metadata: Mapping[str, TraceMetadataValue] | None = None,
    tags: Sequence[str] | None = None,
    span_name: str | None = None,
    parent_span_context: str | None = None,
) -> dict[str, str]:
    """Return observability headers for direct agent-server API calls."""
    context = automation_observability_context(
        metadata=metadata,
        tags=tags,
        span_name=span_name,
        parent_span_context=parent_span_context,
    )
    headers: dict[str, str] = {}
    if context_metadata := context.get("observability_metadata"):
        headers["X-OpenHands-Observability-Metadata"] = json.dumps(
            context_metadata, separators=(",", ":")
        )
    if context_tags := context.get("observability_tags"):
        headers["X-OpenHands-Observability-Tags"] = ",".join(context_tags)
    if context_span_name := context.get("observability_span_name"):
        headers["X-OpenHands-Observability-Span-Name"] = context_span_name
    if context_parent := context.get("observability_parent_span_context"):
        headers["X-OpenHands-Observability-Parent-Span-Context"] = context_parent

    try:
        from opentelemetry.propagate import inject

        inject(headers)
    except Exception:
        logger.debug(
            "Failed to inject OpenTelemetry propagation headers", exc_info=True
        )
    return headers


__all__ = [
    "AUTOMATION_ID_TAG",
    "AUTOMATION_RUN_ID_TAG",
    "AUTOMATION_TAG_KEYS",
    "AUTOMATION_TRIGGER_TAG",
    "AutomationConversationKwargs",
    "AutomationObservabilityContext",
    "automation_conversation_kwargs",
    "automation_conversation_tags",
    "automation_observability_context",
    "automation_observability_headers",
]
