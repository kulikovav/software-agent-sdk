"""Cursor bridge conversation tags derived from automation run context.

cursor-assistant's bridge marks its own conversations with lowercase
``bridge*`` tag keys (see ``internal/backend/server/cloudagent/conversation.go``
in cursor-assistant). The Agent Canvas stamps the same family on the
conversations it creates; this module stamps it on automation-created
conversations, with the values the run context truly knows: the ownership
and archive marks beside the automation run id, and the repository
provenance when the run works with a repository.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any, Final
from urllib.parse import urlparse

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

BRIDGE_MODE_TAG_KEY = "bridgemode"
BRIDGE_MODE_TAG_VALUE = "cloudagent"
BRIDGE_ARCHIVED_TAG_KEY = "bridgearchived"
BRIDGE_ARCHIVED_TAG_VALUE = "false"
BRIDGE_REPO_TAG_KEY = "bridgerepo"
BRIDGE_BASE_TAG_KEY = "bridgebase"

_EVENT_PAYLOAD_ENV = "AUTOMATION_EVENT_PAYLOAD"

# The value grammar cursor-assistant's `ParseRepository` accepts: a
# lower-case multi-label host, safe path segments, and a git ref. A value
# outside it is omitted rather than stamped, because the bridge refuses a
# record whose tags it cannot parse and a wrong repository would be worse
# than an absent one. Keep in sync with the Agent Canvas
# `bridgeRepositoryPath`.
_BRIDGE_HOST_PATTERN = re.compile(
    r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$"
)
_BRIDGE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BRIDGE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# The conversation tags validator (``conversation/types.py``) raises on a
# value over this length at ConversationState construction, so one payload
# field could fail the run's conversation create. A stamp drops such a value
# instead, exactly like the Agent Canvas create path.
_TAG_VALUE_MAX_LENGTH = 256

# Default instance hosts for short `owner/name` repository references. Public
# hosts only: a self-hosted instance must arrive as a full URL or a hosted
# path. Keep in sync with the Agent Canvas `bridgeRepositoryPath`, which
# resolves the same shape.
_PROVIDER_HOSTS = {
    "github": "github.com",
    "gitlab": "gitlab.com",
    "bitbucket": "bitbucket.org",
    "azure_devops": "dev.azure.com",
}

# Marks "the caller has not read the payload"; None means the caller read it
# and it was absent or malformed, so the read must not happen twice.
_PAYLOAD_UNREAD: Final = object()


def read_automation_event_payload() -> dict[str, Any] | None:
    """Read the dispatcher-injected automation event payload.

    Returns None when the payload is absent, is not valid JSON, or is not an
    object; a malformed payload never fails the caller.
    """
    payload_str = os.environ.get(_EVENT_PAYLOAD_ENV)
    if not payload_str:
        return None
    try:
        payload = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        logger.error("Failed to parse AUTOMATION_EVENT_PAYLOAD")
        return None
    return payload if isinstance(payload, dict) else None


def _strip_git_suffix(text: str) -> str:
    return text[:-4] if text.lower().endswith(".git") else text


def _bridge_host(hostname: str) -> str | None:
    host = (hostname or "").strip().lower().rstrip(".")
    if host and _BRIDGE_HOST_PATTERN.match(host):
        return host
    return None


def bridge_ref(ref: str) -> str | None:
    """Normalize one ref to the branch value the bridge reads.

    Mirrors cursor-assistant's `validateRef`: a value its grammar refuses is
    omitted, because an invalid `bridgebase` would fail the bridge's own
    repository parse.
    """
    value = ref.strip().removeprefix("refs/heads/").strip()
    if not value or len(value) > _TAG_VALUE_MAX_LENGTH:
        return None
    if not _BRIDGE_REF_PATTERN.match(value):
        return None
    if ".." in value or "@{" in value or "//" in value:
        return None
    if value.endswith("/") or value.endswith(".lock"):
        return None
    if any(
        segment == "" or segment.startswith(".") or segment == "@"
        for segment in value.split("/")
    ):
        return None
    return value


def _hosted_repo_path(value: str, provider: str | None) -> str | None:
    """Normalize one repository reference to `host/owner/name`.

    A full URL or an already-hosted path keeps its own host. A short
    `owner/name` reference takes the provider's default host, so a reference
    without a resolvable host yields None.
    """
    text = value.strip()
    if "://" in text:
        parsed = urlparse(text)
        host = _bridge_host(parsed.hostname or "")
        segments = [segment for segment in parsed.path.split("/") if segment]
        if host and len(segments) >= 2:
            segments[-1] = _strip_git_suffix(segments[-1])
            if all(_BRIDGE_SEGMENT_PATTERN.match(segment) for segment in segments):
                return f"{host}/{'/'.join(segments)}"
        return None
    segments = [segment for segment in text.split("/") if segment]
    if segments:
        segments[-1] = _strip_git_suffix(segments[-1])
    if len(segments) >= 3:
        host = _bridge_host(segments[0])
        if host and all(
            _BRIDGE_SEGMENT_PATTERN.match(segment) for segment in segments[1:]
        ):
            return f"{host}/{'/'.join(segments[1:])}"
        return None
    if len(segments) == 2 and provider:
        host = _PROVIDER_HOSTS.get(provider.strip().lower())
        if host and all(_BRIDGE_SEGMENT_PATTERN.match(segment) for segment in segments):
            return f"{host}/{'/'.join(segments)}"
    return None


def _safe_repo_path(path: str | None) -> str | None:
    """Keep a repository path only when the tags validator would accept it."""
    if path and len(path) <= _TAG_VALUE_MAX_LENGTH and not _CONTROL_CHARS.search(path):
        return path
    return None


def bridge_repo_tags(
    repository: str,
    ref: str | None = None,
    provider: str | None = None,
) -> dict[str, str]:
    """Build the bridge repository tags for one known repository."""
    tags: dict[str, str] = {}
    path = _safe_repo_path(_hosted_repo_path(repository, provider))
    if path:
        tags[BRIDGE_REPO_TAG_KEY] = path
    if ref and (cleaned := bridge_ref(ref)):
        tags[BRIDGE_BASE_TAG_KEY] = cleaned
    return tags


def _first_str(entry: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _pull_request_base_ref(payload: Mapping[str, Any]) -> str | None:
    pull_request = payload.get("pull_request")
    if isinstance(pull_request, Mapping):
        base = pull_request.get("base")
        if isinstance(base, Mapping):
            return _first_str(base, "ref")
    return None


def _merge_request_base_ref(payload: Mapping[str, Any]) -> str | None:
    attributes = payload.get("object_attributes")
    if isinstance(attributes, Mapping):
        return _first_str(attributes, "target_branch")
    return None


def _repo_entry(entry: Any) -> tuple[str | None, str | None]:
    """Read one repository entry: a reference string or a mapping.

    The mapping covers an automation's configured repository (`url`, `ref`,
    `provider`), a webhook's repository object (`full_name` beside
    `html_url`/`clone_url`), and a webhook's project object
    (`path_with_namespace` beside `web_url`).
    """
    if isinstance(entry, str):
        return _hosted_repo_path(entry, None), None
    if not isinstance(entry, Mapping):
        return None, None
    provider = _first_str(entry, "provider")
    url = _first_str(entry, "url", "web_url", "html_url", "clone_url")
    path = _hosted_repo_path(url, provider) if url else None
    if not path:
        name = _first_str(entry, "full_name", "path_with_namespace")
        path = _hosted_repo_path(name, provider) if name else None
    return path, _first_str(entry, "ref", "default_branch")


def bridge_repo_tags_from_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    """Derive the bridge repository tags from one automation event payload."""
    tags: dict[str, str] = {}
    entry = payload.get("repo") or payload.get("repository") or payload.get("project")
    path, entry_ref = _repo_entry(entry)
    path = _safe_repo_path(path)
    if path:
        tags[BRIDGE_REPO_TAG_KEY] = path
    base_ref = (
        _first_str(payload, "base_ref")
        or _pull_request_base_ref(payload)
        or _merge_request_base_ref(payload)
        or _first_str(payload, "ref")
        or entry_ref
    )
    if base_ref and (cleaned := bridge_ref(base_ref)):
        tags[BRIDGE_BASE_TAG_KEY] = cleaned
    return tags


def automation_bridge_tags(
    tags: Mapping[str, str],
    event_payload: Mapping[str, Any] | None | object = _PAYLOAD_UNREAD,
) -> dict[str, str]:
    """Extend automation-derived conversation tags with the bridge family.

    The ownership and archive marks ride every conversation that carries an
    automation run id, and the repository provenance rides one whose payload
    names a repository. Existing tag values are never overwritten. Pass the
    payload the caller already read to avoid a second read; the default reads
    ``AUTOMATION_EVENT_PAYLOAD`` itself.
    """
    merged = dict(tags)
    payload = (
        event_payload
        if event_payload is not _PAYLOAD_UNREAD
        else read_automation_event_payload()
    )
    if isinstance(payload, Mapping):
        for key, value in bridge_repo_tags_from_payload(payload).items():
            merged.setdefault(key, value)
    if merged.get("automationrunid"):
        merged.setdefault(BRIDGE_MODE_TAG_KEY, BRIDGE_MODE_TAG_VALUE)
        merged.setdefault(BRIDGE_ARCHIVED_TAG_KEY, BRIDGE_ARCHIVED_TAG_VALUE)
    return merged


__all__ = [
    "BRIDGE_ARCHIVED_TAG_KEY",
    "BRIDGE_ARCHIVED_TAG_VALUE",
    "BRIDGE_BASE_TAG_KEY",
    "BRIDGE_MODE_TAG_KEY",
    "BRIDGE_MODE_TAG_VALUE",
    "BRIDGE_REPO_TAG_KEY",
    "automation_bridge_tags",
    "bridge_ref",
    "bridge_repo_tags",
    "bridge_repo_tags_from_payload",
    "read_automation_event_payload",
]
