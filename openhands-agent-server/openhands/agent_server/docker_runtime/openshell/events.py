"""Policy denials and governance context for governed conversations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from openhands.sdk.event.policy import PolicyEvent


GOVERNANCE_CONTEXT = (
    "This conversation runs inside a governed sandbox. Egress is denied by "
    "default; only the model endpoint is allowed. The workspace is ephemeral, "
    "and credentials exist only as placeholders that resolve outside the "
    "sandbox. A denied request names its destination and reason. Ask for an "
    "approval if the task needs another destination; one retry is allowed "
    "after a human approves the request."
)


@dataclass(frozen=True)
class PolicyDenial:
    """One denial record from the gateway's sandbox log."""

    layer: Literal["network", "http"]
    destination: str
    reason: str
    rule: str | None = None

    @property
    def summary(self) -> str:
        rule = f" (rule {self.rule})" if self.rule else ""
        return (
            f"{self.layer} access to {self.destination} was denied: {self.reason}{rule}"
        )


_DENIAL_FIELDS = re.compile(r"([a-z_]+):([^\]\s]+)")
# The gateway quotes some values, such as `rule "POST /v1"`, so a value can
# contain spaces and colons that the unquoted scan would misread.
_QUOTED_FIELDS = re.compile(r'([a-z_]+) "([^"]+)"')


def _fields(text: str) -> dict[str, str]:
    fields = dict(_DENIAL_FIELDS.findall(_QUOTED_FIELDS.sub("", text)))
    fields.update(dict(_QUOTED_FIELDS.findall(text)))
    return fields


def parse_denial(line: str) -> PolicyDenial | None:
    """Parse one sandbox log line, or return ``None`` for an unrelated line."""
    text = line.strip()
    if " DENIED " not in text:
        return None
    prefix, rest = text.split(" DENIED ", 1)
    rest = rest.strip()
    bracket = rest.find(" [")
    detail = rest[bracket:] if bracket >= 0 else ""
    body = rest[:bracket].strip() if bracket >= 0 else rest

    if prefix.startswith("NET:REFUSE"):
        layer: Literal["network", "http"] = "network"
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

    fields = _fields(detail)
    reason = fields.get("reason") or fields.get("policy") or "policy_denied"
    return PolicyDenial(
        layer=layer,
        destination=destination,
        reason=reason,
        rule=fields.get("rule"),
    )


def denied_event(denial: PolicyDenial) -> PolicyEvent:
    """The conversation-visible event for one denial."""
    return PolicyEvent(
        status="denied",
        destination=denial.destination,
        reason=denial.reason,
        rule=denial.rule,
    )
