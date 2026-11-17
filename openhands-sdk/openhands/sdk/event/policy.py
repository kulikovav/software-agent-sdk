from typing import Literal

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class PolicyEvent(Event):
    """A governance decision the sandbox boundary recorded.

    Emitted when the gateway denies egress or a human answers a widening
    request. The event is not sent to the LLM: the system prompt carries the
    standing rules, and this event keeps each decision visible in the event
    log and the UI.
    """

    source: SourceType = "environment"

    status: Literal["denied", "approved", "rejected", "expired"] = Field(
        description="Decision the boundary reached."
    )
    destination: str = Field(
        description="Host, URL, or pattern the decision names."
    )
    reason: str = Field(
        description="Policy, rule, or human reason for the decision."
    )
    rule: str | None = Field(
        default=None, description="Matched policy rule, if the boundary named one."
    )
    provenance: str | None = Field(
        default=None,
        description="Proposal id or label that ties the decision to its request.",
    )

    @property
    def visualize(self) -> Text:
        text = Text()
        text.append(f"Policy {self.status}\n", style="bold")
        text.append(f"{self.destination}: {self.reason}\n")
        if self.rule:
            text.append(f"rule: {self.rule}\n")
        if self.provenance:
            text.append(f"provenance: {self.provenance}\n")
        return text