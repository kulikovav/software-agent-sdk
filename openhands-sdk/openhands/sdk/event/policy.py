from typing import Literal

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


PolicyDecision = Literal["denied", "approved", "rejected", "expired"]


class PolicyEvent(Event):
    """A governance decision the sandbox boundary recorded.

    Emitted when the gateway denies egress or when a human decision answers a
    widening request. The event is not sent to the LLM: the governance context
    in the system prompt carries the standing rules, and this event keeps each
    decision visible in the event log and the UI.
    """

    source: SourceType = "environment"

    status: PolicyDecision = Field(description="Decision the boundary reached.")
    destination: str = Field(
        description="Host, URL, pattern, or scope name the decision covers."
    )
    reason: str = Field(
        description="Policy, rule, expiry, or human reason for the decision."
    )
    rule: str | None = Field(
        default=None, description="Matched policy rule, if the boundary named one."
    )
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Provenance labels tying the decision to its request: the proposal "
            "record id, the requested scope, and the prover verdict."
        ),
    )

    @property
    def visualize(self) -> Text:
        text = Text()
        text.append(f"Policy {self.status}\n", style="bold")
        text.append(f"{self.destination}: {self.reason}\n")
        if self.rule:
            text.append(f"rule: {self.rule}\n")
        for name, value in self.provenance.items():
            text.append(f"{name}: {value}\n")
        return text
