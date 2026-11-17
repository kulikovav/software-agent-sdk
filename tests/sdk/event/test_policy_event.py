from openhands.sdk.event import Event, LLMConvertibleEvent, PolicyEvent


def test_policy_event_serializes_with_its_kind():
    event = PolicyEvent(
        status="approved",
        destination="example.com",
        reason="human approval",
        provenance={"proposal": "chunk-1", "scope": "https://example.com:443"},
    )

    data = event.model_dump(mode="json")

    assert data["kind"] == "PolicyEvent"
    assert Event.model_validate(data) == event


def test_policy_event_visualizes_the_decision():
    event = PolicyEvent(
        status="denied",
        destination="example.com",
        reason="policy_dns_ineligible",
        rule="POST /v1",
    )

    text = event.visualize.plain

    assert "denied" in text
    assert "example.com" in text
    assert "policy_dns_ineligible" in text
    assert "POST /v1" in text


def test_policy_event_never_reaches_the_llm():
    event = PolicyEvent(
        status="expired",
        destination="example.com",
        reason="proposal_expired",
    )

    assert not isinstance(event, LLMConvertibleEvent)
