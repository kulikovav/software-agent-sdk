from openhands.sdk.event import Event, PolicyEvent


def test_policy_event_serializes_with_its_kind():
    event = PolicyEvent(
        status="approved",
        destination="example.com",
        reason="human approval",
        provenance="proposal-1",
    )

    data = event.model_dump(mode="json")

    assert data["kind"] == "PolicyEvent"
    assert Event.model_validate(data) == event


def test_policy_event_visualizes_the_decision():
    event = PolicyEvent(
        status="denied",
        destination="example.com",
        reason="policy_dns_ineligible",
    )

    text = event.visualize.plain

    assert "denied" in text
    assert "example.com" in text
    assert "policy_dns_ineligible" in text
