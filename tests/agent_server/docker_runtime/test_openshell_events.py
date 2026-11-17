from openhands.agent_server.docker_runtime.openshell.events import (
    GOVERNANCE_CONTEXT,
    PolicyDenial,
    denied_event,
    parse_denial,
)


def test_parse_network_refusal():
    denial = parse_denial(
        "NET:REFUSE DENIED example.com [reason:policy_dns_ineligible]"
    )

    assert denial == PolicyDenial(
        layer="network",
        destination="example.com",
        reason="policy_dns_ineligible",
    )


def test_parse_network_connect_denial():
    denial = parse_denial(
        "NET:OPEN  DENIED /usr/bin/curl(0) -> example.com:443 "
        "[reason:transparent_tcp_policy_denied]"
    )

    assert denial == PolicyDenial(
        layer="network",
        destination="example.com:443",
        reason="transparent_tcp_policy_denied",
    )


def test_parse_http_denial_with_policy_and_rule():
    denial = parse_denial(
        'HTTP:POST DENIED  POST https://litellm.example/v1/other '
        '[policy:_provider_litellm engine:l7 rule "POST /other"]'
    )

    assert denial == PolicyDenial(
        layer="http",
        destination="https://litellm.example/v1/other",
        reason="_provider_litellm",
        rule="POST /other",
    )


def test_parse_ignores_an_allowed_line():
    assert parse_denial("HTTP:POST ALLOWED POST https://example/v1/x") is None


def test_summary_names_the_destination_and_reason():
    denial = PolicyDenial(
        layer="http",
        destination="https://example/v1/x",
        reason="canvas_ingress",
        rule="GET /",
    )

    assert "https://example/v1/x" in denial.summary
    assert "canvas_ingress" in denial.summary
    assert "GET /" in denial.summary


def test_governance_context_states_default_deny_and_one_retry():
    assert "denied by default" in GOVERNANCE_CONTEXT
    assert "one retry" in GOVERNANCE_CONTEXT


def test_denied_event_carries_the_decision():
    denial = parse_denial(
        "HTTP:GET DENIED  GET https://example/v1/x "
        "[policy:canvas_ingress engine:l7]"
    )

    event = denied_event(denial)

    assert event.kind == "PolicyEvent"
    assert event.status == "denied"
    assert event.destination == "https://example/v1/x"
    assert event.reason == "canvas_ingress"
