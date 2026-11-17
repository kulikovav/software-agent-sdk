from openshell import SandboxRef, SandboxStatusRef

from openhands.agent_server.docker_runtime.openshell import client as client_module
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway


class FakeSdkClient:
    def __init__(self):
        self.created = None
        self.closed = False

    def create(self, **kwargs):
        self.created = kwargs
        return SandboxRef(
            id="1",
            name=kwargs["name"],
            workspace="default",
            status=SandboxStatusRef(phase=1, current_policy_version=1),
            service_urls={"": "https://default--demo.openshell.localhost:8080/"},
        )

    def wait_ready(self, name, *, workspace, timeout_seconds=300):
        return SandboxRef(
            id="1",
            name=name,
            workspace=workspace,
            status=SandboxStatusRef(phase=2, current_policy_version=1),
        )

    def close(self):
        self.closed = True


def make_gateway(monkeypatch, service_port):
    sdk = FakeSdkClient()
    monkeypatch.setattr(
        client_module.SandboxClient, "from_active_cluster", lambda **kwargs: sdk
    )
    return OpenShellGateway("default", service_port=service_port), sdk


def test_create_keeps_the_service_url_and_rewrites_the_published_port(monkeypatch):
    gateway, sdk = make_gateway(monkeypatch, 17670)

    ref = gateway.create(
        name="demo",
        image="image:tag",
        command=["agent"],
        environment={},
        labels={},
        provider=None,
        expose_port=8000,
    )

    exposures = sdk.created["service_exposures"]
    assert [exposure.target_port for exposure in exposures] == [8000]
    assert dict(ref.service_urls) == {
        "": "https://default--demo.openshell.localhost:17670"
    }


def test_create_without_a_published_port_keeps_the_sdk_url(monkeypatch):
    gateway, _ = make_gateway(monkeypatch, None)

    ref = gateway.create(
        name="demo",
        image="image:tag",
        command=["agent"],
        environment={},
        labels={},
        provider=None,
        expose_port=8000,
    )

    assert dict(ref.service_urls) == {
        "": "https://default--demo.openshell.localhost:8080"
    }