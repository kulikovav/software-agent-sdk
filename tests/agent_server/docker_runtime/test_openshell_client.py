import grpc
import pytest
from openshell import SandboxRef, SandboxStatusRef

from openhands.agent_server.docker_runtime.openshell import client as client_module
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway


class FakeSdkClient:
    def __init__(self, url="https://default--demo.openshell.localhost:8080/"):
        self.url = url
        self.created = None
        self.closed = False

    def create(self, **kwargs):
        self.created = kwargs
        return SandboxRef(
            id="1",
            name=kwargs["name"],
            workspace="default",
            status=SandboxStatusRef(phase=1, current_policy_version=1),
            service_urls={"": self.url},
        )

    def close(self):
        self.closed = True


def make_gateway(monkeypatch, service_port, url=None):
    sdk = FakeSdkClient() if url is None else FakeSdkClient(url)
    monkeypatch.setattr(
        client_module.SandboxClient, "from_active_cluster", lambda **kwargs: sdk
    )
    return OpenShellGateway("default", service_port=service_port), sdk


def create(gateway):
    return gateway.create(
        name="demo",
        image="image:tag",
        command=["agent"],
        environment={},
        labels={},
        provider=None,
        expose_port=8000,
    )


def test_create_without_a_memory_limit_leaves_the_resources_struct_empty(
    monkeypatch,
):
    gateway, sdk = make_gateway(monkeypatch, 17670)

    create(gateway)

    template = sdk.created["spec"].template
    assert not template.resources


def test_create_passes_the_memory_limit_in_the_resources_struct(monkeypatch):
    gateway, sdk = make_gateway(monkeypatch, 17670)

    gateway.create(
        name="demo",
        image="image:tag",
        command=["agent"],
        environment={},
        labels={},
        provider=None,
        expose_port=8000,
        memory="6Gi",
    )

    template = sdk.created["spec"].template
    assert template.resources["limits"]["memory"] == "6Gi"


def test_create_keeps_the_service_url_and_rewrites_the_published_port(monkeypatch):
    gateway, sdk = make_gateway(monkeypatch, 17670)

    ref = create(gateway)

    assert sdk.created is not None
    exposures = sdk.created["service_exposures"]
    assert [exposure.target_port for exposure in exposures] == [8000]
    assert dict(ref.service_urls) == {
        "": "https://default--demo.openshell.localhost:17670"
    }


def test_create_without_a_published_port_keeps_the_sdk_url(monkeypatch):
    gateway, _ = make_gateway(monkeypatch, None)

    ref = create(gateway)

    assert dict(ref.service_urls) == {
        "": "https://default--demo.openshell.localhost:8080"
    }


def test_create_refuses_a_service_url_without_tls(monkeypatch):
    gateway, _ = make_gateway(
        monkeypatch, 17670, url="http://default--demo.openshell.localhost:8080/"
    )

    with pytest.raises(RuntimeError, match="without TLS"):
        create(gateway)


def test_create_adopts_a_sandbox_another_process_created(monkeypatch):
    gateway, sdk = make_gateway(monkeypatch, 17670)
    existing = SandboxRef(
        id="1",
        name="demo",
        workspace="default",
        status=SandboxStatusRef(phase=1, current_policy_version=1),
    )

    def conflict(**_kwargs):
        raise _AlreadyExists()

    monkeypatch.setattr(sdk, "create", conflict)
    monkeypatch.setattr(gateway, "get", lambda name: existing)

    ref = create(gateway)

    assert ref is existing


class _AlreadyExists(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.ALREADY_EXISTS
