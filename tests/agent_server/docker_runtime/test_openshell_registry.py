from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.docker_runtime.openshell.registry import (
    OpenShellConversationRegistry,
)


class FakeGateway:
    def __init__(self, existing=None):
        self.existing = existing
        self.created = []
        self.deleted = []
        self.stopped = []
        self.closed = False

    def find(self, conversation_hex):
        return self.existing

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(
            name=kwargs["name"],
            phase=2,
            service_urls={"": "https://default--oh-test.openshell.localhost:17670/"},
        )

    def get(self, name):
        return SimpleNamespace(name=name, phase=2, service_urls={})

    def stop(self, name):
        self.stopped.append(name)

    def delete(self, name):
        self.deleted.append(name)

    def close(self):
        self.closed = True


def make_registry(tmp_path, gateway):
    config = Config(
        conversation_runtime="openshell",
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspaces",
        openshell_tls_ca=str(tmp_path / "ca.crt"),
        secret_key=SecretStr("outer-key"),
    )
    return OpenShellConversationRegistry(config, gateway=gateway)


def test_build_container_creates_a_sandbox_for_the_service_url(
    tmp_path, monkeypatch
):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)

    container = registry._build_container(conversation_id)

    assert container.host == "https://default--oh-test.openshell.localhost:17670"
    assert container.api_key == identity.api_key.get_secret_value()
    assert container.ca_bundle == str(tmp_path / "ca.crt")

    create = gateway.created[0]
    assert create["name"] == f"oh-{conversation_id.hex}"
    assert create["labels"]["openhands.conversation"] == conversation_id.hex
    assert create["expose_port"] == 8000
    assert create["command"][:2] == ["openhands-agent-server", "--host"]
    assert create["environment"]["HOME"] == "/sandbox"
    assert create["environment"]["OH_CONVERSATIONS_PATH"] == "/sandbox/conversations"
    assert (
        create["environment"][V1_SESSION_API_KEY_ENV]
        == identity.api_key.get_secret_value()
    )


def test_build_container_adopts_an_existing_sandbox(tmp_path, monkeypatch):
    existing = SimpleNamespace(
        name="oh-existing",
        phase=2,
        service_urls={"": "https://existing.openshell.localhost:17670/"},
    )
    gateway = FakeGateway(existing=existing)
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    container = registry._build_container(conversation_id)

    assert gateway.created == []
    assert container.container_id == "oh-existing"
    assert container.host == "https://existing.openshell.localhost:17670"


def test_a_sandbox_without_a_service_url_is_rejected(tmp_path, monkeypatch):
    class NoUrlGateway(FakeGateway):
        def create(self, **kwargs):
            return SimpleNamespace(name=kwargs["name"], phase=2, service_urls={})

    gateway = NoUrlGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    with pytest.raises(RuntimeError, match="service URL"):
        registry._build_container(conversation_id)