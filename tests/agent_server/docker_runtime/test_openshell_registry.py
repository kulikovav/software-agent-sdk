import asyncio
from uuid import uuid4

import grpc
import pytest
from openshell import SandboxRef, SandboxStatusRef
from openshell._proto import openshell_pb2 as pb
from pydantic import SecretStr

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.openshell.registry import (
    OpenShellConversationRegistry,
)


def make_sandbox(name, phase=pb.SANDBOX_PHASE_READY, service_urls=None):
    return SandboxRef(
        id=name,
        name=name,
        workspace="default",
        status=SandboxStatusRef(phase=phase, current_policy_version=1),
        service_urls=service_urls or {},
    )


class FakeGateway(OpenShellGateway):
    def __init__(self, existing=None):
        self.service_port = None
        self.workspace = "default"
        self.existing = existing
        self.created = []
        self.deleted = []
        self.started = []
        self.stopped = []
        self.files = {}
        self.closed = False

    def get(self, name):
        if self.existing is not None and self.existing.name == name:
            return self.existing
        return None

    def find(self, selector):
        return None

    def create(self, **kwargs):
        self.created.append(kwargs)
        return make_sandbox(
            kwargs["name"],
            service_urls={"": "https://default--oh-test.openshell.localhost:17670/"},
        )

    def start(self, name):
        self.started.append(name)
        return make_sandbox(name)

    def stop(self, name):
        self.stopped.append(name)

    def delete(self, name):
        self.deleted.append(name)

    def read_file(self, name, path):
        return self.files.get(path)

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


def test_build_container_creates_a_sandbox_and_records_its_url(tmp_path, monkeypatch):
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

    assert registry.provisioning.load(conversation_id).service_url == container.host


def test_build_container_adopts_the_sandbox_by_name(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)
    created = registry._build_container(conversation_id)

    gateway.existing = make_sandbox(gateway.created[0]["name"])
    gateway.created.clear()
    adopted = registry._build_container(conversation_id)

    assert gateway.created == []
    assert adopted.container_id == created.container_id
    assert adopted.host == created.host


@pytest.mark.asyncio
async def test_concurrent_starts_create_one_sandbox(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    first, second = await asyncio.gather(
        registry.get_or_create(conversation_id),
        registry.get_or_create(conversation_id),
    )

    assert len(gateway.created) == 1
    assert first.container_id == second.container_id


def test_build_container_starts_a_stopped_sandbox(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(update={"service_url": "https://sandbox:17670"})
    )
    name = registry.sandbox_name(conversation_id)
    gateway.existing = make_sandbox(name, phase=pb.SANDBOX_PHASE_STOPPED)

    container = registry._build_container(conversation_id)

    assert gateway.started == [name]
    assert container.host == "https://sandbox:17670"


def test_a_sandbox_without_a_service_url_is_rejected_and_deleted(tmp_path, monkeypatch):
    class NoUrlGateway(FakeGateway):
        def create(self, **kwargs):
            self.created.append(kwargs)
            return make_sandbox(kwargs["name"])

    gateway = NoUrlGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    with pytest.raises(RuntimeError, match="service URL"):
        registry._build_container(conversation_id)

    assert gateway.deleted == [registry.sandbox_name(conversation_id)]


def test_a_transient_gateway_error_keeps_the_sandbox(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    def fail(container):
        raise grpc.RpcError()

    monkeypatch.setattr(registry, "_wait_until_ready", fail)

    with pytest.raises(grpc.RpcError):
        registry._build_container(conversation_id)

    assert gateway.deleted == []


def test_a_failed_adopted_start_keeps_the_sandbox(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(update={"service_url": "https://sandbox:17670"})
    )
    gateway.existing = make_sandbox(registry.sandbox_name(conversation_id))

    def fail(container):
        raise RuntimeError("Conversation container failed to become healthy in time")

    monkeypatch.setattr(registry, "_wait_until_ready", fail)

    with pytest.raises(RuntimeError, match="healthy"):
        registry._build_container(conversation_id)

    assert gateway.deleted == []


def test_copy_conversation_record_writes_the_outer_record(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    gateway.files[f"/sandbox/conversations/{conversation_id.hex}/meta.json"] = (
        '{"id": "conversation"}'
    )

    registry.copy_conversation_record(conversation_id)

    record = registry.conversation_dir(conversation_id) / "meta.json"
    assert record.read_text(encoding="utf-8") == '{"id": "conversation"}'


def test_delete_runtime_deletes_the_sandbox(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()

    registry.delete_runtime(conversation_id)

    assert gateway.deleted == [registry.sandbox_name(conversation_id)]
