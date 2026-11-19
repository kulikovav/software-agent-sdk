import asyncio
import json
from uuid import uuid4

import grpc
import pytest
from openshell import SandboxRef, SandboxStatusRef
from openshell._proto import openshell_pb2 as pb, sandbox_pb2
from pydantic import SecretStr

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.openshell.registry import (
    OpenShellContainer,
    OpenShellConversationRegistry,
)
from openhands.agent_server.docker_runtime.registry import RuntimeArchivedError
from openhands.agent_server.models import (
    ConversationRuntimeStatus,
    StartConversationRequest,
)
from openhands.sdk import LLM, Agent
from openhands.sdk.conversation.request import ConversationContainerOptions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.security.confirmation_policy import NeverConfirm
from openhands.sdk.workspace import LocalWorkspace


def make_sandbox(
    name, phase=pb.SANDBOX_PHASE_READY, service_urls=None, conversation_hex=""
):
    return SandboxRef(
        id=name,
        name=name,
        workspace="default",
        status=SandboxStatusRef(phase=phase, current_policy_version=1),
        labels={"openhands.conversation": conversation_hex} if conversation_hex else {},
        service_urls=service_urls or {},
    )


class FakeGateway(OpenShellGateway):
    def __init__(self, existing=None):
        self.service_port = None
        self.workspace = "default"
        self.existing = existing
        self.found: SandboxRef | None = None
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
        return self.found

    def create(self, **kwargs):
        self.created.append(kwargs)
        return make_sandbox(
            kwargs["name"],
            service_urls={"": "https://default--oh-test.openshell.localhost:17670/"},
            conversation_hex=kwargs["labels"]["openhands.conversation"],
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


def make_registry(tmp_path, gateway, policy: str | None = "governed-sandbox.yaml"):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir(exist_ok=True)
    (policy_dir / "governed-sandbox.yaml").write_text(
        """
version: 1
filesystem_policy:
  read_write:
    - /dev/pts
    - /sandbox
network_policies:
  test_rule:
    endpoints:
      - host: example.invalid
        port: 443
        protocol: rest
        enforcement: enforce
        access: full
        rules:
          - allow:
              method: "*"
              path: /v1/**
    binaries:
      - path: /usr/bin/curl
"""
    )
    (policy_dir / "wide-egress.yaml").write_text(
        """
version: 1
filesystem_policy:
  read_write:
    - /tmp
"""
    )
    config = Config(
        conversation_runtime="openshell",
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspaces",
        openshell_tls_ca=str(tmp_path / "ca.crt"),
        openshell_policy=policy,
        openshell_policy_dir=str(policy_dir),
        secret_key=SecretStr("outer-key"),
    )
    return OpenShellConversationRegistry(config, gateway=gateway)


def test_the_server_defaults_to_the_governed_sandbox_policy():
    assert Config().openshell_policy == "governed-sandbox.yaml"


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
    assert create["name"] == registry.sandbox_name(conversation_id)
    assert len(create["name"]) <= 19
    assert create["labels"]["openhands.conversation"] == conversation_id.hex
    assert create["expose_port"] == 8000
    policy = create["policy"]
    assert list(policy.filesystem.read_write) == ["/dev/pts", "/sandbox"]
    endpoint = policy.network_policies["test_rule"].endpoints[0]
    assert endpoint.enforcement == sandbox_pb2.NETWORK_ENFORCEMENT_MODE_ENFORCE
    assert endpoint.access == sandbox_pb2.NETWORK_ACCESS_PRESET_FULL
    # The image's frozen ``openhands-agent-server`` binary embeds upstream code,
    # so it cannot serve a route this fork installs beside it. Start the module.
    assert create["command"][:4] == ["python", "-m", "openhands.agent_server", "--host"]
    assert create["command"][-2:] == ["--port", "8000"]
    assert create["environment"]["HOME"] == "/sandbox"
    assert create["environment"]["OH_CONVERSATIONS_PATH"] == "/sandbox/conversations"
    assert (
        create["environment"][V1_SESSION_API_KEY_ENV]
        == identity.api_key.get_secret_value()
    )

    assert registry.provisioning.load(conversation_id).service_url == container.host


def test_a_conversation_can_select_another_policy(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(
            update={
                "container": ConversationContainerOptions(policy="wide-egress.yaml")
            }
        )
    )

    registry._build_container(conversation_id)

    policy = gateway.created[0]["policy"]
    assert list(policy.filesystem.read_write) == ["/tmp"]


def test_no_policy_is_sent_when_both_selections_are_unset(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway, policy=None)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    registry._build_container(conversation_id)

    assert gateway.created[0]["policy"] is None


def test_build_container_adopts_the_sandbox_by_name(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)
    created = registry._build_container(conversation_id)

    gateway.existing = make_sandbox(
        gateway.created[0]["name"], conversation_hex=conversation_id.hex
    )
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


def test_build_container_adopts_a_sandbox_found_by_label(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(update={"service_url": "https://sandbox:17670"})
    )
    gateway.found = make_sandbox("oh-legacy", conversation_hex=conversation_id.hex)

    container = registry._build_container(conversation_id)

    assert gateway.created == []
    assert container.container_id == "oh-legacy"
    assert container.host == "https://sandbox:17670"


def test_a_name_collision_is_refused(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)
    gateway.existing = make_sandbox(registry.sandbox_name(conversation_id))

    with pytest.raises(RuntimeError, match="collision"):
        registry._build_container(conversation_id)

    assert gateway.created == []


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
    gateway.existing = make_sandbox(
        name, phase=pb.SANDBOX_PHASE_STOPPED, conversation_hex=conversation_id.hex
    )

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
    gateway.existing = make_sandbox(
        registry.sandbox_name(conversation_id), conversation_hex=conversation_id.hex
    )

    def fail(container):
        raise RuntimeError("Conversation container failed to become healthy in time")

    monkeypatch.setattr(registry, "_wait_until_ready", fail)

    with pytest.raises(RuntimeError, match="healthy"):
        registry._build_container(conversation_id)

    assert gateway.deleted == []


@pytest.mark.asyncio
async def test_a_gone_sandbox_is_not_recreated(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(update={"service_url": "https://sandbox:17670"})
    )

    with pytest.raises(RuntimeArchivedError):
        await registry.get_or_create(conversation_id)

    assert gateway.created == []
    assert registry.is_archived(conversation_id)


def test_a_vanished_sandbox_is_archived_for_resume(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(update={"service_url": "https://sandbox:17670"})
    )

    with pytest.raises(RuntimeArchivedError):
        registry._build_container(conversation_id)

    info = registry.runtime_info(conversation_id)
    assert info.runtime_status == ConversationRuntimeStatus.MISSING
    assert info.can_resume is False


def test_a_provisioning_sandbox_is_not_a_failed_start(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    name = registry.sandbox_name(uuid4())
    handle = OpenShellContainer(
        host="https://sandbox:17670",
        api_key="key",
        container_id=name,
        gateway=gateway,
    )

    gateway.existing = make_sandbox(name, phase=pb.SANDBOX_PHASE_PROVISIONING)
    assert handle.startup_failed() is False

    gateway.existing = make_sandbox(name, phase=pb.SANDBOX_PHASE_ERROR)
    assert handle.startup_failed() is True

    gateway.existing = None
    assert handle.startup_failed() is True


def test_copy_conversation_record_writes_the_outer_record(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    base = f"/sandbox/conversations/{conversation_id.hex}"
    gateway.files[f"{base}/meta.json"] = '{"id": "conversation"}'
    gateway.files[f"{base}/base_state.json"] = '{"agent": {}}'

    registry.copy_conversation_record(conversation_id)

    record = registry.conversation_dir(conversation_id)
    assert (record / "meta.json").read_text(
        encoding="utf-8"
    ) == '{"id": "conversation"}'
    # The catalog builds the conversation from base_state.json.
    assert (record / "base_state.json").read_text(encoding="utf-8") == '{"agent": {}}'


def test_a_failed_created_start_clears_the_recorded_url(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    registry.provisioning.create(conversation_id)

    def not_ready(_container):
        raise RuntimeError("sandbox did not become ready")

    monkeypatch.setattr(registry, "_wait_until_ready", not_ready)
    with pytest.raises(RuntimeError, match="did not become ready"):
        registry._build_container(conversation_id)

    assert gateway.deleted == [registry.sandbox_name(conversation_id)]
    assert registry.provisioning.load(conversation_id).service_url is None

    # A retry recreates the sandbox instead of reading the failure as an
    # archived conversation.
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    container = registry._build_container(conversation_id)

    assert container.host.startswith("https://")
    assert len(gateway.created) == 2


def test_event_reads_are_not_served_from_the_host(tmp_path):
    registry = make_registry(tmp_path, FakeGateway())

    assert registry.serves_persisted_event_reads(uuid4()) is False


def test_governed_reads_go_through_the_conversation_proxy(tmp_path):
    """A persisted read route would answer from the catalog copy, not the sandbox."""
    from fastapi import APIRouter
    from fastapi.routing import APIRoute

    registry = make_registry(tmp_path, FakeGateway())
    router = APIRouter()

    registry.add_execution_routes(router)

    paths = {route.path for route in router.routes if isinstance(route, APIRoute)}
    assert "/conversations/{conversation_id}/{tail:path}" in paths
    assert "/conversations/{conversation_id}/events" not in paths
    assert "/conversations/{conversation_id}/events/{event_id}" not in paths


def test_refresh_catalog_record_replaces_a_stale_status(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    base = f"/sandbox/conversations/{conversation_id.hex}"
    gateway.files[f"{base}/meta.json"] = '{"id": "conversation"}'
    gateway.files[f"{base}/base_state.json"] = '{"execution_status": "finished"}'
    target = registry.conversation_dir(conversation_id) / "base_state.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"execution_status": "error"}', encoding="utf-8")

    registry.refresh_catalog_record(conversation_id)

    assert target.read_text(encoding="utf-8") == '{"execution_status": "finished"}'


def test_refresh_catalog_record_survives_a_gateway_failure(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)

    def gateway_down(name, path):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(gateway, "read_file", gateway_down)

    registry.refresh_catalog_record(uuid4())


@pytest.mark.asyncio
async def test_sync_conversation_record_updates_the_catalog_list_and_detail(tmp_path):
    """A finished turn must reach both the catalog list and the detail record."""
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversations_dir = registry.config.conversations_path
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = StartConversationRequest(
        agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace)),
        confirmation_policy=NeverConfirm(),
    )
    async with ConversationService(
        conversations_dir=conversations_dir, cipher=registry.provisioning.cipher
    ) as creator:
        info, _ = await creator.start_conversation(request)

    conversation_id = info.id
    outer_dir = conversations_dir / conversation_id.hex
    persisted = json.loads((outer_dir / "base_state.json").read_text())

    # The sandbox owns the live record; prepare both sides from the same one.
    base = f"/sandbox/conversations/{conversation_id.hex}"
    gateway.files[f"{base}/meta.json"] = (outer_dir / "meta.json").read_text()
    gateway.files[f"{base}/base_state.json"] = json.dumps(
        {**persisted, "execution_status": "finished"}
    )

    service = ConversationService(
        conversations_dir=conversations_dir,
        cipher=registry.provisioning.cipher,
    )
    registry.configure_service(service)
    async with service:
        # Recreate the stale catalog row the outer server held while the run
        # finished, after the service loaded it so startup recovery ignores it.
        (outer_dir / "base_state.json").write_text(
            json.dumps({**persisted, "execution_status": "running"})
        )
        stale_detail = await service.get_conversation(conversation_id)
        stale_page = await service.search_conversations()
        assert stale_detail is not None
        assert stale_detail.execution_status == ConversationExecutionStatus.RUNNING
        assert (
            next(
                item for item in stale_page.items if item.id == conversation_id
            ).execution_status
            == ConversationExecutionStatus.RUNNING
        )

        await registry.sync_conversation_record(conversation_id)

        detail = await service.get_conversation(conversation_id)
        page = await service.search_conversations()

    assert detail is not None
    assert detail.execution_status == ConversationExecutionStatus.FINISHED
    assert (
        next(item for item in page.items if item.id == conversation_id).execution_status
        == ConversationExecutionStatus.FINISHED
    )


def test_delete_runtime_deletes_the_sandbox(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()

    registry.delete_runtime(conversation_id)

    assert gateway.deleted == [registry.sandbox_name(conversation_id)]


@pytest.mark.asyncio
async def test_shutdown_stops_the_policy_watcher(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)

    watcher = registry._policy_watcher
    assert watcher is not None
    watcher.start()
    await registry.shutdown()

    assert registry._policy_watcher is None
    assert gateway.closed


def test_live_governed_conversations_lists_its_containers(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    container = OpenShellContainer(
        host="https://sandbox:17670",
        api_key="key",
        container_id="oh-x",
        gateway=gateway,
    )
    conversation_id = uuid4()
    registry._containers[conversation_id] = container

    live = registry.live_governed_conversations()

    assert live == [(conversation_id, container)]


def test_policy_state_is_persisted_for_a_restart(tmp_path):
    registry = make_registry(tmp_path, FakeGateway())
    conversation_id = uuid4()

    assert registry.load_policy_state(conversation_id) is None

    registry.save_policy_state(
        conversation_id, {"chunk_status": {"chunk-1": "approved"}}
    )

    # Beside the manifest: the same control-file lifecycle as the identity.
    assert (
        registry.provisioning.policy_state_path(conversation_id)
        == registry.provisioning.control_root / f"{conversation_id.hex}.policy.json"
    )
    assert registry.load_policy_state(conversation_id) == {
        "chunk_status": {"chunk-1": "approved"}
    }
