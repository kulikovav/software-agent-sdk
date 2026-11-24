"""The memory credential a governed conversation receives."""

import base64
import json
import os
import sys
import time
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from openshell import SandboxRef, SandboxStatusRef
from openshell._proto import openshell_pb2 as pb
from pydantic import SecretStr

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.openshell.events import (
    MEMORY_API_URL_ENV,
    MEMORY_TOKEN_ENV,
)
from openhands.agent_server.docker_runtime.openshell.registry import (
    OpenShellConversationRegistry,
)
from openhands.agent_server.docker_runtime.provisioning import MemoryScope
from openhands.agent_server.docker_runtime.routers import _prepare_forward
from openhands.sdk.launch import AgentLaunchError


HOST_SECRET_KEY = "host-secret-key"
OUTER_SESSION_KEY = "outer-session-key"
MEMORY_TOKEN_SECRET = "memory-token-secret"
MEMORY_API_URL = "https://memory.example.test"


def make_sandbox(
    name: str,
    phase: int = pb.SANDBOX_PHASE_READY,
    conversation_hex: str = "",
    service_url: str | None = None,
) -> SandboxRef:
    return SandboxRef(
        id=name,
        name=name,
        workspace="default",
        status=SandboxStatusRef(phase=phase, current_policy_version=1),
        labels={"openhands.conversation": conversation_hex} if conversation_hex else {},
        service_urls={"": service_url} if service_url else {},
    )


class FakeGateway(OpenShellGateway):
    def __init__(self) -> None:
        self.service_port: int | None = None
        self.workspace = "default"
        self.existing: SandboxRef | None = None
        self.created: list[dict] = []
        self.started: list[str] = []
        self.updated: list[tuple[str, dict[str, str]]] = []
        self.deleted: list[str] = []

    def get(self, name: str) -> SandboxRef | None:
        if self.existing is not None and self.existing.name == name:
            return self.existing
        return None

    def find(self, selector: str) -> SandboxRef | None:
        return None

    def create(self, **kwargs) -> SandboxRef:
        self.created.append(kwargs)
        return make_sandbox(
            kwargs["name"],
            conversation_hex=kwargs["labels"]["openhands.conversation"],
            service_url="https://default--oh-test.openshell.localhost:17670/",
        )

    def start(self, name: str) -> SandboxRef:
        self.started.append(name)
        return make_sandbox(name)

    def update_environment(self, name: str, environment) -> None:
        self.updated.append((name, dict(environment)))

    def delete(self, name: str) -> None:
        self.deleted.append(name)

    def close(self) -> None:
        pass


def make_registry(tmp_path, gateway: FakeGateway) -> OpenShellConversationRegistry:
    config = Config(
        conversation_runtime="openshell",
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspaces",
        openshell_policy=None,
        secret_key=SecretStr(HOST_SECRET_KEY),
        session_api_keys=[OUTER_SESSION_KEY],
        memory_api_url=MEMORY_API_URL,
        memory_token_secret=SecretStr(MEMORY_TOKEN_SECRET),
    )
    return OpenShellConversationRegistry(config, gateway=gateway)


def launch(
    gateway: FakeGateway,
    registry: OpenShellConversationRegistry,
    monkeypatch,
    conversation_id: UUID,
    tags: dict[str, str],
) -> dict[str, str]:
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    scope = MemoryScope.from_tags(conversation_id, tags)
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(identity.model_copy(update={"memory_scope": scope}))
    registry._build_container(conversation_id)
    return gateway.created[-1]["environment"]


def claims_of(token: str) -> dict[str, object]:
    version, payload, _signature = token.split(".")
    assert version == "v1"
    padded = payload + "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


def test_a_launched_sandbox_holds_one_token_for_its_scope(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()

    environment = launch(
        gateway,
        registry,
        monkeypatch,
        conversation_id,
        {"automationid": "auto-1", "automationrunid": "run-1"},
    )

    tokens = [value for value in environment.values() if value.startswith("v1.")]
    assert len(tokens) == 1
    claims = claims_of(tokens[0])
    assert claims["namespace"] == "auto-1"
    assert claims["automation_id"] == "auto-1"
    assert claims["run_id"] == "run-1"
    assert int(claims["exp"]) > int(time.time())
    assert environment[MEMORY_API_URL_ENV] == MEMORY_API_URL


def test_a_conversation_without_automation_tags_gets_its_own_namespace():
    conversation_id = uuid4()

    scope = MemoryScope.from_tags(conversation_id, {})

    assert scope.namespace == f"conversation-{conversation_id.hex}"
    assert scope.automation_id == conversation_id.hex
    assert scope.run_id == conversation_id.hex


def test_no_forbidden_key_enters_the_sandbox(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()

    environment = launch(gateway, registry, monkeypatch, conversation_id, {})

    identity = registry.provisioning.load(conversation_id)
    assert environment["OH_SECRET_KEY"] == identity.encryption_key.get_secret_value()
    assert environment["OH_SECRET_KEY"] != HOST_SECRET_KEY
    assert (
        environment[V1_SESSION_API_KEY_ENV] == identity.api_key.get_secret_value()
    )
    values = set(environment.values())
    assert OUTER_SESSION_KEY not in values
    assert HOST_SECRET_KEY not in values
    assert MEMORY_TOKEN_SECRET not in values
    assert environment[MEMORY_TOKEN_ENV] not in {OUTER_SESSION_KEY, HOST_SECRET_KEY}


def test_a_resumed_sandbox_receives_a_fresh_token(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    scope = MemoryScope.from_tags(
        conversation_id, {"automationid": "auto-1", "automationrunid": "run-1"}
    )
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(identity.model_copy(update={"memory_scope": scope}))
    registry._build_container(conversation_id)
    created = gateway.created[0]
    first_token = created["environment"][MEMORY_TOKEN_ENV]

    gateway.existing = make_sandbox(
        created["name"],
        phase=pb.SANDBOX_PHASE_STOPPED,
        conversation_hex=conversation_id.hex,
    )
    monkeypatch.setattr(
        "openhands.agent_server.docker_runtime.provisioning.time",
        SimpleNamespace(time=lambda: 1_900_000_000.0),
    )
    registry._build_container(conversation_id)

    assert gateway.started == [created["name"]]
    name, refreshed = gateway.updated[0]
    assert name == created["name"]
    fresh_token = refreshed[MEMORY_TOKEN_ENV]
    assert fresh_token != first_token
    assert int(claims_of(fresh_token)["exp"]) == 1_900_000_000 + 604800


@pytest.mark.asyncio
async def test_a_launch_derives_the_scope_from_automation_tags(tmp_path):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    body = {
        "conversation_id": str(conversation_id),
        "workspace": {"kind": "LocalWorkspace", "working_dir": "/workspace"},
        "agent": {
            "kind": "Agent",
            "llm": {"model": "test", "api_key": "openshell:resolve:env:model"},
        },
        "tags": {"automationid": "auto-1", "automationrunid": "run-1"},
    }

    _, _, _, _, scope = await _prepare_forward(
        body, registry, existing=False, conversation_id=conversation_id
    )

    assert scope is not None
    assert scope.namespace == "auto-1"
    assert scope.automation_id == "auto-1"
    assert scope.run_id == "run-1"


@pytest.mark.asyncio
async def test_a_refused_launch_delivers_no_token(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a refused launch built a sandbox environment")

    monkeypatch.setattr(registry, "_sandbox_environment", unexpected)
    body = {
        "conversation_id": str(conversation_id),
        "workspace": {"kind": "LocalWorkspace", "working_dir": "/workspace"},
        "agent": {"kind": "Agent", "llm": {"model": "test", "api_key": "sk-real"}},
    }

    with pytest.raises(AgentLaunchError):
        await _prepare_forward(
            body, registry, existing=False, conversation_id=conversation_id
        )

    assert gateway.created == []


def test_the_delivered_token_authenticates_to_the_memory_service(tmp_path, monkeypatch):
    source = os.environ.get("OPENHANDS_MEMORY_SRC")
    if not source:
        pytest.skip("set OPENHANDS_MEMORY_SRC to the openhands-memory source tree")
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()

    environment = launch(
        gateway,
        registry,
        monkeypatch,
        conversation_id,
        {"automationid": "auto-1", "automationrunid": "run-1"},
    )

    sys.path.insert(0, source)
    try:
        from openhands_memory.auth import verify_token
    finally:
        sys.path.remove(source)
    claims = verify_token(MEMORY_TOKEN_SECRET, environment[MEMORY_TOKEN_ENV])
    assert claims is not None
    assert claims.namespace == "auto-1"
    assert claims.automation_id == "auto-1"
    assert claims.run_id == "run-1"