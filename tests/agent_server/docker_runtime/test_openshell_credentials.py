"""The memory credential a governed conversation receives."""

import base64
import json
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from openshell import SandboxRef, SandboxStatusRef
from openshell._proto import openshell_pb2 as pb
from pydantic import SecretStr

from openhands.agent_server.config import (
    MEMORY_API_URL_ENV,
    MEMORY_TOKEN_ENV,
    V1_SESSION_API_KEY_ENV,
    Config,
)
from openhands.agent_server.docker_runtime import routers
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.openshell.registry import (
    OpenShellConversationRegistry,
)
from openhands.agent_server.docker_runtime.provisioning import (
    MemoryScope,
    mint_memory_token,
)
from openhands.agent_server.docker_runtime.routers import _prepare_forward
from openhands.sdk.launch import AgentLaunchError


HOST_SECRET_KEY = "host-secret-key"
OUTER_SESSION_KEY = "outer-session-key"
MEMORY_TOKEN_SECRET = "memory-token-secret"
MEMORY_API_URL = "https://memory.example.test"

# Pinned by both this repository and the memory service: the mint and the
# verifier must accept the same bytes.
GOLDEN_SECRET = "golden-vector-secret"
GOLDEN_NAMESPACE = "automation-demo"
GOLDEN_TOKEN = (
    "v1.eyJhdXRvbWF0aW9uX2lkIjoiYXV0b21hdGlvbi1kZW1vIiwiZXhwIjoxOTAwMDAw"
    "MDAwLCJuYW1lc3BhY2UiOiJhdXRvbWF0aW9uLWRlbW8iLCJydW5faWQiOiJydW4tZGVt"
    "byJ9.ALavbc90AcajCtOlx4nRLCk3PI3riQ-rsdte5E9WUtA"
)
GOLDEN_SCOPE = MemoryScope(
    namespace=GOLDEN_NAMESPACE, automation_id="automation-demo", run_id="run-demo"
)


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

    def update_environment(self, name: str, environment, *, request_id: str) -> None:
        self.updated.append((name, dict(environment)))

    def delete(self, name: str) -> None:
        self.deleted.append(name)

    def close(self) -> None:
        pass


def make_registry(
    tmp_path, gateway: FakeGateway, **config_overrides
) -> OpenShellConversationRegistry:
    config = Config(
        conversation_runtime="openshell",
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspaces",
        openshell_policy=None,
        secret_key=SecretStr(HOST_SECRET_KEY),
        session_api_keys=[OUTER_SESSION_KEY],
        memory_api_url=MEMORY_API_URL,
        memory_token_secret=SecretStr(MEMORY_TOKEN_SECRET),
        **config_overrides,
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


def stop_existing(gateway: FakeGateway, name: str, phase: int, conversation_hex: str):
    gateway.existing = make_sandbox(
        name, phase=phase, conversation_hex=conversation_hex
    )


def test_the_mint_matches_the_memory_service_wire_vector(monkeypatch):
    monkeypatch.setattr(
        "openhands.agent_server.docker_runtime.provisioning.time",
        SimpleNamespace(time=lambda: 1_900_000_000.0),
    )

    assert mint_memory_token(GOLDEN_SECRET, GOLDEN_SCOPE, 0) == GOLDEN_TOKEN


def test_distinct_automation_ids_get_distinct_namespaces():
    automation_ids = ["Prod-Team", "prod team", "PROD TEAM", "Öko"]

    namespaces = {
        MemoryScope.from_tags(uuid4(), {"automationid": value}).namespace
        for value in automation_ids
    }

    assert len(namespaces) == len(automation_ids)


def test_one_automation_keeps_one_namespace_across_conversations():
    first = MemoryScope.from_tags(uuid4(), {"automationid": "auto-1"})
    second = MemoryScope.from_tags(uuid4(), {"automationid": "auto-1"})

    assert first.namespace == second.namespace == "auto-1-4161fa29"


@pytest.mark.parametrize(
    "automation_id",
    ["auto-1", "Prod-Team", "Öko", "!!!", "a" * 200],
)
def test_every_namespace_stays_in_the_service_charset(automation_id):
    namespace = MemoryScope.from_tags(
        uuid4(), {"automationid": automation_id}
    ).namespace

    # The memory service's namespace contract.
    assert re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", namespace)


def test_an_id_without_a_usable_slug_falls_back_to_the_automation_prefix():
    namespace = MemoryScope.from_tags(uuid4(), {"automationid": "!!!"}).namespace

    assert namespace.startswith("automation-")


def test_a_conversation_without_automation_tags_gets_its_own_namespace():
    conversation_id = uuid4()

    scope = MemoryScope.from_tags(conversation_id, {})

    assert scope.namespace == f"conversation-{conversation_id.hex}"
    assert scope.automation_id == conversation_id.hex
    assert scope.run_id == conversation_id.hex


def test_an_automation_id_equal_to_the_conversation_hex_stays_an_automation():
    conversation_id = uuid4()

    scope = MemoryScope.from_tags(
        conversation_id, {"automationid": conversation_id.hex}
    )

    assert scope.namespace != f"conversation-{conversation_id.hex}"
    assert scope.namespace.startswith(f"{conversation_id.hex}-")


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
    assert claims["namespace"] == "auto-1-4161fa29"
    assert claims["automation_id"] == "auto-1"
    assert claims["run_id"] == "run-1"
    assert environment[MEMORY_API_URL_ENV] == MEMORY_API_URL


def test_a_token_lasts_for_the_retention_window(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(
        tmp_path, gateway, conversation_storage_retention_days=30
    )
    conversation_id = uuid4()
    monkeypatch.setattr(
        "openhands.agent_server.docker_runtime.provisioning.time",
        SimpleNamespace(time=lambda: 1_900_000_000.0),
    )

    environment = launch(gateway, registry, monkeypatch, conversation_id, {})

    claims = claims_of(environment[MEMORY_TOKEN_ENV])
    assert int(claims["exp"]) == 1_900_000_000 + 30 * 86400


def test_the_launch_mirrors_the_extension_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSIONS_REF", "bleenk")
    monkeypatch.setenv("EXTENSIONS_PATH", "/opt/extensions")
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)

    environment = launch(gateway, registry, monkeypatch, uuid4(), {})

    assert environment["EXTENSIONS_REF"] == "bleenk"
    assert environment["EXTENSIONS_PATH"] == "/opt/extensions"


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

    stop_existing(
        gateway, created["name"], pb.SANDBOX_PHASE_STOPPED, conversation_id.hex
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
    assert int(claims_of(fresh_token)["exp"]) == (
        1_900_000_000 + registry.config.memory_token_ttl_seconds
    )


def test_a_suspended_sandbox_refreshes_its_environment(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(
            update={"memory_scope": MemoryScope.from_tags(conversation_id, {})}
        )
    )
    registry._build_container(conversation_id)
    created = gateway.created[0]

    stop_existing(
        gateway, created["name"], pb.SANDBOX_PHASE_SUSPENDED, conversation_id.hex
    )
    registry._build_container(conversation_id)

    assert gateway.started == [created["name"]]
    assert [name for name, _ in gateway.updated] == [created["name"]]


def test_a_refresh_without_a_scope_still_delivers_the_whole_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXTENSIONS_REF", "bleenk")
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    registry.provisioning.create(conversation_id)
    registry._build_container(conversation_id)
    created = gateway.created[0]

    stop_existing(
        gateway, created["name"], pb.SANDBOX_PHASE_STOPPED, conversation_id.hex
    )
    registry._build_container(conversation_id)

    _name, refreshed = gateway.updated[0]
    assert refreshed["EXTENSIONS_REF"] == "bleenk"
    assert refreshed["OH_PERSISTENCE_DIR"] == "/sandbox/.openhands"


def test_a_refresh_clears_a_token_the_secret_no_longer_mints(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    monkeypatch.setattr(registry, "_wait_until_ready", lambda container: None)
    identity = registry.provisioning.create(conversation_id)
    registry.provisioning.save(
        identity.model_copy(
            update={"memory_scope": MemoryScope.from_tags(conversation_id, {})}
        )
    )
    registry._build_container(conversation_id)
    created = gateway.created[0]
    assert created["environment"][MEMORY_TOKEN_ENV].startswith("v1.")

    registry.config = registry.config.model_copy(
        update={"memory_token_secret": None}
    )
    stop_existing(
        gateway, created["name"], pb.SANDBOX_PHASE_STOPPED, conversation_id.hex
    )
    registry._build_container(conversation_id)

    _name, refreshed = gateway.updated[0]
    assert refreshed[MEMORY_TOKEN_ENV] == ""


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

    plan = await _prepare_forward(body, registry, existing=False)

    scope = plan.memory_scope
    assert scope is not None
    assert scope.namespace == "auto-1-4161fa29"
    assert scope.automation_id == "auto-1"
    assert scope.run_id == "run-1"


@pytest.mark.asyncio
async def test_the_first_launch_scope_is_persisted_and_kept(tmp_path, monkeypatch):
    gateway = FakeGateway()
    registry = make_registry(tmp_path, gateway)
    conversation_id = uuid4()
    container = SimpleNamespace(
        host="https://default--oh-test.openshell.localhost:17670",
        api_key="inner-key",
        ca_bundle=None,
        container_id="oh-test",
    )
    monkeypatch.setattr(registry, "get_or_create", AsyncMock(return_value=container))
    monkeypatch.setattr(registry, "acquire_lease", lambda _id: lambda: None)

    def copy_record(cid: UUID) -> bool:
        target = registry.conversation_dir(cid) / "meta.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
        return True

    monkeypatch.setattr(registry, "copy_conversation_record", copy_record)

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, _url, **_kwargs):
            return httpx.Response(200, json={"id": str(conversation_id)})

    monkeypatch.setattr(routers.httpx, "AsyncClient", Client)
    body = {
        "conversation_id": str(conversation_id),
        "workspace": {"kind": "LocalWorkspace", "working_dir": "/workspace"},
        "agent": {
            "kind": "Agent",
            "llm": {"model": "test", "api_key": "openshell:resolve:env:model"},
        },
        "tags": {"automationid": "auto-1", "automationrunid": "run-1"},
    }

    await routers.start_docker_conversation(body, registry)
    first = registry.provisioning.load(conversation_id).memory_scope
    await routers.start_docker_conversation(
        {**body, "tags": {"automationid": "auto-2", "automationrunid": "run-2"}},
        registry,
    )
    second = registry.provisioning.load(conversation_id).memory_scope

    assert first is not None
    assert first.namespace == "auto-1-4161fa29"
    assert second == first


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
        await _prepare_forward(body, registry, existing=False)

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
    assert claims.namespace == "auto-1-4161fa29"
    assert claims.automation_id == "auto-1"
    assert claims.run_id == "run-1"