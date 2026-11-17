"""Run one governed agent server per conversation in an OpenShell sandbox."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import grpc
from openshell._proto import openshell_pb2 as pb

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.openshell.events import PolicyEventWatcher
from openhands.agent_server.docker_runtime.provisioning import RuntimeIdentity
from openhands.agent_server.docker_runtime.registry import (
    OBSERVABILITY_ENV_VARS,
    ConversationContainer,
    DockerConversationRegistry,
    RuntimeArchivedError,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

_CONVERSATION_LABEL = "openhands.conversation"
_OWNER_LABEL = "openhands.runtime-owner"
_SANDBOX_HOME = "/sandbox"
_SANDBOX_CONVERSATIONS = f"{_SANDBOX_HOME}/conversations"
_SANDBOX_PERSISTENCE = f"{_SANDBOX_HOME}/.openhands"
_AGENT_PORT = 8000
# The agent-server image also ships a frozen ``openhands-agent-server`` binary
# built from the upstream SDK release. That binary embeds its own copy of the
# server, so it never sees an ``openhands-agent-server`` package installed
# beside it; a fork's routes and fixes would stay invisible. Start the module
# from the image's interpreter instead, which is the package the image
# install actually holds.
_AGENT_COMMAND = [
    "python",
    "-m",
    "openhands.agent_server",
    "--host",
    "0.0.0.0",
    "--port",
    str(_AGENT_PORT),
]
_POLICY_WATCH_INTERVAL_SECONDS = float(
    os.environ.get("OH_POLICY_WATCH_INTERVAL_SECONDS", "5")
)


@dataclass(slots=True)
class OpenShellContainer(ConversationContainer):
    """A container handle whose lifecycle is a gateway sandbox."""

    gateway: OpenShellGateway | None = None

    def is_running(self) -> bool:
        if self.gateway is None:
            return False
        sandbox = self.gateway.get(self.container_id)
        return sandbox is not None and sandbox.phase == pb.SANDBOX_PHASE_READY

    def startup_failed(self) -> bool:
        """A provisioning or starting sandbox is not a failed one."""
        if self.gateway is None:
            return True
        sandbox = self.gateway.get(self.container_id)
        return sandbox is None or sandbox.phase in (
            pb.SANDBOX_PHASE_ERROR,
            pb.SANDBOX_PHASE_STOPPED,
            pb.SANDBOX_PHASE_COMPLETED,
            pb.SANDBOX_PHASE_DELETING,
        )

    def stop(self) -> None:
        if self.gateway is not None:
            self.gateway.stop(self.container_id)


class OpenShellConversationRegistry(DockerConversationRegistry):
    """Provision a sandbox per conversation, then proxy to its agent server.

    The published service URL needs the gateway CA, and conversation state
    lives under ``/sandbox``.
    """

    inner_workspace_root = _SANDBOX_HOME
    governed = True

    def __init__(self, config: Config, gateway: OpenShellGateway | None = None) -> None:
        super().__init__(config)
        self.gateway = gateway or OpenShellGateway(
            config.openshell_workspace,
            cluster=config.openshell_cluster,
            service_port=config.openshell_service_port,
        )
        self._policy_watcher = PolicyEventWatcher(
            self.gateway,
            self,
            interval=_POLICY_WATCH_INTERVAL_SECONDS,
        )

    async def start(self) -> None:
        await super().start()
        watcher = self._policy_watcher
        if watcher is not None:
            watcher.start()

    async def shutdown(self) -> None:
        try:
            watcher, self._policy_watcher = self._policy_watcher, None
            if watcher is not None:
                await watcher.stop()
        finally:
            try:
                await super().shutdown()
            finally:
                self.gateway.close()

    def live_governed_conversations(self) -> list[tuple[UUID, OpenShellContainer]]:
        """Conversations with a container handle the watcher can reach."""
        return [
            (conversation_id, container)
            for conversation_id, container in list(self._containers.items())
            if isinstance(container, OpenShellContainer)
        ]

    def load_policy_state(self, conversation_id: UUID) -> dict[str, Any] | None:
        path = self.provisioning.policy_state_path(conversation_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def save_policy_state(self, conversation_id: UUID, state: dict[str, Any]) -> None:
        path = self.provisioning.policy_state_path(conversation_id)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(state), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            logger.warning(
                "Could not persist the policy state of conversation %s",
                conversation_id,
            )

    def sandbox_name(self, conversation_id: UUID) -> str:
        # The gateway caps a sandbox name at 19 characters, so the name keeps
        # 16 hex digits; the full id lives in the conversation label.
        return f"oh-{conversation_id.hex[:16]}"

    def cleanup_stale_containers(self) -> None:
        """The gateway owns sandbox cleanup; there is no host runtime to scan."""

    def serves_persisted_event_reads(self, _conversation_id: UUID) -> bool:
        """Conversation state lives in the sandbox, so reads proxy to it."""
        return False

    def _build_container(self, conversation_id: UUID) -> ConversationContainer:
        identity = self.provisioning.load(conversation_id)
        name = self.sandbox_name(conversation_id)
        created = False
        # Lookup by label, not name: the label carries the full id (see
        # sandbox_name), so a shortened name still identifies its sandbox.
        sandbox = self.gateway.find(f"{_CONVERSATION_LABEL}={conversation_id.hex}")
        if sandbox is None:
            sandbox = self.gateway.get(name)
            if (
                sandbox is not None
                and sandbox.labels.get(_CONVERSATION_LABEL) != conversation_id.hex
            ):
                raise RuntimeError(
                    f"Sandbox name collision: {name} holds "
                    f"{sandbox.labels.get(_CONVERSATION_LABEL)}, "
                    f"not {conversation_id.hex}"
                )
        if sandbox is None and identity.service_url:
            # The sandbox held the conversation state, so a recorded runtime
            # the gateway no longer has cannot be resumed.
            self.mark_archived(conversation_id)
            raise RuntimeArchivedError("Conversation sandbox no longer exists")
        if sandbox is None:
            sandbox = self.gateway.create(
                name=name,
                image=self.config.conversation_image,
                command=list(_AGENT_COMMAND),
                environment=self._sandbox_environment(identity),
                labels={
                    _CONVERSATION_LABEL: conversation_id.hex,
                    _OWNER_LABEL: self.owner,
                },
                provider=self.config.openshell_provider,
                expose_port=_AGENT_PORT,
            )
            created = True
        elif sandbox.phase != pb.SANDBOX_PHASE_READY:
            sandbox = self.gateway.start(name)
        try:
            # The empty key is the unnamed service exposure.
            url = identity.service_url or sandbox.service_urls.get("")
            if not url:
                raise RuntimeError(
                    "Sandbox has no recorded service URL; "
                    "delete the runtime to recreate it"
                )
            container = OpenShellContainer(
                host=self.gateway.service_url(url),
                api_key=identity.api_key.get_secret_value(),
                container_id=sandbox.name,
                ca_bundle=self.config.openshell_tls_ca,
                gateway=self.gateway,
            )
            if created:
                self.provisioning.save(
                    identity.model_copy(update={"service_url": container.host})
                )
            self._wait_until_ready(container)
        except grpc.RpcError:
            # A gateway restart is not a failed start; keep the sandbox.
            raise
        except Exception:
            if created:
                # The identity already names the sandbox URL; a retry would
                # otherwise read it as an archived runtime.
                self.provisioning.save(
                    identity.model_copy(update={"service_url": None})
                )
                try:
                    self.gateway.delete(name)
                except Exception:
                    logger.exception("Could not delete failed sandbox %s", name)
            raise
        return container

    def _sandbox_environment(self, identity: RuntimeIdentity) -> dict[str, str]:
        environment = {
            "HOME": _SANDBOX_HOME,
            "OH_CONVERSATIONS_PATH": _SANDBOX_CONVERSATIONS,
            "OH_PERSISTENCE_DIR": _SANDBOX_PERSISTENCE,
            "OH_CONVERSATION_RUNTIME": "local",
            "OH_ENABLE_BROWSER": "0",
            "OH_SECRET_KEY": identity.encryption_key.get_secret_value(),
            V1_SESSION_API_KEY_ENV: identity.api_key.get_secret_value(),
        }
        if identity.launched_agent_profile:
            environment["OH_RUNTIME_LAUNCHED_PROFILE"] = (
                identity.launched_agent_profile.model_dump_json()
            )
        for name in OBSERVABILITY_ENV_VARS:
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment

    def _existing_sandbox_name(self, conversation_id: UUID) -> str:
        """The sandbox that holds this conversation, or its reserved name."""
        container = self.get(conversation_id)
        if container is not None:
            return container.container_id
        sandbox = self.gateway.find(f"{_CONVERSATION_LABEL}={conversation_id.hex}")
        return (
            sandbox.name if sandbox is not None else self.sandbox_name(conversation_id)
        )

    def delete_runtime(self, conversation_id: UUID) -> None:
        self.gateway.delete(self._existing_sandbox_name(conversation_id))

    def copy_conversation_record(self, conversation_id: UUID) -> None:
        """Copy the inner conversation record into the outer catalog."""
        sandbox = self._existing_sandbox_name(conversation_id)
        base = f"{_SANDBOX_CONVERSATIONS}/{conversation_id.hex}"
        # The catalog builds each conversation from base_state.json; meta.json
        # only carries the listing fields.
        for name in ("meta.json", "base_state.json"):
            text = self.gateway.read_file(sandbox, f"{base}/{name}")
            if text is None:
                logger.warning(
                    "Conversation %s has no %s in sandbox %s",
                    conversation_id,
                    name,
                    sandbox,
                )
                continue
            target = self.conversation_dir(conversation_id) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, target)
