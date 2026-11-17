"""Run one governed agent server per conversation in an OpenShell sandbox."""

from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import UUID

import grpc
from openshell._proto import openshell_pb2 as pb

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.provisioning import RuntimeIdentity
from openhands.agent_server.docker_runtime.registry import (
    _OBSERVABILITY_ENV_VARS,
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


@dataclass(slots=True)
class OpenShellContainer(ConversationContainer):
    """A container handle whose lifecycle is a gateway sandbox."""

    gateway: OpenShellGateway | None = None

    def is_running(self) -> bool:
        if self.gateway is None:
            return False
        sandbox = self.gateway.get(self.container_id)
        return sandbox is not None and sandbox.phase == pb.SANDBOX_PHASE_READY

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

    async def shutdown(self) -> None:
        await super().shutdown()
        self.gateway.close()

    def sandbox_name(self, conversation_id: UUID) -> str:
        # The gateway caps a sandbox name at 19 characters, so the name keeps
        # 16 hex digits; the full id lives in the conversation label.
        return f"oh-{conversation_id.hex[:16]}"

    def cleanup_stale_containers(self) -> None:
        """The gateway owns sandbox cleanup; there is no host runtime to scan."""

    def _build_container(self, conversation_id: UUID) -> ConversationContainer:
        identity = self.provisioning.load(conversation_id)
        name = self.sandbox_name(conversation_id)
        created = False
        # The label carries the full conversation id, so it identifies the
        # sandbox even when the name had to be shortened.
        sandbox = self.gateway.find(f"{_CONVERSATION_LABEL}={conversation_id.hex}")
        if sandbox is None:
            sandbox = self.gateway.get(name)
            if (
                sandbox is not None
                and sandbox.labels.get(_CONVERSATION_LABEL) != conversation_id.hex
            ):
                raise RuntimeError("Sandbox name collision")
        if sandbox is None and identity.service_url:
            # The sandbox held the conversation state, so a recorded runtime
            # the gateway no longer has cannot be resumed.
            self.mark_archived(conversation_id)
            raise RuntimeArchivedError("Conversation sandbox no longer exists")
        if sandbox is None:
            sandbox = self.gateway.create(
                name=name,
                image=self.config.conversation_image,
                command=[
                    "openhands-agent-server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(_AGENT_PORT),
                ],
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
        for name in _OBSERVABILITY_ENV_VARS:
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
        return sandbox.name if sandbox is not None else self.sandbox_name(conversation_id)

    def delete_runtime(self, conversation_id: UUID) -> None:
        self.gateway.delete(self._existing_sandbox_name(conversation_id))

    def copy_conversation_record(self, conversation_id: UUID) -> None:
        """Copy the inner conversation record into the outer catalog."""
        text = self.gateway.read_file(
            self._existing_sandbox_name(conversation_id),
            f"{_SANDBOX_CONVERSATIONS}/{conversation_id.hex}/meta.json",
        )
        if text is None:
            return
        target = self.conversation_dir(conversation_id) / "meta.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
