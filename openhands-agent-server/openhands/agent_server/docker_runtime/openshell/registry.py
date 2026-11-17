"""Run one governed agent server per conversation in an OpenShell sandbox."""

from __future__ import annotations

import os
import ssl
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen
from uuid import UUID

from openshell._proto import openshell_pb2 as pb

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.docker_runtime.openshell.client import OpenShellGateway
from openhands.agent_server.docker_runtime.provisioning import RuntimeIdentity
from openhands.agent_server.docker_runtime.registry import (
    _OBSERVABILITY_ENV_VARS,
    ConversationContainer,
    DockerConversationRegistry,
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
    """Provision one sandbox per conversation through the OpenShell gateway.

    The gateway publishes the agent server as a loopback HTTP service on its
    own TLS listener, so the proxy target is the service URL plus the gateway
    CA. Conversation state lives under ``/sandbox`` inside the sandbox.
    """

    inner_workspace_root = _SANDBOX_HOME
    governed = True

    def __init__(
        self, config: Config, gateway: OpenShellGateway | None = None
    ) -> None:
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
        return f"oh-{conversation_id.hex}"

    def cleanup_stale_containers(self) -> None:
        """The gateway owns sandbox cleanup; there is no host runtime to scan."""

    def _build_container(self, conversation_id: UUID) -> ConversationContainer:
        identity = self.provisioning.load(conversation_id)
        sandbox = self.gateway.find(conversation_id.hex)
        if sandbox is None:
            sandbox = self.gateway.create(
                name=self.sandbox_name(conversation_id),
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
        url = sandbox.service_urls.get("")
        if not url:
            raise RuntimeError("OpenShell did not publish a service URL")
        container = OpenShellContainer(
            host=url.rstrip("/"),
            api_key=identity.api_key.get_secret_value(),
            container_id=sandbox.name,
            ca_bundle=self.config.openshell_tls_ca,
            gateway=self.gateway,
        )
        try:
            self._wait_until_ready(container)
        except BaseException:
            self.gateway.delete(container.container_id)
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

    def _wait_until_ready(self, container: ConversationContainer) -> None:
        deadline = time.monotonic() + self.config.conversation_container_startup_timeout
        context = (
            ssl.create_default_context(cafile=container.ca_bundle)
            if container.ca_bundle
            else None
        )
        while time.monotonic() < deadline:
            try:
                with urlopen(
                    container.host + "/health", timeout=1, context=context
                ) as response:
                    if 200 <= response.status < 300:
                        return
            except (URLError, TimeoutError, ConnectionError):
                pass
            if not container.is_running():
                raise RuntimeError("Conversation sandbox stopped during startup")
            time.sleep(1)
        raise RuntimeError("Conversation container failed to become healthy in time")