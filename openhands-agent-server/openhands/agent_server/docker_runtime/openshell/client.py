"""Synchronous gateway client for the governed runtime driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

from openshell import SandboxClient, SandboxError, SandboxRef, ServiceExposure
from openshell._proto import openshell_pb2 as pb

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class OpenShellGateway:
    """One gateway connection, shared by the registry's worker threads.

    The SDK is synchronous, so the registry calls every method through
    ``asyncio.to_thread``.
    """

    def __init__(
        self,
        workspace: str,
        *,
        cluster: str | None = None,
        service_port: int | None = None,
    ) -> None:
        self.workspace = workspace
        # The SDK composes service URLs on the gateway's internal listener
        # port. A published gateway listens elsewhere, so the driver rewrites
        # the port when the deployment sets one.
        self.service_port = service_port
        self._client = SandboxClient.from_active_cluster(cluster=cluster)

    def close(self) -> None:
        self._client.close()

    def find(self, conversation_hex: str) -> SandboxRef | None:
        """The sandbox that already owns this conversation, if any."""
        selector = f"openhands.conversation={conversation_hex}"
        sandboxes = self._client.list_all(
            workspace=self.workspace, label_selector=selector
        )
        return sandboxes[0] if sandboxes else None

    def create(
        self,
        *,
        name: str,
        image: str,
        command: Sequence[str],
        environment: Mapping[str, str],
        labels: Mapping[str, str],
        provider: str | None,
        expose_port: int,
    ) -> SandboxRef:
        spec = pb.SandboxSpec(
            template=pb.SandboxTemplate(image=image),
            command=list(command),
            environment=dict(environment),
        )
        if provider:
            spec.providers.append(provider)
        created = self._client.create(
            workspace=self.workspace,
            spec=spec,
            name=name,
            labels=dict(labels),
            service_exposures=[ServiceExposure(target_port=expose_port)],
        )
        ready = self._client.wait_ready(name, workspace=self.workspace)
        # Only the create response carries the URLs; wait_ready drops them.
        urls = dict(created.service_urls or ready.service_urls)
        return replace(
            ready,
            service_urls={
                key: self.service_url(url) for key, url in urls.items()
            },
        )

    def service_url(self, url: str) -> str:
        """Republish a service URL on the gateway's published port."""
        parts = urlsplit(url)
        if self.service_port is None or parts.port in (None, self.service_port):
            return url.rstrip("/")
        netloc = f"{parts.hostname or parts.netloc}:{self.service_port}"
        return urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        ).rstrip("/")

    def get(self, name: str) -> SandboxRef | None:
        try:
            return self._client.get(name, workspace=self.workspace)
        except SandboxError as exc:
            logger.warning("Could not read sandbox %s: %s", name, exc)
            return None

    def stop(self, name: str) -> None:
        self._client.stop(name, workspace=self.workspace)

    def delete(self, name: str) -> None:
        self._client.delete(name, workspace=self.workspace, allow_missing=True)