"""Synchronous gateway client for the governed runtime driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

import grpc
from openshell import SandboxClient, SandboxRef, ServiceExposure
from openshell._proto import openshell_pb2 as pb

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class OpenShellGateway:
    """One gateway connection, shared by the registry's worker threads.

    The SDK is synchronous, so the registry calls the blocking methods from
    worker threads; ``close`` runs on the shutdown path instead.
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

    def find(self, selector: str) -> SandboxRef | None:
        """The first sandbox that matches a label selector, if any."""
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
        # Only the create response carries the service URLs; a later read of
        # the sandbox cannot recover them.
        urls = {key: self.service_url(url) for key, url in created.service_urls.items()}
        for url in urls.values():
            self._require_tls(url)
        return replace(created, service_urls=urls)

    def service_url(self, url: str) -> str:
        """Republish a service URL on the gateway's published port."""
        parts = urlsplit(url)
        if self.service_port is None or parts.port == self.service_port:
            return url.rstrip("/")
        host = parts.hostname or parts.netloc
        netloc = (
            f"[{host}]:{self.service_port}"
            if ":" in host
            else f"{host}:{self.service_port}"
        )
        return urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        ).rstrip("/")

    def get(self, name: str) -> SandboxRef | None:
        """The sandbox, or ``None`` only when the gateway says it is missing."""
        try:
            return self._client.get(name, workspace=self.workspace)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def start(self, name: str) -> SandboxRef:
        return self._client.start(name, workspace=self.workspace)

    def stop(self, name: str) -> None:
        try:
            self._client.stop(name, workspace=self.workspace)
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.NOT_FOUND:
                raise

    def delete(self, name: str) -> None:
        self._client.delete(name, workspace=self.workspace, allow_missing=True)

    def read_file(self, name: str, path: str) -> str | None:
        """One file from a running sandbox, or ``None`` when it is absent."""
        try:
            result = self._client.exec(name, ["cat", path], workspace=self.workspace)
        except grpc.RpcError as exc:
            logger.warning("Could not read %s from sandbox %s: %s", path, name, exc)
            return None
        return result.stdout if result.exit_code == 0 else None

    def _require_tls(self, url: str) -> None:
        # The proxy forwards the inner session key to this URL.
        if urlsplit(url).scheme != "https":
            raise RuntimeError("OpenShell published a service URL without TLS")
