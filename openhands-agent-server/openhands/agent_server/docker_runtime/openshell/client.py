"""Synchronous gateway client for the governed runtime driver."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import grpc
import yaml
from google.protobuf import json_format
from openshell import SandboxClient, SandboxRef, ServiceExposure
from openshell._proto import datamodel_pb2, openshell_pb2 as pb, sandbox_pb2

from openhands.agent_server.docker_runtime.openshell.events import (
    NOTICES_DIR,
    NOTICES_PATH,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# The 0.1.7 SDK pins no wrapper for the policy and log surfaces, so the
# governance loop calls the pinned gRPC stub directly.
_LOG_LINES = 500
_POLICY_TIMEOUT_SECONDS = 30.0

# The policy file uses the CLI's YAML vocabulary. The proto uses the field name
# ``filesystem`` and enum names, so the loader translates only those values.
_ENFORCEMENT = {
    "enforce": "NETWORK_ENFORCEMENT_MODE_ENFORCE",
    "audit": "NETWORK_ENFORCEMENT_MODE_AUDIT",
}
_ACCESS = {
    "read-only": "NETWORK_ACCESS_PRESET_READ_ONLY",
    "read-write": "NETWORK_ACCESS_PRESET_READ_WRITE",
    "full": "NETWORK_ACCESS_PRESET_FULL",
}
_TLS = {
    "skip": "NETWORK_TLS_MODE_SKIP",
    "terminate": "NETWORK_TLS_MODE_TERMINATE",
    "passthrough": "NETWORK_TLS_MODE_PASSTHROUGH",
}


def load_sandbox_policy(path: str | Path) -> sandbox_pb2.SandboxPolicy:
    """Read a repository policy file into the protobuf the gateway accepts."""
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read OpenShell policy {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"OpenShell policy {source} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"OpenShell policy {source} is not a mapping")
    if "filesystem_policy" in data:
        data["filesystem"] = data.pop("filesystem_policy")
    for name, rule in (data.get("network_policies") or {}).items():
        rule.setdefault("name", name)
        for endpoint in rule.get("endpoints", []):
            for key, values in (
                ("enforcement", _ENFORCEMENT),
                ("access", _ACCESS),
                ("tls", _TLS),
            ):
                if key not in endpoint:
                    continue
                value = endpoint[key]
                if key == "tls" and value in ("", None):
                    endpoint.pop(key)
                elif value not in values:
                    raise ValueError(
                        f"OpenShell policy {source} has an unsupported "
                        f"{key} value {value!r}"
                    )
                else:
                    endpoint[key] = values[value]
    try:
        return json_format.ParseDict(data, sandbox_pb2.SandboxPolicy())
    except Exception as exc:
        raise ValueError(f"OpenShell policy {source} is invalid: {exc}") from exc


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
        # port, not the published one.
        self.service_port = service_port
        self._client = SandboxClient.from_active_cluster(cluster=cluster)

    def close(self) -> None:
        self._client.close()

    def find(self, selector: str) -> SandboxRef | None:
        """The sandbox that matches a label selector, if any."""
        sandboxes = self._client.list_all(
            workspace=self.workspace, label_selector=selector
        )
        if not sandboxes:
            return None
        if len(sandboxes) > 1:
            logger.warning(
                "Several sandboxes match %s: %s",
                selector,
                ", ".join(sorted(sandbox.name for sandbox in sandboxes)),
            )
        return sorted(sandboxes, key=lambda sandbox: sandbox.name)[0]

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
        policy: sandbox_pb2.SandboxPolicy | None = None,
        memory: str | None = None,
        cpu: str | None = None,
    ) -> SandboxRef:
        spec = pb.SandboxSpec(
            template=pb.SandboxTemplate(image=image),
            command=list(command),
            environment=dict(environment),
        )
        limits = {
            key: value
            for key, value in (("cpu", cpu), ("memory", memory))
            if value is not None
        }
        if limits:
            # The gateway reads the Kubernetes limits shape out of this Struct
            # (crates/openshell-server/src/compute/mod.rs
            # `extract_typed_resources`) and the podman driver applies
            # `limits.memory` and `limits.cpu`; without them the driver falls
            # back to its built-in 4 GiB memory default and 2-core
            # `DEFAULT_CPU_QUOTA`. Write both keys in one update: two separate
            # updates would let the second overwrite the first's `limits` key.
            spec.template.resources.update({"limits": limits})
        if policy is not None:
            spec.policy.CopyFrom(policy)
        if provider:
            spec.providers.append(provider)
        try:
            created = self._client.create(
                workspace=self.workspace,
                spec=spec,
                name=name,
                labels=dict(labels),
                service_exposures=[ServiceExposure(target_port=expose_port)],
            )
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
                raise
            # Another process won the race for this conversation; adopt its
            # sandbox instead of running a second one.
            existing = self.get(name)
            if existing is None:
                raise
            logger.warning("Adopted sandbox %s created by another process", name)
            return existing
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
        """Stop the sandbox; a missing sandbox stays a no-op."""
        try:
            self._client.stop(name, workspace=self.workspace)
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.NOT_FOUND:
                raise

    def delete(self, name: str) -> None:
        self._client.delete(name, workspace=self.workspace, allow_missing=True)

    def read_file(self, name: str, path: str) -> str | None:
        """One file from a sandbox, or ``None`` when the read fails."""
        try:
            result = self._client.exec(name, ["cat", path], workspace=self.workspace)
        except grpc.RpcError as exc:
            logger.warning("Could not read %s from sandbox %s: %s", path, name, exc)
            return None
        if result.exit_code != 0:
            logger.warning(
                "Could not read %s from sandbox %s: exit %s",
                path,
                name,
                result.exit_code,
            )
            return None
        return result.stdout

    def _policy_scope(self) -> datamodel_pb2.WorkspaceSelector:
        return datamodel_pb2.WorkspaceSelector(workspace=self.workspace)

    def logs(
        self,
        name: str,
        since_epoch: float | None = None,
        lines: int = _LOG_LINES,
    ) -> list[pb.SandboxLogLine]:
        """Recent sandbox log lines, newer than ``since_epoch`` when given."""
        request = pb.GetSandboxLogsRequest(
            workspace_scope=self._policy_scope(),
            sandbox=name,
            lines=lines,
        )
        if since_epoch is not None:
            request.since_time.FromMilliseconds(int(since_epoch * 1000))
        response = self._client._stub.GetSandboxLogs(
            request, timeout=_POLICY_TIMEOUT_SECONDS
        )
        return list(response.logs)

    def draft_chunks(self, name: str) -> list[Any]:
        """The gateway's drafted rule proposals for one sandbox."""
        response = self._client._stub.GetDraftPolicy(
            pb.GetDraftPolicyRequest(
                workspace_scope=self._policy_scope(),
                sandbox=name,
            ),
            timeout=_POLICY_TIMEOUT_SECONDS,
        )
        return list(response.chunks)

    def commit_approved_chunk(self, name: str, chunk_id: str, review_token: str) -> int:
        """Commit one widening request the gateway has already decided.

        The driver only passes a chunk the gateway already approved: it never
        approves a pending or rejected proposal, and it never batches chunks,
        so no automated surface performs a bulk approval. A token-bearing
        (human) approval is committed after the turn ends; an auto approval
        carries an empty token and commits immediately.
        """
        response = self._client._stub.ApproveDraftChunk(
            pb.ApproveDraftChunkRequest(
                workspace_scope=self._policy_scope(),
                sandbox=name,
                chunk_id=chunk_id,
                review_token=review_token,
                request_id=str(uuid.uuid4()),
            ),
            timeout=_POLICY_TIMEOUT_SECONDS,
        )
        return int(response.policy_version)

    def append_notices(self, name: str, text: str) -> None:
        """Append governance notices to the sandbox file the hook reads.

        The text travels on stdin, so no shell quoting is involved.
        """
        result = self._client.exec(
            name,
            ["sh", "-c", f"mkdir -p {NOTICES_DIR} && cat >> {NOTICES_PATH}"],
            workspace=self.workspace,
            stdin=text.encode(),
        )
        if result.exit_code != 0:
            logger.warning(
                "Could not append governance notices in sandbox %s: exit %s",
                name,
                result.exit_code,
            )

    def _require_tls(self, url: str) -> None:
        # The proxy forwards the inner session key to this URL.
        if urlsplit(url).scheme != "https":
            raise RuntimeError("OpenShell published a service URL without TLS")
