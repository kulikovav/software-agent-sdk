"""Persist the credentials owned by one conversation container."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from stat import S_ISREG
from typing import Annotated
from uuid import UUID

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from openhands.agent_server.config import Config
from openhands.agent_server.persistence.store import _get_persistence_dir
from openhands.sdk.conversation.request import ConversationContainerOptions
from openhands.sdk.profiles.agent_profile import LaunchedAgentProfile
from openhands.sdk.utils.cipher import Cipher
from openhands.sdk.utils.pydantic_secrets import SecretField, cipher_context


# The memory service verifies this wire format with the same secret.
_MEMORY_TOKEN_VERSION = "v1"
_MEMORY_NAMESPACE_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MEMORY_CLAIM_MAX_LENGTH = 200
_MEMORY_NAMESPACE_MAX_LENGTH = 128
_INVALID_NAMESPACE_CHARS = re.compile(r"[^a-z0-9._-]+")


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _claim_value(value: str | None) -> str | None:
    """A tag value usable as a token claim, or none when it is empty or too long."""
    claim = (value or "").strip()
    if not claim or len(claim) > _MEMORY_CLAIM_MAX_LENGTH:
        return None
    return claim


def _namespace_for(automation_id: str, conversation_id: UUID) -> str:
    """A namespace bound to one automation, or to one conversation without one."""
    if automation_id == conversation_id.hex:
        return f"conversation-{conversation_id.hex}"
    slug = _INVALID_NAMESPACE_CHARS.sub("-", automation_id.lower()).strip("._-")
    if 0 < len(slug) <= _MEMORY_NAMESPACE_MAX_LENGTH and slug[0].isalnum():
        return slug
    return f"conversation-{conversation_id.hex}"


class MemoryScope(BaseModel):
    """The one namespace a conversation's memory token is bound to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: str = Field(pattern=_MEMORY_NAMESPACE_PATTERN)
    automation_id: str = Field(min_length=1, max_length=_MEMORY_CLAIM_MAX_LENGTH)
    run_id: str = Field(min_length=1, max_length=_MEMORY_CLAIM_MAX_LENGTH)

    @classmethod
    def from_tags(
        cls, conversation_id: UUID, tags: Mapping[str, str] | None
    ) -> MemoryScope:
        """Derive the scope from the automation tags a launch carries."""
        values = tags or {}
        automation_id = (
            _claim_value(values.get("automationid")) or conversation_id.hex
        )
        run_id = _claim_value(values.get("automationrunid")) or conversation_id.hex
        return cls(
            namespace=_namespace_for(automation_id, conversation_id),
            automation_id=automation_id,
            run_id=run_id,
        )


def mint_memory_token(secret: str, scope: MemoryScope, ttl_seconds: int) -> str:
    """Sign one short-lived token for a conversation's memory scope."""
    claims = {
        "namespace": scope.namespace,
        "automation_id": scope.automation_id,
        "run_id": scope.run_id,
        "exp": int(time.time()) + ttl_seconds,
    }
    payload = _encode(
        json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signature = hmac.new(
        secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{_MEMORY_TOKEN_VERSION}.{payload}.{_encode(signature)}"


class RuntimeIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_id: UUID
    api_key: Annotated[SecretStr, SecretField()]
    encryption_key: Annotated[SecretStr, SecretField()]
    workspace_path: Path
    worktree_repository_path: Path | None = None
    worktree_git_common_dir_path: Path | None = None
    launched_agent_profile: LaunchedAgentProfile | None = None
    container: ConversationContainerOptions | None = None
    # The namespace and claims a launch minted its memory token from. Kept so a
    # resumed conversation re-mints into the same scope.
    memory_scope: MemoryScope | None = None
    # The gateway publishes the agent server at this URL. Only a create
    # response carries it, so the driver records it here and reuses it when it
    # adopts the sandbox after a restart.
    service_url: str | None = None

    @property
    def cipher(self) -> Cipher:
        return Cipher(self.encryption_key.get_secret_value())


class RuntimeProvisioningStore:
    def __init__(self, config: Config):
        if config.cipher is None:
            raise ValueError("Docker runtime requires OH_SECRET_KEY")
        self.config = config
        self.cipher = config.cipher
        persistence = _get_persistence_dir(config).resolve()
        self.control_root = persistence / "runtime-control"
        self.data_root = persistence / "runtime-data"
        self._identities: dict[UUID, tuple[tuple[int, int, int], RuntimeIdentity]] = {}
        for root in (self.control_root, self.data_root):
            if root.is_symlink():
                raise ValueError("Runtime storage roots must not be symlinks")
            root.mkdir(parents=True, mode=0o700, exist_ok=True)
            root.chmod(0o700)

    def manifest_path(self, conversation_id: UUID) -> Path:
        return self.control_root / f"{conversation_id.hex}.json"

    def policy_state_path(self, conversation_id: UUID) -> Path:
        # Beside the manifest: the policy-watch cursor survives a restart so
        # decisions are never posted twice, and dies with the conversation.
        return self.control_root / f"{conversation_id.hex}.policy.json"

    def runtime_dir(self, conversation_id: UUID) -> Path:
        return self.direct_child(self.data_root, conversation_id.hex)

    @staticmethod
    def direct_child(root: Path, name: str) -> Path:
        if root.is_symlink():
            raise ValueError("Runtime storage root must not be a symlink")
        child = root / name
        resolved = child.resolve()
        if child.is_symlink() or resolved.parent != root.resolve():
            raise ValueError("Runtime mount must not follow a symlink")
        return resolved

    def load_optional(self, conversation_id: UUID) -> RuntimeIdentity | None:
        """Load an identity, returning ``None`` only when it does not exist."""
        path = self.manifest_path(conversation_id)
        try:
            manifest_stat = path.lstat()
        except FileNotFoundError:
            self._identities.pop(conversation_id, None)
            return None
        except OSError:
            self._identities.pop(conversation_id, None)
            raise ValueError("Conversation runtime identity is unavailable") from None
        if not S_ISREG(manifest_stat.st_mode):
            self._identities.pop(conversation_id, None)
            raise ValueError("Conversation runtime identity is unavailable")
        signature = (
            manifest_stat.st_ino,
            manifest_stat.st_mtime_ns,
            manifest_stat.st_size,
        )
        cached = self._identities.get(conversation_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        identity = RuntimeIdentity.model_validate_json(
            path.read_text(), context=cipher_context(self.cipher)
        )
        if identity.conversation_id != conversation_id:
            raise ValueError("Runtime identity does not match conversation")
        self._identities[conversation_id] = (signature, identity)
        return identity

    def load(self, conversation_id: UUID) -> RuntimeIdentity:
        identity = self.load_optional(conversation_id)
        if identity is None:
            raise ValueError("Conversation runtime identity is unavailable")
        return identity

    def create(
        self, conversation_id: UUID, workspace_path: Path | None = None
    ) -> RuntimeIdentity:
        path = self.manifest_path(conversation_id)
        with FileLock(str(path) + ".lock"):
            if path.exists():
                identity = self.load(conversation_id)
                if (
                    workspace_path is not None
                    and workspace_path.resolve() != identity.workspace_path
                ):
                    raise ValueError("A conversation cannot change workspaces")
                return identity
            conversation_dir = self.direct_child(
                self.config.conversations_path, conversation_id.hex
            )
            if (conversation_dir / "meta.json").exists() or (
                conversation_dir / "base_state.json"
            ).exists():
                raise ValueError(
                    "An existing local conversation cannot become a Docker runtime"
                )
            if workspace_path is None:
                workspace_path = self.direct_child(
                    self.runtime_dir(conversation_id), "workspace"
                )
                workspace_path.mkdir(parents=True, mode=0o700, exist_ok=True)
            else:
                if not workspace_path.is_absolute():
                    raise ValueError("Conversation workspace must be absolute")
                if workspace_path.is_symlink():
                    raise ValueError("Conversation workspace must be a real directory")
                try:
                    workspace_path.mkdir(parents=True, mode=0o700, exist_ok=True)
                except OSError as exc:
                    raise ValueError(
                        "Conversation workspace must be a real directory"
                    ) from exc
                if workspace_path.is_symlink() or not workspace_path.is_dir():
                    raise ValueError("Conversation workspace must be a real directory")
                workspace_path = workspace_path.resolve()
            identity = RuntimeIdentity(
                conversation_id=conversation_id,
                api_key=SecretStr(secrets.token_urlsafe(32)),
                encryption_key=SecretStr(secrets.token_urlsafe(32)),
                workspace_path=workspace_path,
            )
            self._save(identity)
            return identity

    def save(self, identity: RuntimeIdentity) -> None:
        with FileLock(str(self.manifest_path(identity.conversation_id)) + ".lock"):
            self._save(identity)

    def _save(self, identity: RuntimeIdentity) -> None:
        payload = identity.model_dump_json(context=cipher_context(self.cipher))
        fd, temporary = tempfile.mkstemp(dir=self.control_root)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(payload)
            os.replace(temporary, self.manifest_path(identity.conversation_id))
            stat = self.manifest_path(identity.conversation_id).stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            self._identities[identity.conversation_id] = (signature, identity)
        finally:
            Path(temporary).unlink(missing_ok=True)
