"""Contract the conversation routes use for a per-conversation runtime.

The routes need more than a proxy target: leases, session attachments, stop and
delete gates, and the per-conversation identity store. They depend on this
contract instead of the Docker registry so a second driver can implement it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard
from uuid import UUID

from openhands.agent_server.conversation_registry import ConversationRegistry
from openhands.agent_server.docker_runtime.provisioning import RuntimeProvisioningStore
from openhands.agent_server.docker_runtime.registry import ConversationContainer
from openhands.agent_server.models import ConversationRuntimeInfo
from openhands.sdk.utils.cipher import Cipher


if TYPE_CHECKING:
    from fastapi import APIRouter

    from openhands.agent_server.config import Config


class ConversationDriver(Protocol):
    """Runtime operations the conversation routes and sockets require."""

    config: Config
    provisioning: RuntimeProvisioningStore
    # Root the inner agent server sees; a governed runtime keeps it inside the
    # sandbox, so a host path in a start request is refused.
    inner_workspace_root: str
    governed: bool

    def conversation_dir(self, conversation_id: UUID) -> Path: ...

    def workspace_dir(self, conversation_id: UUID) -> Path: ...

    def get(self, conversation_id: UUID) -> ConversationContainer | None: ...

    def is_starting(self, conversation_id: UUID) -> bool: ...

    def runtime_info(self, conversation_id: UUID) -> ConversationRuntimeInfo: ...

    def resolve_persisted_cipher(self, conversation_id: UUID) -> Cipher: ...

    async def refresh_conversation(self, conversation_id: UUID) -> None: ...

    async def start(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def get_or_create(self, conversation_id: UUID) -> ConversationContainer: ...

    async def stop(self, conversation_id: UUID) -> None: ...

    async def begin_delete(self, conversation_id: UUID) -> bool: ...

    async def finish_delete(self, conversation_id: UUID) -> None: ...

    def cleanup_worktree(self, conversation_id: UUID) -> None: ...

    def attach_session(self, conversation_id: UUID) -> None: ...

    def detach_session(self, conversation_id: UUID) -> None: ...

    def acquire_lease(self, conversation_id: UUID) -> Callable[[], None]: ...

    def runtime_idle(self, conversation_id: UUID) -> AsyncIterator[bool]: ...

    def archived_marker(self, conversation_id: UUID) -> Path: ...

    def is_archived(self, conversation_id: UUID) -> bool: ...

    def add_execution_routes(self, router: APIRouter) -> None: ...


# A runtime that lacks any of these cannot serve the conversation routes; the
# host-local registry answers the same 503 the Docker-only check did.
_DRIVER_MEMBERS = (
    "provisioning",
    "get_or_create",
    "acquire_lease",
    "attach_session",
    "detach_session",
    "begin_delete",
    "finish_delete",
    "cleanup_worktree",
    "conversation_dir",
    "workspace_dir",
    "runtime_info",
)


def is_conversation_driver(registry: object) -> TypeGuard[ConversationDriver]:
    """Whether a registry implements the per-conversation runtime contract."""
    if not isinstance(registry, ConversationRegistry):
        return False
    return all(hasattr(registry, member) for member in _DRIVER_MEMBERS)