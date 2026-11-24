"""Conversation routes for the Docker runtime."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, NamedTuple
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, Query, Request, WebSocket
from starlette.responses import JSONResponse, Response, StreamingResponse

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_registry import ConversationRegistry
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import (
    get_config,
    get_conversation_registry,
    get_conversation_service,
)
from openhands.agent_server.docker_runtime.mediation import materialize_secrets
from openhands.agent_server.docker_runtime.openshell.events import (
    governed_hook_config,
    governed_launch_additions,
)
from openhands.agent_server.docker_runtime.provisioning import MemoryScope
from openhands.agent_server.docker_runtime.proxy import (
    bridge_websocket,
    proxy_http,
    strip_auth_query,
)
from openhands.agent_server.docker_runtime.registry import (
    ConversationContainer,
    DockerConversationRegistry,
    RuntimeArchivedError,
)
from openhands.agent_server.event_router import require_feature_enabled
from openhands.agent_server.launch import (
    forward_to_runtime,
    launch_error,
    launch_source,
    request_secrets,
    server_launch_stores,
)
from openhands.agent_server.models import (
    ConversationRuntimeInfo,
    ConversationRuntimeStatus,
    StartConversationRequest,
    UpdateSecretsRequest,
)
from openhands.agent_server.persistence import PersistedSettings, run_store
from openhands.agent_server.utils import safe_rmtree
from openhands.sdk.conversation.request import ConversationContainerOptions
from openhands.sdk.errors import (
    Conflict,
    DomainError,
    Gone,
    Invalid,
    NotFound,
    NotSupported,
    Unavailable,
    Unprocessable,
    UpstreamFailed,
)
from openhands.sdk.git.exceptions import GitCommandError, GitRepositoryError
from openhands.sdk.git.utils import run_git_command, validate_git_repository
from openhands.sdk.launch import (
    AgentLaunchError,
    LaunchSource,
    LaunchStoreError,
    ResolvedLaunch,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.profiles.agent_profile import LaunchedAgentProfile
from openhands.sdk.profiles.resolver import ProfileNotFound
from openhands.sdk.utils.cipher import Cipher
from openhands.sdk.utils.pydantic_secrets import cipher_context, client_context
from openhands.sdk.workspace import LocalWorkspace


logger = get_logger(__name__)


def get_registry(registry: ConversationRegistry) -> DockerConversationRegistry:
    if not isinstance(registry, DockerConversationRegistry):
        raise Unavailable("Container conversation runtime is unavailable")
    return registry


async def _container(
    registry: DockerConversationRegistry, conversation_id: UUID
) -> ConversationContainer:
    if not registry.provisioning.manifest_path(conversation_id).is_file():
        raise NotFound("Conversation not found")
    try:
        return await registry.get_or_create(conversation_id)
    except RuntimeArchivedError as exc:
        raise Gone(
            "Conversation runtime was archived; its history is read-only"
        ) from exc
    except Exception as exc:
        logger.exception("Could not start conversation container %s", conversation_id)
        raise UpstreamFailed("Could not start conversation container") from exc


def _upstream_path(request: Request, path: str) -> str:
    query = strip_auth_query("?" + request.url.query).lstrip("?")
    return f"{path}?{query}" if query else path


async def _proxy_with_session(
    registry: DockerConversationRegistry,
    conversation_id: UUID,
    container: ConversationContainer,
    request: Request,
    *,
    upstream_path: str,
    body: bytes | None = None,
) -> StreamingResponse:
    """Proxy one request while holding an outer session attachment.

    The attachment is released when the streamed response finishes (or the
    client disconnects), so a long-lived proxied stream keeps the container
    from being evicted while a client is still reading from it.
    """
    registry.attach_session(conversation_id)
    release = registry.acquire_lease(conversation_id)

    detached = False

    async def detach() -> None:
        nonlocal detached
        if not detached:
            detached = True
            release()
            registry.detach_session(conversation_id)

    try:
        return await proxy_http(
            request,
            container,
            upstream_path=upstream_path,
            body=body,
            on_close=detach,
        )
    except BaseException:
        await detach()
        raise


docker_conversation_router = APIRouter(
    prefix="/conversations", tags=["Docker Conversations"]
)


def _worktree_git_paths(workspace_path: Path) -> dict[str, Path]:
    """The repository and git common dir a worktree start records, if any."""
    try:
        validate_git_repository(workspace_path)
        repo_root = Path(
            run_git_command(
                ["git", "--no-pager", "rev-parse", "--show-toplevel"],
                workspace_path,
            )
        ).resolve()
        git_common_dir = Path(
            run_git_command(
                ["git", "--no-pager", "rev-parse", "--git-common-dir"],
                workspace_path,
            )
        )
    except (GitCommandError, GitRepositoryError):
        return {}
    if not git_common_dir.is_absolute():
        git_common_dir = workspace_path / git_common_dir
    return {
        "worktree_repository_path": repo_root,
        "worktree_git_common_dir_path": git_common_dir.resolve(),
    }


# Canvas sends ``/workspace`` for a backend it treats as the Docker runtime,
# where that path is the default workspace. A governed runtime maps the alias to
# its own sandbox root; another absolute path is a host path unless it lies
# under a runtime root (see ``_is_runtime_workspace``).
_DEFAULT_WORKSPACE_ALIASES = frozenset({"/workspace"})

# A governed start may carry this gateway placeholder instead of a key; the
# provider resolves it at egress, so the key never enters the sandbox.
_OPEN_SHELL_KEY_PREFIX = "openshell:resolve:"


def _llm_key_of(holder: object) -> str | None:
    """The model key a settings model, an agent, or a raw settings dict carries."""
    key: object = None
    if isinstance(holder, dict):
        llm = holder.get("llm")
        key = llm.get("api_key") if isinstance(llm, dict) else None
    else:
        key = getattr(getattr(holder, "llm", None), "api_key", None)
    if key is None:
        return None
    secret: str = (
        key.get_secret_value()  # type: ignore[attr-defined]
        if hasattr(key, "get_secret_value")
        else str(key)
    )
    return secret or None


def _model_key_values(
    request: StartConversationRequest, source: LaunchSource | None
) -> list[tuple[str, str]]:
    """Every model key this start would send into the runtime, by origin."""
    holders: list[tuple[str, object]] = []
    if source is not None:
        holders.append(
            (
                "profile",
                source.settings if isinstance(source, ResolvedLaunch) else source,
            )
        )
    else:
        holders.append(("agent_settings", request.agent_settings))
        holders.append(("agent", request.agent))
    values: list[tuple[str, str]] = []
    for origin, holder in holders:
        secret = _llm_key_of(holder)
        if secret is not None:
            values.append((origin, secret))
    return values


def _refuse_model_key(
    request: StartConversationRequest, source: LaunchSource | None
) -> None:
    """Refuse a governed start that would carry a real model key inside."""
    for origin, value in _model_key_values(request, source):
        if not value.startswith(_OPEN_SHELL_KEY_PREFIX):
            raise AgentLaunchError(
                f"The {origin} model key would enter the sandbox. A governed "
                "conversation accepts only the gateway placeholder "
                "(openshell:resolve:...); store that placeholder in the LLM "
                "profile instead of a real key."
            )


def _strip_governed_mcp(source: LaunchSource) -> tuple[LaunchSource, list[str]]:
    """Drop every MCP server a governed start would forward (R14).

    Works on both launch sources: resolved agent settings (from a stored
    profile or request ``agent_settings``) and a request-supplied agent. The
    strip is unconditional: governance is policy, not an option.
    """
    if isinstance(source, ResolvedLaunch):
        dropped = sorted(source.settings.mcp_config)
        if not dropped:
            return source, []
        cleared = source.settings.model_copy(update={"mcp_config": {}})
        return ResolvedLaunch(settings=cleared, profile=source.profile), dropped
    dropped = sorted(source.mcp_config)
    if not dropped:
        return source, []
    return source.model_copy(update={"mcp_config": {}}), dropped


def _is_runtime_workspace(
    requested: str, inner_root: str, workspace_root: Path
) -> bool:
    """Whether a requested directory belongs to the runtime, not the host.

    A relative path resolves inside the runtime, never on the host. An
    absolute path belongs to the runtime only under a known runtime root: the
    Docker alias, the driver's inner root, or this server's configured
    workspace root.

    Canvas anchors its default working directory on this server's file home,
    so a governed start names a directory under the configured workspace root.
    The start rewrites every accepted path to the inner root; the sandbox never
    sees the host tree, and any other absolute path stays a refusal.
    """
    if not isinstance(requested, str) or not requested:
        return False
    path = PurePosixPath(requested)
    if not path.is_absolute():
        return True
    roots = (*_DEFAULT_WORKSPACE_ALIASES, inner_root, workspace_root.as_posix())
    return any(path.is_relative_to(PurePosixPath(root)) for root in roots)


def _resolve_workspace(
    registry: DockerConversationRegistry,
    worktree: object,
    workspace: dict[str, Any] | None,
) -> tuple[str, Path | None]:
    """Resolve the requested workspace, refusing a governed host path."""
    inner_root = registry.inner_workspace_root
    # A null working directory means "the driver's root", like a missing one.
    working_dir = (workspace or {}).get("working_dir") or inner_root
    if registry.governed:
        runtime_workspace = _is_runtime_workspace(
            working_dir, inner_root, registry.config.workspace_path
        )
        if worktree or not runtime_workspace:
            raise Unprocessable(
                "A governed conversation runs only in the sandbox workspace"
            )
        working_dir = inner_root
    return working_dir, None if working_dir == inner_root else Path(working_dir)


@docker_conversation_router.post("")
async def start_conversation(
    request: Request,
    include_skills: Annotated[bool, Query()] = False,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> JSONResponse:
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Invalid("Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise Unprocessable("Expected a JSON object")
    status_code, content, conversation_id = await start_docker_conversation(
        body,
        get_registry(conversation_registry),
        include_skills=include_skills,
    )
    if status_code < 400:
        await conversation_service.refresh_persisted_conversation(conversation_id)
    return JSONResponse(content=content, status_code=status_code)


async def start_docker_conversation(
    body: dict[str, Any],
    registry: DockerConversationRegistry,
    *,
    include_skills: bool = False,
) -> tuple[int, Any, UUID]:
    """Start a Docker conversation through the same path as the REST route."""
    body = body.copy()
    requested_runtime = body.pop("conversation_runtime", None)
    served_runtime = "openshell" if registry.governed else "docker"
    if requested_runtime not in (None, served_runtime):
        raise Unprocessable(f"This server only runs {served_runtime} conversations")

    workspace = body.get("workspace")
    if workspace is not None and (
        not isinstance(workspace, dict)
        or workspace.get("kind", "LocalWorkspace") != "LocalWorkspace"
    ):
        raise Unprocessable("Docker conversations require a local workspace")
    working_dir, host_workspace = _resolve_workspace(
        registry, body.get("worktree"), workspace
    )

    try:
        conversation_id = (
            UUID(body["conversation_id"]) if body.get("conversation_id") else uuid4()
        )
    except (TypeError, ValueError) as exc:
        raise Invalid("Invalid conversation_id") from exc
    body["conversation_id"] = str(conversation_id)
    body["workspace"] = {
        "kind": "LocalWorkspace",
        "working_dir": registry.inner_workspace_root,
    }

    # Only a conversation the container already created is left alone on
    # failure; a manifest from a failed earlier start is retried like a new one.
    try:
        existing = (
            registry.conversation_dir(conversation_id).joinpath("meta.json").is_file()
        )
    except ValueError as exc:
        raise Unprocessable(str(exc)) from exc
    release = None
    try:
        plan = await _prepare_forward(body, registry, existing=existing)
        identity = await run_store(
            registry.provisioning.create, conversation_id, host_workspace
        )
        updates: dict[str, Any] = {}
        if plan.memory_scope is not None and (
            not existing or identity.memory_scope is None
        ):
            # A persisted scope is never overwritten, so two runs of one
            # automation always share a namespace.
            updates["memory_scope"] = plan.memory_scope
        if plan.worktree:
            updates.update(
                await asyncio.to_thread(_worktree_git_paths, identity.workspace_path)
            )
        if plan.launched is not None and identity.launched_agent_profile is None:
            updates["launched_agent_profile"] = plan.launched
        # Fixed once the container accepted the conversation; until then a
        # retried start may change them.
        if not existing and identity.container != plan.container:
            updates["container"] = plan.container
        if updates:
            identity = identity.model_copy(update=updates)
            await run_store(registry.provisioning.save, identity)
        container = await registry.get_or_create(conversation_id)
        release = registry.acquire_lease(conversation_id)
        forwarded = await asyncio.to_thread(plan.payload, identity.cipher)
        if plan.worktree and identity.worktree_repository_path is not None:
            forwarded["workspace"] = LocalWorkspace(
                working_dir=identity.workspace_path
            ).model_dump(mode="json")
        async with httpx.AsyncClient(
            timeout=60, verify=container.ca_bundle or True
        ) as client:
            response = await client.post(
                f"{container.host}/api/conversations",
                params={"include_skills": include_skills},
                headers={"X-Session-API-Key": container.api_key},
                json=forwarded,
            )
    except (ProfileNotFound, AgentLaunchError, LaunchStoreError) as exc:
        raise launch_error(exc) from exc
    except ValueError as exc:
        raise Unprocessable(str(exc)) from exc
    except RuntimeArchivedError as exc:
        raise Gone(
            "Conversation runtime was archived; its history is read-only"
        ) from exc
    except httpx.HTTPError as exc:
        if not existing:
            await registry.stop(conversation_id)
        raise UpstreamFailed(
            "Conversation container did not accept the request"
        ) from exc
    except Exception as exc:
        if not existing:
            await registry.stop(conversation_id)
        logger.exception("Could not create conversation container")
        raise UpstreamFailed("Could not create conversation container") from exc
    finally:
        if release is not None:
            release()

    content = response.json() if response.content else None
    if response.is_error:
        if not existing:
            await registry.stop(conversation_id)
        content = {"detail": "Conversation runtime rejected the request"}
    else:
        await asyncio.to_thread(registry.copy_conversation_record, conversation_id)
    return response.status_code, content, conversation_id


class _ForwardPlan(NamedTuple):
    """What the router needs after resolving a start request."""

    payload: Callable[[Cipher], dict[str, Any]]
    launched: LaunchedAgentProfile | None
    container: ConversationContainerOptions | None
    worktree: bool
    memory_scope: MemoryScope | None


async def _prepare_forward(
    body: dict[str, Any],
    registry: DockerConversationRegistry,
    *,
    existing: bool,
) -> _ForwardPlan:
    """Resolve the start request with this host's stores; the container finalizes.

    The profile of a conversation that already exists is not resolved again:
    its container returns the conversation as it is.
    """
    config = registry.config
    start = StartConversationRequest.model_validate(
        body,
        context=(
            cipher_context(config.cipher) if body.get("secrets_encrypted") else None
        ),
    )
    try:
        settings = await run_store(registry.stores.settings.load)
    except (OSError, PermissionError):
        settings = None
    settings = settings or PersistedSettings()
    context = settings.agent_settings.agent_context
    load_memory = context is not None and context.load_memory
    secrets = await asyncio.to_thread(request_secrets, start, config.cipher)
    source = (
        None
        if existing and start.agent_profile_id is not None
        else await asyncio.to_thread(
            launch_source,
            start,
            lambda: server_launch_stores(settings, registry.stores, config.cipher),
            config.cipher,
        )
    )
    if registry.governed:
        _refuse_model_key(start, source)
        if source is not None:
            source, dropped_mcp = _strip_governed_mcp(source)
            if dropped_mcp:
                # R14: the operator should know what governance dropped.
                logger.warning(
                    "Governed conversation drops MCP servers: %s",
                    ", ".join(dropped_mcp),
                )
    if registry.governed and not existing:
        memory_configured = bool(
            registry.config.memory_token_secret and registry.config.memory_api_url
        )
        # Server-side governance injection: the standing rules ride the launch
        # additions; the notices and memory hooks ride the hook config. Both
        # are existing wire fields, so the start schema stays untouched.
        start = start.model_copy(
            update={
                "hook_config": governed_hook_config(
                    start.hook_config, memory_configured=memory_configured
                ),
                "agent_launch_additions": governed_launch_additions(
                    start.agent_launch_additions,
                    memory_configured=memory_configured,
                ),
            }
        )

    def payload(cipher: Cipher) -> dict[str, Any]:
        return forward_to_runtime(
            start, source, secrets, cipher, load_memory=load_memory
        )

    conversation_id = start.conversation_id
    if conversation_id is None:
        raise Unprocessable("A conversation id is required")
    launched = source.profile if isinstance(source, ResolvedLaunch) else None
    memory_scope = (
        MemoryScope.from_tags(conversation_id, start.tags)
        if registry.governed
        else None
    )
    return _ForwardPlan(
        payload=payload,
        launched=launched,
        container=start.container,
        worktree=start.worktree,
        memory_scope=memory_scope,
    )


@docker_conversation_router.get(
    "/{conversation_id}/runtime", response_model=ConversationRuntimeInfo
)
async def runtime_info(
    conversation_id: UUID,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
) -> ConversationRuntimeInfo:
    registry = get_registry(conversation_registry)
    if not registry.conversation_dir(conversation_id).joinpath("meta.json").is_file():
        raise NotFound("Conversation not found")
    return registry.runtime_info(conversation_id)


@docker_conversation_router.post("/{conversation_id}/runtime/credentials")
async def runtime_credentials(
    conversation_id: UUID,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
) -> JSONResponse:
    """Return the key for an authenticated client attached to this runtime."""
    registry = get_registry(conversation_registry)
    # A governed conversation's key lives only in the outer proxy. Serving it
    # here would leak the key; answer 404 like a missing conversation instead.
    if registry.governed:
        raise NotFound("Conversation not found")
    if not registry.conversation_dir(conversation_id).joinpath("meta.json").is_file():
        raise NotFound("Conversation not found")
    identity = await run_store(registry.provisioning.load, conversation_id)
    return JSONResponse({"session_api_key": identity.api_key.get_secret_value()})


@docker_conversation_router.delete("/{conversation_id}/runtime", status_code=204)
async def release_runtime(
    conversation_id: UUID,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Response:
    """Stop the container while retaining conversation and workspace state."""
    registry = get_registry(conversation_registry)
    if not registry.conversation_dir(conversation_id).joinpath("meta.json").is_file():
        raise NotFound("Conversation not found")
    try:
        await registry.stop(conversation_id)
        await conversation_service.refresh_persisted_conversation(conversation_id)
    except Exception as exc:
        logger.exception("Could not release conversation runtime %s", conversation_id)
        raise UpstreamFailed("Could not release conversation runtime") from exc
    return Response(status_code=204)


@docker_conversation_router.post(
    "/{conversation_id}/runtime/reprovision", response_model=ConversationRuntimeInfo
)
async def reprovision_runtime(
    conversation_id: UUID,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
) -> ConversationRuntimeInfo:
    registry = get_registry(conversation_registry)
    await _container(registry, conversation_id)
    return ConversationRuntimeInfo(
        runtime_status=ConversationRuntimeStatus.AVAILABLE, can_resume=True
    )


@docker_conversation_router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Response:
    registry = get_registry(conversation_registry)
    if not registry.provisioning.manifest_path(conversation_id).is_file():
        raise NotFound("Conversation not found")

    if not await registry.begin_delete(conversation_id):
        raise Conflict("Conversation deletion is already in progress")

    # Stopping the container closes its conversation service. The outer server
    # owns the bind-mounted state and removes it after Docker has unmounted it;
    # asking the inner server to remove the mount root leaves that root in a
    # partially deleted state.
    try:
        await registry.stop(conversation_id)
        await asyncio.to_thread(registry.delete_runtime, conversation_id)
        try:
            await asyncio.to_thread(registry.cleanup_worktree, conversation_id)
        except (OSError, subprocess.SubprocessError):
            logger.exception(
                "Could not clean up conversation worktree %s", conversation_id
            )
        runtime_dir = await run_store(
            registry.provisioning.runtime_dir, conversation_id
        )
        await asyncio.to_thread(safe_rmtree, runtime_dir)
        await asyncio.to_thread(safe_rmtree, registry.conversation_dir(conversation_id))
        registry.provisioning.manifest_path(conversation_id).unlink(missing_ok=True)
        registry.provisioning.policy_state_path(conversation_id).unlink(missing_ok=True)
        registry.archived_marker(conversation_id).unlink(missing_ok=True)
        await conversation_service.refresh_persisted_conversation(conversation_id)
    finally:
        await registry.finish_delete(conversation_id)
    return Response(status_code=200)


@docker_conversation_router.api_route(
    "/{conversation_id}",
    methods=["POST", "PUT", "PATCH", "OPTIONS"],
)
async def proxy_conversation_root(
    conversation_id: UUID,
    request: Request,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> StreamingResponse:
    return await proxy_conversation(
        conversation_id, "", request, conversation_registry, conversation_service
    )


@docker_conversation_router.api_route(
    "/{conversation_id}/{tail:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_conversation(
    conversation_id: UUID,
    tail: str,
    request: Request,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> StreamingResponse:
    if tail.split("/", 1)[0] in {
        "credential-bindings",
        "fork",
        "switch_llm",
        "switch_profile",
    }:
        raise NotSupported("This operation is unavailable in Docker runtime mode")
    registry = get_registry(conversation_registry)
    # The runtime's server has no settings of its own to check: the user's
    # switches live here and are followed at once.
    if tail == "reactions" and request.method == "POST":
        await require_feature_enabled(
            registry.stores.settings,
            "conversation_reactions",
            "Conversation reactions are turned off",
        )
    if tail == "ask_user/answer" and request.method == "POST":
        await require_feature_enabled(
            registry.stores.settings, "ask_user", "Decision cards are turned off"
        )
    container = await _container(registry, conversation_id)
    upstream_path = _upstream_path(
        request,
        f"/api/conversations/{conversation_id}" + (f"/{tail}" if tail else ""),
    )
    if tail == "secrets" and request.method == "POST":
        try:
            update = UpdateSecretsRequest.model_validate_json(await request.body())
        except ValueError as exc:
            raise Unprocessable("Invalid secrets payload") from exc
        identity = await run_store(registry.provisioning.load, conversation_id)
        profile = identity.launched_agent_profile
        secrets = update.secrets
        if profile is not None:
            secrets = {
                name: source
                for name, source in secrets.items()
                if profile.allows_secret(name)
            }
        materialized = await asyncio.to_thread(materialize_secrets, secrets)
        body = UpdateSecretsRequest(secrets=materialized).model_dump(
            mode="json", context=client_context("plaintext")
        )
        response = await _proxy_with_session(
            registry,
            conversation_id,
            container,
            request,
            upstream_path=upstream_path,
            body=json.dumps(body).encode(),
        )
    else:
        response = await _proxy_with_session(
            registry,
            conversation_id,
            container,
            request,
            upstream_path=upstream_path,
        )
    if request.method not in {"GET", "HEAD", "OPTIONS"} and response.status_code < 400:
        await registry.sync_conversation_record(conversation_id, conversation_service)
    return response


docker_workspace_router = APIRouter(prefix="/conversations", tags=["Docker Workspace"])


@docker_workspace_router.get("/{conversation_id}/workspace/{file_path:path}")
async def proxy_workspace_file(
    conversation_id: UUID,
    file_path: str,
    request: Request,
    conversation_registry: ConversationRegistry = Depends(get_conversation_registry),
) -> StreamingResponse:
    registry = get_registry(conversation_registry)
    container = await _container(registry, conversation_id)
    return await _proxy_with_session(
        registry,
        conversation_id,
        container,
        request,
        upstream_path=_upstream_path(
            request, f"/api/conversations/{conversation_id}/workspace/{file_path}"
        ),
    )


docker_sockets_router = APIRouter(prefix="/sockets", tags=["Docker WebSockets"])
docker_session_sockets_router = APIRouter(prefix="/sockets", tags=["Docker WebSockets"])


async def _proxy_socket(
    websocket: WebSocket,
    conversation_id: UUID,
    session_api_key: str | None,
    socket_name: str,
    config: Config,
    registry: ConversationRegistry,
    conversation_service: ConversationService,
) -> None:
    from openhands.agent_server.sockets import _accept_authenticated_websocket

    if not await _accept_authenticated_websocket(websocket, session_api_key, config):
        return
    if not isinstance(registry, DockerConversationRegistry):
        await websocket.close(code=1011)
        return
    try:
        container = await _container(registry, conversation_id)
    except DomainError as exc:
        await websocket.close(code=1008 if isinstance(exc, NotFound) else 1011)
        return
    release = registry.acquire_lease(conversation_id)
    query = strip_auth_query("?" + websocket.url.query).lstrip("?")
    path = f"/sockets/{socket_name}/{conversation_id}"
    # Hold the attachment for the whole bridge lifetime so the idle-eviction
    # loop cannot stop the container under a live events/session websocket.
    registry.attach_session(conversation_id)
    try:
        await bridge_websocket(
            websocket,
            container,
            upstream_path=f"{path}?{query}" if query else path,
        )
    finally:
        if release is not None:
            release()
        registry.detach_session(conversation_id)
        # A socket detach is not a turn boundary: pull the record after detach
        # so a turn that finished while attached is visible.
        await registry.sync_conversation_record(conversation_id, conversation_service)


@docker_sockets_router.websocket("/events/{conversation_id}")
async def proxy_events(
    websocket: WebSocket,
    conversation_id: UUID,
    session_api_key: Annotated[str | None, Query(alias="session_api_key")] = None,
    config: Config = Depends(get_config),
    registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> None:
    await _proxy_socket(
        websocket,
        conversation_id,
        session_api_key,
        "events",
        config,
        registry,
        conversation_service,
    )


@docker_session_sockets_router.websocket("/session/{conversation_id}")
async def proxy_session(
    websocket: WebSocket,
    conversation_id: UUID,
    session_api_key: Annotated[str | None, Query(alias="session_api_key")] = None,
    config: Config = Depends(get_config),
    registry: ConversationRegistry = Depends(get_conversation_registry),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> None:
    await _proxy_socket(
        websocket,
        conversation_id,
        session_api_key,
        "session",
        config,
        registry,
        conversation_service,
    )
