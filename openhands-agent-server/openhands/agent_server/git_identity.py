"""Apply the Canvas-owned git identity to a started conversation's workspace.

The frontend saves Git Username / Git Email under
``misc_settings.app_preferences`` (see ``settings_router``: ``misc_settings``
is opaque, frontend-owned data). This module implements the server side of
`OpenHands/OpenHands#17899 <https://github.com/OpenHands/OpenHands/issues/17899>`_;
``ConversationService.start_conversation`` applies the stored identity to the
conversation's workspace.

Precedence, kept deliberate and documented:

- An identity that already resolves in the workspace (repo-local,
  includeIf-conditioned, global, or system scope) is never changed: the
  settings only fill a missing identity, they never overwrite one.
- A repository without an identity gets repo-local ``user.name``/``user.email``
  from the settings. This is repo-wide: ``git config --local`` run from a
  linked worktree writes the shared parent ``.git/config``, so the identity
  also reaches the main checkout and sibling worktrees, and stays there after
  the conversation ends.
- When nothing resolves in any scope and the workspace path exists, the
  runtime's global identity is filled from the settings, so new ``git init``
  work works on a host without a ``~/.gitconfig`` while an existing host
  identity is never overwritten. First writer wins; the fill is per runtime
  and assumes one agent-server user per HOME.
- Everything else keeps git's own resolution order.

The application is best-effort: any failure during configuration is logged and
must not block the conversation from starting.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from openhands.sdk.git.exceptions import GitCommandError
from openhands.sdk.git.utils import run_git_command


logger = logging.getLogger(__name__)

_PREFERENCES_NAMESPACE = "app_preferences"
_NAME_KEY = "git_user_name"
_EMAIL_KEY = "git_user_email"

_GitConfigScope = Literal["--local", "--global"]


def _settings_identity(
    misc_settings: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Return each saved field independently, or None when it is not set."""
    preferences = misc_settings.get(_PREFERENCES_NAMESPACE) if misc_settings else None
    if not isinstance(preferences, Mapping):
        return (None, None)

    def _clean(key: str) -> str | None:
        value = preferences.get(key)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    return (_clean(_NAME_KEY), _clean(_EMAIL_KEY))


def _resolved_value(
    cwd: Path | None, scope: _GitConfigScope | None, key: str
) -> str | None:
    try:
        args = ["git", "config"]
        if scope is not None:
            args.append(scope)
        args.extend(["--get", key])
        return run_git_command(args, cwd=cwd, expected_failure=True)
    except GitCommandError:
        return None


def _set_config(cwd: Path | None, scope: _GitConfigScope, key: str, value: str) -> None:
    run_git_command(["git", "config", scope, key, value], cwd=cwd)


def _repo_root(workspace_dir: Path) -> Path | None:
    try:
        return Path(
            run_git_command(
                ["git", "--no-pager", "rev-parse", "--show-toplevel"],
                cwd=workspace_dir,
                expected_failure=True,
            )
        )
    except GitCommandError:
        return None


def apply_git_identity(
    workspace_dir: Path, misc_settings: Mapping[str, Any] | None
) -> None:
    """Write the stored identity into the conversation's git environment.

    Idempotent per conversation start: every write fills a missing value, so a
    repeat start never rewrites an existing identity.
    """
    try:
        name, email = _settings_identity(misc_settings)
        if name is None and email is None:
            return
        if not workspace_dir.is_absolute():
            # A relative working_dir would resolve against the agent-server's
            # process CWD and could name an unrelated host directory.
            return

        # Probe the resolved identity (all scopes) before anything is written,
        # because a repo-local fill of one's own would otherwise mask whether a
        # global fill is needed at all.
        cwd = workspace_dir if workspace_dir.exists() else None
        repo_root = _repo_root(workspace_dir) if cwd is not None else None
        resolved_name = _resolved_value(cwd, None, "user.name")
        resolved_email = _resolved_value(cwd, None, "user.email")

        if repo_root is not None:
            _fill_missing(
                repo_root,
                "--local",
                {
                    "user.name": (
                        name,
                        _resolved_value(repo_root, "--local", "user.name"),
                    ),
                    "user.email": (
                        email,
                        _resolved_value(repo_root, "--local", "user.email"),
                    ),
                },
            )
        _fill_missing(
            cwd,
            "--global",
            {
                "user.name": (name, resolved_name),
                "user.email": (email, resolved_email),
            },
        )
    except Exception:
        # Best-effort by contract: OSError from a bad cwd (a path that exists
        # but is a file, an unreadable directory, a lost mount), ValueError
        # from a malformed path, or anything else must not block the start.
        logger.warning(
            "Could not apply the stored git identity to the conversation workspace",
            exc_info=True,
        )


def _fill_missing(
    cwd: Path | None,
    scope: _GitConfigScope,
    candidates: Mapping[str, tuple[str | None, str | None]],
) -> None:
    """Set each key whose resolved value is missing; skip the present ones."""
    for key, (setting, current) in candidates.items():
        if setting is not None and current is None:
            _set_config(cwd, scope, key, setting)


__all__ = ["apply_git_identity"]
