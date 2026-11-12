"""Apply the Canvas-owned git identity to a started conversation's workspace.

The frontend saves Git Username / Git Email under
``misc_settings.app_preferences`` (see ``settings_router``: ``misc_settings``
is opaque, frontend-owned data). The identity matters to the agent only, so
``ConversationService.start_conversation`` applies it here, mirroring
`OpenHands/OpenHands#17899 <https://github.com/OpenHands/OpenHands/issues/17899>`_.

Precedence, kept deliberate and documented:

- The workspace repository gets repo-local ``user.name``/``user.email`` from
  the settings. The settings are what the user configured for the agent, so
  they are authoritative for the workspace the agent commits into.
- The runtime's global identity is *filled from the settings only when unset*,
  so a brand-new ``git init`` works on a host without a ``~/.gitconfig`` while
  an existing host identity is never overwritten.
- Everything else keeps git's own resolution order.

The application is best-effort: a failure to configure git must not block the
conversation from starting.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openhands.sdk.git.exceptions import GitCommandError
from openhands.sdk.git.utils import run_git_command


logger = logging.getLogger(__name__)

_PREFERENCES_NAMESPACE = "app_preferences"
_NAME_KEY = "git_user_name"
_EMAIL_KEY = "git_user_email"


def _workspace_identity(
    misc_settings: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Return the (name, email) the frontend saved, or (None, None)."""
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


def _config_value(scope: str, key: str, workspace_dir: Path) -> str | None:
    try:
        return run_git_command(
            ["git", "config", scope, "--get", key], cwd=workspace_dir
        )
    except GitCommandError:
        return None


def _set_configs(
    workspace_dir: Path | None,
    scope: str,
    keys: tuple[tuple[str, str | None], ...],
) -> None:
    for key, value in keys:
        if value is None:
            continue
        run_git_command(
            ["git", "config", scope, key, value],
            cwd=workspace_dir,
        )


def apply_git_identity(
    workspace_dir: Path, misc_settings: Mapping[str, Any] | None
) -> None:
    """Write the stored identity into the conversation's git environment.

    Safe to run per conversation start: every write is idempotent and the
    global scope is only touched when the runtime has no identity yet.
    """
    try:
        name, email = _workspace_identity(misc_settings)
        if name is None and email is None:
            return

        # The workspace repository gets the settings identity outright, so the
        # settings win there even on a host that carries its own identity.
        repo_dir: Path | None = None
        if workspace_dir.exists():
            try:
                repo_dir = Path(
                    run_git_command(
                        ["git", "--no-pager", "rev-parse", "--show-toplevel"],
                        cwd=workspace_dir,
                    )
                )
            except GitCommandError:
                repo_dir = None

        if repo_dir is not None:
            _set_configs(
                repo_dir, "--local", (("user.name", name), ("user.email", email))
            )

        if repo_dir is None and not workspace_dir.exists():
            return

        # Fill, never overwrite: an existing global identity stays untouched.
        global_name = _config_value("--global", "user.name", workspace_dir) or None
        global_email = _config_value("--global", "user.email", workspace_dir) or None
        _set_configs(
            workspace_dir,
            "--global",
            (
                ("user.name", name if global_name is None else None),
                ("user.email", email if global_email is None else None),
            ),
        )
    except GitCommandError:
        logger.warning(
            "Could not apply the stored git identity to the conversation workspace",
            exc_info=True,
        )


__all__ = ["apply_git_identity"]
