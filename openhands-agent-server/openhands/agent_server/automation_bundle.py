"""Deliver an automation bundle into a conversation's workspace.

A governed automation run arrives as one conversation start carrying a
gzipped tar, an entrypoint command, an optional setup script and a config.
The archive comes from an automation author, so it is checked before any
member is extracted; an escaping link is refused during extraction, after
earlier members are written. Neither the archive nor its contents are ever
logged. Nothing here runs the bundle: the entrypoint is dispatched through
the server's bash events, which is the surface run logs already read.
"""

from __future__ import annotations

import base64
import io
import json
import shlex
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

from openhands.sdk.conversation.request import (
    AUTOMATION_BUNDLE_MAX_ARCHIVE_BYTES,
    AutomationBundle,
)
from openhands.sdk.launch import AgentLaunchError


CONFIG_FILENAME: Final[str] = "config.json"
MAX_EXTRACTED_BYTES: Final[int] = 256 * 1024 * 1024
MAX_EXTRACTED_MEMBERS: Final[int] = 10_000
# The conversation's own runtime lifecycle bounds a run. This only stops a
# leaked entrypoint from outliving it.
ENTRYPOINT_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60


class AutomationBundleError(AgentLaunchError):
    """A bundle that cannot be delivered as given.

    Messages never name an archive member: the archive is untrusted input and
    its contents must not reach a log or an error response.
    """

    code = "invalid_automation_bundle"


def _decode_archive(bundle: AutomationBundle) -> bytes:
    try:
        archive = base64.b64decode(bundle.tarball_b64, validate=True)
    except ValueError:
        # `from None` drops the decoder's context; only this message is ours.
        raise AutomationBundleError(
            "The automation bundle archive is not valid base64"
        ) from None
    if len(archive) > AUTOMATION_BUNDLE_MAX_ARCHIVE_BYTES:
        raise AutomationBundleError("The automation bundle archive is too large")
    return archive


def _validated_members(tar: tarfile.TarFile) -> None:
    """Refuse the entries this reader can check, before extraction starts.

    Absolute names and ``..`` traversal, and device or FIFO entries, are
    refused here; escaping links are refused by tarfile's ``data`` filter
    during extraction, after earlier members are written. The size bound is
    applied to the declared member sizes so an archive cannot fill the
    workspace.
    """
    try:
        members = tar.getmembers()
    except (tarfile.TarError, OSError):
        members = None  # a tar error can name a member; see the raise below
    if members is None:
        raise AutomationBundleError(
            "The automation bundle is not a readable tar archive"
        )
    if len(members) > MAX_EXTRACTED_MEMBERS:
        raise AutomationBundleError("The automation bundle has too many entries")
    declared_bytes = 0
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise AutomationBundleError("The automation bundle contains an unsafe path")
        if not (member.isfile() or member.isdir() or member.islnk() or member.issym()):
            raise AutomationBundleError(
                "The automation bundle contains an unsupported entry"
            )
        if member.isfile():
            declared_bytes += member.size
    if declared_bytes > MAX_EXTRACTED_BYTES:
        raise AutomationBundleError(
            "The automation bundle expands to more than the workspace allows"
        )


def _validated_setup_script(setup_script_path: str | None) -> str | None:
    if setup_script_path is None:
        return None
    if not setup_script_path.strip():
        raise AutomationBundleError(
            "The automation bundle's setup script path must not be blank"
        )
    path = PurePosixPath(setup_script_path)
    if path.is_absolute() or ".." in path.parts:
        raise AutomationBundleError(
            "The automation bundle's setup script path must stay inside the archive"
        )
    return setup_script_path


def prepare_automation_bundle(bundle: AutomationBundle, workspace: str | Path) -> None:
    """Validate the archive, extract it into ``workspace`` and write the config.

    Extraction runs with tarfile's ``data`` filter, so an escaping link or a
    device entry is refused, and every member lands under ``workspace``. The
    setup script path is checked here rather than when it runs, so a bad path
    refuses the start before any file exists.
    """
    _validated_setup_script(bundle.setup_script_path)
    archive = _decode_archive(bundle)
    root = Path(workspace)
    failed = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            _validated_members(tar)
            tar.extractall(path=root, filter="data")
    except AutomationBundleError:
        raise
    except (tarfile.TarError, OSError, ValueError):
        # Raised outside the handler: tarfile's filter errors name the member,
        # and an exception chain would keep that name in ``__context__``.
        failed = True
    if failed:
        raise AutomationBundleError("The automation bundle could not be extracted")
    write_workspace_config(bundle.config, root)


def automation_bundle_command(bundle: AutomationBundle) -> str:
    """The setup-then-entrypoint command, run from the extracted root."""
    setup_script_path = _validated_setup_script(bundle.setup_script_path)
    if setup_script_path is None:
        return bundle.entrypoint
    return f"bash {shlex.quote(setup_script_path)} && {bundle.entrypoint}"


def write_workspace_config(config: Mapping[str, Any], workspace: str | Path) -> None:
    """Write ``config`` as ``config.json`` in the conversation workspace.

    A bundle's entrypoint runs from the workspace root and reads this file
    there.
    """
    path = Path(workspace) / CONFIG_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(config), indent=2, sort_keys=True))
    except OSError as exc:
        raise AgentLaunchError(
            "The automation config could not be written into the workspace"
        ) from exc
