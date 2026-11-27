"""Tests for delivering an automation bundle into a conversation workspace.

The archive is untrusted input, so these tests cover the unsafe entries a
bundle may carry and the one-time delivery a start performs. Execution itself
is covered where the start dispatches it: the bash events and the governed
proxy.
"""

import base64
import io
import json
import tarfile
import traceback

import pytest
from pydantic import ValidationError

from openhands.agent_server import automation_bundle
from openhands.agent_server.automation_bundle import (
    AutomationBundleError,
    automation_bundle_command,
    prepare_automation_bundle,
)
from openhands.sdk.conversation.request import AutomationBundle


Member = tuple[tarfile.TarInfo, bytes]


def _file(name: str, content: bytes = b"print('hello')\n") -> Member:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mode = 0o644
    return member, content


def _archive(members: list[Member]) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for member, content in members:
            tar.addfile(member, io.BytesIO(content))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _bundle(
    members: list[Member] | None = None,
    *,
    entrypoint: str = "python3 main.py",
    setup_script_path: str | None = None,
    config: dict | None = None,
) -> AutomationBundle:
    return AutomationBundle(
        entrypoint=entrypoint,
        setup_script_path=setup_script_path,
        config=config if config is not None else {},
        tarball_b64=_archive(members or [_file("main.py")]),
    )


def test_a_bundle_extracts_into_the_workspace_and_writes_its_config(tmp_path):
    workspace = tmp_path / "workspace"
    bundle = _bundle(
        [_file("main.py"), _file("setup.sh", b"pip install .\n")],
        setup_script_path="setup.sh",
        config={"repos": ["owner/repo"], "trigger_label": "bug"},
    )

    prepare_automation_bundle(bundle, workspace)

    assert (workspace / "main.py").read_text() == "print('hello')\n"
    assert (workspace / "setup.sh").read_text() == "pip install .\n"
    assert json.loads((workspace / "config.json").read_text()) == {
        "repos": ["owner/repo"],
        "trigger_label": "bug",
    }


def test_the_entrypoint_runs_after_the_setup_script_in_the_extracted_root():
    assert (
        automation_bundle_command(
            _bundle(setup_script_path="scripts/setup.sh", entrypoint="python3 main.py")
        )
        == "bash scripts/setup.sh && python3 main.py"
    )
    assert automation_bundle_command(_bundle()) == "python3 main.py"


@pytest.mark.parametrize(
    "member",
    [
        _file("/etc/passwd", b"owned\n")[0],
        _file("../escaped.txt", b"owned\n")[0],
    ],
    ids=["absolute", "parent-traversal"],
)
def test_an_archive_member_that_leaves_the_workspace_is_refused(tmp_path, member):
    workspace = tmp_path / "workspace"
    bundle = _bundle([(member, b"owned\n")])

    with pytest.raises(AutomationBundleError, match="unsafe path"):
        prepare_automation_bundle(bundle, workspace)

    assert not (workspace / "escaped.txt").exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_a_symlink_that_escapes_the_workspace_is_refused(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text("do not touch\n")
    link = tarfile.TarInfo("escape")
    link.type = tarfile.SYMTYPE
    link.linkname = "../outside.txt"
    bundle = _bundle([(link, b"")])

    with pytest.raises(AutomationBundleError):
        prepare_automation_bundle(bundle, workspace)

    assert outside.read_text() == "do not touch\n"
    assert not (workspace / "escape").exists()


def test_a_device_entry_is_refused(tmp_path):
    device = tarfile.TarInfo("device")
    device.type = tarfile.CHRTYPE
    bundle = _bundle([(device, b"")])

    with pytest.raises(AutomationBundleError, match="unsupported entry"):
        prepare_automation_bundle(bundle, tmp_path / "workspace")


def test_an_archive_that_expands_past_the_cap_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(automation_bundle, "MAX_EXTRACTED_BYTES", 3)
    bundle = _bundle([_file("big.bin", b"1234")])

    with pytest.raises(AutomationBundleError, match="expands"):
        prepare_automation_bundle(bundle, tmp_path / "workspace")


def test_an_archive_that_is_too_large_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(automation_bundle, "AUTOMATION_BUNDLE_MAX_ARCHIVE_BYTES", 1)

    with pytest.raises(AutomationBundleError, match="too large"):
        prepare_automation_bundle(_bundle(), tmp_path / "workspace")


def test_a_setup_script_outside_the_archive_is_refused(tmp_path):
    bundle = _bundle(setup_script_path="../setup.sh")

    with pytest.raises(AutomationBundleError, match="setup script path"):
        prepare_automation_bundle(bundle, tmp_path / "workspace")


def test_a_malformed_archive_is_refused(tmp_path):
    bundle = AutomationBundle(
        entrypoint="python3 main.py", config={}, tarball_b64="not base64!"
    )

    with pytest.raises(AutomationBundleError, match="base64"):
        prepare_automation_bundle(bundle, tmp_path / "workspace")


def test_an_escaping_link_names_no_member_in_the_error_chain(tmp_path):
    """The archive's contents must not surface through ``__cause__`` either."""
    marker = "ARCHIVEMEMBERMARKER"
    link = tarfile.TarInfo(marker)
    link.type = tarfile.SYMTYPE
    link.linkname = f"../{marker}.txt"
    bundle = _bundle([(link, b"")])

    with pytest.raises(AutomationBundleError) as caught:
        prepare_automation_bundle(bundle, tmp_path / "workspace")

    surfaces = [str(caught.value), repr(caught.value)]
    surfaces.extend(
        "".join(traceback.format_exception(linked))
        for linked in _exception_chain(caught.value)
    )
    assert len(_exception_chain(caught.value)) == 1
    assert all(marker not in surface for surface in surfaces)


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Every exception reachable from ``error`` by cause or context."""
    chain: list[BaseException] = []
    pending = [error]
    while pending:
        current = pending.pop()
        if current in chain:
            continue
        chain.append(current)
        pending.extend(
            link
            for link in (current.__cause__, current.__context__)
            if link is not None
        )
    return chain


def test_a_blank_setup_script_path_is_invalid_input():
    """``bash '' && entrypoint`` would skip the entrypoint; refuse it."""
    with pytest.raises(ValidationError, match="setup_script_path"):
        _bundle(setup_script_path="   ")

    with pytest.raises(AutomationBundleError, match="must not be blank"):
        automation_bundle_command(
            _bundle().model_copy(update={"setup_script_path": "   "})
        )
