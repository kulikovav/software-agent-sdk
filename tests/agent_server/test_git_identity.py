"""The Application-settings git identity reaches the workspace.

Canvas saves Git Username / Git Email under ``misc_settings.app_preferences``.
The conversation start must apply that identity to the workspace:

- a repository without an identity gets repo-local ``user.name``/``user.email``
  from the settings, so the settings fill the workspace the agent commits into;
- the runtime's global identity is filled from the settings only when nothing
  resolves in any git scope, so ``git init``ed sub-repositories work on a host
  without a ``~/.gitconfig`` and an existing host identity is never clobbered;
- any failure while doing so must not block the conversation from starting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openhands.agent_server import git_identity
from openhands.agent_server.git_identity import apply_git_identity
from openhands.agent_server.models import StartConversationRequest
from openhands.agent_server.persistence import PersistedSettings
from openhands.sdk import LLM, Agent
from openhands.sdk.git.exceptions import GitCommandError
from openhands.sdk.git.utils import run_git_command
from openhands.sdk.workspace import LocalWorkspace
from tests.agent_server.test_agent_profile_conv_start import _start


def _git(*args: str, cwd: Path | None = None) -> str:
    return run_git_command(["git", *args], cwd=cwd).strip()


def _git_or_empty(*args: str, cwd: Path | None = None) -> str:
    try:
        return _git(*args, cwd=cwd)
    except GitCommandError:
        return ""


def _repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "workspace"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# test repo\n")
    _git("init", "-q", "-b", "main", cwd=repo_dir)
    return repo_dir


def _settings_identity(
    name: str = "QA Tester", email: str = "qa-tester@example.com"
) -> PersistedSettings:
    return PersistedSettings(
        misc_settings={
            "app_preferences": {"git_user_name": name, "git_user_email": email}
        }
    )


@pytest.fixture(autouse=True)
def isolated_global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point git's global and system scopes at test files, never at real ones."""
    config_file = tmp_path / "gitconfig"
    config_file.touch()
    system_file = tmp_path / "system-gitconfig"
    system_file.touch()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config_file))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_file))


def test_workspace_repo_receives_settings_identity(tmp_path: Path) -> None:
    repo_dir = _repo(tmp_path)
    apply_git_identity(repo_dir, _settings_identity().misc_settings)

    assert _git("config", "user.name", cwd=repo_dir) == "QA Tester"
    assert _git("config", "user.email", cwd=repo_dir) == "qa-tester@example.com"


def test_global_identity_filled_when_unset(tmp_path: Path) -> None:
    repo_dir = _repo(tmp_path)
    apply_git_identity(repo_dir, _settings_identity().misc_settings)

    assert _git("config", "--global", "user.name") == "QA Tester"
    assert _git("config", "--global", "user.email") == "qa-tester@example.com"


def test_repo_with_own_identity_is_kept(tmp_path: Path) -> None:
    repo_dir = _repo(tmp_path)
    _git("config", "--local", "user.name", "Repo Person", cwd=repo_dir)
    _git("config", "--local", "user.email", "repo@example.com", cwd=repo_dir)

    apply_git_identity(repo_dir, _settings_identity().misc_settings)

    assert _git("config", "user.name", cwd=repo_dir) == "Repo Person"
    assert _git("config", "user.email", cwd=repo_dir) == "repo@example.com"


def test_existing_global_identity_is_kept(tmp_path: Path) -> None:
    _git("config", "--global", "user.name", "Host User")
    _git("config", "--global", "user.email", "host@example.com")
    repo_dir = _repo(tmp_path)

    apply_git_identity(repo_dir, _settings_identity().misc_settings)

    # The workspace still gets the settings, but the host identity survives.
    assert _git("config", "user.name", cwd=repo_dir) == "QA Tester"
    assert _git("config", "--global", "user.name") == "Host User"


def test_system_identity_is_not_shadowed(tmp_path: Path) -> None:
    _git("config", "--system", "user.name", "System User")

    apply_git_identity(tmp_path, _settings_identity().misc_settings)

    assert _git_or_empty("config", "--global", "user.name") == ""


def test_empty_settings_leave_git_untouched(tmp_path: Path) -> None:
    repo_dir = _repo(tmp_path)
    for misc in (
        {},
        {"app_preferences": {}},
        {"app_preferences": {"git_user_name": "", "git_user_email": " "}},
    ):
        apply_git_identity(repo_dir, misc)

    assert _git_or_empty("config", "user.name", cwd=repo_dir) == ""
    assert _git_or_empty("config", "--global", "user.name") == ""


def test_workspace_without_git_repo_still_fills_global(tmp_path: Path) -> None:
    plain_dir = tmp_path / "workspace"
    plain_dir.mkdir()

    apply_git_identity(plain_dir, _settings_identity().misc_settings)

    assert _git("config", "--global", "user.name") == "QA Tester"


def test_nonexistent_workspace_still_fills_global(tmp_path: Path) -> None:
    apply_git_identity(tmp_path / "gone", _settings_identity().misc_settings)

    assert _git("config", "--global", "user.name") == "QA Tester"


def test_workspace_path_is_a_file_does_not_crash(tmp_path: Path) -> None:
    plain_file = tmp_path / "workspace"
    plain_file.write_text("not a directory\n")

    apply_git_identity(plain_file, _settings_identity().misc_settings)


def test_git_failure_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_run_git_command(*args: Any, **kwargs: Any) -> str:
        raise OSError("git subprocess cannot run")

    monkeypatch.setattr(git_identity, "run_git_command", broken_run_git_command)

    apply_git_identity(_repo(tmp_path), _settings_identity().misc_settings)


async def test_start_conversation_applies_settings_identity(tmp_path: Path) -> None:
    repo_dir = _repo(tmp_path)
    request = StartConversationRequest(
        agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(repo_dir)),
    )

    await _start(tmp_path, request, persisted=_settings_identity())

    assert _git("config", "user.name", cwd=repo_dir) == "QA Tester"
    assert _git("config", "--global", "user.name") == "QA Tester"


def test_partial_identity_applies_each_field(tmp_path: Path) -> None:
    repo_dir = _repo(tmp_path)
    misc: dict[str, Any] = {
        "app_preferences": {"git_user_name": "Only Name", "git_user_email": ""}
    }

    apply_git_identity(repo_dir, misc)

    assert _git("config", "user.name", cwd=repo_dir) == "Only Name"
    assert _git_or_empty("config", "user.email", cwd=repo_dir) == ""
