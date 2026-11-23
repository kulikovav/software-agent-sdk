"""The terminal's shell command must not use su when it cannot succeed."""

import os

from openhands.tools.terminal.terminal.tmux_terminal import login_shell_command


def test_root_drops_to_the_agent_user(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    assert login_shell_command("openhands") == ["su", "openhands", "-"]
    assert login_shell_command("root") == ["su", "root", "-"]


def test_non_root_starts_bash_instead_of_prompting_for_a_password(monkeypatch):
    # The governed sandbox runs as 10001, so `su openhands -` would wait at a
    # password prompt and the tmux pane would never show its shell prompt.
    monkeypatch.setattr(os, "geteuid", lambda: 10001)

    assert login_shell_command("openhands") == ["/bin/bash"]
    assert login_shell_command("root") == ["/bin/bash"]
    assert login_shell_command(None) == ["/bin/bash"]
