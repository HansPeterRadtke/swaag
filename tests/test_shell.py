from __future__ import annotations

from pathlib import Path

from swaag.environment.process import ProcessResult
from swaag.environment.shell import ShellSession
from swaag.environment.state import ProcessRecord, ShellSessionState


class RecordingProcessManager:
    def __init__(self) -> None:
        self.command = None

    def run(self, command, *, cwd, env, timeout_seconds, metadata=None):
        self.command = list(command)
        record = ProcessRecord(
            process_id="proc_test",
            command=list(command),
            cwd=str(cwd),
            status="completed",
            return_code=0,
            stdout="",
            stderr="",
        )
        return ProcessResult(record=record, stdout="", stderr="")


def test_shell_session_uses_configured_non_login_shell(make_config, tmp_path: Path) -> None:
    config = make_config(environment__shell_executable="/bin/sh")
    process = RecordingProcessManager()
    shell = ShellSession(config, process_manager=process)

    result, _ = shell.execute(
        ShellSessionState(cwd=str(tmp_path)),
        "printf ok",
        workspace_root=tmp_path,
    )

    assert result.exit_code == 0
    assert process.command is not None
    assert process.command[0] == "/bin/sh"
    assert process.command[1] == "-c"
    assert "-l" not in process.command[:2]
