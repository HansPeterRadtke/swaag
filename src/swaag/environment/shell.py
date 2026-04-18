from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from swaag.config import AgentConfig
from swaag.environment.process import ProcessManager, ProcessResult
from swaag.environment.state import ShellSessionState
from swaag.utils import utc_now_iso


class ShellError(RuntimeError):
    pass


@dataclass(slots=True)
class ShellCommandResult:
    command: str
    cwd_before: str
    cwd_after: str
    env_overrides: dict[str, str]
    unset_vars: list[str]
    exit_code: int
    stdout: str
    stderr: str
    process_result: ProcessResult


class ShellSession:
    def __init__(self, config: AgentConfig, *, process_manager: ProcessManager | None = None):
        self.config = config
        self.process_manager = process_manager or ProcessManager()

    def _trim_text(self, text: str) -> str:
        limit = self.config.environment.max_capture_chars
        if len(text) <= limit:
            return text
        return text[:limit]

    def execute(self, shell_state: ShellSessionState, command: str, *, workspace_root: Path) -> tuple[ShellCommandResult, ShellSessionState]:
        cwd_before = shell_state.cwd or str(workspace_root)
        effective_env = os.environ.copy()
        effective_env.update(shell_state.env_overrides)
        for key in shell_state.unset_vars:
            effective_env.pop(key, None)
        with tempfile.TemporaryDirectory(prefix="swaag_shell_") as temp_dir:
            temp_root = Path(temp_dir)
            cwd_path = temp_root / "cwd.txt"
            env_path = temp_root / "env.bin"
            wrapped_command = (
                f"{command}\n"
                "status=$?\n"
                'pwd > "$SWAAG_CWD_FILE"\n'
                'env -0 > "$SWAAG_ENV_FILE"\n'
                "exit $status\n"
            )
            effective_env["SWAAG_CWD_FILE"] = str(cwd_path)
            effective_env["SWAAG_ENV_FILE"] = str(env_path)
            result = self.process_manager.run(
                ["bash", "-lc", wrapped_command],
                cwd=Path(cwd_before),
                env=effective_env,
                timeout_seconds=self.config.runtime.tool_timeout_seconds,
                metadata={"kind": "shell_command"},
            )
            stdout = self._trim_text(result.stdout)
            stderr = self._trim_text(result.stderr)
            result.record.stdout = stdout
            result.record.stderr = stderr
            cwd_after = cwd_before
            if cwd_path.exists():
                cwd_after = cwd_path.read_text(encoding="utf-8").strip() or cwd_before
            env_after: dict[str, str] | None = None
            if env_path.exists():
                env_blob = env_path.read_bytes()
                env_after = {}
                for entry in env_blob.split(b"\x00"):
                    if not entry:
                        continue
                    key, _, value = entry.partition(b"=")
                    env_after[key.decode("utf-8", errors="ignore")] = value.decode("utf-8", errors="ignore")

        overrides: dict[str, str]
        unset_vars: list[str]
        if env_after is None:
            overrides = dict(shell_state.env_overrides)
            unset_vars = list(shell_state.unset_vars)
        else:
            overrides = {}
            unset_vars = []
            for key, value in env_after.items():
                if key in os.environ:
                    if os.environ[key] != value:
                        overrides[key] = value
                else:
                    overrides[key] = value
            for key in shell_state.env_overrides:
                if key not in env_after:
                    unset_vars.append(key)
        updated_state = ShellSessionState(
            cwd=cwd_after,
            env_overrides=overrides,
            unset_vars=sorted(set(shell_state.unset_vars) | set(unset_vars)),
            command_count=shell_state.command_count + 1,
            last_command=command,
            last_exit_code=result.record.return_code,
            updated_at=utc_now_iso(),
        )
        return (
            ShellCommandResult(
                command=command,
                cwd_before=cwd_before,
                cwd_after=cwd_after,
                env_overrides=dict(updated_state.env_overrides),
                unset_vars=list(updated_state.unset_vars),
                exit_code=result.record.return_code or 0,
                stdout=stdout,
                stderr=stderr,
                process_result=result,
            ),
            updated_state,
        )
