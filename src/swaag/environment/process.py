from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from swaag.environment.state import ProcessRecord
from swaag.fsops import ensure_dir, write_text as _fsops_write_text
from swaag.utils import new_id, utc_now_iso


class ProcessError(RuntimeError):
    pass


@dataclass(slots=True)
class ProcessResult:
    record: ProcessRecord
    stdout: str
    stderr: str


@dataclass(slots=True)
class ProcessPollResult:
    record: ProcessRecord
    completed: bool
    status_changed: bool
    stdout: str
    stderr: str


_BACKGROUND_WRAPPER = r"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


control_path = Path(sys.argv[1])
control = json.loads(control_path.read_text(encoding="utf-8"))
stdout_path = Path(control["stdout_path"])
stderr_path = Path(control["stderr_path"])
status_path = Path(control["status_path"])
started_at = _now()
timeout_seconds = int(control["timeout_seconds"])
status_payload = {
    "status": "failed",
    "return_code": None,
    "started_at": started_at,
    "ended_at": _now(),
}
try:
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        try:
            completed = subprocess.run(
                control["command"],
                cwd=control["cwd"],
                env=control["env"],
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            status_payload = {
                "status": "completed" if completed.returncode == 0 else "failed",
                "return_code": completed.returncode,
                "started_at": started_at,
                "ended_at": _now(),
            }
        except subprocess.TimeoutExpired:
            stderr_handle.write(f"Process timed out after {timeout_seconds}s\n")
            status_payload = {
                "status": "timed_out",
                "return_code": None,
                "started_at": started_at,
                "ended_at": _now(),
            }
        except BaseException as exc:
            stderr_handle.write(f"{exc.__class__.__name__}: {exc}\n")
            status_payload = {
                "status": "failed",
                "return_code": None,
                "started_at": started_at,
                "ended_at": _now(),
            }
finally:
    _write_status(status_path, status_payload)
"""


class ProcessManager:
    def __init__(self, *, python_executable: str | None = None):
        self._python_executable = python_executable or sys.executable

    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: int,
        metadata: dict[str, str] | None = None,
    ) -> ProcessResult:
        process_id = new_id("proc")
        started_at = utc_now_iso()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            record = ProcessRecord(
                process_id=process_id,
                command=list(command),
                cwd=str(cwd),
                status="timed_out",
                return_code=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                started_at=started_at,
                ended_at=utc_now_iso(),
                metadata={} if metadata is None else dict(metadata),
            )
            raise ProcessError(f"Process timed out after {timeout_seconds}s") from exc
        status = "completed" if completed.returncode == 0 else "failed"
        record = ProcessRecord(
            process_id=process_id,
            command=list(command),
            cwd=str(cwd),
            status=status,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            started_at=started_at,
            ended_at=utc_now_iso(),
            metadata={} if metadata is None else dict(metadata),
        )
        return ProcessResult(record=record, stdout=completed.stdout, stderr=completed.stderr)

    def start_background(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: int,
        artifacts_dir: Path,
        metadata: dict[str, str] | None = None,
    ) -> ProcessRecord:
        process_id = new_id("proc")
        process_dir = artifacts_dir / process_id
        ensure_dir(process_dir)
        stdout_path = process_dir / "stdout.txt"
        stderr_path = process_dir / "stderr.txt"
        status_path = process_dir / "status.json"
        control_path = process_dir / "control.json"
        control = {
            "command": list(command),
            "cwd": str(cwd),
            "env": dict(env),
            "timeout_seconds": int(timeout_seconds),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "status_path": str(status_path),
        }
        _fsops_write_text(control_path, json.dumps(control), encoding="utf-8")
        started_at = utc_now_iso()
        proc = subprocess.Popen(
            [self._python_executable, "-c", _BACKGROUND_WRAPPER, str(control_path)],
            cwd=str(cwd),
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        return ProcessRecord(
            process_id=process_id,
            command=list(command),
            cwd=str(cwd),
            status="running",
            pid=proc.pid,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            status_path=str(status_path),
            started_at=started_at,
            metadata={} if metadata is None else dict(metadata),
        )

    def poll(self, record: ProcessRecord) -> ProcessPollResult:
        updated = self._with_outputs(record)
        previous_status = record.status
        status_payload = self._read_status_payload(updated)
        if updated.status == "running" and status_payload is not None:
            updated.status = str(status_payload.get("status", updated.status))  # type: ignore[assignment]
            updated.return_code = status_payload.get("return_code")
            updated.started_at = str(status_payload.get("started_at", updated.started_at))
            updated.ended_at = str(status_payload.get("ended_at", updated.ended_at))
            updated = self._with_outputs(updated)
        elif updated.status == "running" and not self._pid_alive(updated.pid):
            updated.status = "failed"
            updated.ended_at = utc_now_iso()
            if not updated.stderr:
                updated.stderr = "background process exited without a status file"
        return ProcessPollResult(
            record=updated,
            completed=updated.status != "running",
            status_changed=updated.status != previous_status,
            stdout=updated.stdout,
            stderr=updated.stderr,
        )

    def kill(self, record: ProcessRecord) -> ProcessRecord:
        if record.pid is not None:
            try:
                os.killpg(int(record.pid), signal.SIGTERM)
            except OSError as exc:
                try:
                    os.kill(int(record.pid), signal.SIGTERM)
                except OSError:
                    if record.status == "running":
                        raise ProcessError(f"Failed to kill process {record.process_id}") from exc
                else:
                    pass
                if record.status == "running" and self._pid_alive(record.pid):
                    raise ProcessError(f"Failed to kill process {record.process_id}") from exc
        updated = self._with_outputs(record)
        updated.status = "killed"
        updated.ended_at = utc_now_iso()
        self._write_status_payload(
            updated,
            {
                "status": "killed",
                "return_code": updated.return_code,
                "started_at": updated.started_at,
                "ended_at": updated.ended_at,
            },
        )
        return self._with_outputs(updated)

    def _with_outputs(self, record: ProcessRecord) -> ProcessRecord:
        return ProcessRecord(
            process_id=record.process_id,
            command=list(record.command),
            cwd=record.cwd,
            status=record.status,
            pid=record.pid,
            return_code=record.return_code,
            stdout=self._read_text(record.stdout_path),
            stderr=self._read_text(record.stderr_path),
            stdout_path=record.stdout_path,
            stderr_path=record.stderr_path,
            status_path=record.status_path,
            started_at=record.started_at,
            ended_at=record.ended_at,
            metadata=dict(record.metadata),
        )

    def _read_text(self, path_text: str) -> str:
        if not path_text:
            return ""
        path = Path(path_text)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _read_status_payload(self, record: ProcessRecord) -> dict | None:
        if not record.status_path:
            return None
        path = Path(record.status_path)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProcessError(f"Invalid background process status payload for {record.process_id}") from exc

    def _write_status_payload(self, record: ProcessRecord, payload: dict[str, object]) -> None:
        if not record.status_path:
            return
        _fsops_write_text(record.status_path, json.dumps(payload), encoding="utf-8")

    def _pid_alive(self, pid: int | None) -> bool:
        if pid is None:
            return False
        try:
            os.kill(int(pid), 0)
        except OSError:
            return False
        return True
