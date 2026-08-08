from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swaag.fsops import atomic_replace, ensure_dir, ensure_parent_dir, remove_file, write_bytes, write_text
from swaag.utils import new_id, utc_now_iso


@dataclass(slots=True, frozen=True)
class TerminalRecord:
    terminal_id: str
    name: str
    root: str
    cwd: str
    shell: str
    worker_pid: int
    shell_pid: int
    active: bool
    return_code: int | None
    created_at: str
    updated_at: str
    output_chars: int


class TerminalStore:
    """Durable session-scoped PTY terminals backed by detached worker processes."""

    def __init__(self, sessions_root: Path, session_id: str):
        self.root = Path(sessions_root).expanduser() / session_id / "terminals"
        ensure_dir(self.root)

    def _dir(self, terminal_id: str) -> Path:
        if not terminal_id or "/" in terminal_id or "\\" in terminal_id or ".." in terminal_id:
            raise ValueError("invalid terminal_id")
        return self.root / terminal_id

    def _meta_path(self, terminal_id: str) -> Path:
        return self._dir(terminal_id) / "metadata.json"

    def _status_path(self, terminal_id: str) -> Path:
        return self._dir(terminal_id) / "status.json"

    def _output_path(self, terminal_id: str) -> Path:
        return self._dir(terminal_id) / "output.log"

    def _inbox(self, terminal_id: str) -> Path:
        return self._dir(terminal_id) / "inbox"

    def _stop_path(self, terminal_id: str) -> Path:
        return self._dir(terminal_id) / "stop"

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        write_text(temp, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        atomic_replace(temp, path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _record(self, terminal_id: str) -> TerminalRecord:
        meta = self._read_json(self._meta_path(terminal_id))
        status = self._read_json(self._status_path(terminal_id)) if self._status_path(terminal_id).exists() else {}
        output_path = self._output_path(terminal_id)
        output_chars = len(output_path.read_text(encoding="utf-8", errors="replace")) if output_path.exists() else 0
        active = bool(status.get("active", False)) and _pid_alive(int(status.get("worker_pid", 0)))
        return TerminalRecord(
            terminal_id=terminal_id,
            name=str(meta.get("name", "")),
            root=str(meta["root"]),
            cwd=str(meta["cwd"]),
            shell=str(meta["shell"]),
            worker_pid=int(status.get("worker_pid", 0)),
            shell_pid=int(status.get("shell_pid", 0)),
            active=active,
            return_code=status.get("return_code"),
            created_at=str(meta["created_at"]),
            updated_at=str(status.get("updated_at", meta["created_at"])),
            output_chars=output_chars,
        )

    def resolve(self, terminal_ref: str) -> str:
        direct = self._dir(terminal_ref)
        if direct.is_dir() and self._meta_path(terminal_ref).exists():
            return terminal_ref
        matches = [r.terminal_id for r in self.list() if r.name == terminal_ref]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"terminal name is ambiguous: {terminal_ref}")
        raise FileNotFoundError(f"Unknown terminal: {terminal_ref}")

    def list(self) -> list[TerminalRecord]:
        records: list[TerminalRecord] = []
        for child in sorted(self.root.iterdir()) if self.root.exists() else []:
            if child.is_dir() and (child / "metadata.json").exists():
                try:
                    records.append(self._record(child.name))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        return sorted(records, key=lambda item: (item.created_at, item.terminal_id))

    def create(self, *, cwd: Path, shell: str, name: str = "") -> TerminalRecord:
        if name and any(item.name == name and item.active for item in self.list()):
            raise ValueError(f"active terminal name already exists: {name}")
        terminal_id = new_id("terminal")
        root = self._dir(terminal_id)
        ensure_dir(root)
        ensure_dir(self._inbox(terminal_id))
        write_text(self._output_path(terminal_id), "")
        created_at = utc_now_iso()
        self._atomic_json(
            self._meta_path(terminal_id),
            {
                "terminal_id": terminal_id,
                "name": name,
                "root": str(root),
                "cwd": str(cwd),
                "shell": shell,
                "created_at": created_at,
            },
        )
        self._atomic_json(
            self._status_path(terminal_id),
            {"worker_pid": 0, "shell_pid": 0, "active": False, "return_code": None, "updated_at": created_at},
        )
        with open(os.devnull, "rb") as devnull_in, open(os.devnull, "ab") as devnull_out:
            subprocess.Popen(
                [sys.executable, "-m", "swaag.environment.terminal", "--worker", str(root)],
                stdin=devnull_in,
                stdout=devnull_out,
                stderr=devnull_out,
                start_new_session=True,
                close_fds=True,
                env=os.environ.copy(),
            )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                record = self._record(terminal_id)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.02)
                continue
            if record.active and record.shell_pid > 0:
                return record
            time.sleep(0.02)
        raise RuntimeError(f"terminal worker failed to start: {terminal_id}")

    def send(self, terminal_ref: str, data: str, *, append_newline: bool = False) -> TerminalRecord:
        terminal_id = self.resolve(terminal_ref)
        record = self._record(terminal_id)
        if not record.active:
            raise RuntimeError(f"terminal is not active: {terminal_ref}")
        payload = data + ("\n" if append_newline else "")
        inbox = self._inbox(terminal_id)
        ensure_dir(inbox)
        msg = inbox / f"{time.time_ns()}_{new_id('input')}.bin"
        write_bytes(msg, payload.encode("utf-8"))
        return self._record(terminal_id)

    def read(self, terminal_ref: str, *, start_offset: int = 0, max_chars: int = 4000) -> dict[str, Any]:
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        terminal_id = self.resolve(terminal_ref)
        record = self._record(terminal_id)
        text = self._output_path(terminal_id).read_text(encoding="utf-8", errors="replace")
        start = min(start_offset, len(text))
        end = min(len(text), start + max_chars)
        return {
            "terminal_id": record.terminal_id,
            "name": record.name,
            "active": record.active,
            "return_code": record.return_code,
            "start_offset": start,
            "end_offset": end,
            "next_offset": end,
            "finished": end >= len(text),
            "total_chars": len(text),
            "text": text[start:end],
        }

    def close(self, terminal_ref: str, *, force_after_seconds: float = 2.0) -> TerminalRecord:
        terminal_id = self.resolve(terminal_ref)
        record = self._record(terminal_id)
        if record.active:
            write_text(self._stop_path(terminal_id), "stop\n")
            deadline = time.monotonic() + max(0.0, force_after_seconds)
            while time.monotonic() < deadline:
                record = self._record(terminal_id)
                if not record.active:
                    return record
                time.sleep(0.05)
            if record.worker_pid > 0 and _pid_alive(record.worker_pid):
                try:
                    os.kill(record.worker_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return self._record(terminal_id)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_status(root: Path, payload: dict[str, Any]) -> None:
    TerminalStore._atomic_json(root / "status.json", payload)


def _worker(root: Path) -> int:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    shell = str(metadata["shell"])
    cwd = str(metadata["cwd"])
    output_path = root / "output.log"
    inbox = root / "inbox"
    stop_path = root / "stop"
    ensure_dir(inbox)
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    process = subprocess.Popen(
        [shell],
        cwd=cwd,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    os.set_blocking(master_fd, False)
    status = {"worker_pid": os.getpid(), "shell_pid": process.pid, "active": True, "return_code": None, "updated_at": utc_now_iso()}
    _write_status(root, status)
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ)
    output_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        while True:
            if stop_path.exists():
                try:
                    os.killpg(process.pid, signal.SIGHUP)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 1.0
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                remove_file(stop_path, missing_ok=True)
            for path in sorted(inbox.glob("*.bin")):
                try:
                    data = path.read_bytes()
                    if data:
                        os.write(master_fd, data)
                finally:
                    remove_file(path, missing_ok=True)
            for key, _ in selector.select(timeout=0.03):
                try:
                    data = os.read(key.fd, 65536)
                except OSError as exc:
                    if exc.errno in {errno.EIO, errno.EBADF}:
                        data = b""
                    else:
                        raise
                if data:
                    os.write(output_fd, data)
            rc = process.poll()
            if rc is not None:
                while True:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    os.write(output_fd, data)
                status.update({"active": False, "return_code": rc, "updated_at": utc_now_iso()})
                _write_status(root, status)
                return int(rc)
            status["updated_at"] = utc_now_iso()
            _write_status(root, status)
    finally:
        try:
            selector.close()
        except Exception:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.close(output_fd)
        except OSError:
            pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        status.update({"active": False, "return_code": process.poll(), "updated_at": utc_now_iso()})
        _write_status(root, status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args(argv)
    if args.worker is None:
        parser.error("--worker is required")
    return _worker(args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
