"""Bounded-memory subprocess capture with durable exact output and explicit limits."""
from __future__ import annotations

import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from swaag.environment.artifacts import TextArtifactStore
from swaag.tools.base import ToolExecutionError
from swaag.types import ToolGeneratedEvent


def capture_command(
    args: list[str], *, store: TextArtifactStore, timeout_seconds: float,
    max_output_bytes: int, preview_chars: int, env: dict[str, str],
    kind: str, cwd: Path | None = None,
) -> tuple[dict[str, Any], list[ToolGeneratedEvent]]:
    """Stop only the owned process tree at a deadline or capture limit.

    A timeout retains all available output. A byte limit is an explicit failed
    command with a retained prefix, never a claim of complete capture. Failed
    artifact/storage publication leaves the raw capture directory recoverable.
    """
    if timeout_seconds <= 0 or max_output_bytes <= 0 or preview_chars <= 0:
        raise ValueError("capture limits must be positive")
    store.root.mkdir(parents=True, exist_ok=True)
    capture_root = Path(tempfile.mkdtemp(prefix=".capture-", dir=store.root))
    paths = {name: capture_root / (name + ".raw") for name in ("stdout", "stderr")}
    process = None
    events: list[ToolGeneratedEvent] = []
    stopped_reason: str | None = None
    total_bytes = 0
    stopped_at: float | None = None
    started = time.monotonic()

    def stop() -> None:
        nonlocal stopped_at
        if stopped_at is None:
            stopped_at = time.monotonic()
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        with paths["stdout"].open("xb") as stdout, paths["stderr"].open("xb") as stderr, selectors.DefaultSelector() as selector:
            handles = {"stdout": stdout, "stderr": stderr}
            process = subprocess.Popen(
                args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            )
            assert process.stdout is not None and process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0 and stopped_reason is None:
                    stopped_reason = "timed_out"
                    stop()
                for key, _ in selector.select(timeout=max(0.001, min(max(remaining, 0.001), 0.1))):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    admitted = min(len(chunk), max_output_bytes - total_bytes)
                    if admitted:
                        handles[key.data].write(chunk[:admitted])
                        total_bytes += admitted
                    if admitted != len(chunk):
                        stopped_reason = "output_limit_exceeded"
                        stop()
                # Descendants cannot leave an unbounded inherited-pipe wait.
                if stopped_at is not None and time.monotonic() - stopped_at > 2:
                    if selector.get_map():
                        stopped_reason = "capture_drain_incomplete"
                    break
            remaining = timeout_seconds - (time.monotonic() - started)
            try:
                process.wait(timeout=max(0.001, remaining))
            except subprocess.TimeoutExpired:
                stopped_reason = stopped_reason or "timed_out"
                stop()
                process.wait(timeout=2)
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        output: dict[str, Any] = {
            "return_code": process.returncode,
            "status": stopped_reason or ("completed" if process.returncode == 0 else "failed"),
            "capture_complete": stopped_reason not in {"output_limit_exceeded", "capture_drain_incomplete"},
            "max_output_bytes": max_output_bytes,
        }
        for name, path in paths.items():
            artifact = store.create_from_file(path, kind=kind + "_" + name)
            preview = store.read(artifact.artifact_id, max_chars=preview_chars)
            output.update({
                name: preview["text"],
                name + "_artifact_id": artifact.artifact_id,
                name + "_sha256": artifact.sha256,
                name + "_chars": artifact.size_chars,
                name + "_bytes": artifact.size_bytes,
                name + "_truncated": not preview["finished"],
                name + "_decoding_errors": artifact.decoding_errors,
            })
            events.append(ToolGeneratedEvent("artifact_created", asdict(artifact)))
        shutil.rmtree(capture_root)
        return output, events
    except BaseException as exc:
        stop()
        if process is not None:
            process.wait(timeout=2)
        raise ToolExecutionError(
            "Command capture failed; retained raw evidence must be inspected before retrying the command.",
            error_type=type(exc).__name__,
            evidence={"capture_complete": False, "recovery_directory": str(capture_root),
                      "captured_bytes": total_bytes, "cause": str(exc)},
            generated_events=events,
        ) from exc
    finally:
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
