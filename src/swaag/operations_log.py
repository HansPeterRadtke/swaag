from __future__ import annotations

import json
import os
import queue
import sys
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig
from swaag.redaction import (
    configured_secret_values,
    is_sensitive_key,
    redact_for_persistence,
)


_ACTIVE: "OperationsLog | None" = None
_ACTIVE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: Any, *, secret_values: tuple[str, ...] = ()) -> Any:
    """Apply central persistence redaction plus the legacy exact `token` key rule."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_key(key_text) or key_text.strip().casefold() == "token":
                # Delegate fingerprinting/text handling to the central redactor;
                # map the ambiguous legacy `token` key through a known-sensitive
                # key without broadening central redaction for non-log persistence.
                central_key = (
                    key_text
                    if is_sensitive_key(key_text)
                    else "bearer_token"
                )
                redacted = redact_for_persistence(
                    {central_key: item}, secret_values=secret_values
                )
                result[key_text] = redacted[central_key]
            else:
                result[key_text] = _redact(item, secret_values=secret_values)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item, secret_values=secret_values) for item in value]
    return redact_for_persistence(value, secret_values=secret_values)


class OperationsLog:
    def __init__(self, config: AgentConfig, *, component: str):
        self.config = config
        self.component = str(component)
        self.path = config.logging.file_path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._secret_values = configured_secret_values(config)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=int(config.logging.queue_capacity)
        )
        self._closed = False
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="swaag-operations-log",
            daemon=True,
        )
        self._thread.start()
        self.event(
            "startup",
            severity="INFO",
            pid=os.getpid(),
            config_fingerprint=config.config_fingerprint(),
            effective_config=_redact(
                config.raw, secret_values=self._secret_values
            ),
            config_sources=dict(config.sources),
            parameter_metadata=config.parameter_metadata(),
        )

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def event(self, event: str, *, severity: str = "INFO", **attributes: Any) -> None:
        if self._closed:
            return
        payload = {
            "timestamp": _now(),
            "severity": severity.upper(),
            "component": self.component,
            "event": str(event),
            "attributes": _redact(
                attributes, secret_values=self._secret_values
            ),
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped += 1
            fallback = {
                "timestamp": _now(),
                "severity": "ERROR",
                "component": "operations_log",
                "event": "queue_overflow",
                "attributes": {
                    "dropped_total": self._dropped,
                    "original_event": str(event),
                },
            }
            try:
                os.write(2, (json.dumps(fallback, sort_keys=True) + "\n").encode("utf-8"))
            except OSError:
                pass

    def shutdown(self, *, reason: str = "normal") -> None:
        if self._closed:
            return
        # Stop ordinary producers first, then enqueue the lifecycle terminator with
        # backpressure instead of the best-effort nonblocking event path. A saturated
        # queue may drop ordinary records, but it must never drop shutdown evidence.
        self._closed = True
        shutdown_payload = {
            "timestamp": _now(),
            "severity": "INFO",
            "component": self.component,
            "event": "shutdown",
            "attributes": {
                "reason": str(reason),
                "dropped_log_events": self._dropped,
            },
        }
        self._queue.put(shutdown_payload)
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            try:
                os.write(2, b'{"severity":"ERROR","event":"operations_log_shutdown_timeout"}\n')
            except OSError:
                pass

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write_line(json.dumps(item, sort_keys=True, separators=(",", ":")))
            except Exception as exc:
                try:
                    os.write(
                        2,
                        (
                            json.dumps(
                                {
                                    "timestamp": _now(),
                                    "severity": "ERROR",
                                    "component": "operations_log",
                                    "event": "write_failure",
                                    "attributes": {
                                        "error_type": type(exc).__name__,
                                        "error": _redact(
                                            str(exc),
                                            secret_values=self._secret_values,
                                        ),
                                    },
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8"),
                    )
                except OSError:
                    pass
            finally:
                self._queue.task_done()

    def _write_line(self, line: str) -> None:
        encoded = (line + "\n").encode("utf-8")
        if self.path.exists() and self.path.stat().st_size + len(encoded) > self.config.logging.max_bytes:
            self._rotate()
        with self.path.open("ab", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def _rotate(self) -> None:
        count = int(self.config.logging.backup_count)
        if count <= 0:
            self.path.unlink(missing_ok=True)
            return
        oldest = self.path.with_name(self.path.name + f".{count}")
        oldest.unlink(missing_ok=True)
        for index in range(count - 1, 0, -1):
            source = self.path.with_name(self.path.name + f".{index}")
            target = self.path.with_name(self.path.name + f".{index + 1}")
            if source.exists():
                source.replace(target)
        if self.path.exists():
            self.path.replace(self.path.with_name(self.path.name + ".1"))


def configure_operations_log(config: AgentConfig, *, component: str) -> OperationsLog:
    global _ACTIVE
    runtime = OperationsLog(config, component=component)
    with _ACTIVE_LOCK:
        previous = _ACTIVE
        _ACTIVE = runtime
    if previous is not None:
        previous.shutdown(reason="reconfigured")
    return runtime


def emit_operation_event(event: str, *, severity: str = "INFO", **attributes: Any) -> None:
    with _ACTIVE_LOCK:
        runtime = _ACTIVE
    if runtime is not None:
        runtime.event(event, severity=severity, **attributes)


def shutdown_operations_log(*, reason: str = "normal") -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        runtime = _ACTIVE
        _ACTIVE = None
    if runtime is not None:
        runtime.shutdown(reason=reason)
