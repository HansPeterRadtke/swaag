"""Mechanical supervision. Backend probes never run on the supervisor loop.

A live process, transport keepalive, busy slot and advancing token counter are
separate evidence. None is a semantic correctness judgment or a kill policy.
"""
from __future__ import annotations

import copy
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from swaag.utils import utc_now_iso


def llama_slot_activity(slots: Any) -> dict[str, Any]:
    if not isinstance(slots, list) or not slots:
        raise ValueError("backend did not expose a nonempty slot inventory")
    rows = []
    for item in slots:
        if not isinstance(item, dict) or not isinstance(item.get("is_processing"), bool):
            raise ValueError("backend slot has no reliable processing flag")
        row = {key: item.get(key) for key in (
            "id", "id_task", "is_processing", "n_prompt_tokens",
            "n_prompt_tokens_processed", "n_prompt_tokens_cache")}
        next_tokens = item.get("next_token", [])
        row["n_decoded"] = next((part.get("n_decoded") for part in next_tokens
                                 if isinstance(part, dict) and isinstance(part.get("n_decoded"), int)), None)
        if not row["is_processing"]:
            row["phase"] = "idle"
        elif isinstance(row["n_decoded"], int) and row["n_decoded"] > 0:
            row["phase"] = "generation"
        elif isinstance(row["n_prompt_tokens_processed"], int) and isinstance(row["n_prompt_tokens"], int):
            cached = row["n_prompt_tokens_cache"] if isinstance(row["n_prompt_tokens_cache"], int) else 0
            row["phase"] = "prefill" if row["n_prompt_tokens_processed"] + cached < row["n_prompt_tokens"] else "generation"
        else:
            row["phase"] = "processing"
        rows.append(row)
    return {"supported": True, "source": "llama.cpp:/slots", "slots": rows,
            "state": "processing" if any(row["is_processing"] for row in rows) else "idle",
            "request_attribution": "backend_only"}


class BackendActivityMonitor:
    """One bounded probe in flight per backend, with nonblocking snapshot reads."""
    def __init__(self, probe: Callable[[], dict[str, Any]], *, interval: float = 1.0):
        self.probe = probe
        self.interval = max(.01, float(interval))
        self._condition = threading.Condition()
        self._references = 0
        self._thread: threading.Thread | None = None
        self._first_sample = threading.Event()
        self._sample: dict[str, Any] = {"supported": False, "state": "unobserved"}
        self._sample_time: float | None = None
        self._progress_time: float | None = None
        self._progress_at: str | None = None
        self._last_counters: dict[tuple, tuple] = {}

    def acquire(self) -> None:
        with self._condition:
            self._references += 1
            if self._thread is None:
                self._first_sample.clear()
                self._thread = threading.Thread(target=self._run, name="swaag-backend-observer", daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            self._references = max(0, self._references - 1)
            self._condition.notify_all()

    @contextmanager
    def observing(self):
        self.acquire()
        try:
            yield self
        finally:
            self.release()

    def wait_for_sample(self, timeout: float) -> bool:
        return self._first_sample.wait(max(0.0, timeout))

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            result = copy.deepcopy(self._sample)
            result["sample_age_seconds"] = None if self._sample_time is None else time.monotonic() - self._sample_time
            result["progress_age_seconds"] = None if self._progress_time is None else time.monotonic() - self._progress_time
            result["progress_observed_at"] = self._progress_at
            result["monitor_running"] = self._thread is not None
            result["stale"] = self._sample_time is None or time.monotonic() - self._sample_time > max(3 * self.interval, 5.0)
            return result

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._references == 0:
                    self._thread = None
                    return
            try:
                sample = dict(self.probe())
            except Exception as exc:
                # Errors describe observation failure, never a proved hung model.
                sample = {"supported": False, "state": "unavailable", "error_type": type(exc).__name__}
            observed = utc_now_iso()
            counters = {(row.get("id"), row.get("id_task")):
                        (row.get("n_prompt_tokens_processed"), row.get("n_decoded"))
                        for row in sample.get("slots", []) if row.get("is_processing")}
            with self._condition:
                advancing = any(isinstance(value, int) and not isinstance(value, bool) and value > (previous if isinstance(previous, int) else 0)
                    for key, values in counters.items()
                    for value, previous in zip(values, self._last_counters.get(key, (0, 0))))
                if advancing:
                    self._progress_time = time.monotonic()
                    self._progress_at = observed
                self._last_counters = counters
                self._sample = {**sample, "observed_at": observed}
                self._sample_time = time.monotonic()
                self._first_sample.set()
                self._condition.wait(timeout=self.interval)


def client_activity_monitor(client: Any) -> BackendActivityMonitor | None:
    if getattr(client, "is_deterministic_test_client", False) or getattr(client, "mode", "") == "replay":
        return None
    delegate = getattr(client, "delegate", None)
    if delegate is not None:
        return client_activity_monitor(delegate)
    getter = getattr(client, "activity_monitor", None)
    return getter() if callable(getter) else None


class RuntimeSupervisor:
    """An independent deterministic loop for orchestrator, workers and backends.

    The loop only copies cached evidence. Network probes have their own bounded
    observers; model calls, worker tools, SQLite writers and semantic decisions
    cannot block it. There is deliberately no automatic stop/restart operation.
    """
    def __init__(self, runtimes: dict[str, Any], *, interval: float = .1):
        self.interval = max(.01, float(interval))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._heartbeat_at: str | None = None
        self._heartbeat_time: float | None = None
        self._roles: dict[int, list[str]] = {}
        self.monitors: dict[str, BackendActivityMonitor] = {}
        for role, runtime in runtimes.items():
            self._roles.setdefault(id(runtime), []).append(role)
            runtime.supervisor = self
            monitor = client_activity_monitor(runtime.client)
            if monitor is not None:
                key = runtime.config.model.base_url.rstrip("/")
                shared = self.monitors.setdefault(key, monitor)
                client = runtime.client
                while getattr(client, "delegate", None) is not None:
                    client = client.delegate
                if hasattr(client, "_backend_monitor"):
                    with client._backend_monitor_lock:
                        client._backend_monitor = shared

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            for monitor in self.monitors.values():
                monitor.acquire()
            self._thread = threading.Thread(target=self._run, name="swaag-central-supervisor", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
        thread.join(timeout=max(1.0, 2 * self.interval))
        with self._lock:
            self._thread = None
            for monitor in self.monitors.values():
                monitor.release()

    def finish_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def observe_runtime(self, runtime: Any, session_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if payload.get("phase") in {"completed", "cancelled", "failed"}:
                self._sessions.pop(session_id, None)
                return
            self._sessions[session_id] = {**copy.deepcopy(payload), "session_id": session_id,
                "roles": list(self._roles.get(id(runtime), [])),
                "backend": runtime.config.model.base_url.rstrip("/"), "observed_monotonic": time.monotonic()}

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            sessions = copy.deepcopy(list(self._sessions.values()))
            loop = {"running": self._thread is not None and self._thread.is_alive(),
                    "heartbeat_at": self._heartbeat_at,
                    "heartbeat_age_seconds": None if self._heartbeat_time is None else now - self._heartbeat_time}
        for row in sessions:
            row["heartbeat_age_seconds"] = now - row.pop("observed_monotonic")
            row["heartbeat_is_computation_evidence"] = False
        return {"supervisor": loop, "active_sessions": sessions,
                "backends": {key: monitor.snapshot() for key, monitor in self.monitors.items()},
                "automatic_semantic_intervention": False}

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._heartbeat_time = time.monotonic()
                self._heartbeat_at = utc_now_iso()
            self._stop.wait(self.interval)
