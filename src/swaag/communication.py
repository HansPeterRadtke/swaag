from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig
from swaag.heartbeat import systemd_notify, watchdog_interval_seconds
from swaag.protocol_adapters import (
    A2AProjectionAdapter,
    AgUiProjectionAdapter,
    OpenWebUiProjectionAdapter,
)
from swaag.runtime import AgentRuntime
from swaag.task_api import TaskApi
from swaag.utils import new_id, utc_now_iso
from swaag.workers import WorkerManager


@dataclass(slots=True, frozen=True)
class CommunicationRequest:
    correlation_id: str
    session_id: str
    message: str
    source: str
    priority: int
    status: str
    created_at: str
    completed_at: str | None = None
    reply: str | None = None


class CommunicationStore:
    def __init__(self, root: Path):
        self.path = Path(root).expanduser() / "communication.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    correlation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    reply TEXT
                );
                CREATE INDEX IF NOT EXISTS requests_pending
                    ON requests(status, priority DESC, created_at, correlation_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def create(self, session_id: str, message: str, *, source: str = "communication") -> CommunicationRequest:
        text = message.strip()
        if not text:
            raise ValueError("communication message must not be empty")
        lowered = text.casefold()
        priority = 100 if lowered == "stop" or lowered.startswith("stop ") else 80 if lowered == "pause" or lowered.startswith("pause ") else 0
        request = CommunicationRequest(new_id("correlation"), session_id, text, source, priority, "queued", utc_now_iso())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO requests(correlation_id,session_id,message,source,priority,status,created_at) VALUES(?,?,?,?,?,?,?)",
                (request.correlation_id, request.session_id, request.message, request.source, request.priority, request.status, request.created_at),
            )
        return request

    def get(self, correlation_id: str) -> CommunicationRequest | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM requests WHERE correlation_id=?", (correlation_id,)).fetchone()
        return CommunicationRequest(**dict(row)) if row else None

    def next_pending(self, session_id: str | None = None) -> CommunicationRequest | None:
        sql = "SELECT * FROM requests WHERE status='queued'"
        params: tuple[object, ...] = ()
        if session_id is not None:
            sql += " AND session_id=?"
            params = (session_id,)
        sql += " ORDER BY priority DESC, created_at, correlation_id LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return CommunicationRequest(**dict(row)) if row else None

    def set_status(self, correlation_id: str, status: str, *, reply: str | None = None) -> None:
        completed = utc_now_iso() if status in {"completed", "failed"} else None
        with self._connect() as connection:
            connection.execute(
                "UPDATE requests SET status=?,completed_at=?,reply=? WHERE correlation_id=?",
                (status, completed, reply, correlation_id),
            )


class CommunicationService:
    """Separate correlated communication/control service using the canonical AgentRuntime."""

    def __init__(self, runtime: AgentRuntime, *, assistant_runtime: AgentRuntime | None = None, max_concurrency: int = 4):
        self.runtime = runtime
        self.assistant_runtime = assistant_runtime
        self.store = CommunicationStore(runtime.config.sessions.root)
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.workers = WorkerManager(runtime, max_workers=max_concurrency)
        self.task_api = TaskApi(self.workers)

    @classmethod
    def from_config(cls, config: AgentConfig) -> "CommunicationService":
        main = AgentRuntime(config)
        assistant = None
        if getattr(config, "communication", None) and config.communication.enabled:
            if config.communication.model_base_url:
                assistant_config = copy.deepcopy(config)
                assistant_config.model.base_url = config.communication.model_base_url
                assistant_config.tools.enabled = list(config.communication.enabled_tools)
                assistant_config.tools.allow_side_effect_tools = False
                assistant = AgentRuntime(assistant_config)
            return cls(main, assistant_runtime=assistant, max_concurrency=config.communication.max_concurrent_requests)
        return cls(main)

    def submit(self, session_ref: str | None, message: str, *, source: str = "communication") -> CommunicationRequest:
        session_id = self.runtime.resolve_session_ref(session_ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No target session available")
        return self.store.create(session_id, message, source=source)

    def status(self, correlation_id: str) -> CommunicationRequest:
        request = self.store.get(correlation_id)
        if request is None:
            raise FileNotFoundError(f"Unknown correlation id: {correlation_id}")
        return request

    def _preempt_active_main_call(self, session_id: str, message: str):
        request = self.runtime.preemption.request_preemption(session_id, message, source="communication")
        if request is None:
            return None
        timeout = max(
            1.0,
            float(self.runtime.config.model.structured_timeout_seconds),
            float(self.runtime.config.model.timeout_seconds),
        )
        interrupted = self.runtime.preemption.wait_for_status(
            request.preemption_id,
            {"interrupted", "failed"},
            timeout_seconds=timeout,
            poll_seconds=0.02,
        )
        if interrupted.status == "failed":
            raise RuntimeError(interrupted.reply or "main model preemption failed")
        self.runtime.preemption.mark_assistant_running(request.preemption_id)
        return request

    def _complete_preemption(self, request, *, target_changed: bool, reply: str | None = None) -> None:
        if request is not None:
            self.runtime.preemption.complete(
                request.preemption_id,
                target_changed=target_changed,
                reply=reply,
            )

    def _fail_preemption(self, request, exc: Exception) -> None:
        if request is not None:
            self.runtime.preemption.fail(request.preemption_id, f"{type(exc).__name__}: {exc}")

    def _same_model_assistant_answer(self, session_id: str, question: str, status: dict[str, Any]) -> str:
        prompt = (
            "You are the communication assistant for another SWAAG agent. Answer the user's status/history question only from the target session evidence below. "
            "Do not alter the target workspace or target session. If evidence is insufficient, say so.\n\n"
            f"Target session id: {session_id}\nStatus: {json.dumps(status, sort_keys=True)}\nQuestion: {question}"
        )
        communication_session = self.runtime.create_or_load_user_session(f"communication-{session_id}")
        return self.runtime.run_turn_in_session(communication_session, prompt).assistant_text

    def process_once(self, *, session_id: str | None = None) -> CommunicationRequest | None:
        request = self.store.next_pending(session_id)
        if request is None:
            return None
        self.store.set_status(request.correlation_id, "processing")
        preemption = None
        try:
            preemption = self._preempt_active_main_call(request.session_id, request.message)
            self.runtime.history.enqueue_control_message(
                request.session_id,
                request.message,
                source=f"communication:{request.correlation_id}",
                control_id=request.correlation_id,
            )
            state = self.runtime.history.rebuild_from_history(request.session_id, write_projections=False)
            result = self.runtime.run_pending_controls_in_session(state)
            reply = result.assistant_text if result is not None else ""
            self.store.set_status(request.correlation_id, "completed", reply=reply)
            self._complete_preemption(preemption, target_changed=True, reply=reply)
        except Exception as exc:
            self._fail_preemption(preemption, exc)
            self.store.set_status(request.correlation_id, "failed", reply=f"{type(exc).__name__}: {exc}")
        return self.status(request.correlation_id)

    async def process_once_async(self, *, session_id: str | None = None) -> CommunicationRequest | None:
        async with self._semaphore:
            return await asyncio.to_thread(self.process_once, session_id=session_id)

    def answer_status_question(self, session_ref: str | None, question: str) -> str:
        session_id = self.runtime.resolve_session_ref(session_ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No target session available")
        state = self.runtime.history.rebuild_from_history(session_id, write_projections=False)
        status = self.runtime.session_status_payload(state)
        if self.assistant_runtime is not None:
            prompt = (
                "You are the communication assistant for another SWAAG agent. Answer the user's status/history question only from the target session evidence below. "
                "Do not alter the target workspace. If evidence is insufficient, say so.\n\n"
                f"Target session id: {session_id}\nStatus: {json.dumps(status, sort_keys=True)}\nQuestion: {question}"
            )
            communication_session = self.assistant_runtime.create_or_load_user_session(f"communication-{session_id}")
            return self.assistant_runtime.run_turn_in_session(communication_session, prompt).assistant_text

        preemption = None
        try:
            preemption = self._preempt_active_main_call(session_id, question)
            if preemption is None:
                return json.dumps(status, sort_keys=True)
            answer = self._same_model_assistant_answer(session_id, question, status)
            self._complete_preemption(preemption, target_changed=False, reply=answer)
            return answer
        except Exception as exc:
            self._fail_preemption(preemption, exc)
            raise

    def protocol_projection(
        self,
        protocol: str,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        worker_id = str(params.get("worker_id", "")).strip()
        if not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        record = self.workers.store.get(worker_id)
        if protocol == "a2a" and operation == "get":
            return {
                "protocol": A2AProjectionAdapter.protocol_version,
                "task": A2AProjectionAdapter().task(record),
            }
        if protocol == "open_webui" and operation == "get":
            return OpenWebUiProjectionAdapter().response(record)
        if protocol == "ag_ui" and operation == "events":
            page = self.task_api.execute("events", params)
            projected = AgUiProjectionAdapter().events(
                record,
                [
                    self.workers.event_from_payload(item)
                    for item in page["events"]
                ],
            )
            return {
                "protocol": "ag-ui",
                "worker_id": worker_id,
                "events": projected,
                "next_sequence": page["next_sequence"],
                "has_more": page["has_more"],
            }
        raise ValueError(f"unsupported protocol operation: {protocol}.{operation}")


    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    op = str(request.get("op", ""))
                    if op == "submit":
                        item = self.submit(request.get("session"), str(request.get("message", "")), source=str(request.get("source", "communication")))
                        response = asdict(item)
                    elif op == "status":
                        response = asdict(self.status(str(request.get("correlation_id", ""))))
                    elif op == "process":
                        item = await self.process_once_async(session_id=request.get("session_id"))
                        response = None if item is None else asdict(item)
                    elif op == "ask_status":
                        async with self._semaphore:
                            answer = await asyncio.to_thread(
                                self.answer_status_question,
                                request.get("session"),
                                str(request.get("question", "What is the current status?")),
                            )
                        response = {"answer": answer}
                    elif op.startswith("task."):
                        params = request.get("params") or {}
                        if not isinstance(params, dict):
                            raise ValueError("task operation params must be an object")
                        async with self._semaphore:
                            response = await asyncio.to_thread(
                                self.task_api.execute,
                                op.removeprefix("task."),
                                params,
                            )
                    elif op.startswith(("ag_ui.", "a2a.", "open_webui.")):
                        params = request.get("params") or {}
                        if not isinstance(params, dict):
                            raise ValueError("protocol operation params must be an object")
                        protocol, operation = op.split(".", 1)
                        response = self.protocol_projection(
                            protocol,
                            operation,
                            params,
                        )
                    else:
                        raise ValueError(f"unknown communication op: {op}")
                    payload = {"ok": True, "result": response}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                writer.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _watchdog_loop(self) -> None:
        interval = watchdog_interval_seconds(default_seconds=10.0)
        while True:
            systemd_notify("WATCHDOG=1", "STATUS=swaag communication service healthy")
            await asyncio.sleep(interval)

    async def serve_tcp(self, host: str, port: int) -> None:
        self.workers.reconcile_orphans()
        server = await asyncio.start_server(self.handle_client, host, port)
        systemd_notify("READY=1", f"STATUS=swaag communication listening on {host}:{port}")
        watchdog_task = asyncio.create_task(self._watchdog_loop(), name="swaag-systemd-watchdog")
        try:
            async with server:
                await server.serve_forever()
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
            self.workers.shutdown(wait=False)
            systemd_notify("STOPPING=1", "STATUS=swaag communication stopping")
