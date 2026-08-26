from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swaag.workers import WorkerManager


class TaskApi:
    """Transport-neutral command/query API over the canonical worker lifecycle."""

    version = "swaag.task.v1"

    def __init__(self, workers: WorkerManager):
        self.workers = workers

    def execute(self, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(payload or {})
        if operation == "create":
            objective = _required_text(args, "objective")
            record = self.workers.create(objective, name=_optional_text(args, "name"))
            if bool(args.get("start", False)):
                record = self.workers.start(record.worker_id)
            return self._record(record)
        if operation == "start":
            return self._record(self.workers.start(_required_text(args, "worker_id")))
        if operation == "get":
            return self.workers.inspect(_required_text(args, "worker_id"))
        if operation == "list":
            return {
                "version": self.version,
                "workers": [
                    asdict(item)
                    for item in self.workers.list(
                        include_archived=bool(args.get("include_archived", False))
                    )
                ],
            }
        if operation == "message":
            record = self.workers.message(
                _required_text(args, "worker_id"),
                _required_text(args, "message"),
                source=_optional_text(args, "source") or "task_api",
                resume_if_idle=bool(args.get("resume_if_idle", True)),
            )
            return self._record(record)
        if operation == "cancel":
            record = self.workers.cancel(
                _required_text(args, "worker_id"),
                reason=_optional_text(args, "reason") or "task API cancellation",
            )
            return self._record(record)
        if operation == "resume":
            record = self.workers.resume(
                _required_text(args, "worker_id"),
                message=_optional_text(args, "message"),
            )
            return self._record(record)
        if operation == "archive":
            return self._record(self.workers.archive(_required_text(args, "worker_id")))
        if operation == "events":
            worker_id = _required_text(args, "worker_id")
            after = args.get("after_sequence", 0)
            if not isinstance(after, int) or isinstance(after, bool) or after < 0:
                raise ValueError("after_sequence must be a non-negative integer")
            events = self.workers.events(worker_id, after_sequence=after)
            return {
                "version": self.version,
                "worker_id": worker_id,
                "events": [asdict(item) for item in events],
                "next_sequence": events[-1].sequence if events else after,
            }
        raise ValueError(f"Unknown task operation: {operation}")

    def _record(self, record) -> dict[str, Any]:
        return {"version": self.version, "worker": asdict(record)}


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value.strip() or None
