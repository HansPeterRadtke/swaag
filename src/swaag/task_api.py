from __future__ import annotations

import base64
import binascii
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
            output_schema = args.get("output_schema")
            if output_schema is not None and not isinstance(output_schema, dict):
                raise ValueError("output_schema must be an object or null")
            mechanical_fields = args.get("mechanical_fields")
            if mechanical_fields is not None and (
                not isinstance(mechanical_fields, dict)
                or any(
                    not isinstance(name, str) or not isinstance(source, str)
                    for name, source in mechanical_fields.items()
                )
            ):
                raise ValueError("mechanical_fields must map output field names to source names")
            attachment_payloads = args.get("attachments", [])
            if not isinstance(attachment_payloads, list):
                raise ValueError("attachments must be an array")
            decoded_attachments = [_attachment_payload(item) for item in attachment_payloads]
            max_bytes = self.workers.runtime.config.attachments.max_upload_bytes
            if any(len(data) > max_bytes for _name, _media_type, data in decoded_attachments):
                raise ValueError(f"attachment exceeds max_upload_bytes: {max_bytes}")
            record = self.workers.create(
                objective,
                name=_optional_text(args, "name"),
                output_schema=output_schema,
                mechanical_fields=mechanical_fields,
            )
            for name, media_type, data in decoded_attachments:
                self.workers.add_attachment(
                    record.worker_id,
                    data,
                    original_name=name,
                    media_type=media_type,
                    source="task_api_create",
                )
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
        if operation == "attachment.add":
            worker_id = _required_text(args, "worker_id")
            name, media_type, data = _attachment_payload(args)
            reference = self.workers.add_attachment(
                worker_id,
                data,
                original_name=name,
                media_type=media_type,
                source=_optional_text(args, "source") or "task_api",
            )
            payload = asdict(reference)
            payload.pop("storage_ref", None)
            return {"version": self.version, "worker_id": worker_id, "attachment": payload}
        if operation == "attachment.list":
            worker_id = _required_text(args, "worker_id")
            attachments = []
            for reference in self.workers.attachments(worker_id):
                payload = asdict(reference)
                payload.pop("storage_ref", None)
                attachments.append(payload)
            return {"version": self.version, "worker_id": worker_id, "attachments": attachments}
        if operation == "events":
            worker_id = _required_text(args, "worker_id")
            after = args.get("after_sequence", 0)
            if not isinstance(after, int) or isinstance(after, bool) or after < 0:
                raise ValueError("after_sequence must be a non-negative integer")
            limit = args.get("limit", 200)
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or not 1 <= limit <= 1000
            ):
                raise ValueError("limit must be an integer between 1 and 1000")
            available = self.workers.events(worker_id, after_sequence=after)
            events = available[:limit]
            return {
                "version": self.version,
                "worker_id": worker_id,
                "events": [asdict(item) for item in events],
                "next_sequence": events[-1].sequence if events else after,
                "has_more": len(available) > len(events),
            }
        raise ValueError(f"Unknown task operation: {operation}")

    def _record(self, record) -> dict[str, Any]:
        payload = asdict(record)
        payload["structured_output"] = self.workers.structured_output(record.worker_id)
        return {"version": self.version, "worker": payload}


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


def _attachment_payload(payload: Any) -> tuple[str, str, bytes]:
    if not isinstance(payload, dict):
        raise ValueError("attachment must be an object")
    name = _required_text(payload, "original_name")
    media_type = _optional_text(payload, "media_type") or ""
    encoded = payload.get("content_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("content_base64 must be a non-empty base64 string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("content_base64 is invalid") from exc
    return name, media_type, data
