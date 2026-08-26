from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from swaag.utils import stable_json_dumps
from swaag.workers import WorkerEvent, WorkerRecord


_A2A_STATES = {
    "created": "TASK_STATE_SUBMITTED",
    "queued": "TASK_STATE_SUBMITTED",
    "working": "TASK_STATE_WORKING",
    "cancellation_requested": "TASK_STATE_WORKING",
    "input_required": "TASK_STATE_INPUT_REQUIRED",
    "completed": "TASK_STATE_COMPLETED",
    "failed": "TASK_STATE_FAILED",
    "canceled": "TASK_STATE_CANCELED",
}


class A2AProjectionAdapter:
    """Projects internal tasks into A2A 1.0 task objects without owning lifecycle state."""

    protocol_version = "1.0"

    def task(self, record: WorkerRecord) -> dict[str, Any]:
        state = _A2A_STATES[record.status]
        text = record.result or record.error or ""
        status: dict[str, Any] = {
            "state": state,
            "timestamp": record.updated_at,
        }
        if text:
            status["message"] = {
                "role": "ROLE_AGENT",
                "parts": [{"text": text}],
                "messageId": f"{record.worker_id}-status-{record.run_count}",
            }
        task: dict[str, Any] = {
            "id": record.worker_id,
            "contextId": record.session_id,
            "status": status,
            "metadata": {
                "swaagStatus": record.status,
                "runCount": record.run_count,
                "archivedAt": record.archived_at,
            },
        }
        if record.status == "completed" and record.result:
            task["artifacts"] = [
                {
                    "artifactId": f"{record.worker_id}-result",
                    "name": "result",
                    "parts": [{"text": record.result}],
                }
            ]
        return task


class AgUiProjectionAdapter:
    """Maps durable worker events to current AG-UI event shapes."""

    def events(
        self, record: WorkerRecord, events: Iterable[WorkerEvent]
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for event in events:
            base = {
                "timestamp": _timestamp_millis(event.timestamp),
                "rawEvent": {
                    "eventId": event.event_id,
                    "eventType": event.event_type,
                    "sequence": event.sequence,
                    "payload": event.payload,
                },
                "metadata": {"swaagWorkerId": record.worker_id},
            }
            if event.event_type == "worker_history_event":
                output.extend(self._history_event(record, event, base))
            elif event.event_type == "worker_started":
                output.append(
                    {
                        **base,
                        "type": "RUN_STARTED",
                        "threadId": record.session_id,
                        "runId": self._run_id(record, event),
                    }
                )
            elif event.event_type == "worker_completed":
                event_result = self._event_text(event, "result", record.result)
                output.extend(self._assistant_result(record, event, base))
                output.append(
                    {
                        **base,
                        "type": "RUN_FINISHED",
                        "threadId": record.session_id,
                        "runId": self._run_id(record, event),
                        "result": event_result,
                        "outcome": {"type": "success"},
                    }
                )
            elif event.event_type in {"worker_failed", "worker_orphaned"}:
                output.append(
                    {
                        **base,
                        "type": "RUN_ERROR",
                        "message": self._event_text(event, "error", record.error)
                        or event.event_type,
                        "code": "SWAAG_WORKER_FAILED",
                    }
                )
            elif event.event_type == "worker_canceled":
                event_error = self._event_text(event, "error", record.error)
                output.append(
                    {
                        **base,
                        "type": "CUSTOM",
                        "name": "swaag.worker.canceled",
                        "value": {"workerId": record.worker_id},
                    }
                )
                output.append(
                    {
                        **base,
                        "type": "RUN_ERROR",
                        "message": event_error or "Worker canceled",
                        "code": "SWAAG_WORKER_CANCELED",
                    }
                )
            elif event.event_type == "worker_input_required":
                event_result = self._event_text(event, "result", record.result)
                output.extend(self._assistant_result(record, event, base))
                output.append(
                    {
                        **base,
                        "type": "CUSTOM",
                        "name": "swaag.worker.input_required",
                        "value": {
                            "workerId": record.worker_id,
                            "message": event_result or "Input required",
                        },
                    }
                )
                output.append(
                    {
                        **base,
                        "type": "RUN_FINISHED",
                        "threadId": record.session_id,
                        "runId": self._run_id(record, event),
                        "result": event_result,
                        "outcome": {
                            "type": "interrupt",
                            "interrupts": [
                                {
                                    "id": f"{record.worker_id}-input-{event.sequence}",
                                    "reason": "human_input",
                                    "message": event_result or "Input required",
                                    "metadata": {
                                        "swaagWorkerId": record.worker_id,
                                        "swaagEventId": event.event_id,
                                    },
                                }
                            ],
                        },
                    }
                )
            else:
                output.append(
                    {
                        **base,
                        "type": "ACTIVITY_SNAPSHOT",
                        "messageId": f"{record.worker_id}-activity",
                        "activityType": "SWAAG_WORKER",
                        "content": {
                            "status": record.status,
                            "eventType": event.event_type,
                            "payload": event.payload,
                        },
                        "replace": True,
                    }
                )
        return output

    @staticmethod
    def _history_event(
        record: WorkerRecord,
        event: WorkerEvent,
        base: dict[str, Any],
    ) -> list[dict[str, Any]]:
        source = event.payload.get("canonical_event")
        if not isinstance(source, dict):
            return [
                {
                    **base,
                    "type": "ACTIVITY_SNAPSHOT",
                    "messageId": f"{record.worker_id}-activity",
                    "activityType": "SWAAG_HISTORY_REFERENCE",
                    "content": event.payload,
                    "replace": True,
                }
            ]
        source_type = str(source.get("type", ""))
        source_payload = source.get("payload")
        if not isinstance(source_payload, dict):
            source_payload = {}
        history_metadata = {
            **dict(base.get("metadata", {})),
            "swaagHistorySequence": source.get("sequence"),
            "swaagHistoryHash": source.get("hash"),
        }
        history_base = {**base, "metadata": history_metadata}
        call_id = str(source_payload.get("call_id") or source.get("id") or event.event_id)
        if source_type == "tool_called":
            tool_name = str(source_payload.get("tool_name", ""))
            arguments = source_payload.get("tool_input", {})
            return [
                {
                    **history_base,
                    "type": "TOOL_CALL_START",
                    "toolCallId": call_id,
                    "toolCallName": tool_name,
                },
                {
                    **history_base,
                    "type": "TOOL_CALL_ARGS",
                    "toolCallId": call_id,
                    "delta": stable_json_dumps(arguments, indent=None),
                },
                {
                    **history_base,
                    "type": "TOOL_CALL_END",
                    "toolCallId": call_id,
                },
            ]
        if source_type in {"tool_result", "tool_error"}:
            if source_type == "tool_result":
                content = source_payload.get("output")
            else:
                content = {
                    "error": source_payload.get("error"),
                    "error_type": source_payload.get("error_type"),
                }
            return [
                {
                    **history_base,
                    "type": "TOOL_CALL_RESULT",
                    "messageId": str(source.get("id") or event.event_id),
                    "toolCallId": call_id,
                    "content": (
                        content
                        if isinstance(content, str)
                        else stable_json_dumps(content, indent=None)
                    ),
                    "role": "tool",
                }
            ]
        if source_type == "assistant_progress":
            text = source_payload.get("assistant_text")
            if isinstance(text, str) and text:
                message_id = str(source.get("id") or event.event_id)
                return [
                    {
                        **history_base,
                        "type": "TEXT_MESSAGE_START",
                        "messageId": message_id,
                        "role": "assistant",
                    },
                    {
                        **history_base,
                        "type": "TEXT_MESSAGE_CONTENT",
                        "messageId": message_id,
                        "delta": text,
                    },
                    {
                        **history_base,
                        "type": "TEXT_MESSAGE_END",
                        "messageId": message_id,
                    },
                ]
        if source_type == "agent_question":
            return [
                {
                    **history_base,
                    "type": "CUSTOM",
                    "name": "swaag.agent.question",
                    "value": source_payload,
                }
            ]
        return [
            {
                **history_base,
                "type": "ACTIVITY_SNAPSHOT",
                "messageId": f"{record.worker_id}-activity",
                "activityType": f"SWAAG_{source_type.upper() or 'HISTORY_EVENT'}",
                "content": source_payload,
                "replace": True,
            }
        ]

    @staticmethod
    def _run_id(record: WorkerRecord, event: WorkerEvent) -> str:
        run_count = event.payload.get("run_count", record.run_count)
        if not isinstance(run_count, int) or run_count < 1:
            run_count = record.run_count
        return f"{record.worker_id}-run-{run_count}"

    @staticmethod
    def _event_text(event: WorkerEvent, key: str, fallback: str | None) -> str | None:
        value = event.payload.get(key)
        return value if isinstance(value, str) else fallback

    @staticmethod
    def _assistant_result(
        record: WorkerRecord, event: WorkerEvent, base: dict[str, Any]
    ) -> list[dict[str, Any]]:
        result = AgUiProjectionAdapter._event_text(event, "result", record.result)
        if not result:
            return []
        message_id = f"{record.worker_id}-result-{event.sequence}"
        return [
            {**base, "type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"},
            {**base, "type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": result},
            {**base, "type": "TEXT_MESSAGE_END", "messageId": message_id},
        ]


class OpenWebUiProjectionAdapter:
    """Builds persistence-safe Open WebUI status events plus a final return value."""

    def response(self, record: WorkerRecord) -> dict[str, Any]:
        done = record.status in {"completed", "failed", "canceled"}
        description = record.result or record.error or f"Worker is {record.status}"
        return {
            "return": record.result or (record.error if done else None),
            "events": [
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                        "hidden": False,
                    },
                }
            ],
            "metadata": {
                "worker_id": record.worker_id,
                "session_id": record.session_id,
                "status": record.status,
            },
        }


def _timestamp_millis(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0
