from __future__ import annotations

import base64
import binascii
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import unquote_to_bytes

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


class A2AProtocolError(ValueError):
    jsonrpc_code = -32602


class A2AUnsupportedOperationError(A2AProtocolError):
    jsonrpc_code = -32004


class A2ATaskNotCancelableError(A2AProtocolError):
    jsonrpc_code = -32002


class A2AContentTypeNotSupportedError(A2AProtocolError):
    jsonrpc_code = -32005


@dataclass(slots=True, frozen=True)
class A2AUserMessage:
    text: str
    message_id: str
    task_id: str | None
    context_id: str | None
    attachments: tuple[dict[str, str], ...]
    return_immediately: bool
    history_length: int | None


@dataclass(slots=True, frozen=True)
class AgUiRunInput:
    thread_id: str
    run_id: str
    parent_run_id: str | None
    message_id: str | None
    text: str
    context_text: str
    attachments: tuple[dict[str, str], ...]
    initial_text: str
    initial_attachments: tuple[dict[str, str], ...]
    resume: tuple[dict[str, Any], ...]


class A2AProjectionAdapter:
    """Projects internal tasks into A2A 1.0 task objects without owning lifecycle state."""

    protocol_version = "1.0"

    def user_message(self, request: dict[str, Any]) -> A2AUserMessage:
        message = request.get("message")
        if not isinstance(message, dict):
            raise ValueError("A2A message must be an object")
        if message.get("role") != "ROLE_USER":
            raise ValueError("A2A message role must be ROLE_USER")
        message_id = message.get("messageId")
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("A2A messageId must be a non-empty string")
        parts = message.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("A2A message parts must be a non-empty array")
        text_parts: list[str] = []
        attachments: list[dict[str, str]] = []
        for index, part in enumerate(parts, start=1):
            if not isinstance(part, dict):
                raise ValueError("Every A2A message part must be an object")
            content_fields = [
                key for key in ("text", "raw", "url", "data") if key in part
            ]
            if len(content_fields) != 1:
                raise ValueError("Every A2A message part must contain exactly one content field")
            if isinstance(part.get("text"), str) and str(part["text"]).strip():
                text_parts.append(str(part["text"]).strip())
                continue
            if "data" in part:
                text_parts.append(stable_json_dumps(part["data"], indent=None))
                continue
            raw = part.get("raw")
            if isinstance(raw, str) and raw:
                attachments.append(
                    {
                        "original_name": str(
                            part.get("filename") or f"attachment-{index}"
                        ),
                        "media_type": str(part.get("mediaType") or ""),
                        "content_base64": raw,
                    }
                )
                continue
            if "url" in part:
                raise A2AUnsupportedOperationError(
                    "A2A URL parts require an authenticated fetch adapter and are not enabled"
                )
            raise ValueError("Unsupported A2A message part")
        text = "\n\n".join(text_parts).strip()
        if not text:
            raise ValueError("A2A message must contain text or structured data")
        configuration = request.get("configuration") or {}
        if not isinstance(configuration, dict):
            raise ValueError("A2A message configuration must be an object")
        return_immediately = configuration.get(
            "returnImmediately",
            configuration.get("return_immediately", False),
        )
        if not isinstance(return_immediately, bool):
            raise ValueError("A2A returnImmediately must be a boolean")
        history_length = configuration.get("historyLength")
        if history_length is not None and (
            not isinstance(history_length, int)
            or isinstance(history_length, bool)
            or history_length < 0
        ):
            raise ValueError("A2A historyLength must be a non-negative integer")
        task_id = message.get("taskId")
        context_id = message.get("contextId")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            raise ValueError("A2A taskId must be a non-empty string when provided")
        if context_id is not None and (
            not isinstance(context_id, str) or not context_id.strip()
        ):
            raise ValueError("A2A contextId must be a non-empty string when provided")
        return A2AUserMessage(
            text=text,
            message_id=message_id.strip(),
            task_id=None if task_id is None else task_id.strip(),
            context_id=None if context_id is None else context_id.strip(),
            attachments=tuple(attachments),
            return_immediately=return_immediately,
            history_length=history_length,
        )

    def task(
        self,
        record: WorkerRecord,
        *,
        history: Iterable[dict[str, Any]] = (),
        include_artifacts: bool = True,
    ) -> dict[str, Any]:
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
                "swaagCompletionMode": record.completion_mode,
                "runCount": record.run_count,
                "archivedAt": record.archived_at,
            },
        }
        history_items = list(history)
        if history_items:
            task["history"] = history_items
        if include_artifacts and record.status == "completed" and record.result:
            task["artifacts"] = [
                {
                    "artifactId": f"{record.worker_id}-result",
                    "name": "result",
                    "parts": [{"text": record.result}],
                }
            ]
        return task

    def updates(
        self,
        record: WorkerRecord,
        events: Iterable[WorkerEvent],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for event in events:
            internal_status = event.payload.get("to_status")
            text = event.payload.get("result") or event.payload.get("error")
            if event.event_type == "worker_created":
                internal_status = "created"
            elif event.event_type == "worker_queued":
                internal_status = "queued"
            elif event.event_type == "worker_history_event":
                source = event.payload.get("canonical_event")
                if not isinstance(source, dict) or source.get("type") != "agent_status":
                    continue
                payload = source.get("payload")
                if not isinstance(payload, dict):
                    continue
                internal_status = "working"
                text = " ".join(
                    str(payload.get(key, "")).strip()
                    for key in ("situation", "action", "reason")
                    if str(payload.get(key, "")).strip()
                )
            if not isinstance(internal_status, str) or internal_status not in _A2A_STATES:
                continue
            if internal_status == "completed" and isinstance(text, str) and text:
                output.append(
                    {
                        "artifactUpdate": {
                            "taskId": record.worker_id,
                            "contextId": record.session_id,
                            "artifact": {
                                "artifactId": f"{record.worker_id}-result",
                                "name": "result",
                                "parts": [{"text": text}],
                            },
                            "append": False,
                            "lastChunk": True,
                        }
                    }
                )
            status: dict[str, Any] = {
                "state": _A2A_STATES[internal_status],
                "timestamp": event.timestamp,
            }
            if isinstance(text, str) and text:
                status["message"] = {
                    "role": "ROLE_AGENT",
                    "parts": [{"text": text}],
                    "messageId": f"{event.event_id}-status",
                    "taskId": record.worker_id,
                    "contextId": record.session_id,
                }
            output.append(
                {
                    "statusUpdate": {
                        "taskId": record.worker_id,
                        "contextId": record.session_id,
                        "status": status,
                    }
                }
            )
        return output


class AgUiProjectionAdapter:
    """Maps durable worker events to current AG-UI event shapes."""

    def user_run(self, request: dict[str, Any]) -> AgUiRunInput:
        thread_id = _required_ag_ui_text(request, "threadId")
        run_id = _required_ag_ui_text(request, "runId")
        parent_run_id = _optional_ag_ui_text(request, "parentRunId")
        messages = request.get("messages")
        tools = request.get("tools")
        context = request.get("context")
        forwarded_props = request.get("forwardedProps")
        if not isinstance(messages, list):
            raise ValueError("AG-UI messages must be an array")
        if not isinstance(tools, list):
            raise ValueError("AG-UI tools must be an array")
        if tools:
            raise ValueError(
                "AG-UI client-side tools are not enabled by this adapter"
            )
        if not isinstance(context, list):
            raise ValueError("AG-UI context must be an array")
        if forwarded_props not in (None, {}):
            raise ValueError(
                "AG-UI forwardedProps are not enabled by this adapter"
            )
        state = request.get("state")
        if state not in (None, {}):
            raise ValueError("AG-UI shared state is not enabled by this adapter")

        context_items: list[dict[str, str]] = []
        for item in context:
            if not isinstance(item, dict):
                raise ValueError("Every AG-UI context item must be an object")
            description = item.get("description")
            value = item.get("value")
            if not isinstance(description, str) or not isinstance(value, str):
                raise ValueError(
                    "AG-UI context description and value must be strings"
                )
            context_items.append({"description": description, "value": value})

        normalized_messages: list[dict[str, Any]] = []
        latest_user: tuple[str, str, list[dict[str, str]]] | None = None
        initial_attachments: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Every AG-UI message must be an object")
            message_id = _required_ag_ui_text(message, "id", prefix="message ")
            role = message.get("role")
            if role not in {
                "developer",
                "system",
                "assistant",
                "user",
                "tool",
                "activity",
                "reasoning",
            }:
                raise ValueError("AG-UI message role is unsupported")
            normalized = dict(message)
            if role == "user":
                text, attachments, references = _ag_ui_user_content(
                    message.get("content"), message_id=message_id
                )
                if references:
                    text = _with_raw_references(text, references)
                normalized["content"] = text
                initial_attachments.extend(attachments)
                latest_user = (message_id, text, attachments)
            normalized_messages.append(normalized)

        resume = request.get("resume", [])
        if not isinstance(resume, list) or any(
            not isinstance(item, dict) for item in resume
        ):
            raise ValueError("AG-UI resume must be an array of objects")
        if not resume and latest_user is None:
            raise ValueError("AG-UI run requires a user message or resume entry")

        context_text = (
            "\n\nAG-UI caller context:\n"
            + stable_json_dumps(context_items, indent=None)
            if context_items
            else ""
        )
        if latest_user is None:
            message_id = None
            text = ""
            attachments: list[dict[str, str]] = []
        else:
            message_id, text, attachments = latest_user
        initial_text = (
            "AG-UI conversation supplied for this new durable thread:\n"
            + stable_json_dumps(normalized_messages, indent=None)
            + context_text
        )
        return AgUiRunInput(
            thread_id=thread_id,
            run_id=run_id,
            parent_run_id=parent_run_id,
            message_id=message_id,
            text=text + context_text,
            context_text=context_text,
            attachments=tuple(attachments),
            initial_text=initial_text,
            initial_attachments=tuple(initial_attachments),
            resume=tuple(dict(item) for item in resume),
        )

    def events(
        self,
        record: WorkerRecord,
        events: Iterable[WorkerEvent],
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
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
                        "threadId": thread_id or record.session_id,
                        "runId": run_id or self._run_id(record, event),
                    }
                )
            elif event.event_type == "worker_completed":
                event_result = self._event_text(event, "result", record.result)
                output.extend(self._assistant_result(record, event, base))
                output.append(
                    {
                        **base,
                        "type": "RUN_FINISHED",
                        "threadId": thread_id or record.session_id,
                        "runId": run_id or self._run_id(record, event),
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
                        "threadId": thread_id or record.session_id,
                        "runId": run_id or self._run_id(record, event),
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
            {
                **base,
                "type": "TEXT_MESSAGE_START",
                "messageId": message_id,
                "role": "assistant",
            },
            {
                **base,
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": result,
            },
            {**base, "type": "TEXT_MESSAGE_END", "messageId": message_id},
        ]


def _required_ag_ui_text(
    payload: dict[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AG-UI {prefix}{key} must be a non-empty string")
    return value.strip()


def _optional_ag_ui_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AG-UI {key} must be a non-empty string when provided")
    return value.strip()


def _ag_ui_user_content(
    content: Any,
    *,
    message_id: str,
) -> tuple[str, list[dict[str, str]], list[str]]:
    if isinstance(content, str):
        if not content.strip():
            raise ValueError("AG-UI user message content must not be empty")
        return content, [], []
    if not isinstance(content, list):
        raise ValueError("AG-UI user content must be text or an input-content array")
    text: list[str] = []
    attachments: list[dict[str, str]] = []
    references: list[str] = []
    for index, item in enumerate(content, start=1):
        if not isinstance(item, dict):
            raise ValueError("Every AG-UI input-content item must be an object")
        content_type = item.get("type")
        if content_type == "text":
            value = item.get("text")
            if not isinstance(value, str):
                raise ValueError("AG-UI text input must contain text")
            text.append(value)
            continue
        if content_type == "binary":
            _ag_ui_legacy_binary(
                item,
                message_id=message_id,
                index=index,
                attachments=attachments,
                references=references,
            )
            continue
        if content_type not in {"image", "audio", "video", "document"}:
            raise ValueError("AG-UI input-content type is unsupported")
        source = item.get("source")
        if not isinstance(source, dict):
            raise ValueError("AG-UI media input requires a source object")
        source_type = source.get("type")
        value = source.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError("AG-UI media source value must be a non-empty string")
        media_type = source.get("mimeType", "")
        if not isinstance(media_type, str):
            raise ValueError("AG-UI media source mimeType must be a string")
        metadata = item.get("metadata")
        name = (
            str(metadata.get("filename"))
            if isinstance(metadata, dict) and metadata.get("filename")
            else _ag_ui_attachment_name(message_id, index, media_type)
        )
        if source_type == "data":
            if not media_type:
                raise ValueError("AG-UI inline media requires mimeType")
            attachments.append(
                {
                    "original_name": name,
                    "media_type": media_type,
                    "content_base64": value,
                }
            )
        elif source_type == "url" and value.startswith("data:"):
            decoded_media_type, encoded = _ag_ui_data_url(value)
            attachments.append(
                {
                    "original_name": name,
                    "media_type": media_type or decoded_media_type,
                    "content_base64": encoded,
                }
            )
        elif source_type == "url":
            references.append(f"{content_type}: {value}")
        else:
            raise ValueError("AG-UI media source type is unsupported")
    rendered = "\n\n".join(text).strip()
    if not rendered and attachments:
        rendered = "Inspect the supplied raw inputs and complete the request."
    if not rendered and not references:
        raise ValueError("AG-UI user message content must not be empty")
    return rendered, attachments, references


def _ag_ui_legacy_binary(
    item: dict[str, Any],
    *,
    message_id: str,
    index: int,
    attachments: list[dict[str, str]],
    references: list[str],
) -> None:
    media_type = item.get("mimeType")
    if not isinstance(media_type, str) or not media_type:
        raise ValueError("AG-UI binary input requires mimeType")
    data = item.get("data")
    url = item.get("url")
    reference_id = item.get("id")
    if not any(isinstance(value, str) and value for value in (data, url, reference_id)):
        raise ValueError("AG-UI binary input requires data, url, or id")
    name = item.get("filename")
    if not isinstance(name, str) or not name:
        name = _ag_ui_attachment_name(message_id, index, media_type)
    if isinstance(data, str) and data:
        attachments.append(
            {
                "original_name": name,
                "media_type": media_type,
                "content_base64": data,
            }
        )
    if isinstance(url, str) and url:
        references.append(f"binary URL: {url}")
    if isinstance(reference_id, str) and reference_id:
        references.append(f"binary ID: {reference_id}")


def _ag_ui_attachment_name(message_id: str, index: int, media_type: str) -> str:
    extension = mimetypes.guess_extension(media_type) or ".bin"
    return f"{message_id}-attachment-{index}{extension}"


def _ag_ui_data_url(value: str) -> tuple[str, str]:
    try:
        header, payload = value.split(",", 1)
        media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
        raw = (
            base64.b64decode(payload, validate=True)
            if ";base64" in header
            else unquote_to_bytes(payload)
        )
    except (binascii.Error, ValueError, UnicodeError) as exc:
        raise ValueError("AG-UI data URL is invalid") from exc
    return media_type, base64.b64encode(raw).decode("ascii")


def _with_raw_references(text: str, references: list[str]) -> str:
    prefix = text.strip() or "Inspect the supplied raw inputs and complete the request."
    return prefix + "\n\nRaw attachment references:\n" + "\n".join(
        f"- {item}" for item in references
    )


class OpenWebUiProjectionAdapter:
    """Builds persistence-safe Open WebUI status events plus a final return value."""

    def response(
        self,
        record: WorkerRecord,
        events: Iterable[WorkerEvent] = (),
    ) -> dict[str, Any]:
        done = record.status in {"completed", "failed", "canceled", "input_required"}
        description = record.result or record.error or f"Worker is {record.status}"
        projected = [
            source
            for event in events
            if (source := self._source_event(event)) is not None
        ]
        projected.append(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                    "hidden": False,
                },
            }
        )
        return {
            "return": record.result or (record.error if done else None),
            "events": projected,
            "metadata": {
                "worker_id": record.worker_id,
                "session_id": record.session_id,
                "status": record.status,
            },
        }

    @staticmethod
    def _source_event(event: WorkerEvent) -> dict[str, Any] | None:
        canonical = event.payload.get("canonical_event")
        if not isinstance(canonical, dict) or canonical.get("type") != "external_source_observed":
            return None
        payload = canonical.get("payload")
        if not isinstance(payload, dict):
            return None
        required = ("source_id", "name", "url", "document")
        if any(not isinstance(payload.get(key), str) for key in required):
            return None
        return {
            "type": "source",
            "data": {
                "source": {
                    "id": payload["source_id"],
                    "name": payload["name"],
                },
                "document": [payload["document"]],
                "metadata": [
                    {
                        "source": payload["url"],
                        "name": payload["name"],
                        "url": payload["url"],
                        "swaagHistorySequence": canonical.get("sequence"),
                        "swaagHistoryHash": canonical.get("hash"),
                        "swaagDocumentTruncated": bool(
                            payload.get("document_truncated", False)
                        ),
                    }
                ],
            },
        }


def _timestamp_millis(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0
