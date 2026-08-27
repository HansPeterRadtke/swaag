"""
title: Swaag Durable Agent
author: Hans-Peter Radtke
description: Connect Open WebUI chats to durable Swaag workers.
version: 0.1.0
requirements: pydantic
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import unquote_to_bytes

from pydantic import BaseModel, Field


EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]
_PIPE_TERMINAL_STATES = {"completed", "failed", "canceled", "input_required"}


class _JsonLineClient:
    def __init__(self, host: str, port: int, timeout_seconds: float):
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    async def request(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.timeout_seconds,
        )
        try:
            payload = json.dumps({"op": operation, "params": params}) + "\n"
            writer.write(payload.encode("utf-8"))
            await asyncio.wait_for(writer.drain(), timeout=self.timeout_seconds)
            line = await asyncio.wait_for(
                reader.readline(), timeout=self.timeout_seconds
            )
            if not line:
                raise RuntimeError("Swaag closed the communication connection")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise RuntimeError("Swaag returned a non-object response")
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error") or "Swaag request failed"))
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Swaag returned a non-object result")
            return result
        finally:
            writer.close()
            await writer.wait_closed()


class Pipe:
    class Valves(BaseModel):
        SWAAG_HOST: str = Field(
            default="127.0.0.1",
            description="Host running the localhost Swaag communication service.",
        )
        SWAAG_PORT: int = Field(
            default=13401,
            ge=1,
            le=65535,
            description="Registered Swaag communication port.",
        )
        REQUEST_TIMEOUT_SECONDS: float = Field(
            default=45.0,
            gt=30.0,
            description="Per-poll transport timeout; worker tasks themselves have no deadline.",
        )
        EMIT_STATUS: bool = Field(
            default=True,
            description="Emit persisted Open WebUI status events while Swaag works.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __files__: list[dict[str, Any]] | None = None,
        __event_emitter__: EventEmitter | None = None,
        **_: Any,
    ) -> str:
        try:
            metadata = __metadata__ if isinstance(__metadata__, dict) else {}
            conversation_id = _first_text(
                metadata.get("chat_id"),
                body.get("chat_id"),
                body.get("conversation_id"),
            )
            request_id = _first_text(
                metadata.get("message_id"),
                body.get("id"),
            )
            if not conversation_id or not request_id:
                raise ValueError(
                    "Open WebUI did not provide stable chat_id/message_id metadata"
                )

            message, attachments, references = _latest_user_payload(body)
            file_attachments, file_references = await _raw_file_attachments(
                __files__ or []
            )
            attachments.extend(file_attachments)
            references.extend(file_references)
            if references:
                message += "\n\nRaw attachment references:\n" + "\n".join(
                    f"- {item}" for item in references
                )

            client = _JsonLineClient(
                self.valves.SWAAG_HOST,
                self.valves.SWAAG_PORT,
                self.valves.REQUEST_TIMEOUT_SECONDS,
            )
            current = await client.request(
                "open_webui.send",
                {
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "message": message,
                    "attachments": attachments,
                },
            )
            await self._emit_projection(__event_emitter__, current)
            cursor = int(current.get("next_sequence", 0))

            while str(current.get("metadata", {}).get("status", "")) not in _PIPE_TERMINAL_STATES:
                page = await client.request(
                    "task.events.wait",
                    {
                        "worker_id": str(current["metadata"]["worker_id"]),
                        "after_sequence": cursor,
                        "limit": 200,
                        "timeout_seconds": 30,
                    },
                )
                cursor = int(page["next_sequence"])
                current = await client.request(
                    "open_webui.get",
                    {"worker_id": str(current["metadata"]["worker_id"])},
                )
                await self._emit_projection(__event_emitter__, current)

            result = current.get("return")
            if isinstance(result, str) and result:
                return result
            status = str(current.get("metadata", {}).get("status", "unknown"))
            return f"Swaag worker ended in {status} without a textual result."
        except Exception as exc:
            message = f"Swaag communication error: {type(exc).__name__}: {exc}"
            await self._emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": True,
                        "hidden": False,
                    },
                },
            )
            return message

    async def _emit_projection(
        self,
        emitter: EventEmitter | None,
        projection: dict[str, Any],
    ) -> None:
        if not self.valves.EMIT_STATUS:
            return
        events = projection.get("events", [])
        if not isinstance(events, list):
            return
        for event in events:
            if isinstance(event, dict):
                await self._emit(emitter, event)

    @staticmethod
    async def _emit(
        emitter: EventEmitter | None,
        event: dict[str, Any],
    ) -> None:
        if emitter is None:
            return
        try:
            await emitter(event)
        except Exception:
            # The durable worker/result channel must survive a closed browser socket.
            return


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _latest_user_payload(
    body: dict[str, Any],
) -> tuple[str, list[dict[str, str]], list[str]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Open WebUI body.messages must be an array")
    message = next(
        (
            item
            for item in reversed(messages)
            if isinstance(item, dict) and item.get("role") == "user"
        ),
        None,
    )
    if message is None:
        raise ValueError("Open WebUI request has no user message")
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        if not text:
            raise ValueError("Open WebUI user message is empty")
        return text, [], []
    if not isinstance(content, list):
        raise ValueError("Open WebUI user message content has an unsupported shape")

    text_parts: list[str] = []
    attachments: list[dict[str, str]] = []
    references: list[str] = []
    for index, part in enumerate(content, start=1):
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            if part["text"].strip():
                text_parts.append(part["text"].strip())
            continue
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, dict) else None
        if not isinstance(url, str) or not url:
            continue
        if url.startswith("data:"):
            media_type, raw = _decode_data_url(url)
            extension = mimetypes.guess_extension(media_type) or ".bin"
            attachments.append(
                {
                    "original_name": f"open-webui-image-{index}{extension}",
                    "media_type": media_type,
                    "content_base64": base64.b64encode(raw).decode("ascii"),
                }
            )
        else:
            references.append(url)
    text = "\n\n".join(text_parts).strip()
    if not text:
        text = "Inspect the supplied raw attachments and complete the request."
    return text, attachments, references


def _decode_data_url(value: str) -> tuple[str, bytes]:
    try:
        header, encoded = value.split(",", 1)
        media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
        raw = (
            base64.b64decode(encoded, validate=True)
            if ";base64" in header
            else unquote_to_bytes(encoded)
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Open WebUI image data URL is invalid") from exc
    return media_type, raw


async def _raw_file_attachments(
    files: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    attachments: list[dict[str, str]] = []
    references: list[str] = []
    for index, entry in enumerate(files, start=1):
        if not isinstance(entry, dict):
            continue
        details = entry.get("file")
        if not isinstance(details, dict):
            details = entry
        path_value = details.get("path")
        name = _first_text(
            details.get("filename"), entry.get("name"), f"attachment-{index}"
        )
        metadata = details.get("meta")
        media_type = (
            _first_text(metadata.get("content_type"))
            if isinstance(metadata, dict)
            else ""
        )
        if not isinstance(path_value, str) or not path_value:
            reference = _first_text(entry.get("url"), details.get("url"))
            if reference:
                references.append(f"{name}: {reference}")
            continue
        path = Path(path_value)
        if not path.is_file():
            references.append(f"{name}: {path_value}")
            continue
        raw = await asyncio.to_thread(path.read_bytes)
        attachments.append(
            {
                "original_name": name,
                "media_type": media_type,
                "content_base64": base64.b64encode(raw).decode("ascii"),
            }
        )
    return attachments, references
