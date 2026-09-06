from __future__ import annotations

from swaag.attachments import AttachmentStore, find_attachment
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import AttachmentReference, ToolExecutionResult
from swaag.utils import stable_json_dumps


def _store(context: ToolContext) -> AttachmentStore:
    return AttachmentStore(
        context.config.sessions.root,
        max_upload_bytes=context.config.attachments.max_upload_bytes,
    )


def _reference(context: ToolContext, attachment_id: str) -> AttachmentReference:
    return find_attachment(context.session_state.attachments, attachment_id)


def _source_references(reference: AttachmentReference) -> list[dict]:
    metadata = reference.metadata
    sequence = metadata.get("source_event_sequence")
    source_hash = metadata.get("source_event_hash")
    if not isinstance(sequence, int) or not isinstance(source_hash, str) or not source_hash:
        return []
    return [
        {
            "session_id": str(metadata.get("source_event_session_id", "")),
            "sequence": sequence,
            "hash": source_hash,
            "event_type": str(metadata.get("source_event_type", "attachment_added")),
        }
    ]


class ListAttachmentsTool(Tool):
    name = "list_attachments"
    description = "List raw attachments for the active task using stable IDs and cheap metadata only; it does not inspect content."
    usage_guidance = "Use when attachment references are relevant but their exact IDs or metadata need confirmation."
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict) -> dict:
        if raw_input:
            raise ToolValidationError("list_attachments takes no arguments")
        return {}

    def execute(self, validated_input: dict, context: ToolContext) -> ToolExecutionResult:
        del validated_input
        store = _store(context)
        output = {
            "attachments": [store.public_metadata(item) for item in context.session_state.attachments],
            "count": len(context.session_state.attachments),
        }
        return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))


class ReadAttachmentTool(Tool):
    name = "read_attachment"
    description = "Read an exact bounded UTF-8 slice from one raw attachment after deciding that direct text inspection is useful."
    usage_guidance = (
        "Use the exact attachment_id. Binary or non-UTF-8 content is not coerced to text. Use an enabled external tool or shell-accessible host utility when semantic inspection of binary content is required. Raw bytes remain authoritative outside context. If finished=false, "
        "advance start_offset to next_offset and continue until the evidence needed by the task is complete."
    )
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {"type": "string"},
            "start_offset": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "max_chars": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["attachment_id", "start_offset", "max_chars"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict) -> dict:
        attachment_id = raw_input.get("attachment_id")
        start_offset = raw_input.get("start_offset")
        start_offset = 0 if start_offset is None else start_offset
        max_chars = raw_input.get("max_chars")
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            raise ToolValidationError("read_attachment.attachment_id must be a non-empty string")
        if (
            not isinstance(start_offset, int)
            or isinstance(start_offset, bool)
            or start_offset < 0
        ):
            raise ToolValidationError(
                "read_attachment.start_offset must be a non-negative integer or null"
            )
        if max_chars is not None and (
            not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0
        ):
            raise ToolValidationError("read_attachment.max_chars must be a positive integer or null")
        return {
            "attachment_id": attachment_id.strip(),
            "start_offset": start_offset,
            "max_chars": max_chars,
        }

    def execute(self, validated_input: dict, context: ToolContext) -> ToolExecutionResult:
        reference = _reference(context, validated_input["attachment_id"])
        data = _store(context).read_bytes(reference)
        try:
            text = data.decode("utf-8")
            text_available = True
        except UnicodeDecodeError:
            text = ""
            text_available = False
        requested = validated_input["max_chars"] or context.config.attachments.preview_chars
        limit = min(int(requested), int(context.config.attachments.preview_chars))
        start = min(validated_input["start_offset"], len(text))
        end = min(len(text), start + limit)
        preview = text[start:end]
        output = {
            "attachment_id": reference.attachment_id,
            "original_name": reference.original_name,
            "media_type": reference.media_type,
            "size_bytes": reference.size_bytes,
            "sha256": reference.sha256,
            "text_available": text_available,
            "text": preview,
            "text_chars": len(text) if text_available else 0,
            "start_offset": start,
            "end_offset": end,
            "next_offset": end,
            "finished": not text_available or end >= len(text),
            "truncated": text_available and (start > 0 or end < len(text)),
            "source_event_references": _source_references(reference),
        }
        return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))


ATTACHMENT_TOOLS = [
    ListAttachmentsTool(),
    ReadAttachmentTool(),
]
