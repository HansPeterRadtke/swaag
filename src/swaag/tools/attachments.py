from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from swaag.attachments import AttachmentStore, find_attachment
from swaag.environment.artifacts import TextArtifactStore
from swaag.fsops import ensure_dir
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import AttachmentReference, ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import new_id, stable_json_dumps


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
    description = "Read a bounded UTF-8 preview from one raw attachment after deciding that direct text inspection is useful."
    usage_guidance = (
        "Use the exact attachment_id. Binary or non-UTF-8 content is not coerced to text; select extract_attachment "
        "or another specialist capability instead. Raw bytes remain authoritative outside context."
    )
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {"type": "string"},
            "max_chars": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["attachment_id", "max_chars"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict) -> dict:
        attachment_id = raw_input.get("attachment_id")
        max_chars = raw_input.get("max_chars")
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            raise ToolValidationError("read_attachment.attachment_id must be a non-empty string")
        if max_chars is not None and (
            not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0
        ):
            raise ToolValidationError("read_attachment.max_chars must be a positive integer or null")
        return {"attachment_id": attachment_id.strip(), "max_chars": max_chars}

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
        preview = text[:limit]
        output = {
            "attachment_id": reference.attachment_id,
            "original_name": reference.original_name,
            "media_type": reference.media_type,
            "size_bytes": reference.size_bytes,
            "sha256": reference.sha256,
            "text_available": text_available,
            "text": preview,
            "text_chars": len(text) if text_available else 0,
            "truncated": text_available and len(preview) < len(text),
            "source_event_references": _source_references(reference),
        }
        return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))


class ExtractAttachmentTool(Tool):
    name = "extract_attachment"
    description = "Run the configured all2text capability on one raw attachment and retain its complete auditable text output as a durable artifact."
    usage_guidance = (
        "Use only after semantically deciding content extraction is needed. Start with profile core for deterministic safe extraction; "
        "choose broader profiles only when their optional providers are materially useful. The returned preview is bounded, while "
        "artifact_id addresses the complete derived text. Extraction is a derived view and never replaces the raw attachment."
    )
    kind = "stateful"
    input_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {"type": "string"},
            "profile": {"type": "string", "enum": ["core", "pip", "tools", "local-models", "full"]},
        },
        "required": ["attachment_id", "profile"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict) -> dict:
        attachment_id = raw_input.get("attachment_id")
        profile = raw_input.get("profile")
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            raise ToolValidationError("extract_attachment.attachment_id must be a non-empty string")
        if profile not in {"core", "pip", "tools", "local-models", "full"}:
            raise ToolValidationError("extract_attachment.profile is invalid")
        return {"attachment_id": attachment_id.strip(), "profile": profile}

    def required_generated_event_types(self, validated_input: dict) -> set[str]:
        return {"artifact_created", "attachment_extracted"}

    def execute(self, validated_input: dict, context: ToolContext) -> ToolExecutionResult:
        reference = _reference(context, validated_input["attachment_id"])
        command = shlex.split(context.config.attachments.all2text_command)
        if not command:
            raise ToolValidationError("attachments.all2text_command is empty")
        executable = command[0]
        resolved_executable = shutil.which(executable) if "/" not in executable else executable
        if not resolved_executable or not Path(resolved_executable).is_file():
            raise FileNotFoundError(
                f"configured all2text command is unavailable: {context.config.attachments.all2text_command}"
            )

        extraction_id = new_id("extraction")
        extraction_root = (
            context.config.sessions.root
            / context.session_state.session_id
            / "attachment_extractions"
            / extraction_id
        )
        source_root = extraction_root / "source"
        output_root = extraction_root / "output"
        ensure_dir(source_root)
        safe_name = Path(reference.original_name.replace("\\", "/")).name.strip() or "attachment.bin"
        if safe_name in {".", ".."} or "\x00" in safe_name:
            safe_name = "attachment.bin"
        shutil.copyfile(_store(context).path_for(reference), source_root / safe_name)
        completed = subprocess.run(
            [resolved_executable, *command[1:], "--profile", validated_input["profile"], str(source_root), str(output_root)],
            cwd=extraction_root,
            text=True,
            capture_output=True,
            timeout=context.config.attachments.extraction_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"all2text failed with exit code {completed.returncode}: {detail}")
        manifest_path = output_root / "_conversion_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [
            item for item in manifest.get("entries", [])
            if isinstance(item, dict) and item.get("relative_path") == safe_name
        ]
        if len(entries) != 1:
            raise RuntimeError(f"all2text manifest did not contain exactly one source entry for {safe_name}")
        entry = entries[0]
        output_path = Path(str(entry.get("output_path", ""))).resolve()
        try:
            output_path.relative_to(output_root.resolve())
        except ValueError as exc:
            raise RuntimeError("all2text returned an output path outside its extraction root") from exc
        full_text = output_path.read_text(encoding="utf-8")
        artifact_store = TextArtifactStore(
            context.config.sessions.root, context.session_state.session_id
        )
        artifact = artifact_store.create(full_text, kind="attachment_extraction")
        manifest_artifact = artifact_store.create(
            stable_json_dumps(manifest, indent=2) + "\n",
            kind="attachment_extraction_manifest",
        )
        preview = full_text[: context.config.attachments.preview_chars]
        manifest_summary = {
            "schema": manifest.get("schema"),
            "summary": manifest.get("summary", {}),
            "entry": {
                key: entry.get(key)
                for key in (
                    "relative_path",
                    "converter_used",
                    "extraction_methods_used",
                    "errors",
                    "warnings",
                    "limitations",
                    "llm_used",
                    "ocr_used",
                    "vlm_used",
                )
            },
        }
        source_references = _source_references(reference)
        output = {
            "attachment_id": reference.attachment_id,
            "attachment_sha256": reference.sha256,
            "extractor": "all2text",
            "profile": validated_input["profile"],
            "artifact_id": artifact.artifact_id,
            "artifact_sha256": artifact.sha256,
            "manifest_artifact_id": manifest_artifact.artifact_id,
            "manifest_artifact_sha256": manifest_artifact.sha256,
            "total_chars": artifact.size_chars,
            "text": preview,
            "truncated": len(preview) < len(full_text),
            "manifest": manifest_summary,
            "source_event_references": source_references,
        }
        events = [
            ToolGeneratedEvent(
                "artifact_created",
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "size_chars": artifact.size_chars,
                    "sha256": artifact.sha256,
                },
            ),
            ToolGeneratedEvent(
                "artifact_created",
                {
                    "artifact_id": manifest_artifact.artifact_id,
                    "kind": manifest_artifact.kind,
                    "size_chars": manifest_artifact.size_chars,
                    "sha256": manifest_artifact.sha256,
                },
            ),
            ToolGeneratedEvent(
                "attachment_extracted",
                {
                    "attachment_id": reference.attachment_id,
                    "attachment_sha256": reference.sha256,
                    "artifact_id": artifact.artifact_id,
                    "manifest_artifact_id": manifest_artifact.artifact_id,
                    "extractor": "all2text",
                    "profile": validated_input["profile"],
                    "manifest": manifest_summary,
                    "source_event_references": source_references,
                },
            ),
        ]
        return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2), events)


ATTACHMENT_TOOLS = [ListAttachmentsTool(), ReadAttachmentTool(), ExtractAttachmentTool()]
