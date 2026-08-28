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
from swaag.tools.base import Tool, ToolContext, ToolExecutionError, ToolValidationError
from swaag.types import AttachmentReference, ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import new_id, sha256_text, stable_json_dumps


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


def _completed_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _all2text_command(command_text: str) -> list[str]:
    command = shlex.split(command_text)
    if not command:
        raise ToolValidationError("attachments.all2text_command is empty")
    executable = command[0]
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(
            f"configured all2text command is unavailable: {command_text}"
        )
    return [str(resolved), *command[1:]]


def _capability_row(value: object) -> dict:
    item = value if isinstance(value, dict) else {}
    details = item.get("details")
    details = details if isinstance(details, dict) else {}
    candidate = details.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    lifecycle = item.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
    return {
        "name": str(item.get("name", "")),
        "kind": str(item.get("kind", "")),
        "family": str(candidate.get("family", "")),
        "enabled": bool(item.get("enabled", False)),
        "available": bool(item.get("available", False)),
        "source": item.get("source"),
        "error": item.get("error"),
        "execution_status": str(
            candidate.get("execution_status", details.get("execution_status", ""))
        ),
        "notes": str(candidate.get("notes", "")),
        "lifecycle": sorted(
            str(name) for name, active in lifecycle.items() if active is True
        ),
    }


def _capability_items(payload: dict, key: str) -> list[dict]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"all2text capabilities field {key!r} must be a list of objects")
    return value


def _attachment_failure(
    context: ToolContext,
    message: str,
    *,
    error_type: str,
    stdout: str = "",
    stderr: str = "",
    evidence: dict | None = None,
    generated_events: list[ToolGeneratedEvent] | None = None,
) -> ToolExecutionError:
    artifact_store = TextArtifactStore(
        context.config.sessions.root,
        context.session_state.session_id,
    )
    exact_evidence = dict(evidence or {})
    events = list(generated_events or [])
    for stream_name, text in (("stdout", stdout), ("stderr", stderr)):
        exact_evidence[f"{stream_name}_chars"] = len(text)
        exact_evidence[f"{stream_name}_sha256"] = sha256_text(text)
        exact_evidence[f"{stream_name}_artifact_id"] = ""
        if not text:
            continue
        artifact = artifact_store.create(
            text,
            kind=f"attachment_extraction_{stream_name}",
        )
        exact_evidence[f"{stream_name}_artifact_id"] = artifact.artifact_id
        events.append(
            ToolGeneratedEvent(
                "artifact_created",
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "size_chars": artifact.size_chars,
                    "sha256": artifact.sha256,
                },
            )
        )
    return ToolExecutionError(
        message,
        error_type=error_type,
        evidence=exact_evidence,
        generated_events=events,
    )


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
        "Use the exact attachment_id. Binary or non-UTF-8 content is not coerced to text; select extract_attachment "
        "or another specialist capability instead. Raw bytes remain authoritative outside context. If finished=false, "
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


class InspectAttachmentCapabilitiesTool(Tool):
    name = "inspect_attachment_capabilities"
    description = (
        "Inspect the configured all2text extractor's actual document, image, OCR, "
        "speech, media, scientific, and other specialist availability without "
        "reading an attachment."
    )
    usage_guidance = (
        "Use when choosing an extraction profile or specialist requires current "
        "host evidence. The compact index includes every reported provider family; "
        "capabilities_artifact_id retains the exact complete discovery response. "
        "Availability does not imply that extraction is useful for the current task."
    )
    kind = "stateful"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict) -> dict:
        if raw_input:
            raise ToolValidationError(
                "inspect_attachment_capabilities takes no arguments"
            )
        return {}

    def required_generated_event_types(self, validated_input: dict) -> set[str]:
        return {"artifact_created"}

    def execute(
        self, validated_input: dict, context: ToolContext
    ) -> ToolExecutionResult:
        del validated_input
        command = _all2text_command(context.config.attachments.all2text_command)
        try:
            completed = subprocess.run(
                [*command, "--capabilities"],
                cwd=context.environment.current_cwd,
                text=True,
                capture_output=True,
                timeout=context.config.attachments.extraction_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _attachment_failure(
                context,
                "all2text capability discovery exceeded the configured timeout",
                error_type=type(exc).__name__,
                stdout=_completed_output(exc.stdout),
                stderr=_completed_output(exc.stderr),
            ) from exc
        if completed.returncode != 0:
            exact_detail = completed.stderr or completed.stdout
            detail = exact_detail[-1000:]
            suffix = (
                ""
                if len(detail) == len(exact_detail)
                else " [bounded tail; read the evidence artifact for complete output]"
            )
            raise _attachment_failure(
                context,
                "all2text capability discovery failed with exit code "
                f"{completed.returncode}: {detail}{suffix}",
                error_type="All2TextCapabilityError",
                stdout=completed.stdout,
                stderr=completed.stderr,
                evidence={"return_code": completed.returncode},
            )
        try:
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict):
                raise ValueError("all2text capabilities must be a JSON object")
            detected_profile = payload.get("profile", {})
            summary = payload.get("summary", {})
            if not isinstance(detected_profile, dict):
                raise ValueError("all2text capabilities field 'profile' must be an object")
            if not isinstance(summary, dict):
                raise ValueError("all2text capabilities field 'summary' must be an object")
            optional_python = _capability_items(payload, "optional_python_libraries")
            external_tools = _capability_items(payload, "external_tools")
            provider_statuses = _capability_items(payload, "provider_statuses")
            provider_families = _capability_items(
                payload, "provider_family_statuses"
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise _attachment_failure(
                context,
                f"all2text capability discovery returned invalid JSON: {exc}",
                error_type=type(exc).__name__,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ) from exc

        artifact_store = TextArtifactStore(
            context.config.sessions.root,
            context.session_state.session_id,
        )
        artifact = artifact_store.create(
            completed.stdout,
            kind="attachment_extraction_capabilities",
        )
        events = [
            ToolGeneratedEvent(
                "artifact_created",
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "size_chars": artifact.size_chars,
                    "sha256": artifact.sha256,
                },
            )
        ]
        stderr_artifact_id = ""
        stderr_sha256 = sha256_text(completed.stderr)
        if completed.stderr:
            stderr_artifact = artifact_store.create(
                completed.stderr,
                kind="attachment_extraction_capabilities_stderr",
            )
            stderr_artifact_id = stderr_artifact.artifact_id
            stderr_sha256 = stderr_artifact.sha256
            events.append(
                ToolGeneratedEvent(
                    "artifact_created",
                    {
                        "artifact_id": stderr_artifact.artifact_id,
                        "kind": stderr_artifact.kind,
                        "size_chars": stderr_artifact.size_chars,
                        "sha256": stderr_artifact.sha256,
                    },
                )
            )

        output = {
            "extractor": "all2text",
            "extraction_profiles": ["core", "pip", "tools", "local-models", "full"],
            "detected_profile": detected_profile,
            "summary": summary,
            "optional_python_libraries": [
                {
                    key: item.get(key)
                    for key in (
                        "name",
                        "module",
                        "extra",
                        "enabled_by_profile",
                        "available",
                        "implemented_in_core",
                        "error",
                    )
                }
                for item in optional_python
            ],
            "external_tools": [
                {
                    key: item.get(key)
                    for key in (
                        "name",
                        "enabled",
                        "available",
                        "source",
                        "used_by_core",
                        "error",
                    )
                }
                for item in external_tools
            ],
            "providers": [_capability_row(item) for item in provider_statuses],
            "provider_families": [
                _capability_row(item) for item in provider_families
            ],
            "capabilities_artifact_id": artifact.artifact_id,
            "capabilities_sha256": artifact.sha256,
            "capabilities_chars": artifact.size_chars,
            "stderr_artifact_id": stderr_artifact_id,
            "stderr_sha256": stderr_sha256,
            "stderr_chars": len(completed.stderr),
        }
        return ToolExecutionResult(
            self.name,
            output,
            stable_json_dumps(output, indent=2),
            events,
        )


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
        command = _all2text_command(context.config.attachments.all2text_command)

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
        try:
            completed = subprocess.run(
                [*command, "--profile", validated_input["profile"], str(source_root), str(output_root)],
                cwd=extraction_root,
                text=True,
                capture_output=True,
                timeout=context.config.attachments.extraction_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _attachment_failure(
                context,
                f"all2text exceeded the {context.config.attachments.extraction_timeout_seconds}-second timeout",
                error_type=type(exc).__name__,
                stdout=_completed_output(exc.stdout),
                stderr=_completed_output(exc.stderr),
                evidence={
                    "attachment_id": reference.attachment_id,
                    "attachment_sha256": reference.sha256,
                    "profile": validated_input["profile"],
                },
            ) from exc
        if completed.returncode != 0:
            exact_detail = completed.stderr or completed.stdout
            detail = exact_detail[-1000:]
            suffix = "" if len(detail) == len(exact_detail) else " [bounded tail; read the evidence artifact for complete output]"
            raise _attachment_failure(
                context,
                f"all2text failed with exit code {completed.returncode}: {detail}{suffix}",
                error_type="All2TextProcessError",
                stdout=completed.stdout,
                stderr=completed.stderr,
                evidence={
                    "attachment_id": reference.attachment_id,
                    "attachment_sha256": reference.sha256,
                    "profile": validated_input["profile"],
                    "return_code": completed.returncode,
                },
            )
        manifest_path = output_root / "_conversion_manifest.json"
        artifact_store = TextArtifactStore(
            context.config.sessions.root, context.session_state.session_id
        )
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _attachment_failure(
                context,
                f"all2text manifest could not be read: {exc}",
                error_type=type(exc).__name__,
                stdout=completed.stdout,
                stderr=completed.stderr,
                evidence={
                    "attachment_id": reference.attachment_id,
                    "attachment_sha256": reference.sha256,
                    "profile": validated_input["profile"],
                    "return_code": completed.returncode,
                },
            ) from exc
        manifest_artifact = artifact_store.create(
            manifest_text,
            kind="attachment_extraction_manifest",
        )
        manifest_event = ToolGeneratedEvent(
            "artifact_created",
            {
                "artifact_id": manifest_artifact.artifact_id,
                "kind": manifest_artifact.kind,
                "size_chars": manifest_artifact.size_chars,
                "sha256": manifest_artifact.sha256,
            },
        )
        try:
            manifest = json.loads(manifest_text)
            if not isinstance(manifest, dict):
                raise ValueError("all2text manifest must be a JSON object")
            entries = [
                item for item in manifest.get("entries", [])
                if isinstance(item, dict) and item.get("relative_path") == safe_name
            ]
            if len(entries) != 1:
                raise ValueError(
                    f"all2text manifest did not contain exactly one source entry for {safe_name}"
                )
            entry = entries[0]
            output_path = Path(str(entry.get("output_path", ""))).resolve()
            output_path.relative_to(output_root.resolve())
            full_text = output_path.read_text(encoding="utf-8")
        except (AttributeError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            raise _attachment_failure(
                context,
                f"all2text output validation failed: {exc}",
                error_type=type(exc).__name__,
                stdout=completed.stdout,
                stderr=completed.stderr,
                evidence={
                    "attachment_id": reference.attachment_id,
                    "attachment_sha256": reference.sha256,
                    "profile": validated_input["profile"],
                    "return_code": completed.returncode,
                    "manifest_artifact_id": manifest_artifact.artifact_id,
                    "manifest_sha256": manifest_artifact.sha256,
                    "manifest_chars": manifest_artifact.size_chars,
                },
                generated_events=[manifest_event],
            ) from exc
        artifact = artifact_store.create(full_text, kind="attachment_extraction")
        preview = full_text[: context.config.attachments.preview_chars]
        stream_evidence: dict[str, object] = {}
        stream_events: list[ToolGeneratedEvent] = []
        for stream_name, text in (
            ("stdout", completed.stdout),
            ("stderr", completed.stderr),
        ):
            stream_evidence[f"{stream_name}_chars"] = len(text)
            stream_evidence[f"{stream_name}_sha256"] = sha256_text(text)
            stream_evidence[f"{stream_name}_artifact_id"] = ""
            if not text:
                continue
            stream_artifact = artifact_store.create(
                text,
                kind=f"attachment_extraction_{stream_name}",
            )
            stream_evidence[f"{stream_name}_artifact_id"] = (
                stream_artifact.artifact_id
            )
            stream_events.append(
                ToolGeneratedEvent(
                    "artifact_created",
                    {
                        "artifact_id": stream_artifact.artifact_id,
                        "kind": stream_artifact.kind,
                        "size_chars": stream_artifact.size_chars,
                        "sha256": stream_artifact.sha256,
                    },
                )
            )
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
            **stream_evidence,
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
            manifest_event,
            *stream_events,
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
                    **stream_evidence,
                    "source_event_references": source_references,
                },
            ),
        ]
        return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2), events)


ATTACHMENT_TOOLS = [
    ListAttachmentsTool(),
    ReadAttachmentTool(),
    InspectAttachmentCapabilitiesTool(),
    ExtractAttachmentTool(),
]
