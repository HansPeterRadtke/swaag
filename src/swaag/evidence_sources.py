from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from swaag.attachments import AttachmentStore, find_attachment
from swaag.config import AgentConfig
from swaag.environment.artifacts import TextArtifactStore
from swaag.history import HistoryInvariantError, HistoryStore
from swaag.events import verify_event_integrity
from swaag.utils import stable_json_dumps, sha256_text
from swaag.types import HistoryEvent, SessionState


def _generated_id(value: str, prefix: str) -> bool:
    stem = f"{prefix}_"
    suffix = value[len(stem) :] if value.startswith(stem) else ""
    return len(suffix) == 12 and all(
        character in "0123456789abcdef" for character in suffix
    )


def _event_reference(event: HistoryEvent) -> dict[str, object]:
    return {
        "session_id": event.session_id,
        "sequence": event.sequence,
        "hash": event.hash,
        "event_type": event.event_type,
    }


class CompletionEvidenceSourceProvider(Protocol):
    source_kind: str

    def inventory(
        self,
        *,
        config: AgentConfig,
        state: SessionState,
        source_events: list[HistoryEvent],
        referenced_values: set[str],
    ) -> list[dict[str, object]]: ...

    def reexpand(
        self,
        *,
        config: AgentConfig,
        state: SessionState,
        source: dict[str, object],
    ) -> dict[str, object]: ...


def is_core_policy_audit_event(event: HistoryEvent) -> bool:
    instruction = event.payload.get("instruction")
    return (
        event.event_type == "prompt_instruction_added"
        and isinstance(instruction, dict)
        and instruction.get("source_kind") == "swaag_core"
        and instruction.get("authority") == "project_policy"
    )


def is_runtime_audit_event(event: HistoryEvent) -> bool:
    return is_core_policy_audit_event(event) or event.event_type in {
        "prompt_built", "model_request_sent", "model_response_received",
    }


class RuntimeAuditEvidenceProvider:
    """Recover harness policy/model envelopes separately from task observations."""

    source_kind = "runtime_audit"

    def inventory(
        self, *, config: AgentConfig, state: SessionState,
        source_events: list[HistoryEvent], referenced_values: set[str],
    ) -> list[dict[str, object]]:
        del config
        rows = []
        for event in source_events:
            source_id = f"{event.session_id}:{event.sequence}"
            if (event.session_id != state.session_id or not is_runtime_audit_event(event)
                    or source_id not in referenced_values):
                continue
            text = stable_json_dumps(asdict(event), indent=None)
            instruction = event.payload.get("instruction", {})
            rows.append({
                "source_kind": self.source_kind,
                "source_id": f"{event.session_id}:{event.sequence}",
                "event_type": event.event_type,
                "operation_kind": event.payload.get("kind"),
                "instruction_id": instruction.get("instruction_id"),
                "instruction_title": instruction.get("title"),
                "instruction_scopes": instruction.get("scopes"),
                "size_chars": len(text), "sha256": sha256_text(text),
                "source_event_references": [_event_reference(event)],
            })
        return rows

    def reexpand(
        self, *, config: AgentConfig, state: SessionState, source: dict[str, object],
    ) -> dict[str, object]:
        references = source.get("source_event_references", [])
        if len(references) != 1:
            raise HistoryInvariantError("Runtime audit requires one exact event reference")
        reference = references[0]
        if reference.get("session_id") != state.session_id:
            raise HistoryInvariantError("Runtime audit cannot cross the operation session")
        sequence = int(reference["sequence"])
        if source.get("source_id") != f"{state.session_id}:{sequence}":
            raise HistoryInvariantError("Runtime audit source identity mismatch")
        events = HistoryStore(config.sessions.root, write_projections=False).read_history_window(
            state.session_id, start_sequence=sequence, limit=1)
        if len(events) != 1:
            raise HistoryInvariantError("Runtime audit event is unavailable")
        event = events[0]
        verify_event_integrity(event, event.prev_hash)
        text = stable_json_dumps(asdict(event), indent=None)
        if (not is_runtime_audit_event(event) or event.hash != reference.get("hash")
                or sha256_text(text) != source.get("sha256")
                or len(text) != source.get("size_chars")):
            raise HistoryInvariantError("Runtime audit differs from its exact source")
        return {**source, "integrity_verified": True, "text": text}


class TextArtifactEvidenceProvider:
    source_kind = "text_artifact"

    def inventory(
        self,
        *,
        config: AgentConfig,
        state: SessionState,
        source_events: list[HistoryEvent],
        referenced_values: set[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        artifact_ids = sorted(
            value for value in referenced_values if _generated_id(value, "artifact")
        )
        for artifact_id in artifact_ids:
            references = [
                event
                for event in source_events
                if event.event_type == "artifact_created"
                and event.payload.get("artifact_id") == artifact_id
            ]
            if not references:
                continue
            source_event = references[-1]
            rows.append(
                {
                    "source_kind": self.source_kind,
                    "source_id": artifact_id,
                    "content_kind": str(source_event.payload.get("kind", "")),
                    "size_chars": int(source_event.payload.get("size_chars", 0)),
                    "sha256": str(source_event.payload.get("sha256", "")),
                    "source_event_references": [
                        _event_reference(event) for event in references
                    ],
                }
            )
        return rows

    def reexpand(
        self,
        *,
        config: AgentConfig,
        state: SessionState,
        source: dict[str, object],
    ) -> dict[str, object]:
        source_id = str(source["source_id"])
        row = dict(source)
        row["integrity_verified"] = True
        artifact = TextArtifactStore(config.sessions.root, state.session_id).get(source_id)
        if (
            artifact.sha256 != str(source.get("sha256", ""))
            or artifact.size_chars != int(source.get("size_chars", -1))
        ):
            raise HistoryInvariantError(
                "Completion evidence artifact metadata differs from its source event"
            )
        row["text"] = Path(artifact.path).read_text(encoding="utf-8")
        return row


class RawAttachmentEvidenceProvider:
    source_kind = "raw_attachment"

    def inventory(
        self,
        *,
        config: AgentConfig,
        state: SessionState,
        source_events: list[HistoryEvent],
        referenced_values: set[str],
    ) -> list[dict[str, object]]:
        del source_events
        referenced_attachment_ids = {
            value
            for value in referenced_values
            if _generated_id(value, "attachment")
        }
        rows: list[dict[str, object]] = []
        for reference in state.attachments:
            if reference.attachment_id not in referenced_attachment_ids:
                continue
            metadata = reference.metadata
            source_references: list[dict[str, object]] = []
            sequence = metadata.get("source_event_sequence")
            source_hash = metadata.get("source_event_hash")
            if isinstance(sequence, int) and isinstance(source_hash, str):
                source_references.append(
                    {
                        "session_id": str(
                            metadata.get("source_event_session_id", state.session_id)
                        ),
                        "sequence": sequence,
                        "hash": source_hash,
                        "event_type": str(
                            metadata.get("source_event_type", "attachment_added")
                        ),
                    }
                )
            rows.append(
                {
                    "source_kind": self.source_kind,
                    "source_id": reference.attachment_id,
                    "original_name": reference.original_name,
                    "media_type": reference.media_type,
                    "size_bytes": reference.size_bytes,
                    "sha256": reference.sha256,
                    "source_event_references": source_references,
                }
            )
        return rows

    def reexpand(
        self,
        *,
        config: AgentConfig,
        state: SessionState,
        source: dict[str, object],
    ) -> dict[str, object]:
        source_id = str(source["source_id"])
        row = dict(source)
        row["integrity_verified"] = True
        reference = find_attachment(state.attachments, source_id)
        data = AttachmentStore(
            config.sessions.root,
            max_upload_bytes=config.attachments.max_upload_bytes,
        ).read_bytes(reference)
        try:
            row["text"] = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            row["requires_specialist_analysis"] = True
            row["specialist_reason"] = "non_utf8_attachment"
            row["read_error"] = (
                "The exact raw bytes are not UTF-8 text; a selected specialist "
                f"reader is required ({exc})."
            )
            row["text"] = ""
        else:
            row["requires_specialist_analysis"] = False
        return row


def default_completion_evidence_source_providers(
) -> tuple[CompletionEvidenceSourceProvider, ...]:
    return (TextArtifactEvidenceProvider(), RawAttachmentEvidenceProvider(), RuntimeAuditEvidenceProvider())
