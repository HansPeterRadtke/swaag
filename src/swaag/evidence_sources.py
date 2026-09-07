from __future__ import annotations

from pathlib import Path
from typing import Protocol

from swaag.attachments import AttachmentStore, find_attachment
from swaag.config import AgentConfig
from swaag.environment.artifacts import TextArtifactStore
from swaag.history import HistoryInvariantError
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
            row["read_error"] = (
                "The exact raw bytes are not UTF-8 text; a selected specialist "
                f"reader is required ({exc})."
            )
            row["text"] = ""
        return row


def default_completion_evidence_source_providers(
) -> tuple[CompletionEvidenceSourceProvider, ...]:
    return (TextArtifactEvidenceProvider(), RawAttachmentEvidenceProvider())
