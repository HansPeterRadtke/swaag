from __future__ import annotations

from collections import Counter
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


RUNTIME_AUDIT_FIELDS = {
    "runtime_context_projected": ("projection_budget_report",),
    "runtime_context_projection_skipped": ("budget_report",),
    "turn_finished": ("budget_reports",),
    "completion_evaluation_unavailable": ("budget_report",),
    "completion_evaluated": (
        "evidence_source_references", "historical_source_event_references",
        "historical_projection_budget_report",
    ),
}


def runtime_audit_fields(event: HistoryEvent) -> tuple[str, ...]:
    return tuple(field for field in RUNTIME_AUDIT_FIELDS.get(event.event_type, ())
                 if field in event.payload)


def is_runtime_audit_event(event: HistoryEvent) -> bool:
    return (is_core_policy_audit_event(event) or event.event_type in {
        "prompt_built", "model_request_sent", "model_response_received",
    } or bool(runtime_audit_fields(event)))


# Their semantic payload fields remain inline; only nested accounting is replaced.
RUNTIME_AUDIT_SEMANTIC_EVENT_TYPES = frozenset(RUNTIME_AUDIT_FIELDS)


RUNTIME_TELEMETRY_EVENT_TYPES = frozenset({
    "context_compiled", "budget_checked", "model_token_progress",
    "inference_request_queued", "inference_request_started", "inference_request_finished",
    "prompt_instructions_selected", "action_prompt_mode_fallback",
    "prompt_built", "model_request_sent", "model_response_received",
    "semantic_reduction_working_set_exceeded", "model_output_budget_exhausted",
    "budget_repaired",
    "model_tokenize_requested", "model_tokenize_result", "token_estimate_used",
    "inference_request_requeued", "model_retry_scheduled", "model_backend_activity",
    "agent_action_rejected", "model_call_failed", "model_call_preempted",
}) | RUNTIME_AUDIT_SEMANTIC_EVENT_TYPES


def is_runtime_telemetry_event(event: HistoryEvent) -> bool:
    return event.event_type in RUNTIME_TELEMETRY_EVENT_TYPES


# Historical observations are available on demand to the foreground checker.
# This is schema selection, never a relevance guess from user wording. Current
# observations and all authoritative user text remain in the initial context.
FOREGROUND_OBSERVATION_TYPES = frozenset({
    "session_created", "environment_initialized", "tool_capabilities_loaded",
    "runtime_system_snapshot", "worker_question_inventory", "agent_action_selected", "agent_status",
    "tool_called", "tool_result", "tool_error", "tool_execution_context",
    "agent_tool_call_completed", "agent_action_terminal", "control_message_processed",
})


def is_foreground_observation_event(event: HistoryEvent) -> bool:
    if event.event_type in FOREGROUND_OBSERVATION_TYPES:
        return True
    message = event.payload.get("message")
    return (event.event_type == "message_added" and isinstance(message, dict)
            and (message.get("role") == "tool" or (
                message.get("role") == "assistant"
                and isinstance(message.get("metadata"), dict)
                and message["metadata"].get("internal_action") is True)))


def is_foreground_detail_event(event: HistoryEvent) -> bool:
    # Visible wording stays inline; its exact provenance can be fetched.
    return is_foreground_observation_event(event) or event.event_type in {
        "message_added", "turn_started", "turn_finished"}


def foreground_history_reference(events: list[HistoryEvent]) -> dict[str, object]:
    selected = sorted(events, key=lambda event: event.sequence)
    if (not selected or len({e.session_id for e in selected}) != 1
            or len({e.sequence for e in selected}) != len(selected)
            or any(not is_foreground_detail_event(e) for e in selected)):
        raise HistoryInvariantError("Foreground history requires exact observations from one session")
    first, last = selected[0], selected[-1]
    text = stable_json_dumps([asdict(e) for e in selected], indent=None)
    return {
        "source_kind": "foreground_history",
        "source_id": f"{first.session_id}:{first.sequence}-{last.sequence}",
        "session_id": first.session_id,
        "event_sequences": [e.sequence for e in selected],
        "event_types": dict(sorted(Counter(e.event_type for e in selected).items())),
        "first_timestamp": first.timestamp, "last_timestamp": last.timestamp,
        "tool_names": sorted({str(e.payload["tool_name"]) for e in selected
                              if isinstance(e.payload.get("tool_name"), str)}),
        "size_chars": len(text), "sha256": sha256_text(text),
        "source_event_references": [_event_reference(first), _event_reference(last)],
    }


class ForegroundHistoryEvidenceProvider:
    source_kind = "foreground_history"

    def inventory(self, *, config, state, source_events, referenced_values):
        del config
        groups, selected, conversation_details = [], [], []
        for event in source_events:
            if event.event_type == "turn_started" and selected:
                groups.append(foreground_history_reference(selected))
                selected = []
            if (event.session_id == state.session_id
                    and f"foreground_history:{event.session_id}:{event.sequence}" in referenced_values
                    and is_foreground_detail_event(event)):
                if is_foreground_observation_event(event):
                    selected.append(event)
                else:
                    conversation_details.append(event)
        if selected:
            groups.append(foreground_history_reference(selected))
        if conversation_details:
            groups.append(foreground_history_reference(conversation_details))
        return groups

    def reexpand(self, *, config, state, source):
        if source.get("session_id") != state.session_id:
            raise HistoryInvariantError("Foreground history cannot cross the operation session")
        sequences = source.get("event_sequences")
        if (not isinstance(sequences, list) or not sequences
                or any(type(n) is not int or n < 1 for n in sequences)
                or sequences != sorted(set(sequences))):
            raise HistoryInvariantError("Foreground history has invalid event sequences")
        events = HistoryStore(config.sessions.root, write_projections=False).read_history_window(
            state.session_id, start_sequence=sequences[0], limit=sequences[-1]-sequences[0]+1)
        wanted = set(sequences)
        selected = [event for event in events if event.sequence in wanted]
        if [event.sequence for event in selected] != sequences:
            raise HistoryInvariantError("Foreground history source is unavailable")
        for event in selected:
            verify_event_integrity(event, event.prev_hash)
        observed = foreground_history_reference(selected)
        if any(source.get(key) != value for key, value in observed.items()):
            raise HistoryInvariantError("Foreground history differs from its exact source")
        return {**source, "integrity_verified": True,
                "text": stable_json_dumps([asdict(e) for e in selected], indent=None)}


def runtime_telemetry_reference(events: list[HistoryEvent], *, legacy_content_digest: bool = False) -> dict[str, object]:
    """One exact audit range; semantic outcome fields also remain in their own views."""
    selected = sorted((e for e in events if is_runtime_telemetry_event(e)),
                      key=lambda e: e.sequence)
    if not selected or len({e.session_id for e in selected}) != 1:
        raise HistoryInvariantError("Runtime telemetry requires one nonempty session range")
    first, last = selected[0], selected[-1]
    if legacy_content_digest:
        text = stable_json_dumps([asdict(e) for e in selected], indent=None)
        digest = {"size_chars": len(text), "sha256": sha256_text(text)}
    else:
        # Event hashes already bind their complete immutable payloads. Hash the
        # exact ordered manifest, then verify each event on expansion; ordinary
        # replies need not copy/serialize every old model request just to index it.
        manifest = stable_json_dumps([(e.sequence, e.hash) for e in selected], indent=None)
        digest = {"integrity_scheme": "event_manifest_v1", "sha256": sha256_text(manifest)}
    return {
        "source_kind": "runtime_telemetry",
        "source_id": f"{first.session_id}:{first.sequence}-{last.sequence}",
        "session_id": first.session_id,
        "first_sequence": first.sequence, "last_sequence": last.sequence,
        "event_count": len(selected),
        "event_types": dict(sorted(Counter(e.event_type for e in selected).items())),
        **digest,
        "source_event_references": [_event_reference(first), _event_reference(last)],
    }


class RuntimeTelemetryEvidenceProvider:
    source_kind = "runtime_telemetry"

    def inventory(self, *, config: AgentConfig, state: SessionState,
                  source_events: list[HistoryEvent], referenced_values: set[str]
                  ) -> list[dict[str, object]]:
        del config
        selected = [e for e in source_events if e.session_id == state.session_id
                    and is_runtime_telemetry_event(e)
                    and f"{e.session_id}:{e.sequence}" in referenced_values]
        return [runtime_telemetry_reference(selected)] if selected else []

    def reexpand(self, *, config: AgentConfig, state: SessionState,
                 source: dict[str, object]) -> dict[str, object]:
        if source.get("session_id") != state.session_id:
            raise HistoryInvariantError("Runtime telemetry cannot cross the operation session")
        first, last = source.get("first_sequence"), source.get("last_sequence")
        if (type(first) is not int or type(last) is not int or first < 1 or last < first):
            raise HistoryInvariantError("Runtime telemetry has an invalid sequence range")
        events = HistoryStore(config.sessions.root, write_projections=False).read_history_window(
            state.session_id, start_sequence=first, limit=last-first+1)
        # References freeze the event-type selection as well as the range/hash.
        # Older references must not acquire newly supported audit event types.
        event_types = source.get("event_types")
        if (not isinstance(event_types, dict) or not event_types
                or any(kind not in RUNTIME_TELEMETRY_EVENT_TYPES
                       or type(count) is not int or count < 1
                       for kind, count in event_types.items())):
            raise HistoryInvariantError("Runtime telemetry has invalid event types")
        selected = [e for e in events if e.event_type in event_types]
        for event in selected:
            verify_event_integrity(event, event.prev_hash)
        scheme = source.get("integrity_scheme")
        if scheme not in {None, "event_manifest_v1"}:
            raise HistoryInvariantError("Unknown runtime telemetry integrity scheme")
        observed = runtime_telemetry_reference(selected, legacy_content_digest=scheme is None)
        if any(source.get(key) != value for key, value in observed.items()):
            raise HistoryInvariantError("Runtime telemetry differs from its exact source")
        return {**source, "integrity_verified": True,
                "text": stable_json_dumps([asdict(e) for e in selected], indent=None)}


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
                    or is_runtime_telemetry_event(event)
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
    return (TextArtifactEvidenceProvider(), RawAttachmentEvidenceProvider(),
            RuntimeAuditEvidenceProvider(), RuntimeTelemetryEvidenceProvider(),
            ForegroundHistoryEvidenceProvider())
