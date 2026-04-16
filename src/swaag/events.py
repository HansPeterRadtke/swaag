from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swaag.types import HistoryEvent
from swaag.utils import new_id, sha256_text, stable_json_dumps, to_jsonable, utc_now_iso

EVENT_SCHEMA_VERSION = 1

ALLOWED_EVENT_TYPES = frozenset(
    {
        "session_created",
        "session_renamed",
        "message_added",
        "turn_started",
        "turn_finished",
        "notes_selected",
        "prompt_built",
        "budget_checked",
        "budget_rejected",
        "summary_created",
        "history_compacted",
        "history_compressed",
        "plan_created",
        "plan_updated",
        "plan_repaired",
        "plan_completed",
        "prompt_analyzed",
        "decision_made",
        "decision_adjusted",
        "task_expanded",
        "role_switched",
        "review_started",
        "review_completed",
        "review_skipped",
        "subagent_spawned",
        "subagent_reported",
        "subagent_selection_resolved",
        "strategy_selected",
        "strategy_selection_resolved",
        "action_selected",
        "action_selection_resolved",
        "failure_classification_resolved",
        "working_memory_updated",
        "memory_extracted",
        "memory_stored",
        "memory_retrieved",
        "memory_flagged",
        "memory_rejected",
        "project_state_updated",
        "context_built",
        "reasoning_started",
        "step_started",
        "step_executed",
        "step_completed",
        "step_failed",
        "reasoning_completed",
        "subsystem_started",
        "subsystem_progress",
        "subsystem_completed",
        "tool_chain_started",
        "tool_chain_step",
        "tool_chain_completed",
        "tool_graph_planned",
        "tool_graph_rejected",
        "evaluation_performed",
        "evaluation_failed",
        "verification_started",
        "verification_completed",
        "verification_type_used",
        "verification_passed",
        "verification_failed",
        "retry_triggered",
        "replan_triggered",
        "drift_detected",
        "recovery_triggered",
        "consistency_checked",
        "consistency_failed",
        "tool_execution_context",
        "model_request_sent",
        "model_request_progress",
        "model_response_received",
        "model_call_failed",
        "model_retry_scheduled",
        "model_tokenize_requested",
        "model_tokenize_result",
        "model_tokenize_failed",
        "token_estimate_used",
        "decision_parsed",
        "answer_derived",
        "output_decomposition_planned",
        "output_unit_generated",
        "output_overflow_recovery_planned",
        "tool_called",
        "tool_result",
        "tool_error",
        "duplicate_action_detected",
        "environment_initialized",
        "filesystem_listed",
        "filesystem_read",
        "filesystem_search",
        "repository_searched",
        "workspace_snapshot",
        "workspace_snapshot_inspected",
        "changes_listed",
        "diff_inspected",
        "shell_command_started",
        "shell_command_completed",
        "process_started",
        "process_polled",
        "process_completed",
        "process_timed_out",
        "process_killed",
        "wait_entered",
        "wait_resumed",
        "control_message_processed",
        "control_action_applied",
        "deferred_task_queued",
        "deferred_task_consumed",
        "code_checkpoint_created",
        "code_checkpoint_restored",
        "note_added",
        "note_replaced",
        "notes_compacted",
        "reader_opened",
        "reader_chunk_read",
        "file_read_requested",
        "buffer_read_requested",
        "file_chunk_read",
        "buffer_chunk_read",
        "file_read_for_edit",
        "edit_previewed",
        "edit_applied",
        "file_write_applied",
        "file_write_failed",
        "state_rebuilt",
        "doctor_health_checked",
        "doctor_tokenize_checked",
        "error",
        "retry",
        "emergency_fallback_used",
        "fatal_system_error",
    }
)

REQUIRED_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "session_created": frozenset({"session_id", "config_fingerprint", "model_base_url", "created_at"}),
    "session_renamed": frozenset({"session_id", "old_name", "new_name", "reason"}),
    "message_added": frozenset({"message"}),
    "turn_started": frozenset({"turn_index", "user_text"}),
    "turn_finished": frozenset({"turn_index", "assistant_text", "tool_steps", "budget_reports"}),
    "notes_selected": frozenset({"included_note_ids", "omitted_note_ids", "tokens", "exact"}),
    "prompt_built": frozenset({"kind", "prompt_mode", "contract", "prompt", "components", "budget_report"}),
    "budget_checked": frozenset({"kind", "prompt_mode", "budget_report", "cap_error"}),
    "budget_rejected": frozenset({"kind", "prompt_mode", "reason", "budget_report"}),
    "summary_created": frozenset({"source_message_count", "summary_message", "summary_budget_report"}),
    "history_compacted": frozenset({"source_message_count", "summary_message", "summary_budget_report"}),
    "history_compressed": frozenset({"source_message_count", "summary_message", "summary_budget_report"}),
    "plan_created": frozenset({"goal", "plan"}),
    "plan_updated": frozenset({"plan", "reason"}),
    "plan_repaired": frozenset({"reason", "required_tools", "repair", "original_plan_id", "update_existing"}),
    "plan_completed": frozenset({"plan_id", "status"}),
    "prompt_analyzed": frozenset({"analysis"}),
    "decision_made": frozenset({"decision"}),
    "decision_adjusted": frozenset({"reason", "tool_name", "decision"}),
    "answer_derived": frozenset({"answer", "source"}),
    "task_expanded": frozenset({"expanded_task"}),
    "role_switched": frozenset({"previous_role", "new_role", "reason"}),
    "review_started": frozenset({"review_kind", "target_id", "role"}),
    "review_completed": frozenset({"review_kind", "target_id", "role", "passed", "reason", "evidence"}),
    "review_skipped": frozenset({"review_kind", "target_id", "reason"}),
    "subagent_spawned": frozenset({"subagent_type", "purpose", "allowed_tools", "token_budget", "target_id"}),
    "subagent_reported": frozenset({"subagent_type", "accepted", "reason", "recommended_action", "artifacts"}),
    "subagent_selection_resolved": frozenset({"purpose", "candidate_types", "selection"}),
    "strategy_selected": frozenset({"strategy"}),
    "strategy_selection_resolved": frozenset({"strategy", "source"}),
    "action_selected": frozenset({"selected_action", "ready_step_ids", "scores", "strategy", "stop_reason"}),
    "action_selection_resolved": frozenset({"selected_action", "candidates", "source"}),
    "failure_classification_resolved": frozenset({"classification", "source"}),
    "working_memory_updated": frozenset({"working_memory", "reason"}),
    "memory_extracted": frozenset({"memory", "source_event_id"}),
    "memory_stored": frozenset({"memory"}),
    "memory_retrieved": frozenset({"query", "memory_ids", "count"}),
    "memory_flagged": frozenset({"source_event_id", "reason", "trust_level", "content_preview"}),
    "memory_rejected": frozenset({"source_event_id", "reason", "trust_level", "content_preview"}),
    "project_state_updated": frozenset({"project_state", "reason"}),
    "context_built": frozenset({"goal", "kind", "prompt_mode", "history_message_count", "note_ids", "semantic_memory_ids", "plan_id", "selection_trace"}),
    "reasoning_started": frozenset({"goal", "max_steps"}),
    "step_started": frozenset({"plan_id", "step_id", "step_title"}),
    "step_executed": frozenset({"plan_id", "step_id", "step_title", "outcome"}),
    "step_completed": frozenset({"plan_id", "step_id", "step_title", "outcome"}),
    "step_failed": frozenset({"plan_id", "step_id", "step_title", "error", "error_type"}),
    "reasoning_completed": frozenset({"goal", "status", "completed_steps", "failed_steps", "reason"}),
    "subsystem_started": frozenset({"subsystem", "step_id", "goal"}),
    "subsystem_progress": frozenset({"subsystem", "step_id", "progress"}),
    "subsystem_completed": frozenset({"subsystem", "step_id", "success", "result_summary"}),
    "tool_chain_started": frozenset({"step_id", "expected_tool", "max_attempts"}),
    "tool_chain_step": frozenset({"step_id", "attempt", "decision"}),
    "tool_chain_completed": frozenset({"step_id", "attempts", "success"}),
    "tool_graph_planned": frozenset({"step_id", "selected_tool", "expected_tool", "chain", "valid", "reason"}),
    "tool_graph_rejected": frozenset({"step_id", "selected_tool", "expected_tool", "chain", "reason"}),
    "evaluation_performed": frozenset({"step_id", "passed", "confidence", "reason"}),
    "evaluation_failed": frozenset({"step_id", "attempt", "reason"}),
    "verification_started": frozenset({"step_id", "verification_type", "required_conditions", "optional_conditions"}),
    "verification_completed": frozenset(
        {
            "step_id",
            "verification_type_used",
            "conditions_met",
            "conditions_failed",
            "evidence",
            "verification_passed",
            "confidence",
            "reason",
        }
    ),
    "verification_type_used": frozenset({"step_id", "verification_type_used"}),
    "verification_passed": frozenset(
        {"step_id", "verification_type_used", "conditions_met", "conditions_failed", "evidence", "confidence", "reason"}
    ),
    "verification_failed": frozenset(
        {"step_id", "verification_type_used", "conditions_met", "conditions_failed", "evidence", "confidence", "reason", "failure_kind"}
    ),
    "retry_triggered": frozenset({"step_id", "reason", "attempt", "failure_kind"}),
    "replan_triggered": frozenset({"step_id", "reason", "replan_count"}),
    "drift_detected": frozenset({"reason", "event_count"}),
    "recovery_triggered": frozenset({"reason", "event_count"}),
    "consistency_checked": frozenset({"working_memory_ok", "semantic_memory_ok", "recovered"}),
    "consistency_failed": frozenset({"component", "reason"}),
    "tool_execution_context": frozenset({"tool_name", "tool_kind", "isolated", "policy"}),
    "model_request_sent": frozenset({"kind", "prompt_mode", "attempt", "request", "budget_report"}),
    "model_request_progress": frozenset({"kind", "prompt_mode", "attempt", "elapsed_seconds", "timeout_seconds", "policy"}),
    "model_response_received": frozenset({"kind", "prompt_mode", "attempt", "completion"}),
    "model_call_failed": frozenset({"kind", "prompt_mode", "attempt", "error", "error_type"}),
    "model_retry_scheduled": frozenset({"kind", "prompt_mode", "next_attempt"}),
    "model_tokenize_requested": frozenset({"text", "text_hash"}),
    "model_tokenize_result": frozenset({"text_hash", "tokens", "exact"}),
    "model_tokenize_failed": frozenset({"text_hash", "error", "error_type"}),
    "token_estimate_used": frozenset({"text_hash", "tokens", "strategy"}),
    "decision_parsed": frozenset({"decision", "prompt_mode"}),
    "tool_called": frozenset({"tool_name", "tool_input"}),
    "output_decomposition_planned": frozenset({"output_class", "reason", "units"}),
    "output_unit_generated": frozenset({"unit", "overflowed", "text"}),
    "output_overflow_recovery_planned": frozenset({"unit", "keep_partial", "reason", "next_units"}),
    "tool_result": frozenset({"tool_name", "raw_input", "validated_input", "output"}),
    "tool_error": frozenset({"tool_name", "tool_input", "error", "error_type"}),
    "duplicate_action_detected": frozenset({"action_key", "count"}),
    "environment_initialized": frozenset({"workspace_root", "cwd", "shell_env_overrides", "shell_unset_vars"}),
    "filesystem_listed": frozenset({"path", "cwd", "entries", "count"}),
    "filesystem_read": frozenset({"path", "relative_path", "text", "size_chars", "cwd"}),
    "filesystem_search": frozenset({"path", "relative_path", "pattern", "regex", "ignore_case", "matches", "match_count", "cwd"}),
    "repository_searched": frozenset({"path", "pattern", "regex", "ignore_case", "matches", "match_count", "matched_files", "cwd"}),
    "workspace_snapshot": frozenset({"workspace_root", "cwd", "files", "created_files", "modified_files", "deleted_files", "captured_at"}),
    "workspace_snapshot_inspected": frozenset({"workspace_root", "cwd", "files", "file_count", "created_files", "modified_files", "deleted_files", "captured_at"}),
    "changes_listed": frozenset({"cwd", "created_files", "modified_files", "deleted_files"}),
    "diff_inspected": frozenset({"path", "relative_path", "changed", "diff", "baseline_source"}),
    "shell_command_started": frozenset({"command", "cwd"}),
    "shell_command_completed": frozenset({"command", "cwd_before", "cwd_after", "env_overrides", "unset_vars", "exit_code", "stdout", "stderr"}),
    "process_started": frozenset({"process_id", "command", "cwd", "status", "started_at"}),
    "process_polled": frozenset({"process_id", "command", "cwd", "status", "started_at", "completed"}),
    "process_completed": frozenset({"process_id", "command", "cwd", "status", "stdout", "stderr", "return_code", "started_at", "ended_at"}),
    "process_timed_out": frozenset({"process_id", "command", "cwd", "status", "stdout", "stderr", "return_code", "started_at", "ended_at"}),
    "process_killed": frozenset({"process_id", "command", "cwd", "status", "started_at", "ended_at"}),
    "wait_entered": frozenset({"reason", "process_ids"}),
    "wait_resumed": frozenset({"reason", "process_ids"}),
    "control_message_processed": frozenset({"control_id", "session_id", "message", "decision"}),
    "control_action_applied": frozenset({"control_id", "session_id", "action", "effect"}),
    "deferred_task_queued": frozenset({"task"}),
    "deferred_task_consumed": frozenset({"task_id", "reason"}),
    "code_checkpoint_created": frozenset({"checkpoint"}),
    "code_checkpoint_restored": frozenset({"checkpoint_id", "restored_to", "workspace_root"}),
    "note_added": frozenset({"note"}),
    "note_replaced": frozenset({"note"}),
    "notes_compacted": frozenset({"removed_note_ids", "compacted_note"}),
    "reader_opened": frozenset({"reader_state"}),
    "reader_chunk_read": frozenset({"reader_state", "chunk"}),
    "file_read_requested": frozenset({"path", "reason"}),
    "buffer_read_requested": frozenset({"source_ref", "reason"}),
    "file_chunk_read": frozenset({"reader_id", "source_ref", "start_offset", "end_offset", "next_offset", "finished", "text"}),
    "buffer_chunk_read": frozenset({"reader_id", "source_ref", "start_offset", "end_offset", "next_offset", "finished", "text"}),
    "file_read_for_edit": frozenset({"path", "size_chars", "text"}),
    "edit_previewed": frozenset({"path", "operation", "details", "changed", "diff", "new_text", "original_text"}),
    "edit_applied": frozenset({"path", "operation", "details", "changed", "diff", "new_text", "original_text"}),
    "file_write_applied": frozenset({"path", "cause_event", "backup_path", "size_chars"}),
    "file_write_failed": frozenset({"path", "cause_event", "error", "error_type"}),
    "state_rebuilt": frozenset({"session_id", "event_count"}),
    "doctor_health_checked": frozenset({"health"}),
    "doctor_tokenize_checked": frozenset({"probe", "tokens"}),
    "error": frozenset({"operation", "error", "error_type"}),
    "retry": frozenset({"operation", "reason", "attempt", "next_attempt"}),
    "emergency_fallback_used": frozenset({"operation", "error", "error_type", "warning"}),
    "fatal_system_error": frozenset({"operation", "error", "error_type", "category", "warning"}),
}


class EventSchemaError(ValueError):
    pass


def validate_event_payload(event_type: str, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    if event_type not in ALLOWED_EVENT_TYPES:
        raise EventSchemaError(f"Unknown event type: {event_type}")
    if not isinstance(payload, dict):
        raise EventSchemaError(f"Event payload for {event_type} must be a dict")
    if not isinstance(metadata, dict):
        raise EventSchemaError(f"Event metadata for {event_type} must be a dict")
    missing = REQUIRED_PAYLOAD_KEYS[event_type] - set(payload)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise EventSchemaError(f"Event payload for {event_type} is missing keys: {missing_text}")


def canonical_event_body(event: HistoryEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, HistoryEvent):
        payload = asdict(event)
    else:
        payload = dict(event)
    payload.pop("hash", None)
    return to_jsonable(payload)


def compute_event_hash(event: HistoryEvent | dict[str, Any]) -> str:
    return sha256_text(stable_json_dumps(canonical_event_body(event)))


def create_event(
    *,
    session_id: str,
    sequence: int,
    event_type: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    prev_hash: str | None,
    timestamp: str | None = None,
) -> HistoryEvent:
    metadata = {} if metadata is None else metadata
    payload = to_jsonable(payload)
    metadata = to_jsonable(metadata)
    validate_event_payload(event_type, payload, metadata)
    event = HistoryEvent(
        id=new_id("event"),
        sequence=sequence,
        session_id=session_id,
        timestamp=timestamp or utc_now_iso(),
        type=event_type,
        version=EVENT_SCHEMA_VERSION,
        payload=payload,
        metadata=metadata,
        prev_hash=prev_hash,
        hash="",
    )
    event.hash = compute_event_hash(event)
    return event


def verify_event_integrity(event: HistoryEvent, expected_prev_hash: str | None) -> None:
    if event.version != EVENT_SCHEMA_VERSION:
        raise EventSchemaError(f"Unsupported event version: {event.version}")
    validate_event_payload(event.event_type, event.payload, event.metadata)
    if event.prev_hash != expected_prev_hash:
        raise EventSchemaError(
            f"Hash chain mismatch at sequence {event.sequence}: expected prev_hash={expected_prev_hash!r}, got {event.prev_hash!r}"
        )
    expected_hash = compute_event_hash(event)
    if event.hash != expected_hash:
        raise EventSchemaError(f"Event hash mismatch at sequence {event.sequence}")
