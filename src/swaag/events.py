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
        "context_compiled",
        "tool_result_projected",
        "tool_result_projection_reused",
        "tool_result_projection_skipped",
        "completion_evaluated",
        "completion_rejected",
        "completion_evaluation_unavailable",
        "caller_structured_output_created",
        "communication_status_requested",
        "communication_status_generated",
        "communication_status_rejected",
        "communication_status_unavailable",
        "budget_checked",
        "budget_rejected",
        "budget_repaired",
        "summary_created",
        "history_compacted",
        "history_compressed",
        "history_retrieved",
        "history_window_read",
        "history_analyzed",
        "artifact_created",
        "artifact_read",
        "external_source_observed",
        "attachment_added",
        "attachment_extracted",
        "terminal_create",
        "terminal_send",
        "terminal_read",
        "terminal_list",
        "terminal_close",
        "plan_created",
        "plan_updated",
        "plan_completed",
        "prompt_analyzed",
        "decision_made",
        "decision_adjusted",
        "clarification_generated",
        "task_expanded",
        "role_switched",
        "review_started",
        "review_completed",
        "review_skipped",
        "strategy_selected",
        "strategy_selection_resolved",
        "action_selected",
        "action_selection_resolved",
        "agent_action_rejected",
        "agent_action_selected",
        "agent_status",
        "agent_question",
        "agent_action_terminal",
        "agent_tool_call_completed",
        "tool_capabilities_loaded",
        "assistant_progress",
        "tool_result_missing",
        "failure_classification_resolved",
        "working_memory_updated",
        "memory_extracted",
        "memory_stored",
        "memory_retrieved",
        "memory_flagged",
        "memory_rejected",
        "project_state_updated",
        "context_built",
        "semantic_retrieval_degraded",
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
        "tool_mismatch_rejected",
        "evaluation_performed",
        "evaluation_failed",
        "verification_started",
        "verification_completed",
        "verification_type_used",
        "verification_passed",
        "verification_failed",
        "unresolved_objective_verification_deferred",
        "retry_triggered",
        "retry_suppressed",
        "replan_triggered",
        "drift_detected",
        "recovery_triggered",
        "consistency_checked",
        "consistency_failed",
        "tool_execution_context",
        "model_request_sent",
        "inference_request_queued",
        "inference_request_started",
        "inference_request_requeued",
        "inference_request_finished",
        "model_request_progress",
        "model_token_progress",
        "model_response_received",
        "model_output_budget_exhausted",
        "model_call_failed",
        "model_call_preempted",
        "model_call_replayed",
        "model_call_replay_invalidated",
        "model_retry_scheduled",
        "model_tokenize_requested",
        "model_tokenize_result",
        "model_tokenize_failed",
        "token_estimate_used",
        "decision_parsed",
        "tool_input_parsed",
        "answer_derived",
        "output_unit_generated",
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
        "wait_completed",
        "wakeup_scheduled",
        "wakeup_cancelled",
        "wakeup_due",
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
        "fatal_system_error",
    }
)

LEGACY_EVENT_TYPES = frozenset(
    {
        'action_selected',
        'action_selection_resolved',
        'answer_derived',
        'consistency_checked',
        'consistency_failed',
        'context_built',
        'decision_adjusted',
        'decision_made',
        'drift_detected',
        'evaluation_failed',
        'evaluation_performed',
        'failure_classification_resolved',
        'memory_extracted',
        'memory_flagged',
        'memory_rejected',
        'memory_retrieved',
        'memory_stored',
        'output_unit_generated',
        'plan_completed',
        'plan_created',
        'plan_updated',
        'project_state_updated',
        'prompt_analyzed',
        'reasoning_completed',
        'reasoning_started',
        'recovery_triggered',
        'replan_triggered',
        'retry_suppressed',
        'review_completed',
        'review_skipped',
        'review_started',
        'role_switched',
        'semantic_retrieval_degraded',
        'step_completed',
        'step_executed',
        'step_started',
        'strategy_selected',
        'strategy_selection_resolved',
        'subsystem_completed',
        'subsystem_progress',
        'subsystem_started',
        'task_expanded',
        'tool_chain_completed',
        'tool_chain_started',
        'tool_chain_step',
        'tool_input_parsed',
        'tool_mismatch_rejected',
        'tool_result_missing',
        'unresolved_objective_verification_deferred',
        'verification_completed',
        'verification_failed',
        'verification_passed',
        'verification_started',
        'verification_type_used',
        'working_memory_updated',
    }
)
READABLE_EVENT_TYPES = ALLOWED_EVENT_TYPES
CURRENT_EVENT_TYPES = ALLOWED_EVENT_TYPES - LEGACY_EVENT_TYPES
ALLOWED_EVENT_TYPES = CURRENT_EVENT_TYPES


REQUIRED_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "session_created": frozenset({"session_id", "config_fingerprint", "model_base_url", "created_at"}),
    "session_renamed": frozenset({"session_id", "old_name", "new_name", "reason"}),
    "message_added": frozenset({"message"}),
    "turn_started": frozenset({"turn_index", "user_text"}),
    "turn_finished": frozenset({"turn_index", "assistant_text", "tool_steps", "budget_reports"}),
    "notes_selected": frozenset({"included_note_ids", "omitted_note_ids", "tokens", "exact"}),
    "prompt_built": frozenset({"kind", "prompt_mode", "contract", "prompt", "components", "budget_report"}),
    "context_compiled": frozenset({"kind", "prompt_mode", "accounting"}),
    "tool_result_projected": frozenset({"source_event_sequence", "source_event_hash", "tool_name", "target_tokens", "original_tokens", "projected_tokens", "overflow_tokens", "projection"}),
    "tool_result_projection_reused": frozenset(
        {
            "source_event_sequence",
            "source_event_hash",
            "projection_event_sequence",
            "target_tokens",
            "projected_tokens",
        }
    ),
    "tool_result_projection_skipped": frozenset({"source_event_sequence", "source_event_hash", "reason", "target_tokens", "original_tokens", "overflow_tokens", "budget_report"}),
    "completion_evaluated": frozenset({"complete", "reason", "remaining_work"}),
    "completion_rejected": frozenset({"action_index", "reason", "remaining_work"}),
    "completion_evaluation_unavailable": frozenset({"reason", "budget_report"}),
    "caller_structured_output_created": frozenset(
        {"schema", "semantic_output", "evidence_source_references"}
    ),
    "communication_status_requested": frozenset(
        {
            "target_session_id",
            "question",
            "mechanical_status",
            "source_event_references",
        }
    ),
    "communication_status_generated": frozenset(
        {
            "target_session_id",
            "question",
            "status",
            "mechanical_status",
            "source_event_references",
            "evidence_projected",
        }
    ),
    "communication_status_rejected": frozenset(
        {"target_session_id", "attempt", "reason"}
    ),
    "communication_status_unavailable": frozenset(
        {
            "target_session_id",
            "question",
            "error",
            "error_type",
            "source_event_references",
        }
    ),
    "budget_checked": frozenset({"kind", "prompt_mode", "budget_report", "cap_error"}),
    "budget_rejected": frozenset({"kind", "prompt_mode", "reason", "budget_report"}),
    "budget_repaired": frozenset({"kind", "reason", "requested_response_tokens", "capped_response_tokens"}),
    "summary_created": frozenset({"source_message_count", "summary_message", "summary_budget_report"}),
    "history_compacted": frozenset({"source_message_count", "summary_message", "summary_budget_report"}),
    "history_compressed": frozenset({"source_message_count", "summary_message", "summary_budget_report"}),
    "history_retrieved": frozenset({"session_id", "query", "match_count", "sequences"}),
    "history_analyzed": frozenset({"session_id", "query", "source_sequences", "candidate_root_cause_count"}),
    "history_window_read": frozenset({"session_id", "start_sequence", "event_count", "sequences"}),
    "artifact_created": frozenset({"artifact_id", "kind", "size_chars", "sha256"}),
    "artifact_read": frozenset({"artifact_id", "start_offset", "end_offset", "finished"}),
    "external_source_observed": frozenset(
        {
            "source_id",
            "name",
            "url",
            "document",
            "document_truncated",
            "tool_name",
        }
    ),
    "attachment_added": frozenset({"attachment"}),
    "attachment_extracted": frozenset(
        {"attachment_id", "attachment_sha256", "artifact_id", "extractor", "profile", "manifest"}
    ),
    "terminal_create": frozenset({"operation", "terminal_id", "terminal_ref", "name", "active"}),
    "terminal_send": frozenset({"operation", "terminal_id", "terminal_ref", "name", "active"}),
    "terminal_read": frozenset({"operation", "terminal_id", "terminal_ref", "name", "active"}),
    "terminal_list": frozenset({"operation", "terminal_id", "terminal_ref", "name", "active"}),
    "terminal_close": frozenset({"operation", "terminal_id", "terminal_ref", "name", "active"}),
    "plan_created": frozenset({"goal", "plan"}),
    "plan_updated": frozenset({"plan", "reason"}),
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
    "strategy_selected": frozenset({"strategy"}),
    "strategy_selection_resolved": frozenset({"strategy", "source"}),
    "action_selected": frozenset({"selected_action", "ready_step_ids", "scores", "strategy", "stop_reason"}),
    "action_selection_resolved": frozenset({"selected_action", "candidates", "source"}),
    "agent_action_rejected": frozenset({"action_index", "validation_attempt", "reason"}),
    "agent_action_selected": frozenset({"action_index", "action", "occurrence"}),
    "agent_status": frozenset({"action_index", "situation", "action", "reason", "importance", "importance_rank"}),
    "agent_question": frozenset({"action_index", "question", "criticality", "reason", "assumption_if_unanswered"}),
    "agent_action_terminal": frozenset({"action_index", "continue_loop", "silent_completion"}),
    "agent_tool_call_completed": frozenset({"action_index", "tool_call_index", "tool_name", "success"}),
    "tool_capabilities_loaded": frozenset({"action_index", "requested_tool_names", "loaded_tool_names"}),
    "assistant_progress": frozenset({"action_index", "assistant_text"}),
    "tool_result_missing": frozenset({"tool_name", "action_index"}),
    "model_output_budget_exhausted": frozenset(
        {
            "kind",
            "finish_reason",
            "reserved_response_tokens",
            "prompt_tokens",
            "completion_tokens",
        }
    ),
    "failure_classification_resolved": frozenset({"classification", "source"}),
    "working_memory_updated": frozenset({"working_memory", "reason"}),
    "memory_extracted": frozenset({"memory", "source_event_id"}),
    "memory_stored": frozenset({"memory"}),
    "memory_retrieved": frozenset({"query", "memory_ids", "count"}),
    "memory_flagged": frozenset({"source_event_id", "reason", "trust_level", "content_preview"}),
    "memory_rejected": frozenset({"source_event_id", "reason", "trust_level", "content_preview"}),
    "project_state_updated": frozenset({"project_state", "reason"}),
    "context_built": frozenset({"goal", "kind", "prompt_mode", "history_message_count", "note_ids", "semantic_memory_ids", "plan_id", "selection_trace"}),
    "semantic_retrieval_degraded": frozenset(
        {
            "operation",
            "kind",
            "goal",
            "prompt_mode",
            "for_planning",
            "retrieval_backend",
            "fallback_backend",
            "error",
        }
    ),
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
    "tool_mismatch_rejected": frozenset({"step_id", "selected_tool", "expected_tool", "reason"}),
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
    "unresolved_objective_verification_deferred": frozenset(
        {"missing_check_groups", "final_step_id", "reason"}
    ),
    "retry_triggered": frozenset({"step_id", "reason", "attempt", "failure_kind"}),
    "retry_suppressed": frozenset({"step_id", "reason", "verification_reason"}),
    "replan_triggered": frozenset({"step_id", "reason", "replan_count"}),
    "drift_detected": frozenset({"reason", "event_count"}),
    "recovery_triggered": frozenset({"reason", "event_count"}),
    "consistency_checked": frozenset({"working_memory_ok", "semantic_memory_ok", "recovered"}),
    "consistency_failed": frozenset({"component", "reason"}),
    "tool_execution_context": frozenset({"tool_name", "tool_kind", "isolated", "policy"}),
    "wakeup_scheduled": frozenset({"wakeup_id", "wake_at", "reason"}),
    "wakeup_cancelled": frozenset({"wakeup_id", "cancelled_at"}),
    "wakeup_due": frozenset({"wakeup_id", "wake_at", "reason"}),
    "model_request_sent": frozenset({"kind", "prompt_mode", "attempt", "request", "budget_report"}),
    "inference_request_queued": frozenset(
        {"request_id", "call_id", "kind", "source", "priority", "backend_key"}
    ),
    "inference_request_started": frozenset(
        {
            "request_id",
            "call_id",
            "kind",
            "source",
            "priority",
            "attempt",
            "queue_wait_seconds",
            "backend_capacity",
            "capacity_source",
        }
    ),
    "inference_request_requeued": frozenset(
        {"request_id", "call_id", "kind", "reason", "attempt"}
    ),
    "inference_request_finished": frozenset(
        {"request_id", "call_id", "kind", "status", "attempt", "error"}
    ),
    "model_request_progress": frozenset({"kind", "prompt_mode", "attempt", "elapsed_seconds", "timeout_seconds", "policy"}),
    "model_token_progress": frozenset({"kind", "prompt_mode", "attempt", "elapsed_seconds", "completion_tokens", "tokens_per_second", "first_token_seconds", "token_timeout_seconds"}),
    "model_response_received": frozenset({"kind", "prompt_mode", "attempt", "completion"}),
    "model_call_failed": frozenset({"kind", "prompt_mode", "attempt", "error", "error_type"}),
    "model_call_preempted": frozenset({"kind", "prompt_mode", "attempt", "call_id", "preemption_id", "request_sha256"}),
    "model_call_replayed": frozenset({"kind", "call_id", "preemption_id", "request_sha256", "request"}),
    "model_call_replay_invalidated": frozenset({"kind", "call_id", "preemption_id", "request_sha256"}),
    "model_retry_scheduled": frozenset({"kind", "prompt_mode", "next_attempt"}),
    # text_hash alone keeps old histories valid while new events add text_chars
    # and intentionally omit raw model-visible text.
    "model_tokenize_requested": frozenset({"text_hash"}),
    "model_tokenize_result": frozenset({"text_hash", "tokens", "exact"}),
    "model_tokenize_failed": frozenset({"text_hash", "error", "error_type"}),
    "token_estimate_used": frozenset({"text_hash", "tokens", "strategy"}),
    "decision_parsed": frozenset({"decision", "prompt_mode"}),
    "clarification_generated": frozenset({"source", "text"}),
    "tool_input_parsed": frozenset({"tool_name", "tool_input", "prompt_mode"}),
    "tool_called": frozenset({"tool_name", "tool_input"}),
    "output_unit_generated": frozenset({"unit", "overflowed", "text"}),
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
    "wait_completed": frozenset({"reason", "requested_seconds", "requested_duration", "elapsed_seconds"}),
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
    "fatal_system_error": frozenset({"operation", "error", "error_type", "category", "warning"}),
}


class EventSchemaError(ValueError):
    pass


def validate_event_payload(event_type: str, payload: dict[str, Any], metadata: dict[str, Any], *, allow_legacy: bool = False) -> None:
    allowed = READABLE_EVENT_TYPES if allow_legacy else ALLOWED_EVENT_TYPES
    if event_type not in allowed:
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
    validate_event_payload(event.event_type, event.payload, event.metadata, allow_legacy=True)
    if event.prev_hash != expected_prev_hash:
        raise EventSchemaError(
            f"Hash chain mismatch at sequence {event.sequence}: expected prev_hash={expected_prev_hash!r}, got {event.prev_hash!r}"
        )
    expected_hash = compute_event_hash(event)
    if event.hash != expected_hash:
        raise EventSchemaError(f"Event hash mismatch at sequence {event.sequence}")
