from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from swaag.environment.state import EnvironmentState, ProcessRecord, ShellSessionState, WorkspaceState
from swaag.events import EventSchemaError, create_event, verify_event_integrity
from swaag.planner import mark_step_completed, mark_step_failed, mark_step_in_progress
from swaag.roles import default_agent_roles
from swaag.security import with_trust_level
from swaag.types import (
    AgentRoleDefinition,
    DerivedFileWrite,
    FileView,
    HistoryEvent,
    MemoryEntity,
    MemoryFact,
    MemoryRelationship,
    Message,
    Note,
    Plan,
    PlanStep,
    PromptAnalysis,
    DecisionOutcome,
    DeferredTask,
    ExpandedTask,
    StrategySelection,
    ProjectState,
    ReaderState,
    SemanticMemoryItem,
    SessionMetrics,
    SessionState,
    CodeCheckpoint,
    WorkingMemory,
)
from swaag.utils import new_id, stable_json_dumps, to_jsonable, utc_now_iso


class HistoryCorruptionError(RuntimeError):
    pass


class HistoryInvariantError(RuntimeError):
    pass


CHECKPOINT_FILE_NAME = "checkpoint.json"
ACTIVE_RUN_FILE_NAME = "active_run.json"
CONTROL_INBOX_DIR_NAME = "control_inbox"
CONTROL_PROCESSED_DIR_NAME = "control_processed"


def _default_session_name(session_id: str) -> str:
    return f"session-{session_id.split('_')[-1][:8]}"


def _slugify_session_name(text: str, *, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    if not cleaned:
        return ""
    return cleaned[:limit].strip("-")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_agent_roles() -> list[AgentRoleDefinition]:
    return default_agent_roles()


class HistoryStore:
    def __init__(self, root: Path, *, write_projections: bool = True):
        self.root = Path(root).expanduser()
        self.write_projections = write_projections

    def guard(self, state: SessionState, operation_name: str) -> HistoryGuard:
        return HistoryGuard(self, state, operation_name)

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def history_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "complete_history.jsonl"

    def current_state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "current_state.json"

    def notes_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "notes.json"

    def reader_state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "reader_state.json"

    def history_index_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "history_index.json"

    def checkpoint_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / CHECKPOINT_FILE_NAME

    def active_run_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / ACTIVE_RUN_FILE_NAME

    def control_inbox_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / CONTROL_INBOX_DIR_NAME

    def control_processed_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / CONTROL_PROCESSED_DIR_NAME

    def code_checkpoints_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "code_checkpoints"

    def create(
        self,
        *,
        config_fingerprint: str,
        model_base_url: str,
        session_id: str | None = None,
        session_name: str | None = None,
        session_name_source: str = "placeholder",
    ) -> SessionState:
        session_id = session_id or new_id("session")
        final_name = (session_name or "").strip() or _default_session_name(session_id)
        state = SessionState(
            session_id=session_id,
            created_at="",
            updated_at="",
            config_fingerprint="",
            model_base_url="",
            session_name=final_name,
            session_name_source=session_name_source if final_name else "placeholder",
        )
        self.record_event(
            state,
            "session_created",
            {
                "session_id": session_id,
                "config_fingerprint": config_fingerprint,
                "model_base_url": model_base_url,
                "created_at": utc_now_iso(),
                "session_name": final_name,
                "session_name_source": session_name_source if final_name else "placeholder",
            },
        )
        return state

    def create_or_load(self, *, config_fingerprint: str, model_base_url: str, session_id: str | None = None) -> SessionState:
        if session_id and self.history_path(session_id).exists():
            return self.rebuild_from_history(session_id, write_projections=False)
        return self.create(config_fingerprint=config_fingerprint, model_base_url=model_base_url, session_id=session_id)

    def create_or_load_user_session(
        self,
        *,
        config_fingerprint: str,
        model_base_url: str,
        session_ref: str | None = None,
        prefer_latest: bool = False,
    ) -> SessionState:
        resolved = self.resolve_session_ref(session_ref, latest_if_none=prefer_latest)
        if resolved is not None:
            return self.rebuild_from_history(resolved, write_projections=False)
        if session_ref is None:
            return self.create(config_fingerprint=config_fingerprint, model_base_url=model_base_url)
        unique_name = self._unique_session_name(session_ref)
        return self.create(
            config_fingerprint=config_fingerprint,
            model_base_url=model_base_url,
            session_name=unique_name,
            session_name_source="explicit",
        )

    def list_sessions(self) -> list[str]:
        return [entry["session_id"] for entry in self.list_session_entries()]

    def list_session_entries(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        entries: list[dict[str, Any]] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not self.history_path(path.name).exists():
                continue
            entries.append(self._session_entry(path.name))
        entries.sort(
            key=lambda item: (
                str(item.get("updated_at", "")),
                str(item.get("created_at", "")),
                str(item.get("session_id", "")),
            ),
            reverse=True,
        )
        return entries

    def latest_session_id(self) -> str | None:
        entries = self.list_session_entries()
        return str(entries[0]["session_id"]) if entries else None

    def resolve_session_ref(self, session_ref: str | None, *, latest_if_none: bool = False) -> str | None:
        if session_ref is None:
            return self.latest_session_id() if latest_if_none else None
        ref = session_ref.strip()
        if not ref:
            return self.latest_session_id() if latest_if_none else None
        if ref == "latest":
            return self.latest_session_id()
        if self.history_path(ref).exists():
            return ref
        lowered = ref.casefold()
        matches = [entry for entry in self.list_session_entries() if str(entry.get("session_name", "")).casefold() == lowered]
        if len(matches) > 1:
            raise HistoryInvariantError(f"Session name is ambiguous: {session_ref}")
        return str(matches[0]["session_id"]) if matches else None

    def rename_session(self, session_ref: str, new_name: str, *, reason: str = "cli_rename") -> SessionState:
        session_id = self.resolve_session_ref(session_ref, latest_if_none=False)
        if session_id is None:
            raise FileNotFoundError(f"Unknown session: {session_ref}")
        desired_name = new_name.strip()
        if not desired_name:
            raise ValueError("new_name must not be empty")
        collision = self.resolve_session_ref(desired_name, latest_if_none=False)
        if collision is not None and collision != session_id:
            raise ValueError(f"Session name already exists: {desired_name}")
        state = self.rebuild_from_history(session_id, write_projections=False)
        old_name = state.session_name or _default_session_name(session_id)
        if old_name == desired_name:
            return state
        self.record_event(
            state,
            "session_renamed",
            {
                "session_id": session_id,
                "old_name": old_name,
                "new_name": desired_name,
                "reason": reason,
            },
        )
        return state

    def ensure_human_readable_name(self, state: SessionState, seed_text: str) -> str:
        if state.session_name and state.session_name_source != "placeholder":
            return state.session_name
        derived = _slugify_session_name(seed_text) or _default_session_name(state.session_id)
        unique = self._unique_session_name(derived, exclude_session_id=state.session_id)
        old_name = state.session_name or _default_session_name(state.session_id)
        if old_name == unique and state.session_name_source != "placeholder":
            return old_name
        self.record_event(
            state,
            "session_renamed",
            {
                "session_id": state.session_id,
                "old_name": old_name,
                "new_name": unique,
                "reason": "auto_name_from_first_prompt",
            },
        )
        return unique

    def _session_entry(self, session_id: str) -> dict[str, Any]:
        fallback = {
            "session_id": session_id,
            "session_name": _default_session_name(session_id),
            "session_name_source": "placeholder",
            "created_at": "",
            "updated_at": "",
            "turn_count": 0,
            "event_count": 0,
            "active": self.active_run_path(session_id).exists(),
        }
        index_path = self.history_index_path(session_id)
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return {**fallback, **payload, "active": self.active_run_path(session_id).exists()}
        state_path = self.current_state_path(session_id)
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return {
                    **fallback,
                    "session_name": str(payload.get("session_name") or fallback["session_name"]),
                    "session_name_source": str(payload.get("session_name_source") or "placeholder"),
                    "created_at": str(payload.get("created_at", "")),
                    "updated_at": str(payload.get("updated_at", "")),
                    "turn_count": int(payload.get("turn_count", 0)),
                    "event_count": int(payload.get("event_count", 0)),
                    "active_plan_id": payload.get("active_plan", {}).get("plan_id") if isinstance(payload.get("active_plan"), dict) else None,
                    "active": self.active_run_path(session_id).exists(),
                }
        return fallback

    def _unique_session_name(self, candidate: str, *, exclude_session_id: str | None = None) -> str:
        base = candidate.strip() or "session"
        existing = {
            str(entry.get("session_name", "")).casefold(): str(entry.get("session_id", ""))
            for entry in self.list_session_entries()
        }
        if existing.get(base.casefold()) in {None, exclude_session_id}:
            return base
        suffix = 2
        while True:
            derived = f"{base}-{suffix}"
            owner = existing.get(derived.casefold())
            if owner in {None, exclude_session_id}:
                return derived
            suffix += 1

    def set_active_run(self, session_id: str, *, run_id: str, user_text: str) -> None:
        payload = {
            "run_id": run_id,
            "session_id": session_id,
            "user_text": user_text,
            "started_at": utc_now_iso(),
            "pid": os.getpid(),
        }
        self._write_projection(self.active_run_path(session_id), payload)

    def clear_active_run(self, session_id: str, *, run_id: str | None = None) -> None:
        path = self.active_run_path(session_id)
        if not path.exists():
            return
        if run_id is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("run_id") not in {None, run_id}:
                return
        path.unlink(missing_ok=True)

    def read_active_run(self, session_id: str) -> dict[str, Any] | None:
        path = self.active_run_path(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def enqueue_control_message(self, session_id: str, text: str, *, source: str = "cli") -> dict[str, Any]:
        control_id = new_id("control")
        payload = {
            "control_id": control_id,
            "session_id": session_id,
            "message": text.strip(),
            "source": source,
            "created_at": utc_now_iso(),
        }
        path = self.control_inbox_dir(session_id) / f"{control_id}.json"
        self._write_projection(path, payload)
        return payload

    def list_pending_control_messages(self, session_id: str) -> list[dict[str, Any]]:
        inbox = self.control_inbox_dir(session_id)
        if not inbox.exists():
            return []
        messages: list[dict[str, Any]] = []
        for path in sorted(inbox.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                payload["_path"] = str(path)
                messages.append(payload)
        return messages

    def mark_control_message_processed(self, session_id: str, control_id: str) -> None:
        inbox_path = self.control_inbox_dir(session_id) / f"{control_id}.json"
        processed_path = self.control_processed_dir(session_id) / f"{control_id}.json"
        if not inbox_path.exists():
            return
        _ensure_directory(processed_path.parent)
        os.replace(inbox_path, processed_path)

    def iter_history(
        self,
        session_id: str,
        *,
        start_sequence: int = 1,
        end_sequence: int | None = None,
    ) -> Iterator[HistoryEvent]:
        path = self.history_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown session: {session_id}")
        with path.open("r", encoding="utf-8") as handle:
            seen_ids: set[str] = set()
            prev_hash: str | None = None
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HistoryCorruptionError(f"Invalid history JSON at line {line_number} in {path}") from exc
                try:
                    event = HistoryEvent(**payload)
                except TypeError as exc:
                    raise HistoryCorruptionError(f"Invalid history event shape at line {line_number} in {path}: {payload!r}") from exc
                try:
                    verify_event_integrity(event, prev_hash)
                except EventSchemaError as exc:
                    raise HistoryCorruptionError(f"Invalid history event at line {line_number} in {path}: {exc}") from exc
                if event.id in seen_ids:
                    raise HistoryCorruptionError(f"Duplicate history event id at line {line_number} in {path}: {event.id}")
                seen_ids.add(event.id)
                prev_hash = event.hash
                if event.sequence < start_sequence:
                    continue
                if end_sequence is not None and event.sequence > end_sequence:
                    break
                yield event

    def read_history(self, session_id: str) -> list[HistoryEvent]:
        return list(self.iter_history(session_id))

    def read_history_window(self, session_id: str, *, start_sequence: int, limit: int) -> list[HistoryEvent]:
        if start_sequence <= 0:
            raise ValueError("start_sequence must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")
        end_sequence = start_sequence + limit - 1
        return list(self.iter_history(session_id, start_sequence=start_sequence, end_sequence=end_sequence))

    def iter_history_chunks(
        self,
        session_id: str,
        *,
        chunk_size: int,
        start_sequence: int = 1,
        end_sequence: int | None = None,
    ) -> Iterator[list[HistoryEvent]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        batch: list[HistoryEvent] = []
        for event in self.iter_history(session_id, start_sequence=start_sequence, end_sequence=end_sequence):
            batch.append(event)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def rebuild_from_history(
        self,
        session_id: str,
        *,
        write_projections: bool = False,
        prefer_checkpoint: bool = True,
        end_sequence: int | None = None,
        chunk_size: int | None = None,
    ) -> SessionState:
        if write_projections:
            raise HistoryInvariantError("rebuild_from_history does not write projections directly; record a follow-up event instead")
        if end_sequence is not None and end_sequence <= 0:
            raise ValueError("end_sequence must be positive")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        use_checkpoint = prefer_checkpoint and end_sequence is None
        state = self._load_checkpoint(session_id) if use_checkpoint else None
        if state is None:
            state = SessionState(
                session_id=session_id,
                created_at="",
                updated_at="",
                config_fingerprint="",
                model_base_url="",
            )
        next_sequence = state.event_count + 1
        saw_event = state.event_count > 0
        chunk_iterator: Iterable[list[HistoryEvent]] | Iterable[HistoryEvent]
        if chunk_size is None:
            chunk_iterator = self.iter_history(session_id, start_sequence=next_sequence, end_sequence=end_sequence)
            for expected_sequence, event in enumerate(chunk_iterator, start=next_sequence):
                saw_event = True
                if event.sequence != expected_sequence:
                    raise HistoryCorruptionError(
                        f"History sequence gap for session {session_id}: expected {expected_sequence}, got {event.sequence}"
                    )
                self._apply_event(state, event)
        else:
            expected_sequence = next_sequence
            for batch in self.iter_history_chunks(session_id, chunk_size=chunk_size, start_sequence=next_sequence, end_sequence=end_sequence):
                for event in batch:
                    saw_event = True
                    if event.sequence != expected_sequence:
                        raise HistoryCorruptionError(
                            f"History sequence gap for session {session_id}: expected {expected_sequence}, got {event.sequence}"
                        )
                    self._apply_event(state, event)
                    expected_sequence += 1
        if not saw_event:
            raise HistoryCorruptionError(f"History is empty for session: {session_id}")
        return state

    def replay_window(self, session_id: str, *, end_sequence: int, chunk_size: int | None = None) -> SessionState:
        return self.rebuild_from_history(
            session_id,
            write_projections=False,
            prefer_checkpoint=False,
            end_sequence=end_sequence,
            chunk_size=chunk_size,
        )

    def record_event(
        self,
        state: SessionState,
        event_type: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        derived_writes: Iterable[DerivedFileWrite] = (),
    ) -> HistoryEvent:
        payload = to_jsonable(payload)
        metadata = with_trust_level(event_type, payload, metadata)
        event = self._next_event(state, event_type, payload, metadata)
        self._append_marshaled_event(state, event)
        for write_plan in derived_writes:
            self._apply_derived_write(state, write_plan, cause_event=event.event_type)
        return event

    def _next_event(self, state: SessionState, event_type: str, payload: dict[str, Any], metadata: dict[str, Any]) -> HistoryEvent:
        return create_event(
            session_id=state.session_id,
            sequence=state.event_count + 1,
            event_type=event_type,
            payload=payload,
            metadata=metadata,
            prev_hash=state.last_event_hash,
            timestamp=utc_now_iso(),
        )

    def _append_marshaled_event(self, state: SessionState, event: HistoryEvent) -> None:
        _ensure_directory(self._session_dir(state.session_id))
        encoded = stable_json_dumps(asdict(event)) + "\n"
        with self.history_path(state.session_id).open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._apply_event(state, event)
        if self.write_projections:
            self._write_projections(state)

    def _apply_derived_write(self, state: SessionState, write_plan: DerivedFileWrite, *, cause_event: str) -> None:
        target = Path(write_plan.path).expanduser()
        try:
            _ensure_directory(target.parent)
            if write_plan.backup_content is not None:
                backup_path = target.with_name(target.name + write_plan.backup_suffix)
                self._atomic_write(backup_path, write_plan.backup_content, encoding=write_plan.encoding)
            self._atomic_write(target, write_plan.content, encoding=write_plan.encoding)
        except Exception as exc:
            failure_event = self._next_event(
                state,
                "file_write_failed",
                {
                    "path": str(target),
                    "cause_event": cause_event,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                {},
            )
            self._append_marshaled_event(state, failure_event)
            raise
        success_event = self._next_event(
            state,
            "file_write_applied",
            {
                "path": str(target),
                "cause_event": cause_event,
                "backup_path": str(target.with_name(target.name + write_plan.backup_suffix)) if write_plan.backup_content is not None else None,
                "size_chars": len(write_plan.content),
            },
            {},
        )
        self._append_marshaled_event(state, success_event)

    def _atomic_write(self, path: Path, content: str, *, encoding: str) -> None:
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(content, encoding=encoding)
        os.replace(tmp_path, path)

    def _write_projections(self, state: SessionState) -> None:
        state_payload = self._state_payload(state)
        self._write_projection(self.current_state_path(state.session_id), state_payload)
        self._write_projection(self.checkpoint_path(state.session_id), state_payload)
        self._write_projection(self.notes_path(state.session_id), to_jsonable([asdict(note) for note in state.notes]))
        self._write_projection(
            self.reader_state_path(state.session_id),
            to_jsonable({key: asdict(value) for key, value in state.reader_states.items()}),
        )
        self._write_projection(
            self.history_index_path(state.session_id),
            {
                "session_id": state.session_id,
                "session_name": state.session_name or _default_session_name(state.session_id),
                "session_name_source": state.session_name_source,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "event_count": state.event_count,
                "last_event_hash": state.last_event_hash,
                "turn_count": state.turn_count,
                "compaction_count": state.compaction_count,
                "edit_count": state.edit_count,
                "active_plan_id": state.active_plan.plan_id if state.active_plan is not None else None,
                "semantic_memory_count": len(state.semantic_memory),
                "semantic_fact_count": len(state.semantic_facts),
                "checkpoint_event_count": state.event_count,
                "deferred_task_count": len(state.deferred_tasks),
                "code_checkpoint_count": len(state.code_checkpoints),
                "latest_user_message": next((message.content for message in reversed(state.messages) if message.role == "user"), ""),
                "metrics": to_jsonable(asdict(state.metrics)),
            },
        )

    def _write_projection(self, path: Path, payload: Any) -> None:
        _ensure_directory(path.parent)
        self._atomic_write(path, stable_json_dumps(payload, indent=2), encoding="utf-8")

    def append_auxiliary_log(self, relative_path: str, payload: Any) -> Path:
        path = self.root / relative_path
        _ensure_directory(path.parent)
        encoded = stable_json_dumps(to_jsonable(payload)) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def query_history_details(
        self,
        session_ref: str | None,
        query_text: str,
        *,
        topic_hint: str = "",
        max_results: int = 8,
        token_score: int = 2,
        exact_score: int = 4,
        type_bonus: int = 1,
        preview_chars: int = 320,
    ) -> dict[str, Any]:
        session_id = self.resolve_session_ref(session_ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No session available")
        query = " ".join(part for part in [query_text.strip(), topic_hint.strip()] if part.strip())
        lowered = query.casefold()
        tokens = [token for token in re.findall(r"[A-Za-z0-9_./:-]+", lowered) if len(token) >= 2]
        quoted_groups = re.findall(r'"([^\"]+)"|\'([^\']+)\'', query)
        exact_terms = [part.strip() for group in quoted_groups for part in group if part.strip()]
        ranked: list[tuple[int, HistoryEvent, str]] = []
        preferred_types = {
            "tool_called",
            "tool_result",
            "shell_command_started",
            "shell_command_completed",
            "process_started",
            "process_completed",
            "process_timed_out",
            "process_killed",
            "file_write_applied",
            "edit_applied",
            "filesystem_read",
            "workspace_snapshot",
        }
        for event in self.iter_history(session_id):
            haystack = stable_json_dumps({"type": event.event_type, "payload": event.payload}).casefold()
            score = 0
            matched_terms: list[str] = []
            for term in tokens:
                if term in haystack:
                    score += token_score
                    matched_terms.append(term)
            for term in exact_terms:
                if term.casefold() in haystack:
                    score += exact_score
                    matched_terms.append(term)
            if event.event_type in preferred_types:
                score += type_bonus
            if score <= 0:
                continue
            preview = stable_json_dumps(event.payload)
            ranked.append((score, event, preview[:preview_chars]))
        ranked.sort(key=lambda item: (item[0], item[1].sequence), reverse=True)
        matches = [
            {
                "sequence": event.sequence,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "payload": to_jsonable(event.payload),
                "preview": preview,
            }
            for _, event, preview in ranked[:max_results]
        ]
        return {
            "session_id": session_id,
            "session_name": self._session_entry(session_id).get("session_name", ""),
            "query": query_text,
            "topic_hint": topic_hint,
            "match_count": len(matches),
            "matches": matches,
        }

    def _state_payload(self, state: SessionState) -> dict[str, Any]:
        return to_jsonable(asdict(state))

    def _load_checkpoint(self, session_id: str) -> SessionState | None:
        path = self.checkpoint_path(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            return _state_from_payload(payload)
        except Exception:
            return None

    def _apply_event(self, state: SessionState, event: HistoryEvent) -> None:
        payload = event.payload
        state.event_count = event.sequence
        state.updated_at = event.timestamp
        state.last_event_hash = event.hash
        if not state.agent_roles:
            state.agent_roles = _default_agent_roles()
        self._update_metrics(state, event)

        if event.event_type == "session_created":
            state.session_id = str(payload["session_id"])
            state.created_at = str(payload["created_at"])
            state.updated_at = event.timestamp
            state.config_fingerprint = str(payload["config_fingerprint"])
            state.model_base_url = str(payload["model_base_url"])
            state.session_name = str(payload.get("session_name") or _default_session_name(state.session_id))
            state.session_name_source = str(payload.get("session_name_source") or "placeholder")
            if not state.agent_roles:
                state.agent_roles = _default_agent_roles()
            return

        if event.event_type == "session_renamed":
            state.session_name = str(payload["new_name"])
            state.session_name_source = str(payload.get("reason") or "explicit")
            return

        if event.event_type == "message_added":
            state.messages.append(Message(**payload["message"]))
            return

        if event.event_type in {"history_compacted", "history_compressed"}:
            source_count = int(payload["source_message_count"])
            summary_message = Message(**payload["summary_message"])
            state.messages = [summary_message, *state.messages[source_count:]]
            state.compaction_count += 1
            return

        if event.event_type == "turn_finished":
            state.turn_count = int(payload["turn_index"])
            return

        if event.event_type == "deferred_task_queued":
            task = DeferredTask(**payload["task"])
            state.deferred_tasks = [item for item in state.deferred_tasks if item.task_id != task.task_id]
            state.deferred_tasks.append(task)
            return

        if event.event_type == "deferred_task_consumed":
            task_id = str(payload["task_id"])
            state.deferred_tasks = [item for item in state.deferred_tasks if item.task_id != task_id]
            return

        if event.event_type == "code_checkpoint_created":
            checkpoint = CodeCheckpoint(**payload["checkpoint"])
            state.code_checkpoints = [item for item in state.code_checkpoints if item.checkpoint_id != checkpoint.checkpoint_id]
            state.code_checkpoints.append(checkpoint)
            return

        if event.event_type == "code_checkpoint_restored":
            return

        if event.event_type == "note_added":
            note = Note(**payload["note"])
            state.notes = [item for item in state.notes if item.note_id != note.note_id]
            state.notes.append(note)
            return

        if event.event_type == "role_switched":
            state.active_role = str(payload["new_role"])
            return

        if event.event_type == "note_replaced":
            note = Note(**payload["note"])
            replaced = False
            new_notes: list[Note] = []
            for item in state.notes:
                if item.note_id == note.note_id:
                    new_notes.append(note)
                    replaced = True
                else:
                    new_notes.append(item)
            if not replaced:
                new_notes.append(note)
            state.notes = new_notes
            return

        if event.event_type == "notes_compacted":
            removed_ids = set(payload["removed_note_ids"])
            compacted_note = Note(**payload["compacted_note"])
            state.notes = [item for item in state.notes if item.note_id not in removed_ids]
            state.notes.append(compacted_note)
            return

        if event.event_type == "reader_opened":
            reader = ReaderState(**payload["reader_state"])
            state.reader_states[reader.reader_id] = reader
            return

        if event.event_type == "reader_chunk_read":
            reader = ReaderState(**payload["reader_state"])
            state.reader_states[reader.reader_id] = reader
            return

        if event.event_type == "environment_initialized":
            state.environment.workspace.root = str(payload["workspace_root"])
            state.environment.workspace.cwd = str(payload["cwd"])
            state.environment.shell.cwd = str(payload["cwd"])
            state.environment.shell.env_overrides = {str(key): str(value) for key, value in payload.get("shell_env_overrides", {}).items()}
            state.environment.shell.unset_vars = [str(item) for item in payload.get("shell_unset_vars", [])]
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "filesystem_listed":
            listed = [str(item) for item in payload.get("entries", [])]
            state.environment.workspace.cwd = str(payload.get("cwd", state.environment.workspace.cwd))
            state.environment.workspace.listed_files = sorted(set(state.environment.workspace.listed_files) | set(listed))
            state.environment.workspace.last_snapshot_at = event.timestamp
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "filesystem_read":
            rel = str(payload["relative_path"])
            text_value = str(payload["text"])
            state.environment.workspace.cwd = str(payload.get("cwd", state.environment.workspace.cwd))
            state.environment.workspace.known_files[rel] = text_value
            state.environment.workspace.listed_files = sorted(set(state.environment.workspace.listed_files) | {rel})
            state.environment.workspace.last_snapshot_at = event.timestamp
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "workspace_snapshot":
            files_payload = {str(key): str(value) for key, value in payload.get("files", {}).items()}
            state.environment.workspace.root = str(payload["workspace_root"])
            state.environment.workspace.cwd = str(payload["cwd"])
            state.environment.workspace.known_files = files_payload
            state.environment.workspace.listed_files = sorted(files_payload)
            state.environment.workspace.created_files = [str(item) for item in payload.get("created_files", [])]
            state.environment.workspace.modified_files = [str(item) for item in payload.get("modified_files", [])]
            state.environment.workspace.deleted_files = [str(item) for item in payload.get("deleted_files", [])]
            state.environment.workspace.last_snapshot_at = str(payload.get("captured_at", event.timestamp))
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "shell_command_completed":
            state.environment.shell.cwd = str(payload["cwd_after"])
            state.environment.shell.env_overrides = {str(key): str(value) for key, value in payload.get("env_overrides", {}).items()}
            state.environment.shell.unset_vars = [str(item) for item in payload.get("unset_vars", [])]
            state.environment.shell.command_count += 1
            state.environment.shell.last_command = str(payload["command"])
            state.environment.shell.last_exit_code = int(payload.get("exit_code", 0))
            state.environment.shell.updated_at = event.timestamp
            state.environment.last_updated = event.timestamp
            return

        if event.event_type in {"process_started", "process_polled", "process_completed", "process_timed_out", "process_killed"}:
            process_id = str(payload["process_id"])
            state.environment.processes[process_id] = ProcessRecord(**{key: value for key, value in payload.items() if key in ProcessRecord.__dataclass_fields__})
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "wait_entered":
            state.environment.waiting = True
            state.environment.waiting_reason = str(payload["reason"])
            state.environment.waiting_process_ids = [str(item) for item in payload.get("process_ids", [])]
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "wait_resumed":
            state.environment.waiting = False
            state.environment.waiting_reason = ""
            state.environment.waiting_process_ids = []
            state.environment.last_updated = event.timestamp
            return

        if event.event_type == "file_chunk_read":
            path = str(payload["source_ref"])
            view = state.file_views.get(path) or FileView(path=path)
            view.last_chunk_text = str(payload["text"])
            view.last_start_offset = int(payload["start_offset"])
            view.last_end_offset = int(payload["end_offset"])
            view.last_next_offset = int(payload["next_offset"])
            view.last_operation = "file_chunk_read"
            view.updated_at = event.timestamp
            state.file_views[path] = view
            return

        if event.event_type == "file_read_for_edit":
            path = str(payload["path"])
            view = state.file_views.get(path) or FileView(path=path)
            view.content = str(payload["text"])
            view.last_operation = "file_read_for_edit"
            view.updated_at = event.timestamp
            state.file_views[path] = view
            _update_environment_file(state, path, str(payload["text"]), event.timestamp)
            return

        if event.event_type == "edit_previewed":
            path = str(payload["path"])
            view = state.file_views.get(path) or FileView(path=path)
            view.last_operation = "edit_previewed"
            view.updated_at = event.timestamp
            view.metadata["last_preview_diff"] = str(payload["diff"])
            view.metadata["last_preview_operation"] = str(payload["operation"])
            state.file_views[path] = view
            return

        if event.event_type == "edit_applied":
            path = str(payload["path"])
            state.pending_file_writes[path] = str(payload["new_text"])
            view = state.file_views.get(path) or FileView(path=path)
            view.last_operation = "edit_applied"
            view.updated_at = event.timestamp
            view.metadata["pending_diff"] = str(payload["diff"])
            view.metadata["pending_operation"] = str(payload["operation"])
            state.file_views[path] = view
            state.edit_count += 1
            return

        if event.event_type == "file_write_applied":
            path = str(payload["path"])
            view = state.file_views.get(path) or FileView(path=path)
            if path in state.pending_file_writes:
                view.content = state.pending_file_writes.pop(path)
            view.last_operation = "file_write_applied"
            view.updated_at = event.timestamp
            view.metadata["backup_path"] = payload.get("backup_path")
            state.file_views[path] = view
            if view.content is not None:
                _update_environment_file(state, path, view.content, event.timestamp)
            return

        if event.event_type == "file_write_failed":
            path = str(payload["path"])
            state.pending_file_writes.pop(path, None)
            view = state.file_views.get(path) or FileView(path=path)
            view.last_operation = "file_write_failed"
            view.updated_at = event.timestamp
            view.metadata["write_error"] = str(payload["error"])
            state.file_views[path] = view
            state.environment.last_updated = event.timestamp
            return

        if event.event_type in {"plan_created", "plan_updated"}:
            state.active_plan = _plan_from_payload(payload["plan"])
            return

        if event.event_type == "plan_completed":
            if state.active_plan is not None and state.active_plan.plan_id == str(payload["plan_id"]):
                state.active_plan.status = str(payload["status"])
                state.active_plan.updated_at = event.timestamp
            return

        if event.event_type == "step_started":
            if state.active_plan is not None and state.active_plan.plan_id == str(payload.get("plan_id", state.active_plan.plan_id)):
                step_id = str(payload["step_id"])
                current = next((item for item in state.active_plan.steps if item.step_id == step_id), None)
                if current is not None and current.status == "pending":
                    state.active_plan = mark_step_in_progress(state.active_plan, step_id)
            return

        if event.event_type == "step_completed":
            if state.active_plan is not None and state.active_plan.plan_id == str(payload.get("plan_id", state.active_plan.plan_id)):
                step_id = str(payload["step_id"])
                current = next((item for item in state.active_plan.steps if item.step_id == step_id), None)
                if current is not None and current.status == "running":
                    state.active_plan = mark_step_completed(state.active_plan, step_id)
            return

        if event.event_type == "step_failed":
            if state.active_plan is not None and state.active_plan.plan_id == str(payload.get("plan_id", state.active_plan.plan_id)):
                step_id = str(payload["step_id"])
                current = next((item for item in state.active_plan.steps if item.step_id == step_id), None)
                if current is not None and current.status == "running":
                    state.active_plan = mark_step_failed(state.active_plan, step_id)
            return

        if event.event_type == "prompt_analyzed":
            state.prompt_analysis = PromptAnalysis(**payload["analysis"])
            return

        if event.event_type in {"decision_made", "decision_adjusted"}:
            state.latest_decision = DecisionOutcome(**payload["decision"])
            return

        if event.event_type == "task_expanded":
            state.expanded_task = ExpandedTask(**payload["expanded_task"])
            return

        if event.event_type == "strategy_selected":
            state.active_strategy = StrategySelection(**payload["strategy"])
            return

        if event.event_type == "working_memory_updated":
            state.working_memory = WorkingMemory(**payload["working_memory"])
            return

        if event.event_type == "memory_stored":
            item = SemanticMemoryItem(**payload["memory"])
            state.semantic_memory = [existing for existing in state.semantic_memory if existing.memory_id != item.memory_id]
            state.semantic_memory.append(item)
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            for entity_payload in metadata.get("entities", []):
                entity = MemoryEntity(
                    entity_id=str(entity_payload["entity_id"]),
                    name=str(entity_payload["name"]),
                    entity_type=str(entity_payload["entity_type"]),
                    source_event_id=item.source_event_id,
                    trust_level=item.trust_level,
                    confidence=float(metadata.get("confidence", 1.0)),
                    metadata={"memory_id": item.memory_id},
                )
                state.semantic_entities[entity.entity_id] = entity
            for relationship_payload in metadata.get("relationships", []):
                relationship = MemoryRelationship(
                    relationship_id=str(relationship_payload["relationship_id"]),
                    source_entity_id=str(relationship_payload["source_entity_id"]),
                    relation_type=str(relationship_payload["relation_type"]),
                    target_entity_id=str(relationship_payload["target_entity_id"]),
                    source_event_id=item.source_event_id,
                    trust_level=item.trust_level,
                    confidence=float(metadata.get("confidence", 1.0)),
                    metadata={"memory_id": item.memory_id},
                )
                if all(existing.relationship_id != relationship.relationship_id for existing in state.semantic_relationships):
                    state.semantic_relationships.append(relationship)
            for fact_payload in metadata.get("facts", []):
                fact = MemoryFact(
                    fact_id=str(fact_payload["fact_id"]),
                    fact_type=str(fact_payload["fact_type"]),
                    content=str(fact_payload["content"]),
                    source_event_id=item.source_event_id,
                    trust_level=item.trust_level,
                    confidence=float(metadata.get("confidence", 1.0)),
                    metadata={"memory_id": item.memory_id},
                )
                if all(existing.fact_id != fact.fact_id for existing in state.semantic_facts):
                    state.semantic_facts.append(fact)
            if item.memory_kind == "procedural":
                state.procedural_patterns = [existing for existing in state.procedural_patterns if existing.memory_id != item.memory_id]
                state.procedural_patterns.append(item)
            return

        if event.event_type == "project_state_updated":
            state.project_state = ProjectState(**payload["project_state"])
            return

        if event.event_type in {
            "summary_created",
            "turn_started",
            "prompt_built",
            "budget_checked",
            "budget_rejected",
            "model_tokenize_requested",
            "model_tokenize_result",
            "model_tokenize_failed",
            "token_estimate_used",
            "model_request_sent",
            "model_request_progress",
            "model_response_received",
            "model_retry_scheduled",
            "model_call_failed",
            "decision_parsed",
            "tool_execution_context",
            "output_decomposition_planned",
            "output_unit_generated",
            "output_overflow_recovery_planned",
            "tool_called",
            "tool_result",
            "tool_error",
            "duplicate_action_detected",
            "filesystem_search",
            "repository_searched",
            "shell_command_started",
            "workspace_snapshot_inspected",
            "changes_listed",
            "diff_inspected",
            "state_rebuilt",
            "notes_selected",
            "buffer_chunk_read",
            "file_read_requested",
            "buffer_read_requested",
            "doctor_health_checked",
            "doctor_tokenize_checked",
            "review_started",
            "review_completed",
            "review_skipped",
            "subagent_spawned",
            "subagent_reported",
            "subagent_selection_resolved",
            "memory_extracted",
            "memory_retrieved",
            "memory_flagged",
            "memory_rejected",
            "context_built",
            "reasoning_started",
            "action_selected",
            "step_executed",
            "reasoning_completed",
            "answer_derived",
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
            "strategy_selection_resolved",
            "failure_classification_resolved",
            "plan_repaired",
            "error",
            "retry",
            "emergency_fallback_used",
            "fatal_system_error",
            "control_message_processed",
            "control_action_applied",
        }:
            return

        raise HistoryCorruptionError(f"Unknown event type during rebuild: {event.event_type}")

    def _update_metrics(self, state: SessionState, event: HistoryEvent) -> None:
        metrics = state.metrics
        payload = event.payload
        if event.event_type == "model_request_sent" and payload.get("kind") != "doctor_health":
            metrics.model_calls += 1
            if str(payload.get("requested_contract_mode", "")) == "json_schema" and str(payload.get("effective_contract_mode", "")) == "plain":
                metrics.post_validate_fallbacks += 1
            if str(payload.get("effective_contract_mode", "")) == "json_schema":
                metrics.server_schema_requests += 1
        elif event.event_type == "tool_called":
            metrics.tool_calls += 1
        elif event.event_type == "tool_result":
            tool_name = str(payload.get("tool_name", ""))
            metrics.tool_success_counts[tool_name] = metrics.tool_success_counts.get(tool_name, 0) + 1
        elif event.event_type == "tool_error":
            metrics.tool_failures += 1
            tool_name = str(payload.get("tool_name", ""))
            metrics.tool_failure_counts[tool_name] = metrics.tool_failure_counts.get(tool_name, 0) + 1
        elif event.event_type == "step_started":
            metrics.steps_started += 1
        elif event.event_type == "step_completed":
            metrics.steps_completed += 1
        elif event.event_type == "step_failed":
            metrics.steps_failed += 1
        elif event.event_type == "verification_passed":
            metrics.verification_passes += 1
        elif event.event_type == "verification_failed":
            metrics.verification_failures += 1
        elif event.event_type == "verification_type_used":
            verification_type = str(payload.get("verification_type_used", ""))
            if verification_type:
                metrics.verification_type_distribution[verification_type] = (
                    metrics.verification_type_distribution.get(verification_type, 0) + 1
                )
        elif event.event_type == "model_request_progress":
            metrics.model_request_progress_events += 1
        elif event.event_type == "model_retry_scheduled":
            metrics.model_retry_events += 1
        elif event.event_type == "retry_triggered":
            metrics.retries += 1
        elif event.event_type == "replan_triggered":
            metrics.replans += 1
        elif event.event_type == "budget_rejected":
            metrics.budget_rejections += 1
        elif event.event_type == "token_estimate_used":
            metrics.token_estimate_uses += 1
        elif event.event_type == "strategy_selected":
            metrics.strategy_switches += 1
        elif event.event_type == "action_selected":
            metrics.action_count += 1
            selected_action = str(payload.get("selected_action", ""))
            action_costs = {
                "execute_step": 1.5,
                "answer_directly": 1.0,
                "retry_step": 1.25,
                "replan": 1.75,
                "stop": 0.25,
            }
            metrics.total_cost_units += action_costs.get(selected_action, 1.0)
        elif event.event_type == "prompt_built":
            budget_report = payload.get("budget_report", {})
            if isinstance(budget_report, dict):
                metrics.input_tokens += int(budget_report.get("input_tokens", 0))
                metrics.reserved_response_tokens += int(budget_report.get("reserved_response_tokens", 0))
        elif event.event_type == "reasoning_completed":
            status = str(payload.get("status", ""))
            reason = str(payload.get("reason", ""))
            metrics.last_reasoning_status = status
            metrics.last_reasoning_reason = reason
            metrics.stop_reason_counts[reason] = metrics.stop_reason_counts.get(reason, 0) + 1
            if status == "completed":
                metrics.successful_turns += 1
            else:
                metrics.failed_turns += 1
            if reason == "tool_call_budget_reached":
                metrics.tool_call_budget_hits += 1
            if reason == "max_iterations_reached":
                metrics.max_iteration_stops += 1
            if reason == "no_progress_possible":
                metrics.no_progress_stops += 1
        if event.event_type in {"step_failed", "verification_failed", "tool_error", "error"}:
            key = payload.get("error_type") or payload.get("failure_kind") or event.event_type
            key = str(key)
            metrics.failure_counts[key] = metrics.failure_counts.get(key, 0) + 1
        total_verifications = metrics.verification_passes + metrics.verification_failures
        if total_verifications > 0:
            metrics.verification_success_rate = metrics.verification_passes / total_verifications
            metrics.verification_failure_rate = metrics.verification_failures / total_verifications
            llm_fallback_count = metrics.verification_type_distribution.get("llm_fallback", 0)
            metrics.llm_fallback_rate = llm_fallback_count / total_verifications


class HistoryGuard:
    def __init__(self, store: HistoryStore, state: SessionState, operation_name: str):
        self._store = store
        self._state = state
        self._operation_name = operation_name
        self._start_sequence = state.event_count
        self._recorded_types: list[str] = []

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        derived_writes: Iterable[DerivedFileWrite] = (),
    ) -> HistoryEvent:
        event = self._store.record_event(self._state, event_type, payload, metadata=metadata, derived_writes=derived_writes)
        self._recorded_types.append(event.event_type)
        return event

    def require_any(self, *event_types: str) -> None:
        if not any(event_type in self._recorded_types for event_type in event_types):
            expected = ", ".join(event_types)
            raise HistoryInvariantError(
                f"Operation {self._operation_name} completed without required history event(s): {expected}"
            )

    def require_all(self, *event_types: str) -> None:
        missing = [event_type for event_type in event_types if event_type not in self._recorded_types]
        if missing:
            missing_text = ", ".join(missing)
            raise HistoryInvariantError(
                f"Operation {self._operation_name} completed without required history event(s): {missing_text}"
            )

    def ensure_progress(self) -> None:
        if self._state.event_count <= self._start_sequence:
            raise HistoryInvariantError(f"Operation {self._operation_name} completed without recording any history event")


def _plan_from_payload(payload: dict[str, Any]) -> Plan:
    normalized_steps = []
    for step_payload in payload.get("steps", []):
        step_payload = dict(step_payload)
        step_payload.setdefault("goal", step_payload.get("title", ""))
        step_payload.setdefault("input_text", "")
        step_payload.setdefault("expected_outputs", [step_payload.get("expected_output", "")] if step_payload.get("expected_output") else [])
        step_payload.setdefault("verification_type", "llm_fallback")
        step_payload.setdefault("verification_checks", [])
        step_payload.setdefault("required_conditions", [])
        step_payload.setdefault("optional_conditions", [])
        step_payload.setdefault("input_refs", [])
        step_payload.setdefault("output_refs", [])
        if "done_condition" not in step_payload:
            if step_payload.get("kind") == "respond":
                step_payload["done_condition"] = "assistant_response_nonempty"
            elif step_payload.get("expected_tool"):
                step_payload["done_condition"] = f"tool_result:{step_payload['expected_tool']}"
            else:
                step_payload["done_condition"] = "reasoning_result_nonempty"
        normalized_steps.append(step_payload)
    steps = [PlanStep(**step_payload) for step_payload in normalized_steps]
    return Plan(
        plan_id=str(payload["plan_id"]),
        goal=str(payload["goal"]),
        steps=steps,
        success_criteria=str(payload["success_criteria"]),
        fallback_strategy=str(payload.get("fallback_strategy", "")),
        status=str(payload["status"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        current_step_id=payload.get("current_step_id"),
    )


def _state_from_payload(payload: dict[str, Any]) -> SessionState:
    return SessionState(
        session_id=str(payload["session_id"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        config_fingerprint=str(payload["config_fingerprint"]),
        model_base_url=str(payload["model_base_url"]),
        session_name=str(payload.get("session_name") or _default_session_name(str(payload["session_id"]))),
        session_name_source=str(payload.get("session_name_source", "placeholder")),
        messages=[Message(**item) for item in payload.get("messages", [])],
        notes=[Note(**item) for item in payload.get("notes", [])],
        reader_states={key: ReaderState(**value) for key, value in payload.get("reader_states", {}).items()},
        file_views={key: FileView(**value) for key, value in payload.get("file_views", {}).items()},
        pending_file_writes={str(key): str(value) for key, value in payload.get("pending_file_writes", {}).items()},
        active_plan=_plan_from_payload(payload["active_plan"]) if payload.get("active_plan") else None,
        prompt_analysis=PromptAnalysis(**payload["prompt_analysis"]) if payload.get("prompt_analysis") else None,
        latest_decision=DecisionOutcome(**payload["latest_decision"]) if payload.get("latest_decision") else None,
        expanded_task=ExpandedTask(**payload["expanded_task"]) if payload.get("expanded_task") else None,
        active_strategy=StrategySelection(**payload["active_strategy"]) if payload.get("active_strategy") else None,
        working_memory=WorkingMemory(**payload.get("working_memory", {})),
        semantic_memory=[SemanticMemoryItem(**item) for item in payload.get("semantic_memory", [])],
        semantic_entities={key: MemoryEntity(**value) for key, value in payload.get("semantic_entities", {}).items()},
        semantic_relationships=[MemoryRelationship(**item) for item in payload.get("semantic_relationships", [])],
        semantic_facts=[MemoryFact(**item) for item in payload.get("semantic_facts", [])],
        procedural_patterns=[SemanticMemoryItem(**item) for item in payload.get("procedural_patterns", [])],
        project_state=ProjectState(**payload.get("project_state", {})),
        environment=_environment_from_payload(payload.get("environment", {})),
        deferred_tasks=[DeferredTask(**item) for item in payload.get("deferred_tasks", [])],
        code_checkpoints=[CodeCheckpoint(**item) for item in payload.get("code_checkpoints", [])],
        metrics=SessionMetrics(**payload.get("metrics", {})),
        agent_roles=[AgentRoleDefinition(**item) for item in payload.get("agent_roles", [])] or _default_agent_roles(),
        active_role=str(payload.get("active_role", "primary")),
        turn_count=int(payload.get("turn_count", 0)),
        compaction_count=int(payload.get("compaction_count", 0)),
        event_count=int(payload.get("event_count", 0)),
        edit_count=int(payload.get("edit_count", 0)),
        last_event_hash=payload.get("last_event_hash"),
    )


def _environment_from_payload(payload: dict[str, Any]) -> EnvironmentState:
    workspace_payload = payload.get("workspace", {}) if isinstance(payload, dict) else {}
    shell_payload = payload.get("shell", {}) if isinstance(payload, dict) else {}
    processes_payload = payload.get("processes", {}) if isinstance(payload, dict) else {}
    return EnvironmentState(
        workspace=WorkspaceState(
            root=str(workspace_payload.get("root", "")),
            cwd=str(workspace_payload.get("cwd", "")),
            known_files={str(key): str(value) for key, value in workspace_payload.get("known_files", {}).items()},
            listed_files=[str(item) for item in workspace_payload.get("listed_files", [])],
            modified_files=[str(item) for item in workspace_payload.get("modified_files", [])],
            created_files=[str(item) for item in workspace_payload.get("created_files", [])],
            deleted_files=[str(item) for item in workspace_payload.get("deleted_files", [])],
            last_snapshot_at=str(workspace_payload.get("last_snapshot_at", "")),
        ),
        shell=ShellSessionState(
            cwd=str(shell_payload.get("cwd", "")),
            env_overrides={str(key): str(value) for key, value in shell_payload.get("env_overrides", {}).items()},
            unset_vars=[str(item) for item in shell_payload.get("unset_vars", [])],
            command_count=int(shell_payload.get("command_count", 0)),
            last_command=str(shell_payload.get("last_command", "")),
            last_exit_code=shell_payload.get("last_exit_code"),
            updated_at=str(shell_payload.get("updated_at", "")),
        ),
        processes={
            str(key): ProcessRecord(**value)
            for key, value in processes_payload.items()
            if isinstance(value, dict)
        },
        waiting=bool(payload.get("waiting", False)),
        waiting_reason=str(payload.get("waiting_reason", "")),
        waiting_process_ids=[str(item) for item in payload.get("waiting_process_ids", [])],
        last_updated=str(payload.get("last_updated", "")),
    )


def _relative_environment_path(state: SessionState, path_text: str) -> str:
    workspace_root = state.environment.workspace.root
    try:
        path = Path(path_text).expanduser().resolve()
        if workspace_root:
            return str(path.relative_to(Path(workspace_root).expanduser().resolve()))
    except Exception:
        pass
    return path_text


def _update_environment_file(state: SessionState, path_text: str, content: str, timestamp: str) -> None:
    rel = _relative_environment_path(state, path_text)
    workspace = state.environment.workspace
    existed = rel in workspace.known_files
    workspace.known_files[rel] = content
    workspace.listed_files = sorted(set(workspace.listed_files) | {rel})
    if workspace.root and Path(path_text).expanduser().is_absolute():
        path = Path(path_text).expanduser()
        try:
            path.relative_to(Path(workspace.root).expanduser())
            if not existed:
                workspace.created_files = sorted(set(workspace.created_files) | {rel})
            elif rel not in workspace.created_files:
                workspace.modified_files = sorted(set(workspace.modified_files) | {rel})
        except ValueError:
            pass
    workspace.last_snapshot_at = timestamp
    state.environment.last_updated = timestamp


def replay_history(history_file: str | Path) -> SessionState:
    path = Path(history_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    session_dir = path.parent
    store = HistoryStore(session_dir.parent, write_projections=False)
    return store.rebuild_from_history(session_dir.name, write_projections=False, prefer_checkpoint=False)
