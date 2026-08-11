from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from swaag.environment.state import EnvironmentState, ProcessRecord, ShellSessionState, WorkspaceState
from swaag.events import ALLOWED_EVENT_TYPES, READABLE_EVENT_TYPES, EventSchemaError, create_event, verify_event_integrity
from swaag.types import (
    CodeCheckpoint,
    DeferredTask,
    DerivedFileWrite,
    FileView,
    HistoryEvent,
    Message,
    Note,
    ReaderState,
    SessionMetrics,
    SessionState,
)
from swaag.fsops import append_text, ensure_dir, write_text as _fsops_write_text
from swaag.utils import new_id, stable_json_dumps, to_jsonable, utc_now_iso


class HistoryCorruptionError(RuntimeError):
    pass


class HistoryInvariantError(RuntimeError):
    pass


CHECKPOINT_FILE_NAME = "checkpoint.json"
ACTIVE_RUN_FILE_NAME = "active_run.json"
CONTROL_INBOX_DIR_NAME = "control_inbox"
CONTROL_PROCESSED_DIR_NAME = "control_processed"

_STATEFUL_REBUILD_EVENT_TYPES = frozenset(
    {
        "session_created",
        "session_renamed",
        "message_added",
        "history_compacted",
        "history_compressed",
        "turn_finished",
        "deferred_task_queued",
        "deferred_task_consumed",
        "code_checkpoint_created",
        "code_checkpoint_restored",
        "note_added",
        "note_replaced",
        "notes_compacted",
        "reader_opened",
        "reader_chunk_read",
        "environment_initialized",
        "filesystem_listed",
        "filesystem_read",
        "workspace_snapshot",
        "shell_command_completed",
        "process_started",
        "process_polled",
        "process_completed",
        "process_timed_out",
        "process_killed",
        "wait_entered",
        "wait_resumed",
        "file_chunk_read",
        "file_read_for_edit",
        "edit_previewed",
        "edit_applied",
        "file_write_applied",
        "file_write_failed",
    }
)

_IGNORED_REBUILD_EVENT_TYPES = READABLE_EVENT_TYPES - _STATEFUL_REBUILD_EVENT_TYPES


def _default_session_name(session_id: str) -> str:
    return f"session-{session_id.split('_')[-1][:8]}"


def _slugify_session_name(text: str, *, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    if not cleaned:
        return ""
    return cleaned[:limit].strip("-")


def _ensure_directory(path: Path) -> Path:
    ensure_dir(path)
    return path



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

    def enqueue_control_message(self, session_id: str, text: str, *, source: str = "cli", control_id: str | None = None) -> dict[str, Any]:
        control_id = control_id or new_id("control")
        inbox_path = self.control_inbox_dir(session_id) / f"{control_id}.json"
        processed_path = self.control_processed_dir(session_id) / f"{control_id}.json"
        for existing_path in (inbox_path, processed_path):
            if existing_path.exists():
                payload = json.loads(existing_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        payload = {
            "control_id": control_id,
            "session_id": session_id,
            "message": text.strip(),
            "source": source,
            "created_at": utc_now_iso(),
        }
        self._write_projection(inbox_path, payload)
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
        metadata = to_jsonable(dict(metadata or {}))
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
        append_text(self.history_path(state.session_id), encoded, encoding="utf-8")
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
        _fsops_write_text(path, content, encoding=encoding)

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
        append_text(path, encoded, encoding="utf-8")
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
        self._update_metrics(state, event)

        if event.event_type == "session_created":
            state.session_id = str(payload["session_id"])
            state.created_at = str(payload["created_at"])
            state.config_fingerprint = str(payload["config_fingerprint"])
            state.model_base_url = str(payload["model_base_url"])
            state.session_name = str(payload.get("session_name") or _default_session_name(state.session_id))
            state.session_name_source = str(payload.get("session_name_source") or "placeholder")
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
        if event.event_type == "note_replaced":
            note = Note(**payload["note"])
            state.notes = [note if item.note_id == note.note_id else item for item in state.notes]
            if not any(item.note_id == note.note_id for item in state.notes):
                state.notes.append(note)
            return
        if event.event_type == "notes_compacted":
            removed = {str(item) for item in payload["removed_note_ids"]}
            compacted = Note(**payload["compacted_note"])
            state.notes = [item for item in state.notes if item.note_id not in removed]
            state.notes.append(compacted)
            return
        if event.event_type in {"reader_opened", "reader_chunk_read"}:
            reader = ReaderState(**payload["reader_state"])
            state.reader_states[reader.reader_id] = reader
            return
        if event.event_type == "environment_initialized":
            cwd = str(payload["cwd"])
            state.environment.workspace.root = str(payload["workspace_root"])
            state.environment.workspace.cwd = cwd
            state.environment.workspace.listed_files = [str(item) for item in payload.get("listed_files", [])]
            state.environment.workspace.listing_truncated = bool(payload.get("listing_truncated", False))
            state.environment.shell.cwd = cwd
            state.environment.shell.env_overrides = {str(k): str(v) for k, v in payload.get("shell_env_overrides", {}).items()}
            state.environment.shell.unset_vars = [str(item) for item in payload.get("shell_unset_vars", [])]
            state.environment.last_updated = event.timestamp
            return
        if event.event_type == "filesystem_listed":
            listed = [str(item) for item in payload.get("entries", [])]
            state.environment.workspace.cwd = str(payload.get("cwd", state.environment.workspace.cwd))
            state.environment.workspace.listed_files = sorted(set(state.environment.workspace.listed_files) | set(listed))
            if str(payload.get("path", "")) in {".", ""} and str(payload.get("cwd", "")) == state.environment.workspace.root:
                state.environment.workspace.listing_truncated = False
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
            files = {str(k): str(v) for k, v in payload.get("files", {}).items()}
            state.environment.workspace.root = str(payload["workspace_root"])
            state.environment.workspace.cwd = str(payload["cwd"])
            if str(payload.get("snapshot_mode") or "full") == "delta":
                known = dict(state.environment.workspace.known_files)
                known.update(files)
                for key in payload.get("deleted_files", []):
                    known.pop(str(key), None)
                state.environment.workspace.known_files = known
                state.environment.workspace.listed_files = sorted(known)
            else:
                state.environment.workspace.known_files = files
                state.environment.workspace.listed_files = sorted(files)
            state.environment.workspace.created_files = [str(item) for item in payload.get("created_files", [])]
            state.environment.workspace.modified_files = [str(item) for item in payload.get("modified_files", [])]
            state.environment.workspace.deleted_files = [str(item) for item in payload.get("deleted_files", [])]
            state.environment.workspace.last_snapshot_at = str(payload.get("captured_at", event.timestamp))
            if any((state.environment.workspace.created_files, state.environment.workspace.modified_files, state.environment.workspace.deleted_files)):
                state.edit_count += 1
            state.environment.last_updated = event.timestamp
            return
        if event.event_type == "shell_command_completed":
            state.environment.shell.cwd = str(payload["cwd_after"])
            state.environment.shell.env_overrides = {str(k): str(v) for k, v in payload.get("env_overrides", {}).items()}
            state.environment.shell.unset_vars = [str(item) for item in payload.get("unset_vars", [])]
            state.environment.shell.command_count += 1
            state.environment.shell.last_command = str(payload["command"])
            state.environment.shell.last_exit_code = int(payload.get("exit_code", 0))
            state.environment.shell.updated_at = event.timestamp
            if payload.get("created_files") or payload.get("modified_files") or payload.get("deleted_files"):
                state.edit_count += 1
            state.environment.last_updated = event.timestamp
            return
        if event.event_type in {"process_started", "process_polled", "process_completed", "process_timed_out", "process_killed"}:
            process_id = str(payload["process_id"])
            fields = ProcessRecord.__dataclass_fields__
            state.environment.processes[process_id] = ProcessRecord(**{k: v for k, v in payload.items() if k in fields})
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
            text_value = str(payload["text"])
            view = state.file_views.get(path) or FileView(path=path)
            view.content = text_value
            view.last_operation = "file_read_for_edit"
            view.updated_at = event.timestamp
            state.file_views[path] = view
            _update_environment_file(state, path, text_value, event.timestamp)
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
            return
        if event.event_type in _IGNORED_REBUILD_EVENT_TYPES:
            return
        raise HistoryCorruptionError(f"Unknown event type during rebuild: {event.event_type}")

    def _update_metrics(self, state: SessionState, event: HistoryEvent) -> None:
        metrics = state.metrics
        payload = event.payload
        if event.event_type == "model_request_sent" and payload.get("kind") != "doctor_health":
            metrics.model_calls += 1
            if str(payload.get("requested_contract_mode", "")) == "json_schema" and str(payload.get("effective_contract_mode", "")) != "json_schema":
                metrics.unconstrained_contract_violations += 1
            if str(payload.get("effective_contract_mode", "")) == "json_schema":
                metrics.server_schema_requests += 1
        elif event.event_type == "tool_called":
            metrics.tool_calls += 1
        elif event.event_type == "tool_result":
            name = str(payload.get("tool_name", ""))
            metrics.tool_success_counts[name] = metrics.tool_success_counts.get(name, 0) + 1
        elif event.event_type == "tool_error":
            metrics.tool_failures += 1
            name = str(payload.get("tool_name", ""))
            metrics.tool_failure_counts[name] = metrics.tool_failure_counts.get(name, 0) + 1
        elif event.event_type in {"model_request_progress", "model_token_progress"}:
            metrics.model_request_progress_events += 1
        elif event.event_type == "model_retry_scheduled":
            metrics.model_retry_events += 1
        elif event.event_type == "retry_triggered":
            metrics.retries += 1
        elif event.event_type == "budget_rejected":
            metrics.budget_rejections += 1
        elif event.event_type == "token_estimate_used":
            metrics.token_estimate_uses += 1
        elif event.event_type == "agent_action_selected":
            metrics.action_count += 1
        elif event.event_type == "prompt_built":
            report = payload.get("budget_report", {})
            if isinstance(report, dict):
                metrics.input_tokens += int(report.get("input_tokens", 0))
                metrics.reserved_response_tokens += int(report.get("reserved_response_tokens", 0))
        elif event.event_type == "turn_finished":
            if str(payload.get("status", "completed")) == "completed":
                metrics.successful_turns += 1
            else:
                metrics.failed_turns += 1
        if event.event_type in {"tool_error", "error"}:
            key = str(payload.get("error_type") or event.event_type)
            metrics.failure_counts[key] = metrics.failure_counts.get(key, 0) + 1


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
        pending_file_writes={str(k): str(v) for k, v in payload.get("pending_file_writes", {}).items()},
        environment=_environment_from_payload(payload.get("environment", {})),
        deferred_tasks=[DeferredTask(**item) for item in payload.get("deferred_tasks", [])],
        code_checkpoints=[CodeCheckpoint(**item) for item in payload.get("code_checkpoints", [])],
        metrics=SessionMetrics(**{
            key: value
            for key, value in payload.get("metrics", {}).items()
            if key in SessionMetrics.__dataclass_fields__
        }),
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
            listing_truncated=bool(workspace_payload.get("listing_truncated", False)),
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
