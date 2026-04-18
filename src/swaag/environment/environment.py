from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from swaag.config import AgentConfig
from swaag.editing import TextEditor
from swaag.environment.browser import aubro_available, run_aubro_command
from swaag.environment.filesystem import FilesystemError, FilesystemManager
from swaag.environment.process import ProcessManager
from swaag.environment.shell import ShellSession
from swaag.environment.state import EnvironmentState, ProcessRecord
from swaag.environment.workspace import WorkspaceManager
from swaag.reader import SequentialReader
from swaag.tools.base import ToolValidationError
from swaag.types import DerivedFileWrite, SessionState, ToolExecutionResult, ToolGeneratedEvent
from swaag.fsops import ensure_dir, write_text as _fsops_write_text
from swaag.utils import stable_json_dumps, utc_now_iso

if TYPE_CHECKING:
    from swaag.tools.base import ToolContext


@dataclass(slots=True)
class BackgroundProcessUpdate:
    record: ProcessRecord
    tool_result: ToolExecutionResult | None
    generated_events: list[ToolGeneratedEvent]
    completed: bool


class AgentEnvironment:
    def __init__(self, config: AgentConfig, session_state: SessionState):
        self.config = config
        self.session_state = session_state
        workspace_root = self._workspace_root()
        self.filesystem = FilesystemManager(config, workspace_root)
        self.workspace = WorkspaceManager(self.filesystem)
        self.process = ProcessManager()
        self.shell = ShellSession(config, process_manager=self.process)

    def initialize_events(self) -> list[ToolGeneratedEvent]:
        env = self.session_state.environment
        if env.workspace.root:
            return []
        state = self.workspace.initialize_state()
        return [
            ToolGeneratedEvent(
                "environment_initialized",
                {
                    "workspace_root": state.root,
                    "cwd": state.cwd,
                    "shell_env_overrides": {},
                    "shell_unset_vars": [],
                },
            )
        ]

    def list_files(self, path_text: str = ".") -> ToolExecutionResult:
        entries = self.filesystem.list_files(path_text, cwd=self.current_cwd)
        return ToolExecutionResult(
            tool_name="list_files",
            output={"path": path_text, "entries": entries, "count": len(entries)},
            display_text=f"list_files result: {stable_json_dumps({'path': path_text, 'entries': entries}, indent=2)}",
            generated_events=[
                ToolGeneratedEvent(
                    "filesystem_listed",
                    {
                        "path": path_text,
                        "cwd": self.current_cwd,
                        "entries": entries,
                        "count": len(entries),
                    },
                )
            ],
        )

    def read_file(self, path_text: str) -> ToolExecutionResult:
        path, text = self.filesystem.read_text(path_text, cwd=self.current_cwd)
        rel = self.filesystem.relative_path(path)
        return ToolExecutionResult(
            tool_name="read_file",
            output={"path": str(path), "relative_path": rel, "text": text, "size_chars": len(text)},
            display_text=f"read_file result: {stable_json_dumps({'path': str(path), 'size_chars': len(text)}, indent=2)}",
            generated_events=[
                ToolGeneratedEvent("file_read_requested", {"path": str(path), "reason": "read_file"}),
                ToolGeneratedEvent(
                    "filesystem_read",
                    {"path": str(path), "relative_path": rel, "text": text, "size_chars": len(text), "cwd": self.current_cwd},
                ),
            ],
        )

    def search_in_file(self, *, path_text: str, pattern: str, regex: bool = False, ignore_case: bool = False, max_matches: int = 50) -> ToolExecutionResult:
        path, matches = self.filesystem.search_in_file(
            path_text,
            cwd=self.current_cwd,
            pattern=pattern,
            regex=regex,
            ignore_case=ignore_case,
            max_matches=max_matches,
        )
        rel = self.filesystem.relative_path(path)
        output = {
            "path": str(path),
            "relative_path": rel,
            "pattern": pattern,
            "regex": regex,
            "ignore_case": ignore_case,
            "matches": matches,
            "match_count": len(matches),
        }
        return ToolExecutionResult(
            tool_name="search_in_file",
            output=output,
            display_text=f"search_in_file result: {stable_json_dumps(output, indent=2)}",
            generated_events=[
                ToolGeneratedEvent("file_read_requested", {"path": str(path), "reason": "search_in_file"}),
                ToolGeneratedEvent("filesystem_search", output | {"cwd": self.current_cwd}),
            ],
        )

    def search_repo(self, *, pattern: str, path_text: str = ".", regex: bool = False, ignore_case: bool = False, max_matches: int = 100) -> ToolExecutionResult:
        matches = self.filesystem.search_repo(
            pattern=pattern,
            path_text=path_text,
            cwd=self.current_cwd,
            regex=regex,
            ignore_case=ignore_case,
            max_matches=max_matches,
        )
        output = {
            "path": path_text,
            "pattern": pattern,
            "regex": regex,
            "ignore_case": ignore_case,
            "matches": matches,
            "match_count": len(matches),
            "matched_files": sorted({str(item["relative_path"]) for item in matches}),
        }
        return ToolExecutionResult(
            tool_name="search_repo",
            output=output,
            display_text=f"search_repo result: {stable_json_dumps(output, indent=2)}",
            generated_events=[
                ToolGeneratedEvent("repository_searched", output | {"cwd": self.current_cwd}),
            ],
        )

    def read_text_chunk(self, validated_input: dict[str, Any], context: "ToolContext") -> ToolExecutionResult:
        reader = SequentialReader(self.config)
        state = context.session_state
        generated: list[ToolGeneratedEvent] = []

        if validated_input["reader_id"] is not None:
            try:
                reader_state = state.reader_states[validated_input["reader_id"]]
            except KeyError as exc:
                raise ToolValidationError(f"Unknown reader_id: {validated_input['reader_id']}") from exc
            buffer_text = None
            if reader_state.source_kind == "buffer":
                note_id = reader_state.source_ref.removeprefix("note:")
                note = next((item for item in state.notes if item.note_id == note_id), None)
                if note is None:
                    raise ToolValidationError(f"Cannot continue note reader; note not found: {note_id}")
                buffer_text = note.content
        elif validated_input["path"] is not None:
            resolved = self.filesystem.resolve_path(validated_input["path"], cwd=self.current_cwd)
            reader_state = reader.open_file(
                str(resolved),
                chunk_chars=validated_input["chunk_chars"],
                overlap_chars=validated_input["overlap_chars"],
            )
            generated.append(ToolGeneratedEvent("reader_opened", {"reader_state": asdict(reader_state)}))
            generated.append(ToolGeneratedEvent("file_read_requested", {"path": str(resolved), "reason": "open_reader", "reader_id": reader_state.reader_id, "offset": 0}))
            buffer_text = None
        else:
            note = next((item for item in state.notes if item.note_id == validated_input["note_id"]), None)
            if note is None:
                raise ToolValidationError(f"Unknown note_id: {validated_input['note_id']}")
            reader_state = reader.open_buffer(
                f"note:{note.note_id}",
                chunk_chars=validated_input["chunk_chars"],
                overlap_chars=validated_input["overlap_chars"],
            )
            generated.append(ToolGeneratedEvent("reader_opened", {"reader_state": asdict(reader_state)}))
            generated.append(ToolGeneratedEvent("buffer_read_requested", {"source_ref": f"note:{note.note_id}", "reason": "open_reader", "reader_id": reader_state.reader_id, "offset": 0}))
            buffer_text = note.content

        if reader_state.source_kind == "file":
            _, _ = self.filesystem.read_text(reader_state.source_ref, cwd=self.current_cwd)
        chunk, updated_state = reader.read_next(reader_state, buffer_text=buffer_text)
        generated.append(ToolGeneratedEvent("reader_chunk_read", {"reader_state": asdict(updated_state), "chunk": asdict(chunk)}))
        event_type = "file_chunk_read" if chunk.source_kind == "file" else "buffer_chunk_read"
        generated.append(ToolGeneratedEvent(event_type, {"reader_id": chunk.reader_id, "source_ref": chunk.source_ref, "start_offset": chunk.start_offset, "end_offset": chunk.end_offset, "next_offset": chunk.next_offset, "finished": chunk.finished, "text": chunk.text}))
        if chunk.source_kind == "file":
            generated.append(ToolGeneratedEvent("filesystem_read", {"path": chunk.source_ref, "relative_path": self.filesystem.relative_path(Path(chunk.source_ref)), "text": chunk.text, "size_chars": len(chunk.text), "cwd": self.current_cwd, "reader_id": chunk.reader_id, "start_offset": chunk.start_offset, "end_offset": chunk.end_offset}))
        output = {
            "reader_id": chunk.reader_id,
            "source_kind": chunk.source_kind,
            "source_ref": chunk.source_ref,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "next_offset": chunk.next_offset,
            "finished": chunk.finished,
            "text": chunk.text,
        }
        return ToolExecutionResult(tool_name="read_text", output=output, display_text=f"read_text result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def preview_or_apply_edit(self, validated_input: dict[str, Any], context: "ToolContext") -> ToolExecutionResult:
        path = self.filesystem.resolve_path(validated_input["path"], cwd=self.current_cwd)
        _, original_text = self.filesystem.read_text(str(path), cwd=self.current_cwd)
        kwargs = {key: validated_input[key] for key in ["start", "end", "position", "replacement", "insertion", "pattern"] if key in validated_input}
        preview = TextEditor.apply(original_text, validated_input["operation"], **kwargs)
        payload = {
            "path": str(path),
            "operation": validated_input["operation"],
            "details": preview.details,
            "changed": preview.changed,
            "diff": preview.diff,
            "new_text": preview.new_text,
            "original_text": preview.original_text,
        }
        generated = [
            ToolGeneratedEvent("file_read_requested", {"path": str(path), "reason": "edit_preview"}),
            ToolGeneratedEvent("file_read_for_edit", {"path": str(path), "size_chars": len(preview.original_text), "text": preview.original_text}),
        ]
        if validated_input.get("dry_run", False) or not preview.changed:
            generated.append(ToolGeneratedEvent("edit_previewed", payload))
        else:
            if not context.config.editor.allow_writes:
                raise PermissionError("edit_text writes are disabled by editor policy")
            generated.append(
                ToolGeneratedEvent(
                    "edit_applied",
                    payload,
                    derived_writes=[
                        DerivedFileWrite(
                            path=str(path),
                            content=preview.new_text,
                            backup_content=preview.original_text if context.config.editor.create_backups else None,
                            backup_suffix=context.config.editor.backup_suffix,
                        )
                    ],
                )
            )
        output = {"path": str(path), "operation": validated_input["operation"], "changed": preview.changed, "diff": preview.diff, "details": preview.details}
        return ToolExecutionResult(tool_name="edit_text", output=output, display_text=f"edit_text result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def write_file(self, path_text: str, content: str, *, create: bool = True) -> ToolExecutionResult:
        path = self.filesystem.resolve_path(path_text, cwd=self.current_cwd)
        original_text = ""
        if path.exists():
            _, original_text = self.filesystem.read_text(str(path), cwd=self.current_cwd)
        elif not create:
            raise ToolValidationError(f"write_file target does not exist: {path}")
        generated = [
            ToolGeneratedEvent("file_read_requested", {"path": str(path), "reason": "write_file"}),
            ToolGeneratedEvent("file_read_for_edit", {"path": str(path), "size_chars": len(original_text), "text": original_text}),
            ToolGeneratedEvent(
                "edit_applied",
                {
                    "path": str(path),
                    "operation": "write_file",
                    "details": {"create": create},
                    "changed": original_text != content,
                    "diff": "",
                    "new_text": content,
                    "original_text": original_text,
                },
                derived_writes=[
                    DerivedFileWrite(
                        path=str(path),
                        content=content,
                        backup_content=original_text if original_text and self.config.editor.create_backups else None,
                        backup_suffix=self.config.editor.backup_suffix,
                    )
                ],
            ),
        ]
        output = {"path": str(path), "written": True, "size_chars": len(content)}
        return ToolExecutionResult(tool_name="write_file", output=output, display_text=f"write_file result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def inspect_diff(self, path_text: str) -> ToolExecutionResult:
        path = self.filesystem.resolve_path(path_text, cwd=self.current_cwd)
        rel = self.filesystem.relative_path(path)
        actual_text = path.read_text(encoding="utf-8") if path.exists() else ""
        view = self.session_state.file_views.get(str(path)) or self.session_state.file_views.get(rel)
        remembered_text = self.session_state.environment.workspace.known_files.get(rel, "")
        diff = ""
        source = "workspace_snapshot"
        if view is not None:
            preview_diff = str(view.metadata.get("pending_diff") or view.metadata.get("last_preview_diff") or "")
            if preview_diff:
                diff = preview_diff
                source = view.last_operation or "file_view"
            elif view.content is not None:
                diff = "".join(
                    unified_diff(
                        view.content.splitlines(keepends=True),
                        actual_text.splitlines(keepends=True),
                        fromfile="remembered",
                        tofile="current",
                    )
                )
                source = "file_view_content"
        if not diff:
            diff = "".join(
                unified_diff(
                    remembered_text.splitlines(keepends=True),
                    actual_text.splitlines(keepends=True),
                    fromfile="workspace_snapshot",
                    tofile="current",
                )
            )
        output = {
            "path": str(path),
            "relative_path": rel,
            "changed": bool(diff),
            "diff": diff,
            "baseline_source": source,
        }
        return ToolExecutionResult(
            tool_name="inspect_diff",
            output=output,
            display_text=f"inspect_diff result: {stable_json_dumps(output, indent=2)}",
            generated_events=[ToolGeneratedEvent("diff_inspected", output)],
        )

    def list_changes(self) -> ToolExecutionResult:
        workspace = self.session_state.environment.workspace
        output = {
            "cwd": self.current_cwd,
            "created_files": list(workspace.created_files),
            "modified_files": list(workspace.modified_files),
            "deleted_files": list(workspace.deleted_files),
        }
        return ToolExecutionResult(
            tool_name="list_changes",
            output=output,
            display_text=f"list_changes result: {stable_json_dumps(output, indent=2)}",
            generated_events=[ToolGeneratedEvent("changes_listed", output)],
        )

    def workspace_snapshot(self) -> ToolExecutionResult:
        snapshot = self.filesystem.snapshot()
        output = {
            "workspace_root": str(self.filesystem.workspace_root),
            "cwd": self.current_cwd,
            "files": snapshot,
            "file_count": len(snapshot),
            "created_files": list(self.session_state.environment.workspace.created_files),
            "modified_files": list(self.session_state.environment.workspace.modified_files),
            "deleted_files": list(self.session_state.environment.workspace.deleted_files),
        }
        return ToolExecutionResult(
            tool_name="workspace_snapshot",
            output=output,
            display_text=f"workspace_snapshot result: {stable_json_dumps({'workspace_root': output['workspace_root'], 'file_count': output['file_count']}, indent=2)}",
            generated_events=[ToolGeneratedEvent("workspace_snapshot_inspected", output | {"captured_at": utc_now_iso()})],
        )

    def detect_stuck_patterns(self, history_events: list[Any]) -> list[str]:
        repeated_failures: list[str] = []
        recent_commands = [
            event.payload
            for event in history_events
            if event.event_type == "shell_command_completed"
        ][-3:]
        if len(recent_commands) >= 2:
            same_command = len({item.get("command") for item in recent_commands}) == 1
            same_exit = len({item.get("exit_code") for item in recent_commands}) == 1
            no_changes = all(
                not item.get("created_files") and not item.get("modified_files") and not item.get("deleted_files")
                for item in recent_commands
            )
            if same_command and same_exit and no_changes:
                repeated_failures.append("repeated_useless_command")
        recent_edits = [
            event.payload
            for event in history_events
            if event.event_type in {"edit_previewed", "edit_applied"}
        ][-3:]
        if recent_edits and all(not bool(item.get("changed", True)) for item in recent_edits):
            repeated_failures.append("repeated_noop_edit")
        recent_tests = [
            event.payload
            for event in history_events
            if event.event_type == "process_completed" and event.payload.get("metadata", {}).get("kind") == "run_tests"
        ][-3:]
        if len(recent_tests) >= 2:
            same_test = len({tuple(item.get("command", [])) for item in recent_tests}) == 1
            all_failed = all(item.get("return_code") not in {0, None} for item in recent_tests)
            if same_test and all_failed:
                repeated_failures.append("repeated_failed_tests_without_change")
        return repeated_failures

    def run_shell_command(self, command: str, *, background: bool = False) -> ToolExecutionResult:
        if background:
            return self._start_background_shell_command(command)
        before = self.filesystem.snapshot()
        result, updated_shell = self.shell.execute(self.session_state.environment.shell, command, workspace_root=self.filesystem.workspace_root)
        after = self.filesystem.snapshot()
        snapshot = self.workspace.snapshot(before=before, after=after, cwd=result.cwd_after)
        process_record = asdict(result.process_result.record)
        generated = [
            ToolGeneratedEvent("process_started", process_record),
            ToolGeneratedEvent("shell_command_started", {"command": command, "cwd": result.cwd_before}),
            ToolGeneratedEvent("process_completed", process_record | {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.exit_code}),
            ToolGeneratedEvent(
                "shell_command_completed",
                {
                    "command": command,
                    "cwd_before": result.cwd_before,
                    "cwd_after": result.cwd_after,
                    "env_overrides": result.env_overrides,
                    "unset_vars": result.unset_vars,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            ),
            ToolGeneratedEvent(
                "workspace_snapshot",
                {
                    "workspace_root": snapshot.root,
                    "cwd": snapshot.cwd,
                    "snapshot_mode": "delta",
                    "files": snapshot.files,
                    "created_files": snapshot.created_files,
                    "modified_files": snapshot.modified_files,
                    "deleted_files": snapshot.deleted_files,
                    "captured_at": snapshot.captured_at,
                },
            ),
        ]
        output = {
            "command": command,
            "cwd_before": result.cwd_before,
            "cwd_after": result.cwd_after,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "created_files": snapshot.created_files,
            "modified_files": snapshot.modified_files,
            "deleted_files": snapshot.deleted_files,
            "background": False,
        }
        return ToolExecutionResult(tool_name="shell_command", output=output, display_text=f"shell_command result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def run_tests(self, command: list[str], *, background: bool = False) -> ToolExecutionResult:
        if not command:
            raise ToolValidationError("run_tests.command must not be empty")
        if background:
            return self._start_background_tests(command)
        process_result = self.process.run(
            command,
            cwd=Path(self.current_cwd),
            env=self.effective_env,
            timeout_seconds=self.config.runtime.tool_timeout_seconds,
            metadata={"kind": "run_tests"},
        )
        record = asdict(process_result.record)
        generated = [
            ToolGeneratedEvent("process_started", record),
            ToolGeneratedEvent("process_completed", record | {"stdout": process_result.stdout, "stderr": process_result.stderr, "return_code": process_result.record.return_code}),
        ]
        output = {
            "command": command,
            "cwd": self.current_cwd,
            "exit_code": process_result.record.return_code,
            "stdout": process_result.stdout,
            "stderr": process_result.stderr,
            "passed": process_result.record.return_code == 0,
            "background": False,
        }
        return ToolExecutionResult(tool_name="run_tests", output=output, display_text=f"run_tests result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def _start_background_tests(self, command: list[str]) -> ToolExecutionResult:
        record = self.process.start_background(
            command,
            cwd=Path(self.current_cwd),
            env=self.effective_env,
            timeout_seconds=self.config.runtime.tool_timeout_seconds,
            artifacts_dir=self._process_artifacts_root(),
            metadata={"kind": "run_tests", "background": "true"},
        )
        record_payload = asdict(record)
        output = {
            "command": command,
            "cwd": self.current_cwd,
            "background": True,
            "process_id": record.process_id,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "passed": False,
        }
        return ToolExecutionResult(
            tool_name="run_tests",
            output=output,
            display_text=f"run_tests started in background: {stable_json_dumps(output, indent=2)}",
            generated_events=[ToolGeneratedEvent("process_started", record_payload)],
            completed=False,
        )

    def _start_background_shell_command(self, command: str) -> ToolExecutionResult:
        before = self.filesystem.snapshot()
        record = self.process.start_background(
            ["bash", "-lc", command],
            cwd=Path(self.current_cwd),
            env=self.effective_env,
            timeout_seconds=self.config.runtime.tool_timeout_seconds,
            artifacts_dir=self._process_artifacts_root(),
            metadata={
                "kind": "shell_command",
                "background": "true",
                "shell_command": command,
                "cwd_before": self.current_cwd,
            },
        )
        before_snapshot_path = Path(self._process_artifacts_root()) / record.process_id / "before_snapshot.json"
        _fsops_write_text(before_snapshot_path, json.dumps(before), encoding="utf-8")
        record.metadata["before_snapshot_path"] = str(before_snapshot_path)
        record_payload = asdict(record)
        output = {
            "command": command,
            "cwd_before": self.current_cwd,
            "cwd_after": self.current_cwd,
            "background": True,
            "process_id": record.process_id,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "created_files": [],
            "modified_files": [],
            "deleted_files": [],
        }
        generated = [
            ToolGeneratedEvent("process_started", record_payload),
            ToolGeneratedEvent("shell_command_started", {"command": command, "cwd": self.current_cwd, "background": True}),
        ]
        return ToolExecutionResult(
            tool_name="shell_command",
            output=output,
            display_text=f"shell_command started in background: {stable_json_dumps({'command': command, 'process_id': record.process_id}, indent=2)}",
            generated_events=generated,
            completed=False,
        )

    def _finalize_background_shell_command(self, record: ProcessRecord) -> tuple[ToolExecutionResult, list[ToolGeneratedEvent]]:
        before_snapshot_path = record.metadata.get("before_snapshot_path", "")
        before = {}
        if before_snapshot_path:
            snapshot_path = Path(before_snapshot_path)
            if snapshot_path.exists():
                before = json.loads(snapshot_path.read_text(encoding="utf-8"))
        after = self.filesystem.snapshot()
        snapshot = self.workspace.snapshot(
            before=before,
            after=after,
            cwd=record.metadata.get("cwd_before", record.cwd),
        )
        generated = [
            ToolGeneratedEvent(
                "shell_command_completed",
                {
                    "command": record.metadata.get("shell_command", " ".join(record.command)),
                    "cwd_before": record.metadata.get("cwd_before", record.cwd),
                    "cwd_after": record.metadata.get("cwd_before", record.cwd),
                    "env_overrides": {},
                    "unset_vars": [],
                    "exit_code": record.return_code,
                    "stdout": record.stdout,
                    "stderr": record.stderr,
                    "background": True,
                },
            ),
            ToolGeneratedEvent(
                "workspace_snapshot",
                {
                    "workspace_root": snapshot.root,
                    "cwd": snapshot.cwd,
                    "snapshot_mode": "delta",
                    "files": snapshot.files,
                    "created_files": snapshot.created_files,
                    "modified_files": snapshot.modified_files,
                    "deleted_files": snapshot.deleted_files,
                    "captured_at": snapshot.captured_at,
                },
            ),
        ]
        output = {
            "command": record.metadata.get("shell_command", " ".join(record.command)),
            "cwd_before": record.metadata.get("cwd_before", record.cwd),
            "cwd_after": record.metadata.get("cwd_before", record.cwd),
            "exit_code": record.return_code,
            "stdout": record.stdout,
            "stderr": record.stderr,
            "created_files": snapshot.created_files,
            "modified_files": snapshot.modified_files,
            "deleted_files": snapshot.deleted_files,
            "background": True,
            "process_id": record.process_id,
        }
        return (
            ToolExecutionResult(
                tool_name="shell_command",
                output=output,
                display_text=f"shell_command result: {stable_json_dumps(output, indent=2)}",
            ),
            generated,
        )

    def poll_background_process(self, process_id: str) -> BackgroundProcessUpdate:
        try:
            existing = self.session_state.environment.processes[process_id]
        except KeyError as exc:
            raise ToolValidationError(f"Unknown process_id: {process_id}") from exc
        poll = self.process.poll(existing)
        record_payload = asdict(poll.record)
        generated = [
            ToolGeneratedEvent(
                "process_polled",
                record_payload | {"completed": poll.completed},
            )
        ]
        tool_result: ToolExecutionResult | None = None
        if poll.completed and poll.status_changed:
            if poll.record.status == "timed_out":
                generated.append(
                    ToolGeneratedEvent(
                        "process_timed_out",
                        record_payload | {"stdout": poll.stdout, "stderr": poll.stderr, "return_code": poll.record.return_code},
                    )
                )
            elif poll.record.status == "killed":
                generated.append(ToolGeneratedEvent("process_killed", record_payload))
            else:
                generated.append(
                    ToolGeneratedEvent(
                        "process_completed",
                        record_payload | {"stdout": poll.stdout, "stderr": poll.stderr, "return_code": poll.record.return_code},
                    )
                )
            kind = poll.record.metadata.get("kind", "")
            if kind == "run_tests":
                tool_result = ToolExecutionResult(
                    tool_name="run_tests",
                    output={
                        "command": list(poll.record.command),
                        "cwd": poll.record.cwd,
                        "exit_code": poll.record.return_code,
                        "stdout": poll.stdout,
                        "stderr": poll.stderr,
                        "passed": poll.record.return_code == 0,
                        "background": True,
                        "process_id": poll.record.process_id,
                    },
                    display_text=f"run_tests result: {stable_json_dumps({'command': poll.record.command, 'exit_code': poll.record.return_code, 'passed': poll.record.return_code == 0, 'background': True}, indent=2)}",
                )
            elif kind == "shell_command":
                tool_result, shell_events = self._finalize_background_shell_command(poll.record)
                generated.extend(shell_events)
        return BackgroundProcessUpdate(
            record=poll.record,
            tool_result=tool_result,
            generated_events=generated,
            completed=poll.completed,
        )

    def kill_background_process(self, process_id: str) -> BackgroundProcessUpdate:
        try:
            existing = self.session_state.environment.processes[process_id]
        except KeyError as exc:
            raise ToolValidationError(f"Unknown process_id: {process_id}") from exc
        killed = self.process.kill(existing)
        record_payload = asdict(killed)
        generated = [ToolGeneratedEvent("process_killed", record_payload)]
        return BackgroundProcessUpdate(record=killed, tool_result=None, generated_events=generated, completed=True)

    def browser_search(self, *, query: str, engine: str, limit: int) -> ToolExecutionResult:
        result = run_aubro_command(
            config=self.config,
            process_manager=self.process,
            command_suffix=["search", query, "--engine", engine, "--limit", str(limit), "--headless"],
            cwd=Path(self.current_cwd),
            env=self.effective_env,
        )
        payload = result.payload
        raw_results = payload.get("results", [])
        raw_attempts = payload.get("attempts", [])
        if not isinstance(raw_results, list):
            raise ToolValidationError("aubro search returned invalid results payload")
        if not isinstance(raw_attempts, list):
            raise ToolValidationError("aubro search returned invalid attempts payload")
        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", ""))[: self.config.environment.aubro_max_text_chars],
            }
            for item in raw_results[: self.config.environment.aubro_max_results]
            if isinstance(item, dict)
        ]
        attempts = [
            {
                "engine": str(item.get("engine", "")),
                "url": str(item.get("url", "")),
                "results": int(item.get("results", 0)),
                "blocked": bool(item.get("blocked", False)),
            }
            for item in raw_attempts[: self.config.environment.aubro_max_results]
            if isinstance(item, dict)
        ]
        output = {
            "query": str(payload.get("query", query)),
            "engine": str(payload.get("engine", engine)),
            "url": str(payload.get("url", "")),
            "result_count": len(results),
            "results": results,
            "attempts": attempts,
        }
        record = asdict(result.process_result.record)
        generated = [
            ToolGeneratedEvent("process_started", record),
            ToolGeneratedEvent("process_completed", record | {"stdout": result.process_result.stdout, "stderr": result.process_result.stderr, "return_code": result.process_result.record.return_code}),
        ]
        return ToolExecutionResult(tool_name="browser_search", output=output, display_text=f"browser_search result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def browser_browse(self, *, url: str) -> ToolExecutionResult:
        result = run_aubro_command(
            config=self.config,
            process_manager=self.process,
            command_suffix=["browse", url, "--headless"],
            cwd=Path(self.current_cwd),
            env=self.effective_env,
        )
        payload = result.payload
        raw_links = payload.get("links", [])
        if not isinstance(raw_links, list):
            raise ToolValidationError("aubro browse returned invalid links payload")
        links = [
            {
                "text": str(item.get("text", ""))[: self.config.environment.aubro_max_text_chars],
                "href": str(item.get("href", "")),
            }
            for item in raw_links[: self.config.environment.aubro_max_links]
            if isinstance(item, dict)
        ]
        output = {
            "url": str(payload.get("url", url)),
            "title": str(payload.get("title", "")),
            "backend": str(payload.get("backend", "")),
            "blocked": bool(payload.get("blocked", False)),
            "block_reason": "" if payload.get("block_reason") in {None, ""} else str(payload.get("block_reason")),
            "text_excerpt": str(payload.get("text", ""))[: self.config.environment.aubro_max_text_chars],
            "link_count": len(raw_links),
            "links": links,
            "form_count": len(payload.get("forms", [])) if isinstance(payload.get("forms"), list) else 0,
            "button_count": len(payload.get("buttons", [])) if isinstance(payload.get("buttons"), list) else 0,
        }
        record = asdict(result.process_result.record)
        generated = [
            ToolGeneratedEvent("process_started", record),
            ToolGeneratedEvent("process_completed", record | {"stdout": result.process_result.stdout, "stderr": result.process_result.stderr, "return_code": result.process_result.record.return_code}),
        ]
        return ToolExecutionResult(tool_name="browser_browse", output=output, display_text=f"browser_browse result: {stable_json_dumps(output, indent=2)}", generated_events=generated)

    def browser_tools_available(self) -> bool:
        return aubro_available(self.config)

    @property
    def current_cwd(self) -> str:
        env = self.session_state.environment
        return env.shell.cwd or env.workspace.cwd or str(self.filesystem.workspace_root)

    @property
    def effective_env(self) -> dict[str, str]:
        import os
        env = os.environ.copy()
        shell_state = self.session_state.environment.shell
        env.update(shell_state.env_overrides)
        for key in shell_state.unset_vars:
            env.pop(key, None)
        return env

    def _workspace_root(self) -> Path:
        roots = list(self.config.tools.read_roots)
        if roots:
            return roots[0].resolve()
        return Path.cwd().resolve()

    def _process_artifacts_root(self) -> Path:
        root = Path(self.config.sessions.root)
        if not root.is_absolute():
            root = (self.filesystem.workspace_root / root).resolve()
        target = root / self.session_state.session_id / "processes"
        ensure_dir(target)
        return target
