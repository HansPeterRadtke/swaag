from __future__ import annotations

import ast
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swaag.editing import EditError, TextEditor
from swaag.notes import compact_notes, enforce_limits, make_note, render_notes
from swaag.reader import ReaderError, SequentialReader
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import DerivedFileWrite, ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the provided text exactly."
    kind = "pure"
    requires_artifacts = ("text",)
    provides_artifacts = ("text",)
    allowed_followers = ("respond",)
    output_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        text = raw_input.get("text")
        if not isinstance(text, str):
            raise ToolValidationError("echo.text must be a string")
        return {"text": text}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return ToolExecutionResult(tool_name=self.name, output={"text": validated_input["text"]}, display_text=tool_result_display(self.name, {"text": validated_input["text"]}))


class TimeNowTool(Tool):
    name = "time_now"
    description = "Return the current local and UTC time from the machine running the agent."
    kind = "pure"
    provides_artifacts = ("time_info",)
    allowed_followers = ("respond",)
    output_schema = {
        "type": "object",
        "properties": {
            "local_time": {"type": "string"},
            "utc_time": {"type": "string"},
            "timezone": {"type": "string"},
        },
        "required": ["local_time", "utc_time", "timezone"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        if raw_input not in ({}, None):
            if raw_input:
                raise ToolValidationError("time_now takes no arguments")
        return {}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)
        output = {
            "local_time": now_local.isoformat(),
            "utc_time": now_utc.isoformat(),
            "timezone": str(now_local.tzinfo),
        }
        return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output))


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression using +, -, *, /, //, %, **, unary +/-, and parentheses."
    kind = "pure"
    requires_artifacts = ("expression",)
    provides_artifacts = ("numeric_result",)
    allowed_followers = ("respond",)
    output_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "result": {"type": "number"},
        },
        "required": ["expression", "result"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    }

    _allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Constant,
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        expression = raw_input.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolValidationError("calculator.expression must be a non-empty string")
        return {"expression": expression.strip()}

    def _safe_eval(self, expression: str) -> Any:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, self._allowed_nodes):
                raise ToolValidationError(f"Unsupported calculator syntax: {node.__class__.__name__}")
        return eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        result = self._safe_eval(validated_input["expression"])
        output = {"expression": validated_input["expression"], "result": result}
        return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output))


class ReadTextTool(Tool):
    name = "read_text"
    description = "Read a local file or note in bounded sequential chunks with continuation state."
    kind = "stateful"
    requires_artifacts = ("path_or_reader",)
    provides_artifacts = ("text", "path")
    allowed_followers = ("notes", "edit_text", "echo", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "reader_id": {"type": "string"},
            "source_kind": {"type": "string", "enum": ["file", "buffer"]},
            "source_ref": {"type": "string"},
            "start_offset": {"type": "integer", "minimum": 0},
            "end_offset": {"type": "integer", "minimum": 0},
            "next_offset": {"type": "integer", "minimum": 0},
            "finished": {"type": "boolean"},
            "text": {"type": "string"},
        },
        "required": ["reader_id", "source_kind", "source_ref", "start_offset", "end_offset", "next_offset", "finished", "text"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "note_id": {"type": "string"},
            "reader_id": {"type": "string"},
            "chunk_chars": {"type": "integer", "minimum": 1},
            "overlap_chars": {"type": "integer", "minimum": 0},
        },
        "required": [],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        allowed = {"path", "note_id", "reader_id", "chunk_chars", "overlap_chars"}
        if not set(raw_input).issubset(allowed):
            raise ToolValidationError("read_text received unknown arguments")
        path = raw_input.get("path")
        note_id = raw_input.get("note_id")
        reader_id = raw_input.get("reader_id")
        chunk_chars = raw_input.get("chunk_chars")
        overlap_chars = raw_input.get("overlap_chars")
        if sum(value is not None for value in [path, note_id, reader_id]) != 1:
            raise ToolValidationError("read_text requires exactly one of path, note_id, or reader_id")
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ToolValidationError("read_text.path must be a non-empty string")
        if note_id is not None and (not isinstance(note_id, str) or not note_id.strip()):
            raise ToolValidationError("read_text.note_id must be a non-empty string")
        if reader_id is not None and (not isinstance(reader_id, str) or not reader_id.strip()):
            raise ToolValidationError("read_text.reader_id must be a non-empty string")
        if chunk_chars is not None and (not isinstance(chunk_chars, int) or chunk_chars <= 0):
            raise ToolValidationError("read_text.chunk_chars must be a positive integer")
        if overlap_chars is not None and (not isinstance(overlap_chars, int) or overlap_chars < 0):
            raise ToolValidationError("read_text.overlap_chars must be a non-negative integer")
        return {
            "path": path,
            "note_id": note_id,
            "reader_id": reader_id,
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
        }

    def pre_execute_events(self, validated_input: dict[str, Any], context: ToolContext) -> list[ToolGeneratedEvent]:
        return []

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        required = {"reader_chunk_read"}
        if validated_input["path"] is not None:
            required.add("file_chunk_read")
        elif validated_input["note_id"] is not None:
            required.add("buffer_chunk_read")
        return required

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.read_text_chunk(validated_input, context)


class NotesTool(Tool):
    name = "notes"
    description = "List, add, replace, and compact durable working notes with strict size limits."
    kind = "stateful"
    requires_artifacts = ("note_action",)
    provides_artifacts = ("notes_state", "text")
    allowed_followers = ("calculator", "echo", "edit_text", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "object"}},
            "removed_note_ids": {"type": "array", "items": {"type": "string"}},
            "compacted_note": {"type": "object"},
            "compacted": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "add", "replace", "compact"]},
            "note_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        action = raw_input.get("action")
        if action not in {"list", "add", "replace", "compact"}:
            raise ToolValidationError("notes.action must be one of list, add, replace, compact")
        note_id = raw_input.get("note_id")
        title = raw_input.get("title")
        content = raw_input.get("content")
        if action == "add" and (not isinstance(title, str) or not isinstance(content, str)):
            raise ToolValidationError("notes add requires string title and content")
        if action == "replace" and (not isinstance(note_id, str) or not isinstance(title, str) or not isinstance(content, str)):
            raise ToolValidationError("notes replace requires note_id, title, and content")
        if action == "compact" and any(raw_input.get(name) is not None for name in ["note_id", "title", "content"]):
            raise ToolValidationError("notes compact takes only action")
        if action == "list" and any(raw_input.get(name) is not None for name in ["note_id", "title", "content"]):
            raise ToolValidationError("notes list takes only action")
        return {"action": action, "note_id": note_id, "title": title, "content": content}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        state = context.session_state
        action = validated_input["action"]
        generated: list[ToolGeneratedEvent] = []

        if action == "list":
            output = {"notes": [{"note_id": note.note_id, "title": note.title, "content": note.content} for note in state.notes]}
            return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output))

        if action == "add":
            note = make_note(context.config, title=validated_input["title"], content=validated_input["content"])
            generated.append(ToolGeneratedEvent("note_added", {"note": asdict(note)}))
            notes_after = enforce_limits(context.config, [*state.notes, note])
            if len(notes_after) < len(state.notes) + 1:
                compaction = compact_notes(context.config, [*state.notes, note])
                if compaction is not None:
                    removed_ids, compacted_note = compaction
                    generated.append(
                        ToolGeneratedEvent(
                            "notes_compacted",
                            {"removed_note_ids": removed_ids, "compacted_note": asdict(compacted_note)},
                        )
                    )
            output = {"note_id": note.note_id, "title": note.title, "content": note.content}
            return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output), generated_events=generated)

        if action == "replace":
            existing = next((note for note in state.notes if note.note_id == validated_input["note_id"]), None)
            if existing is None:
                raise ToolValidationError(f"Unknown note_id: {validated_input['note_id']}")
            replacement = make_note(
                context.config,
                title=validated_input["title"],
                content=validated_input["content"],
                note_id=existing.note_id,
            )
            replacement.created_at = existing.created_at
            generated.append(ToolGeneratedEvent("note_replaced", {"note": asdict(replacement)}))
            output = {"note_id": replacement.note_id, "title": replacement.title, "content": replacement.content}
            return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output), generated_events=generated)

        compaction = compact_notes(context.config, state.notes)
        if compaction is None:
            output = {"notes": [{"note_id": note.note_id, "title": note.title, "content": note.content} for note in state.notes], "compacted": False}
            return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output))
        removed_ids, compacted_note = compaction
        generated.append(ToolGeneratedEvent("notes_compacted", {"removed_note_ids": removed_ids, "compacted_note": asdict(compacted_note)}))
        output = {"removed_note_ids": removed_ids, "compacted_note": asdict(compacted_note), "compacted": True}
        return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output), generated_events=generated)


class EditTextTool(Tool):
    name = "edit_text"
    description = "Preview or apply a bounded text edit to a local UTF-8 text file."
    kind = "side_effect"
    requires_artifacts = ("path", "text")
    provides_artifacts = ("edited_text", "path")
    allowed_followers = ("respond", "read_text")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "operation": {"type": "string"},
            "changed": {"type": "boolean"},
            "diff": {"type": "string"},
            "details": {"type": "object"},
        },
        "required": ["path", "operation", "changed", "diff", "details"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "operation": {"type": "string", "enum": ["replace_range", "insert_at", "delete_range", "replace_pattern_once", "replace_pattern_all"]},
            "dry_run": {"type": "boolean"},
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "position": {"type": "integer", "minimum": 0},
            "replacement": {"type": "string"},
            "insertion": {"type": "string"},
            "pattern": {"type": "string"},
        },
        "required": ["path", "operation"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        operation = raw_input.get("operation")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("edit_text.path must be a non-empty string")
        if operation not in {"replace_range", "insert_at", "delete_range", "replace_pattern_once", "replace_pattern_all"}:
            raise ToolValidationError("edit_text.operation is invalid")
        validated = {"path": path, "operation": operation, "dry_run": bool(raw_input.get("dry_run", False))}
        for field in ["start", "end", "position"]:
            value = raw_input.get(field)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ToolValidationError(f"edit_text.{field} must be a non-negative integer")
            if value is not None:
                validated[field] = value
        for field in ["replacement", "insertion", "pattern"]:
            value = raw_input.get(field)
            if value is not None and not isinstance(value, str):
                raise ToolValidationError(f"edit_text.{field} must be a string")
            if value is not None:
                validated[field] = value
        return validated

    def effective_kind(self, validated_input: dict[str, Any]) -> str:
        return "stateful" if validated_input.get("dry_run", False) else self.kind

    def pre_execute_events(self, validated_input: dict[str, Any], context: ToolContext) -> list[ToolGeneratedEvent]:
        return []

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        required = {"file_read_for_edit"}
        if validated_input.get("dry_run", False):
            required.add("edit_previewed")
        else:
            required.add("edit_applied")
        return required

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.preview_or_apply_edit(validated_input, context)


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files visible in the persistent workspace."
    kind = "stateful"
    requires_artifacts = ("path",)
    provides_artifacts = ("file_list",)
    allowed_followers = ("read_file", "read_text", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "entries": {"type": "array", "items": {"type": "string"}},
            "count": {"type": "integer"},
        },
        "required": ["path", "entries", "count"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("list_files.path must be a non-empty string")
        return {"path": path}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"filesystem_listed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.list_files(validated_input["path"])


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a full UTF-8 file from the persistent workspace."
    kind = "stateful"
    requires_artifacts = ("path",)
    provides_artifacts = ("text", "path")
    allowed_followers = ("write_file", "edit_text", "respond", "run_tests")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "relative_path": {"type": "string"},
            "text": {"type": "string"},
            "size_chars": {"type": "integer"},
        },
        "required": ["path", "relative_path", "text", "size_chars"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("read_file.path must be a non-empty string")
        return {"path": path}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"file_read_requested", "filesystem_read"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.read_file(validated_input["path"])


class SearchInFileTool(Tool):
    name = "search_in_file"
    description = "Search one workspace file for a literal string or regex and return exact match locations."
    kind = "stateful"
    requires_artifacts = ("path", "pattern")
    provides_artifacts = ("matches",)
    allowed_followers = ("read_file", "read_text", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "relative_path": {"type": "string"},
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
            "matches": {"type": "array", "items": {"type": "object"}},
            "match_count": {"type": "integer"},
        },
        "required": ["path", "relative_path", "pattern", "regex", "ignore_case", "matches", "match_count"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
            "max_matches": {"type": "integer"},
        },
        "required": ["path", "pattern"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        pattern = raw_input.get("pattern")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("search_in_file.path must be a non-empty string")
        if not isinstance(pattern, str) or not pattern:
            raise ToolValidationError("search_in_file.pattern must be a non-empty string")
        max_matches = raw_input.get("max_matches", 50)
        if not isinstance(max_matches, int) or max_matches <= 0:
            raise ToolValidationError("search_in_file.max_matches must be a positive integer")
        return {
            "path": path.strip(),
            "pattern": pattern,
            "regex": bool(raw_input.get("regex", False)),
            "ignore_case": bool(raw_input.get("ignore_case", False)),
            "max_matches": max_matches,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"file_read_requested", "filesystem_search"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.search_in_file(
            path_text=validated_input["path"],
            pattern=validated_input["pattern"],
            regex=validated_input["regex"],
            ignore_case=validated_input["ignore_case"],
            max_matches=validated_input["max_matches"],
        )


class SearchRepoTool(Tool):
    name = "search_repo"
    description = "Search across workspace files for a literal string or regex and return exact matches."
    kind = "stateful"
    requires_artifacts = ("pattern",)
    provides_artifacts = ("matches", "paths")
    allowed_followers = ("read_file", "read_text", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
            "matches": {"type": "array", "items": {"type": "object"}},
            "match_count": {"type": "integer"},
            "matched_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["path", "pattern", "regex", "ignore_case", "matches", "match_count", "matched_files"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
            "max_matches": {"type": "integer"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        pattern = raw_input.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolValidationError("search_repo.pattern must be a non-empty string")
        path = raw_input.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("search_repo.path must be a non-empty string")
        max_matches = raw_input.get("max_matches", 100)
        if not isinstance(max_matches, int) or max_matches <= 0:
            raise ToolValidationError("search_repo.max_matches must be a positive integer")
        return {
            "path": path.strip(),
            "pattern": pattern,
            "regex": bool(raw_input.get("regex", False)),
            "ignore_case": bool(raw_input.get("ignore_case", False)),
            "max_matches": max_matches,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"repository_searched"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.search_repo(
            pattern=validated_input["pattern"],
            path_text=validated_input["path"],
            regex=validated_input["regex"],
            ignore_case=validated_input["ignore_case"],
            max_matches=validated_input["max_matches"],
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write full UTF-8 file contents through the persistent environment."
    kind = "side_effect"
    requires_artifacts = ("path", "text")
    provides_artifacts = ("edited_text", "path")
    allowed_followers = ("read_file", "read_text", "respond", "run_tests")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "written": {"type": "boolean"},
            "size_chars": {"type": "integer"},
        },
        "required": ["path", "written", "size_chars"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create": {"type": "boolean"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        content = raw_input.get("content")
        create = bool(raw_input.get("create", True))
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("write_file.path must be a non-empty string")
        if not isinstance(content, str):
            raise ToolValidationError("write_file.content must be a string")
        return {"path": path, "content": content, "create": create}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"file_read_for_edit", "edit_applied"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.write_file(validated_input["path"], validated_input["content"], create=validated_input["create"])


class InspectDiffTool(Tool):
    name = "inspect_diff"
    description = "Inspect the current diff for one workspace file against the last remembered environment state."
    kind = "stateful"
    requires_artifacts = ("path",)
    provides_artifacts = ("diff",)
    allowed_followers = ("edit_text", "write_file", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "relative_path": {"type": "string"},
            "changed": {"type": "boolean"},
            "diff": {"type": "string"},
            "baseline_source": {"type": "string"},
        },
        "required": ["path", "relative_path", "changed", "diff", "baseline_source"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("inspect_diff.path must be a non-empty string")
        return {"path": path.strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"diff_inspected"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.inspect_diff(validated_input["path"])


class ListChangesTool(Tool):
    name = "list_changes"
    description = "List created, modified, and deleted files from the persistent workspace state."
    kind = "stateful"
    requires_artifacts = ()
    provides_artifacts = ("changes",)
    allowed_followers = ("read_file", "inspect_diff", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "created_files": {"type": "array", "items": {"type": "string"}},
            "modified_files": {"type": "array", "items": {"type": "string"}},
            "deleted_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["cwd", "created_files", "modified_files", "deleted_files"],
        "additionalProperties": False,
    }
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        return {}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"changes_listed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.list_changes()


class WorkspaceSnapshotTool(Tool):
    name = "workspace_snapshot"
    description = "Return a structured snapshot of the current workspace state."
    kind = "stateful"
    requires_artifacts = ()
    provides_artifacts = ("workspace_snapshot",)
    allowed_followers = ("search_repo", "read_file", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "workspace_root": {"type": "string"},
            "cwd": {"type": "string"},
            "files": {"type": "object"},
            "file_count": {"type": "integer"},
            "created_files": {"type": "array", "items": {"type": "string"}},
            "modified_files": {"type": "array", "items": {"type": "string"}},
            "deleted_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["workspace_root", "cwd", "files", "file_count", "created_files", "modified_files", "deleted_files"],
        "additionalProperties": False,
    }
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        return {}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"workspace_snapshot_inspected"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.workspace_snapshot()


class ShellCommandTool(Tool):
    name = "shell_command"
    description = "Run a shell command in the persistent session workspace."
    kind = "side_effect"
    requires_artifacts = ("command",)
    provides_artifacts = ("command_result",)
    allowed_followers = ("run_tests", "read_file", "list_files", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd_before": {"type": "string"},
            "cwd_after": {"type": "string"},
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "created_files": {"type": "array", "items": {"type": "string"}},
            "modified_files": {"type": "array", "items": {"type": "string"}},
            "deleted_files": {"type": "array", "items": {"type": "string"}},
            "background": {"type": "boolean"},
            "process_id": {"type": "string"},
        },
        "required": ["command", "cwd_before", "cwd_after", "exit_code", "stdout", "stderr", "created_files", "modified_files", "deleted_files", "background"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "background": {"type": "boolean"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        command = raw_input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolValidationError("shell_command.command must be a non-empty string")
        background = bool(raw_input.get("background", False))
        return {"command": command.strip(), "background": background}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        if validated_input.get("background"):
            return {"shell_command_started", "process_started"}
        return {"shell_command_started", "shell_command_completed", "workspace_snapshot", "process_started", "process_completed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.run_shell_command(
            validated_input["command"],
            background=bool(validated_input.get("background", False)),
        )


class RunTestsTool(Tool):
    name = "run_tests"
    description = "Run a test command inside the persistent workspace and capture structured results."
    kind = "stateful"
    requires_artifacts = ("command",)
    provides_artifacts = ("test_result",)
    allowed_followers = ("read_file", "edit_text", "write_file", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "passed": {"type": "boolean"},
            "background": {"type": "boolean"},
            "process_id": {"type": "string"},
        },
        "required": ["command", "cwd", "exit_code", "stdout", "stderr", "passed", "background"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"}},
            "background": {"type": "boolean"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        command = raw_input.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ToolValidationError("run_tests.command must be a non-empty list of strings")
        return {"command": list(command), "background": bool(raw_input.get("background", False))}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        if validated_input.get("background"):
            return {"process_started"}
        return {"process_started", "process_completed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.run_tests(
            validated_input["command"],
            background=bool(validated_input.get("background", False)),
        )


class BrowserSearchTool(Tool):
    name = "browser_search"
    description = "Search the web through the external aubro browser automation layer and return structured top results."
    kind = "stateful"
    requires_artifacts = ("search_query",)
    provides_artifacts = ("search_results",)
    allowed_followers = ("browser_browse", "notes", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "engine": {"type": "string"},
            "url": {"type": "string"},
            "result_count": {"type": "integer"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["title", "url", "snippet"],
                    "additionalProperties": False,
                },
            },
            "attempts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "engine": {"type": "string"},
                        "url": {"type": "string"},
                        "results": {"type": "integer"},
                        "blocked": {"type": "boolean"},
                    },
                    "required": ["engine", "url", "results", "blocked"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["query", "engine", "url", "result_count", "results", "attempts"],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "engine": {"type": "string", "enum": ["auto", "privau", "bing", "duckduckgo"]},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        query = raw_input.get("query")
        engine = raw_input.get("engine", "auto")
        limit = raw_input.get("limit", 5)
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("browser_search.query must be a non-empty string")
        if engine not in {"auto", "privau", "bing", "duckduckgo"}:
            raise ToolValidationError("browser_search.engine must be one of auto, privau, bing, duckduckgo")
        if not isinstance(limit, int) or limit <= 0:
            raise ToolValidationError("browser_search.limit must be a positive integer")
        return {"query": query.strip(), "engine": engine, "limit": limit}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"process_started", "process_completed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        limit = min(validated_input["limit"], context.config.environment.aubro_max_results)
        return context.environment.browser_search(
            query=validated_input["query"],
            engine=validated_input["engine"],
            limit=limit,
        )


class BrowserBrowseTool(Tool):
    name = "browser_browse"
    description = "Browse one URL through the external aubro browser automation layer and return a structured page summary."
    kind = "stateful"
    requires_artifacts = ("url",)
    provides_artifacts = ("page_summary", "text")
    allowed_followers = ("notes", "respond")
    output_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "backend": {"type": "string"},
            "blocked": {"type": "boolean"},
            "block_reason": {"type": "string"},
            "text_excerpt": {"type": "string"},
            "link_count": {"type": "integer"},
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "href": {"type": "string"},
                    },
                    "required": ["text", "href"],
                    "additionalProperties": False,
                },
            },
            "form_count": {"type": "integer"},
            "button_count": {"type": "integer"},
        },
        "required": [
            "url",
            "title",
            "backend",
            "blocked",
            "block_reason",
            "text_excerpt",
            "link_count",
            "links",
            "form_count",
            "button_count",
        ],
        "additionalProperties": False,
    }
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        url = raw_input.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolValidationError("browser_browse.url must be a non-empty string")
        if not url.startswith(("http://", "https://")):
            raise ToolValidationError("browser_browse.url must start with http:// or https://")
        return {"url": url.strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"process_started", "process_completed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.browser_browse(url=validated_input["url"])


BUILTIN_TOOLS = [
    EchoTool(),
    TimeNowTool(),
    CalculatorTool(),
    BrowserSearchTool(),
    BrowserBrowseTool(),
    ListFilesTool(),
    ReadFileTool(),
    SearchInFileTool(),
    SearchRepoTool(),
    ReadTextTool(),
    NotesTool(),
    EditTextTool(),
    WriteFileTool(),
    InspectDiffTool(),
    ListChangesTool(),
    WorkspaceSnapshotTool(),
    RunTestsTool(),
    ShellCommandTool(),
]


def tool_result_display(tool_name: str, output: dict[str, Any]) -> str:
    return f"{tool_name} result: {stable_json_dumps(output, indent=2)}"
