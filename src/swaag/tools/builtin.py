from __future__ import annotations

import ast
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swaag.editing import EditError, TextEditor
from swaag.notes import compact_notes, enforce_limits, make_note, render_notes
from swaag.reader import ReaderError, SequentialReader
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.scheduler import WakeupStore, parse_duration
from swaag.types import DerivedFileWrite, ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import sha256_text, stable_json_dumps


def _closed_input(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _string_or_null() -> dict[str, Any]:
    return _nullable({"type": "string"})


def _integer_or_null() -> dict[str, Any]:
    return _nullable({"type": "integer"})


def _string_array_or_null() -> dict[str, Any]:
    return _nullable({"type": "array", "items": {"type": "string"}})


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the provided text exactly."
    kind = "pure"
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
    input_schema = _closed_input(
        {
            "path": _string_or_null(),
            "paths": _string_array_or_null(),
            "note_id": _string_or_null(),
            "reader_id": _string_or_null(),
            "chunk_chars": _integer_or_null(),
            "overlap_chars": _integer_or_null(),
            "start_offset": _integer_or_null(),
            "end_offset": _integer_or_null(),
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        allowed = {"path", "paths", "note_id", "reader_id", "chunk_chars", "overlap_chars", "start_offset", "end_offset"}
        if not set(raw_input).issubset(allowed):
            raise ToolValidationError("read_text received unknown arguments")
        path = raw_input.get("path")
        paths = raw_input.get("paths")
        if paths is not None and path is not None:
            raise ToolValidationError("read_text requires exactly one of path, paths, note_id, or reader_id")
        if paths is not None:
            path = "\n".join(str(item) for item in paths)
        elif isinstance(path, list):
            raise ToolValidationError("read_text.path must be a string or null")
        note_id = raw_input.get("note_id")
        reader_id = raw_input.get("reader_id")
        chunk_chars = raw_input.get("chunk_chars")
        overlap_chars = raw_input.get("overlap_chars")
        if sum(value is not None for value in [path, note_id, reader_id]) != 1:
            raise ToolValidationError("read_text requires exactly one of path, paths, note_id, or reader_id")
        if paths is not None:
            if not paths or not all(isinstance(item, str) and item.strip() for item in paths):
                raise ToolValidationError("read_text.path list must contain non-empty strings")
        elif path is not None and (not isinstance(path, str) or not path.strip()):
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
            "paths": list(paths) if paths is not None else None,
            "note_id": note_id,
            "reader_id": reader_id,
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
        }

    def pre_execute_events(self, validated_input: dict[str, Any], context: ToolContext) -> list[ToolGeneratedEvent]:
        return []

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        required = {"reader_chunk_read"}
        if validated_input.get("paths") is not None:
            required.add("buffer_chunk_read")
        elif validated_input["path"] is not None:
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
    input_schema = _closed_input(
        {
            "action": {"type": "string", "enum": ["list", "add", "replace", "compact"]},
            "note_id": _string_or_null(),
            "title": _string_or_null(),
            "content": _string_or_null(),
        }
    )

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
    usage_guidance = (
        "Return one concrete edit with path, operation, dry_run, and null for inapplicable nullable fields. "
        "dry_run=false applies the edit; do not add write_file just to persist it. "
        "Prefer replace_exact when you have observed the exact current text to replace: set old_text to the current literal text and new_text to the desired replacement; it requires exactly one match and fails closed on zero or multiple matches. "
        "replace_pattern_once/all replace the entire matched text; replacement is the complete final text for the match, so preserve required syntax such as return, indentation, and delimiters. Prefer replace_exact after you have read the exact text. "
        "If the pattern is absent, choose an edit that matches the current file text; absence fails closed even if replacement text already appears. "
        "Use replace_range only as a low-level fallback when exact text replacement is unsuitable; it needs start, end, expected_text, and replacement. delete_range needs start, end, and expected_text. "
        "expected_text must exactly equal the current file text in the selected range; range offsets are zero-based character offsets. "
        "insert_at needs position and insertion. "
        "After applying an edit, inspect the returned diff and hashes; require tool_effect_verified evidence that the current file exactly matches the intended result, and run a relevant command or test when the task requires additional confirmation."
    )
    kind = "side_effect"
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "operation": {"type": "string"},
            "changed": {"type": "boolean"},
            "diff": {"type": "string"},
            "details": {"type": "object"},
            "before_sha256": {"type": "string"},
            "after_sha256": {"type": "string"},
        },
        "required": ["path", "operation", "changed", "diff", "details", "before_sha256", "after_sha256"],
        "additionalProperties": False,
    }
    input_schema = _closed_input(
        {
            "path": {"type": "string"},
            "operation": {"type": "string", "enum": ["replace_exact", "replace_range", "insert_at", "delete_range", "replace_pattern_once", "replace_pattern_all"]},
            "dry_run": {"type": "boolean"},
            "old_text": _string_or_null(),
            "new_text": _string_or_null(),
            "start": _integer_or_null(),
            "end": _integer_or_null(),
            "position": _integer_or_null(),
            "expected_text": _string_or_null(),
            "replacement": _string_or_null(),
            "insertion": _string_or_null(),
            "pattern": _string_or_null(),
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        operation = raw_input.get("operation")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("edit_text.path must be a non-empty string")
        if operation not in {"replace_exact", "replace_range", "insert_at", "delete_range", "replace_pattern_once", "replace_pattern_all"}:
            raise ToolValidationError("edit_text.operation is invalid")
        def _non_negative_int(field: str) -> int:
            value = raw_input.get(field)
            if not isinstance(value, int) or value < 0:
                raise ToolValidationError(f"edit_text.{field} must be a non-negative integer")
            return value

        def _string_field(field: str, *, required: bool = True) -> str:
            value = raw_input.get(field)
            if value is None and not required:
                return ""
            if not isinstance(value, str):
                raise ToolValidationError(f"edit_text.{field} must be a string")
            value = value.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
            if len(value) > 2000:
                raise ToolValidationError(f"edit_text.{field} must be at most 2000 characters")
            return value

        validated = {"path": path, "operation": operation, "dry_run": bool(raw_input.get("dry_run", False))}
        if operation == "replace_exact":
            old_text = _string_field("old_text")
            if old_text == "":
                raise ToolValidationError("edit_text.replace_exact requires non-empty old_text")
            validated["old_text"] = old_text
            validated["new_text"] = _string_field("new_text")
        elif operation == "replace_range":
            validated["start"] = _non_negative_int("start")
            validated["end"] = _non_negative_int("end")
            validated["expected_text"] = _string_field("expected_text")
            validated["replacement"] = _string_field("replacement")
        elif operation == "insert_at":
            validated["position"] = _non_negative_int("position")
            validated["insertion"] = _string_field("insertion")
        elif operation == "delete_range":
            validated["start"] = _non_negative_int("start")
            validated["end"] = _non_negative_int("end")
            validated["expected_text"] = _string_field("expected_text")
        elif operation in {"replace_pattern_once", "replace_pattern_all"}:
            if raw_input.get("pattern") is None or raw_input.get("replacement") is None:
                raise ToolValidationError(f"edit_text.{operation} requires pattern and replacement")
            validated["pattern"] = _string_field("pattern")
            validated["replacement"] = _string_field("replacement")
        return validated

    def effective_kind(self, validated_input: dict[str, Any]) -> str:
        return "stateful" if validated_input.get("dry_run", False) else self.kind

    def pre_execute_events(self, validated_input: dict[str, Any], context: ToolContext) -> list[ToolGeneratedEvent]:
        return []

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        # A no-op edit is represented as edit_previewed even when dry_run is false,
        # while a real write emits edit_applied. The common invariant is that the
        # current file was read before the edit decision.
        return {"file_read_for_edit"}

    def verify_effect(
        self,
        result: ToolExecutionResult,
        environment,
    ) -> tuple[bool, dict[str, Any]]:
        output = result.output
        path_text = output.get("path")
        changed = output.get("changed")
        before_sha256 = output.get("before_sha256")
        after_sha256 = output.get("after_sha256")
        evidence: dict[str, Any] = {
            "tool_name": result.tool_name,
            "path": path_text,
            "operation": output.get("operation"),
            "changed": changed,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
        }
        if result.tool_name != self.name:
            evidence["reason"] = "tool_name_mismatch"
            return False, evidence
        if not isinstance(path_text, str) or not path_text.strip():
            evidence["reason"] = "missing_path"
            return False, evidence
        if not isinstance(before_sha256, str) or not before_sha256 or not isinstance(after_sha256, str) or not after_sha256:
            evidence["reason"] = "missing_effect_hashes"
            return False, evidence
        try:
            resolved, current_text = environment.filesystem.read_text(path_text, cwd=environment.current_cwd)
        except Exception as exc:  # noqa: BLE001 - evidence must fail closed for any filesystem error.
            evidence["reason"] = "current_file_unreadable"
            evidence["error"] = str(exc)
            return False, evidence
        current_sha256 = sha256_text(current_text)
        evidence["resolved_path"] = str(resolved)
        evidence["current_sha256"] = current_sha256
        evidence["persisted"] = current_sha256 == after_sha256
        evidence["real_change"] = changed is True and before_sha256 != after_sha256
        passed = bool(evidence["persisted"]) and bool(evidence["real_change"])
        if not passed:
            evidence["reason"] = "tool_effect_not_persisted"
        return passed, evidence

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.preview_or_apply_edit(validated_input, context)


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files visible in the persistent workspace."
    kind = "stateful"
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
    input_schema = _closed_input({"path": {"type": "string"}})

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
    max_plan_output_refs = 1
    description = "Read a full UTF-8 file from the persistent workspace."
    usage_guidance = "One file per call and one output_ref per plan step. Use separate ordered steps when multiple files are needed."
    kind = "stateful"
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
    input_schema = _closed_input(
        {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
            "max_matches": _integer_or_null(),
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        path = raw_input.get("path")
        pattern = raw_input.get("pattern")
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("search_in_file.path must be a non-empty string")
        if not isinstance(pattern, str) or not pattern:
            raise ToolValidationError("search_in_file.pattern must be a non-empty string")
        max_matches = raw_input.get("max_matches")
        if max_matches is None:
            max_matches = 50
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
    input_schema = _closed_input(
        {
            "path": _string_or_null(),
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
            "max_matches": _integer_or_null(),
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        pattern = raw_input.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolValidationError("search_repo.pattern must be a non-empty string")
        path = raw_input.get("path")
        if path is None:
            path = "."
        if not isinstance(path, str) or not path.strip():
            raise ToolValidationError("search_repo.path must be a non-empty string")
        max_matches = raw_input.get("max_matches")
        if max_matches is None:
            max_matches = 100
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
    usage_guidance = (
        "Return path, complete final file content, and create as a boolean. "
        "Use this only when replacing or creating the whole file is the intended action with concrete content. "
        "Do not pass artifact placeholders; use observed file text or choose a narrower edit tool when appropriate. "
        "The runtime automatically installs a persisted-hash tool_effect_verified check; use command_success only for a distinct executable correctness test."
    )
    kind = "side_effect"
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "written": {"type": "boolean"},
            "size_chars": {"type": "integer"},
            "changed": {"type": "boolean"},
            "existed_before": {"type": "boolean"},
            "before_sha256": {"type": "string"},
            "after_sha256": {"type": "string"},
        },
        "required": ["path", "written", "size_chars", "changed", "existed_before", "before_sha256", "after_sha256"],
        "additionalProperties": False,
    }
    input_schema = _closed_input(
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create": {"type": "boolean"},
        }
    )

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

    def verify_effect(
        self,
        result: ToolExecutionResult,
        environment,
    ) -> tuple[bool, dict[str, Any]]:
        output = result.output
        path_text = output.get("path")
        changed = output.get("changed")
        existed_before = output.get("existed_before")
        before_sha256 = output.get("before_sha256")
        after_sha256 = output.get("after_sha256")
        evidence: dict[str, Any] = {
            "tool_name": result.tool_name,
            "path": path_text,
            "changed": changed,
            "existed_before": existed_before,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
        }
        if result.tool_name != self.name:
            evidence["reason"] = "tool_name_mismatch"
            return False, evidence
        if not isinstance(path_text, str) or not path_text.strip():
            evidence["reason"] = "missing_path"
            return False, evidence
        if not isinstance(before_sha256, str) or not before_sha256 or not isinstance(after_sha256, str) or not after_sha256:
            evidence["reason"] = "missing_effect_hashes"
            return False, evidence
        try:
            resolved, current_text = environment.filesystem.read_text(path_text, cwd=environment.current_cwd)
        except Exception as exc:  # noqa: BLE001 - evidence must fail closed for any filesystem error.
            evidence["reason"] = "current_file_unreadable"
            evidence["error"] = str(exc)
            return False, evidence
        current_sha256 = sha256_text(current_text)
        evidence["resolved_path"] = str(resolved)
        evidence["current_sha256"] = current_sha256
        evidence["persisted"] = current_sha256 == after_sha256
        evidence["real_change"] = changed is True and (existed_before is False or before_sha256 != after_sha256)
        passed = bool(evidence["persisted"]) and bool(evidence["real_change"])
        if not passed:
            evidence["reason"] = "tool_effect_not_persisted"
        return passed, evidence

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        return context.environment.write_file(validated_input["path"], validated_input["content"], create=validated_input["create"])


class InspectDiffTool(Tool):
    name = "inspect_diff"
    description = "Inspect the current diff for one workspace file against the last remembered environment state."
    kind = "stateful"
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
    usage_guidance = (
        "Return one non-interactive shell command directly executable in the current workspace. "
        "Do not return only an interpreter name. Set background to true only for work that should continue after the call returns."
    )
    kind = "side_effect"
    output_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd_before": {"type": "string"},
            "cwd_after": {"type": "string"},
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "stdout_chars": {"type": "integer"},
            "stderr_chars": {"type": "integer"},
            "stdout_truncated": {"type": "boolean"},
            "stderr_truncated": {"type": "boolean"},
            "stdout_artifact_id": {"type": "string"},
            "stderr_artifact_id": {"type": "string"},
            "stdout_sha256": {"type": "string"},
            "stderr_sha256": {"type": "string"},
            "created_files": {"type": "array", "items": {"type": "string"}},
            "modified_files": {"type": "array", "items": {"type": "string"}},
            "deleted_files": {"type": "array", "items": {"type": "string"}},
            "background": {"type": "boolean"},
            "process_id": {"type": "string"},
        },
        "required": ["command", "cwd_before", "cwd_after", "exit_code", "stdout", "stderr", "created_files", "modified_files", "deleted_files", "background"],
        "additionalProperties": False,
    }
    input_schema = _closed_input(
        {
            "command": {"type": "string"},
            "background": {"type": "boolean"},
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        command = raw_input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolValidationError("shell_command.command must be a non-empty string")
        stripped = command.strip()
        lowered = stripped.lower()
        test_runner_patterns = (
            r"(?:^|[;&|()\s])(?:pytest|py\.test)(?:$|\s)",
            r"(?:^|[;&|()\s])(?:python|python3|[^\s/]+/python(?:3(?:\.\d+)?)?)\s+-m\s+(?:pytest|unittest)(?:$|\s)",
            r"(?:^|[;&|()\s])(?:tox|nox)(?:$|\s)",
        )
        if any(re.search(pattern, lowered) for pattern in test_runner_patterns):
            raise ToolValidationError(
                "shell_command must not run test runners; use run_tests so pass/fail, exit code, stdout, and stderr are returned as structured verification evidence"
            )
        background = bool(raw_input.get("background", False))
        return {"command": stripped, "background": background}

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
    objective_verification_check_types = (
        "tool_result_success",
        "tool_output_nonempty",
        "tool_output_schema_valid",
    )
    description = "Run a test command inside the persistent workspace and capture structured results."
    usage_guidance = (
        "Use an argv array and boolean background. For required checks, require tool_result_success. For diagnostics where failure is acceptable, inspect passed, exit_code, stdout, and stderr before deciding the next action."
    )
    kind = "stateful"
    output_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "stdout_chars": {"type": "integer"},
            "stderr_chars": {"type": "integer"},
            "stdout_truncated": {"type": "boolean"},
            "stderr_truncated": {"type": "boolean"},
            "stdout_artifact_id": {"type": "string"},
            "stderr_artifact_id": {"type": "string"},
            "stdout_sha256": {"type": "string"},
            "stderr_sha256": {"type": "string"},
            "passed": {"type": "boolean"},
            "background": {"type": "boolean"},
            "process_id": {"type": "string"},
        },
        "required": ["command", "cwd", "exit_code", "stdout", "stderr", "passed", "background"],
        "additionalProperties": False,
    }
    input_schema = _closed_input(
        {
            "command": {"type": "array", "items": {"type": "string"}},
            "background": {"type": "boolean"},
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        command = raw_input.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ToolValidationError("run_tests.command must be a non-empty list of strings")
        normalized_command = list(command)
        executable_name = Path(normalized_command[0]).name
        if executable_name in {"python", "python3"}:
            normalized_command[0] = sys.executable
        elif executable_name in {"pytest", "py.test"}:
            normalized_command = [sys.executable, "-m", "pytest", *normalized_command[1:]]
        return {"command": normalized_command, "background": bool(raw_input.get("background", False))}

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
    input_schema = _closed_input(
        {
            "query": {"type": "string"},
            "engine": {"type": "string", "enum": ["auto", "privau", "bing", "duckduckgo"]},
            "limit": _integer_or_null(),
        }
    )

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        query = raw_input.get("query")
        engine = raw_input.get("engine", "auto")
        limit = raw_input.get("limit")
        if limit is None:
            limit = 5
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


class PollProcessTool(Tool):
    name = "poll_process"
    description = "Poll one background process and return its exact current status and captured output."
    kind = "stateful"
    usage_guidance = "Use the process_id returned by a background run_tests or shell_command call."
    input_schema = _closed_input({"process_id": {"type": "string"}})

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        process_id = raw_input.get("process_id")
        if not isinstance(process_id, str) or not process_id.strip():
            raise ToolValidationError("poll_process.process_id must be a non-empty string")
        return {"process_id": process_id.strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"process_polled"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        update = context.environment.poll_background_process(validated_input["process_id"])
        record = asdict(update.record)
        output = {
            "process_id": update.record.process_id,
            "status": update.record.status,
            "completed": update.completed,
            "return_code": update.record.return_code,
            "stdout": update.record.stdout,
            "stderr": update.record.stderr,
            "record": record,
            "completed_tool_result": (
                {
                    "tool_name": update.tool_result.tool_name,
                    "output": update.tool_result.output,
                    "display_text": update.tool_result.display_text,
                }
                if update.tool_result is not None
                else None
            ),
        }
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=tool_result_display(self.name, output),
            generated_events=update.generated_events,
        )


class KillProcessTool(Tool):
    name = "kill_process"
    description = "Terminate one tracked background process by process_id and return the exact resulting state."
    kind = "side_effect"
    usage_guidance = "Use only when stopping the tracked process is required by the user or current task."
    input_schema = _closed_input({"process_id": {"type": "string"}})

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        process_id = raw_input.get("process_id")
        if not isinstance(process_id, str) or not process_id.strip():
            raise ToolValidationError("kill_process.process_id must be a non-empty string")
        return {"process_id": process_id.strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"process_killed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        update = context.environment.kill_background_process(validated_input["process_id"])
        output = {
            "process_id": update.record.process_id,
            "status": update.record.status,
            "completed": update.completed,
            "return_code": update.record.return_code,
            "stdout": update.record.stdout,
            "stderr": update.record.stderr,
            "record": asdict(update.record),
        }
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=tool_result_display(self.name, output),
            generated_events=update.generated_events,
        )


class WaitSecondsTool(Tool):
    name = "wait_seconds"
    description = "Wait synchronously for a bounded duration. Accept either numeric seconds or a human duration such as '250 ms', '2 minutes', or '1 hour'."
    kind = "pure"
    input_schema = _closed_input({"seconds": {"anyOf": [{"type": "number"}, {"type": "null"}]}, "duration": {"anyOf": [{"type": "string"}, {"type": "null"}]}})

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        seconds = raw_input.get("seconds")
        duration = raw_input.get("duration")
        has_seconds = seconds is not None
        has_duration = isinstance(duration, str) and bool(duration.strip())
        if has_seconds == has_duration:
            raise ToolValidationError("wait_seconds requires exactly one of seconds or duration")
        if has_seconds:
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
                raise ToolValidationError("wait_seconds.seconds must be a number or null")
            if seconds < 0:
                raise ToolValidationError("wait_seconds.seconds must be non-negative")
            return {"seconds": float(seconds), "duration": None}
        if duration is not None and not isinstance(duration, str):
            raise ToolValidationError("wait_seconds.duration must be a string or null")
        try:
            parsed = parse_duration(str(duration).strip())
        except ValueError as exc:
            raise ToolValidationError(str(exc)) from exc
        return {"seconds": parsed.total_seconds(), "duration": str(duration).strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"wait_entered", "wait_resumed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        seconds = float(validated_input["seconds"])
        maximum = max(0.0, float(context.config.runtime.tool_timeout_seconds) - 1.0)
        if seconds > maximum:
            raise ToolValidationError(
                f"wait_seconds.seconds exceeds the current bounded maximum of {maximum:g} seconds"
            )
        started = time.monotonic()
        entered = ToolGeneratedEvent(
            "wait_entered",
            {"reason": f"wait_seconds:{seconds:g}", "process_ids": []},
        )
        time.sleep(seconds)
        elapsed = time.monotonic() - started
        resumed = ToolGeneratedEvent(
            "wait_resumed",
            {"reason": f"wait_seconds:{seconds:g}", "process_ids": []},
        )
        output = {"requested_seconds": seconds, "requested_duration": validated_input.get("duration"), "elapsed_seconds": elapsed}
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=tool_result_display(self.name, output),
            generated_events=[entered, resumed],
        )


class ScheduleWakeupTool(Tool):
    name = "schedule_wakeup"
    description = "Persist a wakeup for this session using either a human duration or an absolute ISO-8601 time. The wakeup survives process restarts."
    usage_guidance = "Provide exactly one of duration or wake_at. Durations support milliseconds, seconds, minutes, hours, days, weeks, months, and years."
    kind = "stateful"
    input_schema = _closed_input({
        "duration": _string_or_null(),
        "wake_at": _string_or_null(),
        "reason": {"type": "string"},
    })

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        duration = raw_input.get("duration")
        wake_at = raw_input.get("wake_at")
        reason = raw_input.get("reason")
        if duration is not None and not isinstance(duration, str):
            raise ToolValidationError("schedule_wakeup.duration must be a string or null")
        if wake_at is not None and not isinstance(wake_at, str):
            raise ToolValidationError("schedule_wakeup.wake_at must be a string or null")
        if not isinstance(reason, str) or not reason.strip():
            raise ToolValidationError("schedule_wakeup.reason must be a non-empty string")
        if bool(duration and duration.strip()) == bool(wake_at and wake_at.strip()):
            raise ToolValidationError("schedule_wakeup requires exactly one of duration or wake_at")
        return {"duration": duration.strip() if duration else None, "wake_at": wake_at.strip() if wake_at else None, "reason": reason.strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"wakeup_scheduled"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        wakeup = WakeupStore(context.config.sessions.root).schedule(
            session_id=context.session_state.session_id, **validated_input
        )
        output = asdict(wakeup)
        event = ToolGeneratedEvent("wakeup_scheduled", {"wakeup_id": wakeup.wakeup_id, "wake_at": wakeup.wake_at, "reason": wakeup.reason})
        return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output), generated_events=[event])


class ListWakeupsTool(Tool):
    name = "list_wakeups"
    description = "List durable wakeups for this session, including whether each wakeup is already due."
    kind = "pure"
    input_schema = _closed_input({"include_cancelled": {"type": "boolean"}})

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        value = raw_input.get("include_cancelled", False)
        if not isinstance(value, bool):
            raise ToolValidationError("list_wakeups.include_cancelled must be a boolean")
        return {"include_cancelled": value}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store = WakeupStore(context.config.sessions.root)
        items = store.list(session_id=context.session_state.session_id, include_cancelled=validated_input["include_cancelled"])
        due_ids = {item.wakeup_id for item in store.due(session_id=context.session_state.session_id)}
        output = {"wakeups": [{**asdict(item), "due": item.wakeup_id in due_ids} for item in items]}
        return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output))


class CancelWakeupTool(Tool):
    name = "cancel_wakeup"
    description = "Cancel a durable wakeup belonging to this session."
    kind = "stateful"
    input_schema = _closed_input({"wakeup_id": {"type": "string"}})

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        wakeup_id = raw_input.get("wakeup_id")
        if not isinstance(wakeup_id, str) or not wakeup_id.strip():
            raise ToolValidationError("cancel_wakeup.wakeup_id must be a non-empty string")
        return {"wakeup_id": wakeup_id.strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"wakeup_cancelled"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        wakeup = WakeupStore(context.config.sessions.root).cancel(
            session_id=context.session_state.session_id, wakeup_id=validated_input["wakeup_id"]
        )
        output = asdict(wakeup)
        event = ToolGeneratedEvent("wakeup_cancelled", {"wakeup_id": wakeup.wakeup_id, "cancelled_at": wakeup.cancelled_at})
        return ToolExecutionResult(tool_name=self.name, output=output, display_text=tool_result_display(self.name, output), generated_events=[event])


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
    PollProcessTool(),
    KillProcessTool(),
    WaitSecondsTool(),
    ScheduleWakeupTool(),
    ListWakeupsTool(),
    CancelWakeupTool(),
]


def tool_result_display(tool_name: str, output: dict[str, Any]) -> str:
    return f"{tool_name} result: {stable_json_dumps(output, indent=2)}"
