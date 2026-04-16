from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path


class EditError(ValueError):
    pass


@dataclass(slots=True)
class EditPreview:
    changed: bool
    operation: str
    path: str | None
    original_text: str
    new_text: str
    diff: str
    details: dict


class TextEditor:
    @staticmethod
    def replace_range(text: str, start: int, end: int, replacement: str) -> EditPreview:
        TextEditor._validate_range(text, start, end)
        new_text = text[:start] + replacement + text[end:]
        return TextEditor._preview("replace_range", None, text, new_text, {"start": start, "end": end, "replacement": replacement})

    @staticmethod
    def insert_at(text: str, position: int, insertion: str) -> EditPreview:
        TextEditor._validate_range(text, position, position)
        new_text = text[:position] + insertion + text[position:]
        return TextEditor._preview("insert_at", None, text, new_text, {"position": position, "insertion": insertion})

    @staticmethod
    def delete_range(text: str, start: int, end: int) -> EditPreview:
        TextEditor._validate_range(text, start, end)
        new_text = text[:start] + text[end:]
        return TextEditor._preview("delete_range", None, text, new_text, {"start": start, "end": end})

    @staticmethod
    def replace_pattern_once(text: str, pattern: str, replacement: str) -> EditPreview:
        if not pattern:
            raise EditError("pattern must not be empty")
        count = text.count(pattern)
        if count == 0:
            raise EditError("pattern not found")
        if count > 1:
            raise EditError("pattern is ambiguous")
        new_text = text.replace(pattern, replacement, 1)
        return TextEditor._preview(
            "replace_pattern_once",
            None,
            text,
            new_text,
            {"pattern": pattern, "replacement": replacement, "match_count": count},
        )

    @staticmethod
    def replace_pattern_all(text: str, pattern: str, replacement: str) -> EditPreview:
        if not pattern:
            raise EditError("pattern must not be empty")
        count = text.count(pattern)
        if count == 0:
            raise EditError("pattern not found")
        new_text = text.replace(pattern, replacement)
        return TextEditor._preview(
            "replace_pattern_all",
            None,
            text,
            new_text,
            {"pattern": pattern, "replacement": replacement, "match_count": count},
        )

    @staticmethod
    def preview_file(path: str, operation: str, **kwargs) -> EditPreview:
        file_path = Path(path).expanduser()
        original = file_path.read_text(encoding="utf-8")
        preview = TextEditor.apply(original, operation, **kwargs)
        return EditPreview(
            changed=preview.changed,
            operation=preview.operation,
            path=str(file_path.resolve()),
            original_text=preview.original_text,
            new_text=preview.new_text,
            diff=preview.diff,
            details=preview.details,
        )

    @staticmethod
    def apply(text: str, operation: str, **kwargs) -> EditPreview:
        operations = {
            "replace_range": TextEditor.replace_range,
            "insert_at": TextEditor.insert_at,
            "delete_range": TextEditor.delete_range,
            "replace_pattern_once": TextEditor.replace_pattern_once,
            "replace_pattern_all": TextEditor.replace_pattern_all,
        }
        try:
            handler = operations[operation]
        except KeyError as exc:
            raise EditError(f"unknown edit operation: {operation}") from exc
        return handler(text, **kwargs)

    @staticmethod
    def _validate_range(text: str, start: int, end: int) -> None:
        if start < 0 or end < 0:
            raise EditError("range positions must be non-negative")
        if start > end:
            raise EditError("start must be <= end")
        if end > len(text):
            raise EditError("range end exceeds text length")

    @staticmethod
    def _preview(operation: str, path: str | None, original: str, new: str, details: dict) -> EditPreview:
        diff = "".join(
            unified_diff(
                original.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        return EditPreview(
            changed=original != new,
            operation=operation,
            path=path,
            original_text=original,
            new_text=new,
            diff=diff,
            details=details,
        )
