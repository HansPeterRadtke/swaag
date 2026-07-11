from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
import re


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
        candidates = TextEditor._pattern_candidates(pattern)
        for candidate in candidates:
            count = text.count(candidate)
            if count > 1:
                raise EditError("pattern is ambiguous")
            if count == 1:
                new_text = text.replace(candidate, replacement, 1)
                return TextEditor._preview(
                    "replace_pattern_once",
                    None,
                    text,
                    new_text,
                    {"pattern": pattern, "matched_pattern": candidate, "replacement": replacement, "match_count": count},
                )
        for candidate in candidates:
            matches = TextEditor._loose_indented_block_matches(text, candidate)
            if len(matches) > 1:
                raise EditError("pattern is ambiguous")
            if matches:
                start, end, indent = matches[0]
                replacement_text = TextEditor._indent_replacement(replacement, indent, text[start:end].endswith("\n"))
                new_text = text[:start] + replacement_text + text[end:]
                return TextEditor._preview(
                    "replace_pattern_once",
                    None,
                    text,
                    new_text,
                    {"pattern": pattern, "matched_pattern": candidate, "replacement": replacement, "match_count": 1, "match_style": "loose_indentation"},
                )
        for candidate in candidates:
            regex_matches = TextEditor._regex_pattern_matches(text, candidate)
            if len(regex_matches) > 1:
                raise EditError("pattern is ambiguous")
            if regex_matches:
                start, end = regex_matches[0]
                new_text = text[:start] + replacement + text[end:]
                return TextEditor._preview(
                    "replace_pattern_once",
                    None,
                    text,
                    new_text,
                    {"pattern": pattern, "matched_pattern": candidate, "replacement": replacement, "match_count": 1, "match_style": "regex"},
                )
        for candidate in TextEditor._pattern_candidates(replacement):
            if text.count(candidate) == 1:
                return TextEditor._preview(
                    "replace_pattern_once",
                    None,
                    text,
                    text,
                    {
                        "pattern": pattern,
                        "matched_pattern": None,
                        "replacement": replacement,
                        "match_count": 0,
                        "already_applied": True,
                    },
                )
        raise EditError("pattern not found")

    @staticmethod
    def replace_pattern_all(text: str, pattern: str, replacement: str) -> EditPreview:
        if not pattern:
            raise EditError("pattern must not be empty")
        for candidate in TextEditor._pattern_candidates(pattern):
            count = text.count(candidate)
            if count > 0:
                new_text = text.replace(candidate, replacement)
                return TextEditor._preview(
                    "replace_pattern_all",
                    None,
                    text,
                    new_text,
                    {"pattern": pattern, "matched_pattern": candidate, "replacement": replacement, "match_count": count},
                )
            matches = TextEditor._loose_indented_block_matches(text, candidate)
            if matches:
                new_text = text
                for start, end, indent in reversed(matches):
                    new_text = new_text[:start] + TextEditor._indent_replacement(replacement, indent, new_text[start:end].endswith("\n")) + new_text[end:]
                return TextEditor._preview(
                    "replace_pattern_all",
                    None,
                    text,
                    new_text,
                    {"pattern": pattern, "matched_pattern": candidate, "replacement": replacement, "match_count": len(matches), "match_style": "loose_indentation"},
                )
            regex_matches = TextEditor._regex_pattern_matches(text, candidate)
            if regex_matches:
                new_text = text
                for start, end in reversed(regex_matches):
                    new_text = new_text[:start] + replacement + new_text[end:]
                return TextEditor._preview(
                    "replace_pattern_all",
                    None,
                    text,
                    new_text,
                    {"pattern": pattern, "matched_pattern": candidate, "replacement": replacement, "match_count": len(regex_matches), "match_style": "regex"},
                )
        raise EditError("pattern not found")

    @staticmethod
    def _pattern_candidates(pattern: str) -> list[str]:
        candidates: list[str] = [pattern]
        unescaped = TextEditor._unescape_regex_literal(pattern)
        if unescaped != pattern:
            candidates.append(unescaped)
        compact = TextEditor._collapse_repeated_regex_whitespace(unescaped)
        if compact not in candidates:
            candidates.append(compact)
        return [item for index, item in enumerate(candidates) if item and item not in candidates[:index]]

    @staticmethod
    def _unescape_regex_literal(pattern: str) -> str:
        replacements = {
            r"\n": "\n", r"\t": "\t", r"\r": "\r",
            r"\(": "(", r"\)": ")", r"\[": "[", r"\]": "]", r"\{": "{", r"\}": "}",
            r"\+": "+", r"\-": "-", r"\*": "*", r"\?": "?", r"\.": ".", r"\|": "|",
            r"\^": "^", r"\$": "$", r"\=": "=", r"\:": ":", r"\,": ",", r"\'": "'", r'\"': '"',
        }
        result = pattern
        for escaped, literal in replacements.items():
            result = result.replace(escaped, literal)
        return result

    @staticmethod
    def _collapse_repeated_regex_whitespace(pattern: str) -> str:
        return re.sub(r"(?:\\s\*|\\n|\\t){2,}", "", pattern)

    @staticmethod
    def _regex_pattern_matches(text: str, pattern: str) -> list[tuple[int, int]]:
        if not pattern or len(pattern) > 2000:
            return []
        if "\\" not in pattern and not any(token in pattern for token in (".*", "\\s", "[", "(")):
            return []
        try:
            compiled = re.compile(pattern, re.MULTILINE)
        except re.error:
            return []
        matches = [(match.start(), match.end()) for match in compiled.finditer(text) if match.start() != match.end()]
        return matches

    @staticmethod
    def _loose_indented_block_matches(text: str, pattern: str) -> list[tuple[int, int, str]]:
        pattern_lines = [line.strip() for line in pattern.expandtabs().splitlines() if line.strip()]
        if not pattern_lines:
            return []
        lines = text.splitlines(keepends=True)
        starts: list[int] = []
        offset = 0
        for line in lines:
            starts.append(offset)
            offset += len(line)
        matches: list[tuple[int, int, str]] = []
        for index in range(0, len(lines) - len(pattern_lines) + 1):
            window = lines[index : index + len(pattern_lines)]
            if [line.expandtabs().strip() for line in window] != pattern_lines:
                continue
            first_nonempty = next((line for line in window if line.strip()), window[0])
            indent = first_nonempty[: len(first_nonempty) - len(first_nonempty.lstrip())]
            start = starts[index]
            end = starts[index + len(pattern_lines)] if index + len(pattern_lines) < len(starts) else len(text)
            matches.append((start, end, indent))
        return matches

    @staticmethod
    def _indent_replacement(replacement: str, indent: str, preserve_trailing_newline: bool) -> str:
        lines = replacement.splitlines()
        if not lines:
            return "\n" if preserve_trailing_newline else ""
        rendered = "\n".join(
            (line if not line.strip() or line.startswith((" ", "\t")) else indent + line)
            for line in lines
        )
        if preserve_trailing_newline and not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

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
