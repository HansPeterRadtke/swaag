from __future__ import annotations

from pathlib import Path

import pytest

from swaag.editing import EditError, TextEditor


def test_replace_range_and_insert_delete() -> None:
    assert TextEditor.replace_range("abcdef", 1, 3, "ZZ").new_text == "aZZdef"
    assert TextEditor.insert_at("abc", 1, "ZZ").new_text == "aZZbc"
    assert TextEditor.delete_range("abcdef", 2, 4).new_text == "abef"


def test_replace_pattern_once_rejects_ambiguous_match() -> None:
    with pytest.raises(EditError):
        TextEditor.replace_pattern_once("abc abc", "abc", "x")


def test_replace_pattern_all_and_unicode() -> None:
    preview = TextEditor.replace_pattern_all("ä\r\nä\r\n", "ä", "ö")
    assert preview.new_text == "ö\r\nö\r\n"


def test_noop_edit_has_changed_false() -> None:
    preview = TextEditor.replace_range("abc", 1, 1, "")
    assert preview.changed is False


def test_preview_file_roundtrip(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    preview = TextEditor.preview_file(str(file_path), "replace_pattern_all", pattern="hello", replacement="world")
    assert "-hello" in preview.diff
    assert "+world" in preview.diff
    assert file_path.read_text(encoding="utf-8") == "hello"
