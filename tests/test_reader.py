from __future__ import annotations

from pathlib import Path

import pytest

from swaag.reader import ReaderError, SequentialReader


def test_reader_chunk_boundaries_and_overlap(make_config, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("abcdefghij", encoding="utf-8")
    reader = SequentialReader(make_config(reader__default_chunk_chars=4, reader__default_overlap_chars=1, reader__max_chunk_chars=10))
    state = reader.open_file(str(file_path))
    chunk1, state = reader.read_next(state)
    chunk2, state = reader.read_next(state)

    assert chunk1.text == "abcd"
    assert chunk1.next_offset == 3
    assert chunk2.start_offset == 3
    assert chunk2.text == "defg"


def test_reader_handles_empty_file(make_config, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    reader = SequentialReader(make_config(reader__default_chunk_chars=4, reader__max_chunk_chars=10))
    state = reader.open_file(str(file_path))
    chunk, state = reader.read_next(state)

    assert chunk.text == ""
    assert chunk.finished is True
    assert state.finished is True


def test_reader_handles_utf8_text(make_config, tmp_path: Path) -> None:
    file_path = tmp_path / "utf8.txt"
    file_path.write_text("äöüß漢字", encoding="utf-8")
    reader = SequentialReader(make_config(reader__default_chunk_chars=3, reader__max_chunk_chars=10))
    state = reader.open_file(str(file_path))
    chunk, _ = reader.read_next(state)

    assert chunk.text == "äöü"


def test_reader_rejects_invalid_overlap(make_config) -> None:
    reader = SequentialReader(make_config(reader__max_chunk_chars=10))
    with pytest.raises(ReaderError):
        reader.open_buffer("buffer", chunk_chars=5, overlap_chars=5)
