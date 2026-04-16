from __future__ import annotations

from pathlib import Path

from swaag.config import AgentConfig
from swaag.types import ReaderChunk, ReaderState, SessionState, SourceKind
from swaag.utils import new_id, utc_now_iso


class ReaderError(ValueError):
    pass


class SequentialReader:
    def __init__(self, config: AgentConfig):
        self.config = config

    def open_file(
        self,
        path: str,
        *,
        chunk_chars: int | None = None,
        overlap_chars: int | None = None,
        reader_id: str | None = None,
    ) -> ReaderState:
        resolved = self._resolve_path(path)
        chunk_chars, overlap_chars = self._validated_chunk_config(chunk_chars, overlap_chars)
        return ReaderState(
            reader_id=reader_id or new_id("reader"),
            source_kind="file",
            source_ref=str(resolved),
            offset=0,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            finished=False,
            updated_at=utc_now_iso(),
        )

    def open_buffer(
        self,
        name: str,
        *,
        chunk_chars: int | None = None,
        overlap_chars: int | None = None,
        reader_id: str | None = None,
    ) -> ReaderState:
        chunk_chars, overlap_chars = self._validated_chunk_config(chunk_chars, overlap_chars)
        return ReaderState(
            reader_id=reader_id or new_id("reader"),
            source_kind="buffer",
            source_ref=name,
            offset=0,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            finished=False,
            updated_at=utc_now_iso(),
        )

    def read_next(self, reader_state: ReaderState, *, buffer_text: str | None = None) -> tuple[ReaderChunk, ReaderState]:
        if reader_state.finished:
            return (
                ReaderChunk(
                    reader_id=reader_state.reader_id,
                    source_kind=reader_state.source_kind,
                    source_ref=reader_state.source_ref,
                    start_offset=reader_state.offset,
                    end_offset=reader_state.offset,
                    next_offset=reader_state.offset,
                    chunk_chars=reader_state.chunk_chars,
                    overlap_chars=reader_state.overlap_chars,
                    finished=True,
                    text="",
                ),
                ReaderState(
                    reader_id=reader_state.reader_id,
                    source_kind=reader_state.source_kind,
                    source_ref=reader_state.source_ref,
                    offset=reader_state.offset,
                    chunk_chars=reader_state.chunk_chars,
                    overlap_chars=reader_state.overlap_chars,
                    finished=True,
                    last_chunk=reader_state.last_chunk,
                    updated_at=utc_now_iso(),
                    metadata=dict(reader_state.metadata),
                ),
            )

        text = self._load_text(reader_state, buffer_text=buffer_text)
        start = reader_state.offset
        end = min(len(text), start + reader_state.chunk_chars)
        chunk_text = text[start:end]
        finished = end >= len(text)
        next_offset = end if finished else max(0, end - reader_state.overlap_chars)
        updated = ReaderState(
            reader_id=reader_state.reader_id,
            source_kind=reader_state.source_kind,
            source_ref=reader_state.source_ref,
            offset=next_offset,
            chunk_chars=reader_state.chunk_chars,
            overlap_chars=reader_state.overlap_chars,
            finished=finished,
            last_chunk=chunk_text,
            updated_at=utc_now_iso(),
            metadata=dict(reader_state.metadata),
        )
        return (
            ReaderChunk(
                reader_id=reader_state.reader_id,
                source_kind=reader_state.source_kind,
                source_ref=reader_state.source_ref,
                start_offset=start,
                end_offset=end,
                next_offset=next_offset,
                chunk_chars=reader_state.chunk_chars,
                overlap_chars=reader_state.overlap_chars,
                finished=finished,
                text=chunk_text,
            ),
            updated,
        )

    def _load_text(self, reader_state: ReaderState, *, buffer_text: str | None = None) -> str:
        if reader_state.source_kind == "file":
            path = self._resolve_path(reader_state.source_ref)
            if not path.exists() or not path.is_file():
                raise ReaderError(f"Reader file does not exist: {path}")
            return path.read_text(encoding="utf-8")
        if buffer_text is None:
            raise ReaderError("buffer_text is required for buffer readers")
        return buffer_text

    def _resolve_path(self, path_text: str) -> Path:
        path = Path(path_text).expanduser()
        resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
        for root in self.config.tools.read_roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        raise ReaderError(f"Reader path is outside allowed roots: {resolved}")

    def _validated_chunk_config(self, chunk_chars: int | None, overlap_chars: int | None) -> tuple[int, int]:
        chunk = int(self.config.reader.default_chunk_chars if chunk_chars is None else chunk_chars)
        overlap = int(self.config.reader.default_overlap_chars if overlap_chars is None else overlap_chars)
        if chunk <= 0:
            raise ReaderError("chunk_chars must be positive")
        if chunk > self.config.reader.max_chunk_chars:
            raise ReaderError(f"chunk_chars exceeds configured maximum: {chunk}")
        if overlap < 0:
            raise ReaderError("overlap_chars must be non-negative")
        if overlap >= chunk:
            raise ReaderError("overlap_chars must be smaller than chunk_chars")
        return chunk, overlap
