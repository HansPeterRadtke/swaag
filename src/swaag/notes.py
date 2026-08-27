from __future__ import annotations

from dataclasses import asdict

from swaag.config import AgentConfig
from swaag.types import Note, NotePromptSelection, SessionState
from swaag.utils import new_id, utc_now_iso


class NoteError(ValueError):
    pass


def validate_note_fields(config: AgentConfig, *, title: str, content: str) -> tuple[str, str]:
    title = title.strip()
    content = content.strip()
    if not title:
        raise NoteError("note title must not be empty")
    if not content:
        raise NoteError("note content must not be empty")
    if len(title) > 200:
        raise NoteError("note title exceeds the 200-character storage limit")
    if len(content) > config.notes.max_note_chars:
        raise NoteError(
            "note content exceeds the configured max_note_chars storage limit: "
            f"{config.notes.max_note_chars}"
        )
    return title, content


def make_note(config: AgentConfig, *, title: str, content: str, note_id: str | None = None) -> Note:
    title, content = validate_note_fields(config, title=title, content=content)
    now = utc_now_iso()
    return Note(note_id=note_id or new_id("note"), title=title, content=content, created_at=now, updated_at=now)


def note_total_chars(notes: list[Note]) -> int:
    return sum(len(note.title) + len(note.content) for note in notes)


def enforce_limits(config: AgentConfig, notes: list[Note]) -> list[Note]:
    result = list(notes)
    if len(result) > config.notes.max_notes:
        raise NoteError(
            "note count exceeds the configured max_notes storage limit: "
            f"{config.notes.max_notes}"
        )
    total = note_total_chars(result)
    if total > config.notes.max_total_chars:
        raise NoteError(
            "notes exceed the configured max_total_chars storage limit: "
            f"{total}>{config.notes.max_total_chars}"
        )
    return result


def compact_notes(
    config: AgentConfig,
    notes: list[Note],
    *,
    title: str,
    content: str,
) -> tuple[list[str], Note] | None:
    if len(notes) < 2:
        return None
    compacted = make_note(
        config,
        title=title,
        content=content,
    )
    enforce_limits(config, [compacted])
    return [note.note_id for note in notes], compacted


def render_notes(notes: list[Note]) -> str:
    if not notes:
        return ""
    return "\n\n".join(f"[{note.note_id}] {note.title}\n{note.content}" for note in notes)


def select_notes_for_prompt(
    config: AgentConfig,
    notes: list[Note],
    counter,
    *,
    max_tokens: int | None = None,
) -> NotePromptSelection:
    token_limit = None if max_tokens is None else max(int(max_tokens), 0)
    included: list[Note] = []
    omitted: list[str] = []
    rendered = ""
    tokens = 0
    exact = True
    for note in reversed(notes):
        candidate_list = list(reversed([note, *included]))
        candidate_text = render_notes(candidate_list)
        counted = counter.count_text(candidate_text)
        if token_limit is None or counted.tokens <= token_limit:
            included.insert(0, note)
            rendered = candidate_text
            tokens = counted.tokens
            exact = exact and counted.exact
        else:
            omitted.append(note.note_id)
    return NotePromptSelection(
        included_notes=included,
        omitted_note_ids=sorted(omitted),
        rendered_text=rendered,
        tokens=tokens,
        exact=exact,
    )


def snapshot_notes(state: SessionState) -> list[dict]:
    return [asdict(note) for note in state.notes]
