from __future__ import annotations

from dataclasses import asdict

from swaag.config import AgentConfig
from swaag.types import Note, NotePromptSelection, SessionState
from swaag.utils import new_id, utc_now_iso


class NoteError(ValueError):
    pass


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def validate_note_fields(config: AgentConfig, *, title: str, content: str) -> tuple[str, str]:
    title = title.strip()
    content = content.strip()
    if not title:
        raise NoteError("note title must not be empty")
    if not content:
        raise NoteError("note content must not be empty")
    return _truncate(title, 200), _truncate(content, config.notes.max_note_chars)


def make_note(config: AgentConfig, *, title: str, content: str, note_id: str | None = None) -> Note:
    title, content = validate_note_fields(config, title=title, content=content)
    now = utc_now_iso()
    return Note(note_id=note_id or new_id("note"), title=title, content=content, created_at=now, updated_at=now)


def note_total_chars(notes: list[Note]) -> int:
    return sum(len(note.title) + len(note.content) for note in notes)


def enforce_limits(config: AgentConfig, notes: list[Note]) -> list[Note]:
    result = list(notes)
    while len(result) > config.notes.max_notes:
        result.pop(0)
    while note_total_chars(result) > config.notes.max_total_chars and len(result) > 1:
        result.pop(0)
    if note_total_chars(result) > config.notes.max_total_chars and result:
        last = result[-1]
        allowed = max(1, config.notes.max_total_chars - len(last.title))
        result[-1] = Note(
            note_id=last.note_id,
            title=last.title,
            content=_truncate(last.content, allowed),
            created_at=last.created_at,
            updated_at=last.updated_at,
            metadata=dict(last.metadata),
        )
    return result


def compact_notes(config: AgentConfig, notes: list[Note]) -> tuple[list[str], Note] | None:
    if len(notes) < 2:
        return None
    removed = notes[:-1]
    combined = []
    for note in removed:
        combined.append(f"[{note.title}] {note.content}")
    compacted = make_note(
        config,
        title="Compacted notes",
        content="\n".join(combined)[: config.notes.compact_target_chars],
    )
    return [note.note_id for note in removed], compacted


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
    token_limit = config.context.note_prompt_token_cap if max_tokens is None else max(int(max_tokens), 0)
    included: list[Note] = []
    omitted: list[str] = []
    rendered = ""
    tokens = 0
    exact = True
    for note in reversed(notes):
        candidate_list = list(reversed([note, *included]))
        candidate_text = render_notes(candidate_list)
        counted = counter.count_text(candidate_text)
        if counted.tokens <= token_limit:
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
