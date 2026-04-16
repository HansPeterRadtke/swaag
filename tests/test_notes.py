from __future__ import annotations

from swaag.notes import compact_notes, enforce_limits, make_note, select_notes_for_prompt
from swaag.tokens import ExactTokenCounter


def test_make_note_enforces_per_note_limit(make_config) -> None:
    config = make_config(notes__max_note_chars=5)
    note = make_note(config, title="Title", content="123456789")
    assert note.content == "12345"


def test_enforce_limits_trims_old_notes(make_config) -> None:
    config = make_config(notes__max_notes=2, notes__max_total_chars=20)
    notes = [
        make_note(config, title="A", content="11111"),
        make_note(config, title="B", content="22222"),
        make_note(config, title="C", content="33333"),
    ]
    limited = enforce_limits(config, notes)
    assert len(limited) <= 2
    assert limited[-1].title == "C"


def test_compact_notes_combines_older_notes(make_config) -> None:
    config = make_config(notes__compact_target_chars=50)
    notes = [make_note(config, title="A", content="111"), make_note(config, title="B", content="222")]
    compacted = compact_notes(config, notes)
    assert compacted is not None
    removed_ids, compacted_note = compacted
    assert removed_ids == [notes[0].note_id]
    assert "[A] 111" in compacted_note.content


def test_select_notes_for_prompt_respects_budget(make_config) -> None:
    config = make_config(context__note_prompt_token_cap=6)
    notes = [make_note(config, title="A", content="one two three"), make_note(config, title="B", content="four five six")]
    selection = select_notes_for_prompt(config, notes, ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0))
    assert selection.tokens <= config.context.note_prompt_token_cap
    assert selection.included_notes
    assert set(selection.omitted_note_ids).issubset({note.note_id for note in notes})
