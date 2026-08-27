from __future__ import annotations

import pytest

from swaag.notes import (
    NoteError,
    compact_notes,
    enforce_limits,
    make_note,
    select_notes_for_prompt,
)
from swaag.tokens import ExactTokenCounter


def test_make_note_rejects_per_note_overflow_without_truncation(make_config) -> None:
    config = make_config(notes__max_note_chars=5)
    with pytest.raises(NoteError, match="max_note_chars"):
        make_note(config, title="Title", content="123456789")


def test_enforce_limits_rejects_overflow_without_dropping_old_notes(make_config) -> None:
    config = make_config(notes__max_notes=2, notes__max_total_chars=20)
    notes = [
        make_note(config, title="A", content="11111"),
        make_note(config, title="B", content="22222"),
        make_note(config, title="C", content="33333"),
    ]
    with pytest.raises(NoteError, match="max_notes"):
        enforce_limits(config, notes)
    assert [note.title for note in notes] == ["A", "B", "C"]


def test_compact_notes_uses_complete_semantic_result_without_clipping(make_config) -> None:
    config = make_config(notes__compact_target_chars=50)
    notes = [make_note(config, title="A", content="111"), make_note(config, title="B", content="222")]
    compacted = compact_notes(
        config,
        notes,
        title="Consolidated",
        content="A remains 111; B remains 222.",
    )
    assert compacted is not None
    removed_ids, compacted_note = compacted
    assert removed_ids == [note.note_id for note in notes]
    assert compacted_note.content == "A remains 111; B remains 222."


def test_select_notes_for_prompt_respects_budget(make_config) -> None:
    config = make_config(context__note_prompt_token_cap=6)
    notes = [make_note(config, title="A", content="one two three"), make_note(config, title="B", content="four five six")]
    selection = select_notes_for_prompt(config, notes, ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0))
    assert selection.tokens <= config.context.note_prompt_token_cap
    assert selection.included_notes
    assert set(selection.omitted_note_ids).issubset({note.note_id for note in notes})


def test_runtime_context_exposes_recoverable_ids_for_omitted_notes(make_config) -> None:
    from swaag.runtime import AgentRuntime

    config = make_config(context__note_prompt_token_cap=1)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    note = make_note(config, title="Constraint", content="retain exact marker")
    state.notes.append(note)

    components = runtime._runtime_context_components(
        state,
        ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    omitted = next(
        component
        for component in components
        if component.name == "omitted_durable_note_references"
    )

    assert note.note_id in omitted.text
    assert "notes list" in omitted.text
