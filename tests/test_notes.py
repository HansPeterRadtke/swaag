from __future__ import annotations

import pytest

from swaag.grammar import yes_no_contract
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
    config = make_config()
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
    config = make_config()
    notes = [make_note(config, title="A", content="one two three"), make_note(config, title="B", content="four five six")]
    selection = select_notes_for_prompt(
        config,
        notes,
        ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0),
        max_tokens=6,
    )
    assert selection.tokens <= 6
    assert selection.included_notes
    assert set(selection.omitted_note_ids).issubset({note.note_id for note in notes})


def test_runtime_context_includes_all_notes_before_measured_overflow(make_config) -> None:
    from swaag.runtime import AgentRuntime

    config = make_config()
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
    durable = next(
        component
        for component in components
        if component.name == "durable_notes"
    )

    assert note.note_id in durable.text
    assert "retain exact marker" in durable.text
    assert not hasattr(config.context, "note_prompt_token_cap")


def test_measured_note_overflow_projects_semantically_and_records_recovery(
    make_config,
    monkeypatch,
) -> None:
    from swaag.runtime import AgentRuntime

    config = make_config(model__context_limit=350)
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    for index in range(8):
        state.notes.append(
            make_note(
                config,
                title=f"Constraint {index}",
                content=(f"exact-note-{index} " * 60).strip(),
            )
        )
    components = runtime._runtime_context_components(state, runtime._counter(state))
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="history_analysis",
        system_instruction="Inspect the runtime context.",
        components=components,
    )
    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(350, "test"),
    )
    assert compilation.overflow_tokens > 0

    def non_reducing(_state, **kwargs):
        return kwargs["source_text"], compilation.report

    monkeypatch.setattr(runtime, "_reduce_text_hierarchically", non_reducing)
    skipped = runtime._project_runtime_context_for_overflow(
        state,
        original_request="preserve the relevant constraints",
        compilation=compilation,
        existing_projections={},
        remaining_calls=[8],
    )
    assert skipped is None
    skipped_event = runtime.history.read_history(state.session_id)[-1]
    assert skipped_event.event_type == "runtime_context_projection_skipped"

    def reduce(_state, **kwargs):
        assert "exact-note-7" in kwargs["source_text"]
        return "Relevant exact notes are recoverable with the notes capability.", compilation.report

    monkeypatch.setattr(runtime, "_reduce_text_hierarchically", reduce)
    projected = runtime._project_runtime_context_for_overflow(
        state,
        original_request="preserve the relevant constraints",
        compilation=compilation,
        existing_projections={},
        remaining_calls=[8],
    )
    assert projected is not None
    assert projected[0] == "durable_notes"
    projected_event = runtime.history.read_history(state.session_id)[-1]
    assert projected_event.event_type == "runtime_context_projected"
    assert projected_event.payload["source_locator"]["recovery_tool"] == "notes"
