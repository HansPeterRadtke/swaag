from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest

from swaag.grammar import yes_no_contract
from swaag.model import CompletionRequestPolicy
from swaag.notes import (
    NoteError,
    compact_notes,
    enforce_limits,
    make_note,
)
from swaag.runtime import AgentRuntime
from swaag.tokens import ExactTokenCounter
from swaag.types import CompletionResult, ContractSpec, PromptComponent


class _NoteSelectionClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.selected_note_ids: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.fail_selection = False

    def context_limit_resolution(self) -> tuple[int, str]:
        return 12_000, "test"

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy(
            "test", "server_schema", contract.mode, 10, 0.01
        )

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
        }

    def send_completion(
        self, payload: dict[str, Any], **_kwargs
    ) -> CompletionResult:
        self.requests.append(payload)
        assert payload["contract"] == "note_selection"
        if self.fail_selection:
            raise ValueError("simulated note selector failure")
        text = json.dumps(
            {
                "operation_categories": ["software implementation", "testing"],
                "selected_note_ids": self.selected_note_ids,
                "reason": "The upcoming action changes and tests code.",
            }
        )
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_make_note_rejects_per_note_overflow_without_truncation(make_config) -> None:
    config = make_config(notes__max_note_chars=5)
    with pytest.raises(NoteError, match="max_note_chars"):
        make_note(config, title="Title", content="123456789")

    with pytest.raises(NoteError, match="category"):
        make_note(
            config,
            title="Title",
            content="valid",
            categories=["x" * 121],
        )


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
        categories=["cross-cutting constraints"],
    )
    assert compacted is not None
    removed_ids, compacted_note = compacted
    assert removed_ids == [note.note_id for note in notes]
    assert compacted_note.content == "A remains 111; B remains 222."
    assert compacted_note.categories == ["cross-cutting constraints"]


def test_runtime_context_includes_all_notes_before_measured_overflow(make_config) -> None:
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


def test_action_note_selector_uses_exact_context_and_semantic_categories(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000)
    client = _NoteSelectionClient()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    programming = make_note(
        config,
        title="Parser investigation",
        content="The parser fails on escaped separators; reproduce before editing.",
        categories=["software implementation", "parser work"],
    )
    reporting = make_note(
        config,
        title="Final report detail",
        content="The user wants the final report to explain measured latency.",
        categories=["user reporting"],
    )
    state.notes.extend([programming, reporting])
    client.selected_note_ids = [programming.note_id]
    components = runtime._runtime_context_components(
        state,
        runtime._counter(state),
    )
    assembly = runtime.prompts.build_agent_action_prompt(
        [],
        [],
        original_request="Repair the parser and run focused tests.",
        pending_user_messages=[],
        prompt_mode="standard",
        context_components=components,
    )

    selected = runtime._select_action_notes(state, assembly)

    assert [note.note_id for note in selected] == [programming.note_id]
    assert len(client.requests) == 1
    selector_prompt = str(client.requests[0]["prompt"])
    assert "Repair the parser and run focused tests." in selector_prompt
    assert programming.content in selector_prompt
    assert reporting.content in selector_prompt
    selected_components = runtime._runtime_context_components(
        state,
        runtime._counter(state),
        selected_notes=selected,
    )
    durable = next(
        component
        for component in selected_components
        if component.name == "durable_notes"
    )
    assert programming.content in durable.text
    assert reporting.content not in durable.text
    event = next(
        event
        for event in reversed(runtime.history.read_history(state.session_id))
        if event.event_type == "notes_selected"
    )
    assert event.payload["semantic_selection"] is True
    assert event.payload["operation_categories"] == [
        "software implementation",
        "testing",
    ]
    assert event.payload["included_note_ids"] == [programming.note_id]
    assert event.payload["omitted_note_ids"] == [reporting.note_id]


def test_action_preflight_includes_only_semantically_selected_exact_notes(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000)
    client = _NoteSelectionClient()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    relevant = make_note(
        config,
        title="Current repair",
        content="Retain parser marker alpha during the repair.",
        categories=["software implementation"],
    )
    unrelated = make_note(
        config,
        title="Later presentation",
        content="Explain chart marker beta in the final audio report.",
        categories=["audio reporting"],
    )
    state.notes.extend([relevant, unrelated])
    client.selected_note_ids = [relevant.note_id]

    prepared = runtime._prepare_action_call(
        state,
        original_request="Repair the parser.",
        pending_messages=[],
        tool_specs=[],
        contract=yes_no_contract(),
        validation_feedback="",
        minimum_output_tokens=64,
    )

    durable = next(
        component
        for component in prepared.assembly.components
        if component.name == "durable_notes"
    )
    assert relevant.content in durable.text
    assert unrelated.content not in durable.text
    assert prepared.report.fits is True


def test_action_note_selector_failure_conservatively_includes_every_note(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000, model__max_retries=0)
    client = _NoteSelectionClient()
    client.fail_selection = True
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    state.notes.extend(
        [
            make_note(config, title="First", content="first exact note"),
            make_note(config, title="Second", content="second exact note"),
        ]
    )
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="action",
        system_instruction="Choose the next action.",
        components=[
            PromptComponent(
                name="request",
                category="current_user",
                text="Continue safely.",
            )
        ],
    )

    selected = runtime._select_action_notes(state, assembly)

    assert [note.note_id for note in selected] == [
        note.note_id for note in state.notes
    ]
    events = runtime.history.read_history(state.session_id)
    failure = next(
        event for event in events if event.event_type == "note_selection_failed"
    )
    assert failure.payload["fallback"] == "include_all_notes"
    selected_event = next(
        event for event in events if event.event_type == "notes_selected"
    )
    assert selected_event.payload["selection_fallback"] is True


def test_note_categories_and_removal_rebuild_with_exact_event_provenance(
    make_config,
) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    note = make_note(
        config,
        title="Durable discrepancy",
        content="Observed library behavior differs from its documentation.",
        categories=["version research", "implementation risk"],
    )
    added = runtime.history.record_event(
        state,
        "note_added",
        {"note": asdict(note)},
    )

    rebuilt = runtime.history.rebuild_from_history(
        state.session_id,
        prefer_checkpoint=False,
    )
    assert rebuilt.notes[0].categories == [
        "version research",
        "implementation risk",
    ]
    assert rebuilt.notes[0].metadata["source_event_sequence"] == added.sequence
    assert rebuilt.notes[0].metadata["source_event_hash"] == added.hash

    runtime.history.record_event(
        rebuilt,
        "note_removed",
        {"note_id": note.note_id},
    )
    final = runtime.history.rebuild_from_history(
        state.session_id,
        prefer_checkpoint=False,
    )
    assert final.notes == []


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
