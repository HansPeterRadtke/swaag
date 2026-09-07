from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from swaag.grammar import note_selection_contract
from swaag.notes import MAX_NOTE_CATEGORIES, MAX_NOTE_CATEGORY_CHARS, render_notes
from swaag.preemption import ModelCallStateChanged, RunCancellationRequested
from swaag.tools.base import SemanticCallRequest
from swaag.types import Note, PromptAssembly, PromptComponent, SessionState
from swaag.utils import sha256_text, stable_json_dumps

if TYPE_CHECKING:
    from swaag.runtime import AgentRuntime


class NoteContextManager:
    def __init__(self, runtime: "AgentRuntime") -> None:
        self.runtime = runtime

    def select_for_action(
        self, state: SessionState, assembly: PromptAssembly
    ) -> list[Note]:
        runtime = self.runtime
        candidates = list(state.notes)
        if not candidates:
            runtime.history.record_event(
                state,
                "notes_selected",
                {
                    "included_note_ids": [],
                    "omitted_note_ids": [],
                    "tokens": 0,
                    "exact": True,
                    "semantic_selection": False,
                    "selection_fallback": False,
                    "operation_categories": [],
                    "selection_reason": "No durable note candidates exist.",
                    "candidate_note_references": [],
                },
            )
            return []

        target_rows = [
            asdict(component)
            for component in assembly.components
            if component.category != "wrapper"
            and component.include_in_context
            and component.category != "notes"
        ]
        target_context = stable_json_dumps(
            {"call_kind": "action", "components": target_rows},
            indent=2,
        )
        target_context_sha256 = sha256_text(target_context)
        candidate_rows = [asdict(note) for note in candidates]
        candidate_references = [
            {
                "note_id": note.note_id,
                "sha256": sha256_text(
                    stable_json_dumps(asdict(note), indent=None)
                ),
            }
            for note in candidates
        ]
        system_template = runtime.config.prompts.note_selection_system_template
        user_template = runtime.config.prompts.note_selection_template
        request = SemanticCallRequest(
            kind="note_selection",
            system_instruction=runtime.prompts.template_text(system_template),
            components=[
                PromptComponent(
                    name="note_selection_task",
                    category="system_prompt_instruction",
                    text=runtime.prompts.template_text(user_template).format(
                        target_context_sha256=target_context_sha256,
                        target_context=target_context,
                        candidate_notes=stable_json_dumps(candidate_rows, indent=2),
                    ),
                )
            ],
            contract=note_selection_contract(note.note_id for note in candidates),
            minimum_output_tokens=128,
            desired_output_tokens=384,
            prompt_template_names=(system_template, user_template),
        )
        semantic_selection = True
        selection_fallback = False
        operation_categories: list[str] = []
        selection_reason = ""
        try:
            payload = runtime._execute_tool_semantic_call(state, request)
            for raw_category in payload["operation_categories"]:
                category = str(raw_category).strip()
                if not category or len(category) > MAX_NOTE_CATEGORY_CHARS:
                    raise ValueError(
                        "note selector returned an invalid operation category"
                    )
                if category not in operation_categories:
                    operation_categories.append(category)
            if len(operation_categories) > MAX_NOTE_CATEGORIES:
                raise ValueError("note selector returned too many operation categories")
            selected_ids = {str(note_id) for note_id in payload["selected_note_ids"]}
            selected = [
                note for note in candidates if note.note_id in selected_ids
            ]
            selection_reason = str(payload["reason"])
        except (ModelCallStateChanged, RunCancellationRequested):
            raise
        except Exception as exc:
            semantic_selection = False
            selection_fallback = True
            selected = candidates
            selection_reason = (
                "Semantic selector failed; every exact note candidate was included."
            )
            runtime.history.record_event(
                state,
                "note_selection_failed",
                {
                    "target_context_sha256": target_context_sha256,
                    "candidate_note_references": candidate_references,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "fallback": "include_all_notes",
                },
            )

        selected_text = render_notes(selected)
        counted = runtime._counter(state).count_text(selected_text)
        selected_ids = {note.note_id for note in selected}
        runtime.history.record_event(
            state,
            "notes_selected",
            {
                "included_note_ids": [note.note_id for note in selected],
                "omitted_note_ids": [
                    note.note_id
                    for note in candidates
                    if note.note_id not in selected_ids
                ],
                "tokens": counted.tokens,
                "exact": counted.exact,
                "semantic_selection": semantic_selection,
                "selection_fallback": selection_fallback,
                "operation_categories": operation_categories,
                "selection_reason": selection_reason,
                "selection_target_context_sha256": target_context_sha256,
                "candidate_note_references": candidate_references,
            },
        )
        return selected
