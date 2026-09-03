from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from swaag.grammar import yes_no_contract
from swaag.model import CompletionRequestPolicy
from swaag.prompt_instructions import (
    PromptInstructionError,
    enforce_prompt_instruction_limits,
    make_prompt_instruction,
)
from swaag.prompt_instruction_store import (
    PromptInstructionStore,
    PromptInstructionStoreError,
)
from swaag.runtime import AgentRuntime
from swaag.tokens import ExactTokenCounter
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry
from swaag.types import CompletionResult, ContractSpec, PromptComponent


def _tool_input(
    action: str,
    *,
    instruction_store: str = "session",
    instruction_id: str | None = None,
    title: str | None = None,
    content: str | None = None,
    scopes: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict:
    return {
        "action": action,
        "instruction_store": instruction_store,
        "instruction_id": instruction_id,
        "title": title,
        "content": content,
        "scopes": scopes,
        "categories": categories,
    }


class _InstructionSelectionClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.selected: list[dict[str, str]] = []
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
        assert payload["contract"] == "prompt_instruction_selection"
        if self.fail_selection:
            raise ValueError("simulated semantic selector failure")
        text = json.dumps(
            {
                "operation_categories": ["programming", "implementation"],
                "selected_instructions": self.selected,
                "reason": "The next call is implementing and testing code.",
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


def test_prompt_instruction_scopes_and_storage_limits_fail_closed(make_config) -> None:
    config = make_config(prompt_instructions__max_instruction_chars=5)

    with pytest.raises(PromptInstructionError, match="scope must be one of"):
        make_prompt_instruction(
            config,
            title="Invalid scope",
            content="valid",
            scopes=["coding_fixture"],
        )
    with pytest.raises(PromptInstructionError, match="max_instruction_chars"):
        make_prompt_instruction(
            config,
            title="Too long",
            content="123456",
            scopes=["action"],
        )
    with pytest.raises(PromptInstructionError, match="category"):
        make_prompt_instruction(
            config,
            title="Invalid category",
            content="valid",
            scopes=["action"],
            categories=["x" * 121],
        )
    with pytest.raises(PromptInstructionError, match="only strings"):
        make_prompt_instruction(
            config,
            title="Invalid category type",
            content="valid",
            scopes=["action"],
            categories=[42],  # type: ignore[list-item]
        )

    config = make_config(prompt_instructions__max_instructions=1)
    instructions = [
        make_prompt_instruction(
            config,
            title=f"Rule {index}",
            content="exact rule",
            scopes=["action"],
        )
        for index in range(2)
    ]
    with pytest.raises(PromptInstructionError, match="max_instructions"):
        enforce_prompt_instruction_limits(config, instructions)


def test_prompt_instruction_tool_crud_is_durable_and_scoped(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    registry = ToolRegistry()

    _, added = registry.dispatch(
        "prompt_instructions",
        _tool_input(
            "add",
            title="Reporting rule",
            content="Do not include hashes unless explicitly requested.",
            scopes=["communication_status", "caller_structured_output"],
            categories=["user-reporting"],
        ),
        config,
        state,
    )
    added_event = added.generated_events[0]
    runtime.history.record_event(state, added_event.event_type, added_event.payload)
    instruction_id = added.output["instruction_id"]

    _, replaced = registry.dispatch(
        "prompt_instructions",
        _tool_input(
            "replace",
            instruction_id=instruction_id,
            title="Reporting rule",
            content="Report only meaningful user-facing evidence.",
            scopes=["communication_status"],
            categories=["user-reporting", "status-reporting"],
        ),
        config,
        state,
    )
    replaced_event = replaced.generated_events[0]
    runtime.history.record_event(
        state, replaced_event.event_type, replaced_event.payload
    )
    rebuilt = runtime.history.rebuild_from_history(
        state.session_id, prefer_checkpoint=False
    )
    assert len(rebuilt.prompt_instructions) == 1
    assert rebuilt.prompt_instructions[0].content.endswith("evidence.")
    assert rebuilt.prompt_instructions[0].scopes == ["communication_status"]
    assert rebuilt.prompt_instructions[0].categories == [
        "user-reporting",
        "status-reporting",
    ]

    _, removed = registry.dispatch(
        "prompt_instructions",
        _tool_input("remove", instruction_id=instruction_id),
        config,
        rebuilt,
    )
    removed_event = removed.generated_events[0]
    runtime.history.record_event(rebuilt, removed_event.event_type, removed_event.payload)
    final = runtime.history.rebuild_from_history(
        state.session_id, prefer_checkpoint=False
    )
    assert final.prompt_instructions == []


def test_prompt_instruction_tool_rejects_duplicate_capacity_without_mutation(
    make_config,
) -> None:
    config = make_config(prompt_instructions__max_instructions=1)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    state.prompt_instructions.append(
        make_prompt_instruction(
            config,
            title="Existing",
            content="exact existing rule",
            scopes=["action"],
        )
    )

    with pytest.raises(ToolValidationError, match="failed without modifying"):
        ToolRegistry().dispatch(
            "prompt_instructions",
            _tool_input(
                "add",
                title="Duplicate",
                content="exact existing rule",
                scopes=["action"],
                categories=["programming"],
            ),
            config,
            state,
        )

    assert len(state.prompt_instructions) == 1
    assert state.prompt_instructions[0].title == "Existing"


def test_central_compiler_injects_every_matching_instruction_into_system_role(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000)
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    action = make_prompt_instruction(
        config,
        title="Action correction",
        content="Use the learned exact tool workaround.",
        scopes=["action"],
    )
    universal = make_prompt_instruction(
        config,
        title="Universal correction",
        content="Preserve verbatim user constraints.",
        scopes=["all"],
    )
    unrelated = make_prompt_instruction(
        config,
        title="Status only",
        content="Explain worker status briefly.",
        scopes=["communication_status"],
    )
    state.prompt_instructions.extend([action, universal, unrelated])
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="action",
        system_instruction="Choose the next action.",
        components=[
            PromptComponent(
                name="request",
                category="current_user",
                text="Continue the task.",
            )
        ],
    )

    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(12_000, "test"),
    )

    exact = next(
        item for item in assembly.components if item.name == "durable_prompt_instructions"
    )
    assert action.content in exact.text
    assert universal.content in exact.text
    assert unrelated.content not in exact.text
    assert any(
        item.name == "durable_prompt_instructions"
        for item in compilation.report.breakdown
    )
    system_message = runtime._assembly_chat_messages(assembly)[0]
    assert system_message["role"] == "system"
    assert action.content in system_message["content"]
    assert any(
        item.source == "durable_prompt_instructions:action"
        for item in assembly.prompt_artifacts
    )
    selected_event = runtime.history.read_history(state.session_id)[-1]
    assert selected_event.event_type == "prompt_instructions_selected"
    assert selected_event.payload["instruction_ids"] == [
        universal.instruction_id,
        action.instruction_id,
    ]
    assert selected_event.payload["exact"] is True
    event_count = state.event_count

    runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(12_000, "test"),
    )
    assert state.event_count == event_count


def test_central_compiler_semantically_selects_fine_grained_instruction_categories(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000)
    client = _InstructionSelectionClient()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    programming = make_prompt_instruction(
        config,
        title="Programming discipline",
        content="Reproduce defects and test every implementation change.",
        scopes=["action"],
        categories=["programming", "implementation", "testing"],
    )
    reporting = make_prompt_instruction(
        config,
        title="User reporting",
        content="Report complete user-relevant information without internal noise.",
        scopes=["action"],
        categories=["user-reporting"],
    )
    call_wide = make_prompt_instruction(
        config,
        title="Action-wide safety",
        content="Never claim an unverified result.",
        scopes=["action"],
        categories=[],
    )
    state.prompt_instructions.extend([programming, reporting, call_wide])
    client.selected = [
        {
            "instruction_store": "session",
            "instruction_id": programming.instruction_id,
        }
    ]
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="action",
        system_instruction="Choose and execute the next useful action.",
        components=[
            PromptComponent(
                name="request",
                category="current_user",
                text="Implement the parser repair and run its tests.",
            )
        ],
    )

    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(12_000, "test"),
    )

    assert compilation.report.fits is True
    exact = next(
        item for item in assembly.components if item.name == "durable_prompt_instructions"
    )
    assert programming.content in exact.text
    assert call_wide.content in exact.text
    assert reporting.content not in exact.text
    assert len(client.requests) == 1
    selector_prompt = str(client.requests[0]["prompt"])
    assert programming.content in selector_prompt
    assert reporting.content in selector_prompt
    assert "Implement the parser repair and run its tests." in selector_prompt
    assert "[DURABLE MODEL-AUTHORED INSTRUCTIONS" not in selector_prompt
    selected = next(
        event
        for event in reversed(runtime.history.read_history(state.session_id))
        if event.event_type == "prompt_instructions_selected"
    )
    assert selected.payload["semantic_selection"] is True
    assert selected.payload["operation_categories"] == [
        "programming",
        "implementation",
    ]
    assert selected.payload["instruction_ids"] == [
        call_wide.instruction_id,
        programming.instruction_id,
    ]


def test_instruction_selector_failure_conservatively_includes_every_candidate(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000, model__max_retries=0)
    client = _InstructionSelectionClient()
    client.fail_selection = True
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    first = make_prompt_instruction(
        config,
        title="Programming",
        content="Test implementation changes.",
        scopes=["action"],
        categories=["programming"],
    )
    second = make_prompt_instruction(
        config,
        title="Reporting",
        content="Keep user reports complete.",
        scopes=["action"],
        categories=["user-reporting"],
    )
    state.prompt_instructions.extend([first, second])
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

    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(12_000, "test"),
    )

    assert compilation.report.fits is True
    exact = next(
        item for item in assembly.components if item.name == "durable_prompt_instructions"
    )
    assert first.content in exact.text
    assert second.content in exact.text
    events = runtime.history.read_history(state.session_id)
    failure = next(
        event
        for event in events
        if event.event_type == "prompt_instruction_selection_failed"
    )
    assert failure.payload["fallback"] == "include_all_scoped_candidates"
    selected = next(
        event for event in events if event.event_type == "prompt_instructions_selected"
    )
    assert selected.payload["semantic_selection"] is False
    assert selected.payload["selection_fallback"] is True


def test_default_config_exposes_prompt_instruction_capability(make_config) -> None:
    config = make_config()
    assert "prompt_instructions" in config.tools.enabled
    assert config.prompt_instructions.max_instructions > 0
    assert "prompt_instructions" in ToolRegistry().tool_names(config)


def test_user_prompt_instructions_cross_session_boundaries_and_remain_event_sourced(
    make_config,
) -> None:
    config = make_config(model__context_limit=12_000)
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    author = runtime.create_or_load_session()
    reader = runtime.create_or_load_session()
    _, added = ToolRegistry().dispatch(
        "prompt_instructions",
        _tool_input(
            "add",
            instruction_store="user",
            title="User reporting correction",
            content="Do not expose operational identifiers unless requested.",
            scopes=["communication_status"],
        ),
        config,
        author,
    )
    event = added.generated_events[0]
    runtime.history.record_event(author, event.event_type, event.payload)
    instruction_id = added.output["instruction_id"]

    rebuilt_author = runtime.history.rebuild_from_history(
        author.session_id,
        prefer_checkpoint=False,
    )
    assert rebuilt_author.prompt_instructions == []
    shared = PromptInstructionStore(config.sessions.root, config).list()
    assert [item.instruction_id for item in shared] == [instruction_id]
    assert shared[0].metadata["instruction_store"] == "user"

    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="communication_status",
        system_instruction="Explain status.",
        components=[
            PromptComponent(
                name="status_request",
                category="current_user",
                text="What is happening?",
            )
        ],
    )
    runtime._compile_context(
        reader,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(12_000, "test"),
    )
    exact = next(
        item
        for item in assembly.components
        if item.name == "durable_prompt_instructions"
    )
    assert "User reporting correction" in exact.text
    assert '"instruction_store": "user"' in exact.text
    selected = runtime.history.read_history(reader.session_id)[-1]
    assert selected.payload["instruction_sources"] == [
        {
            "instruction_store": "user",
            "instruction_id": instruction_id,
        }
    ]

    created_at = shared[0].created_at
    _, replaced = ToolRegistry().dispatch(
        "prompt_instructions",
        _tool_input(
            "replace",
            instruction_store="user",
            instruction_id=instruction_id,
            title="User reporting correction",
            content="Report only meaningful evidence unless details are requested.",
            scopes=["communication_status"],
        ),
        config,
        reader,
    )
    replaced_event = replaced.generated_events[0]
    runtime.history.record_event(
        reader,
        replaced_event.event_type,
        replaced_event.payload,
    )
    replaced_instruction = PromptInstructionStore(
        config.sessions.root,
        config,
    ).list()[0]
    assert replaced_instruction.created_at == created_at
    assert replaced_instruction.content.startswith("Report only")

    _, removed = ToolRegistry().dispatch(
        "prompt_instructions",
        _tool_input(
            "remove",
            instruction_store="user",
            instruction_id=instruction_id,
        ),
        config,
        reader,
    )
    removed_event = removed.generated_events[0]
    runtime.history.record_event(reader, removed_event.event_type, removed_event.payload)
    assert PromptInstructionStore(config.sessions.root, config).list() == []
    actions = [
        item.action
        for item in PromptInstructionStore(config.sessions.root, config).events()
    ]
    assert actions == ["add", "replace", "remove"]


def test_user_prompt_instruction_store_rejects_tampered_event_chain(
    make_config,
) -> None:
    config = make_config()
    store = PromptInstructionStore(config.sessions.root, config)
    store.add(
        title="Exact correction",
        content="Preserve all requested evidence.",
        scopes=["action"],
        origin_session_id="session_origin",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE prompt_instruction_events SET instruction_id='tampered' WHERE sequence=1"
        )
        connection.commit()
    with pytest.raises(PromptInstructionStoreError, match="hash verification failed"):
        store.list()


def test_trusted_prompt_instruction_authority_requires_provenance_and_orders_deterministically(make_config):
    from swaag.prompt_instructions import (
        make_prompt_instruction,
        sort_prompt_instructions_by_authority,
    )
    config = make_config()
    learned = make_prompt_instruction(
        config,
        title="Learned",
        content="Prefer concise output.",
        scopes=["communication_status"],
    )
    with pytest.raises(PromptInstructionError, match="require source_kind and source_ref"):
        make_prompt_instruction(
            config,
            title="Recording",
            content="Always include the blocker.",
            scopes=["communication_status"],
            authority="voice_recording",
        )
    recording_old = make_prompt_instruction(
        config,
        title="Recording old",
        content="Use format A.",
        scopes=["communication_status"],
        authority="voice_recording",
        source_kind="voicebutton_recording",
        source_ref="recording-1",
        specificity=50,
    )
    recording_new_specific = make_prompt_instruction(
        config,
        title="Recording specific",
        content="Use format B for benchmark status.",
        scopes=["communication_status"],
        authority="voice_recording",
        source_kind="voicebutton_recording",
        source_ref="recording-2",
        specificity=90,
    )
    correction = make_prompt_instruction(
        config,
        title="Explicit correction",
        content="Never claim semantic certainty from runtime state.",
        scopes=["communication_status"],
        authority="explicit_user_correction",
        source_kind="direct_user_correction",
        source_ref="conversation:turn-7",
        specificity=80,
    )
    ordered = sort_prompt_instructions_by_authority(
        [learned, recording_old, recording_new_specific, correction]
    )
    assert [item.title for item in ordered] == [
        "Explicit correction",
        "Recording specific",
        "Recording old",
        "Learned",
    ]


def test_trusted_store_ingestion_is_explicit_and_legacy_rows_remain_learned(make_config):
    config = make_config()
    store = PromptInstructionStore(config.sessions.root, config)
    legacy = store.add(
        title="Learned user-store rule",
        content="Prefer direct language.",
        scopes=["communication_status"],
        origin_session_id="session-model",
    ).instruction
    assert legacy is not None and legacy.authority == "learned_model"
    trusted = store.add_trusted(
        title="Recorded project rule",
        content="Status answers must distinguish mechanical facts from semantic conclusions.",
        scopes=["communication_status"],
        authority="voice_recording",
        source_kind="voicebutton_recording",
        source_ref="/data/recordings/project/2026-09-02T20-00.ogg",
        specificity=70,
    ).instruction
    assert trusted is not None
    rebuilt = store.list()
    assert rebuilt[0].authority == "learned_model"
    assert rebuilt[1].authority == "voice_recording"
    assert rebuilt[1].source_ref.endswith(".ogg")
    with pytest.raises(PromptInstructionError, match="trusted ingestion cannot create authority"):
        store.add_trusted(
            title="Fake",
            content="x",
            scopes=["communication_status"],
            authority="learned_model",
            source_kind="fake",
            source_ref="fake",
        )


def test_trusted_instruction_bypasses_semantic_selector(make_config):
    config = make_config()
    client = _InstructionSelectionClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    runtime.prompt_instruction_store.add_trusted(
        title="Recording authority",
        content="Include exact blocker evidence.",
        scopes=["communication_status"],
        categories=["reporting"],
        authority="voice_recording",
        source_kind="voicebutton_recording",
        source_ref="recording:42",
        specificity=80,
    )
    runtime.prompt_instruction_store.add(
        title="Learned categorized rule",
        content="Use bullet summaries.",
        scopes=["communication_status"],
        categories=["reporting"],
        origin_session_id=state.session_id,
    )
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="communication_status",
        system_instruction="Explain status.",
        components=[PromptComponent(name="q", category="current_user", text="What happened?")],
    )
    client.selected = []
    runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(12_000, "test"),
    )
    exact = next(c for c in assembly.components if c.name == "durable_prompt_instructions")
    assert "Recording authority" in exact.text
    assert "Learned categorized rule" not in exact.text
    assert "Trusted recording/user/project instructions are never semantically deselected" in exact.text
