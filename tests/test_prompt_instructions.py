from __future__ import annotations

import sqlite3

import pytest

from swaag.grammar import yes_no_contract
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
from swaag.types import PromptComponent


def _tool_input(
    action: str,
    *,
    instruction_store: str = "session",
    instruction_id: str | None = None,
    title: str | None = None,
    content: str | None = None,
    scopes: list[str] | None = None,
) -> dict:
    return {
        "action": action,
        "instruction_store": instruction_store,
        "instruction_id": instruction_id,
        "title": title,
        "content": content,
        "scopes": scopes,
    }


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
        action.instruction_id,
        universal.instruction_id,
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
