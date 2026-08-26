from __future__ import annotations

from swaag.prompts import PromptBuilder
from swaag.types import Message


def test_tool_message_is_separate_accounting_component_with_source_event(make_config):
    builder = PromptBuilder(make_config())
    messages = [
        Message(role="user", content="do it", created_at="t0"),
        Message(
            role="tool",
            name="read_file",
            content="important result",
            created_at="t1",
            metadata={"source_event_sequence": 42, "source_event_hash": "abc"},
        ),
    ]
    assembly = builder.build_agent_action_prompt(
        messages,
        [],
        original_request="do it",
        pending_user_messages=[],
        prompt_mode="standard",
    )
    matches = [component for component in assembly.components if component.name == "current_turn_tool_event_42"]
    assert len(matches) == 1
    assert matches[0].category == "tool_result"
    assert "SOURCE EVENT sequence=42 hash=abc" in matches[0].text
    assert "important result" in matches[0].text


def test_non_tool_messages_remain_separate_components(make_config):
    builder = PromptBuilder(make_config())
    messages = [
        Message(role="user", content="current", created_at="t0"),
        Message(role="assistant", content="working", created_at="t1"),
    ]
    assembly = builder.build_agent_action_prompt(
        messages,
        [],
        original_request="current",
        pending_user_messages=[],
        prompt_mode="standard",
    )
    assert any(c.name == "current_turn_1" and c.category == "turn_context" for c in assembly.components)
