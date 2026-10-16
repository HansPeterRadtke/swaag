from __future__ import annotations

from swaag.prompts import PromptBuilder
from swaag.types import Message


def test_compact_headers_preserve_order_verbatim_content_and_retrieval_identity(make_config):
    builder = PromptBuilder(make_config())
    messages = [
        Message(role='user', content='Keep  both spaces.\nDo not publish.', created_at='t0',
                metadata={'source_message_event_sequence': 41}),
        Message(role='tool', name='read_file', content='FAILED\n  exact evidence\n', created_at='t1',
                metadata={'source_event_sequence': 42, 'source_event_hash': 'abc'}),
        Message(role='assistant', content='The check failed.', created_at='t2'),
    ]
    components = builder.message_prompt_components(messages, prefix='history',
        category='history', header='Exact history:', compact_provenance=True)[1:]
    assert [c.text for c in components] == [
        '[USER event 41]\nKeep  both spaces.\nDo not publish.\n\n',
        '[TOOL:read_file event 42]\nFAILED\n  exact evidence\n\n\n',
        '[ASSISTANT]\nThe check failed.\n\n',
    ]
    assert components[1].name == 'history_tool_event_42'
    assert components[1].category == 'tool_result'


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
