from __future__ import annotations

from swaag.prompts import PromptBuilder
from swaag.types import Message


def test_decision_prompt_does_not_duplicate_sections(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [
        Message(role="user", content="first", created_at="t1"),
        Message(role="assistant", content="reply", created_at="t2"),
        Message(role="user", content="second", created_at="t3"),
    ]
    prompt = builder.build_decision_prompt(messages, [("echo", "Echo text", {"type": "object"})], prompt_mode="standard")

    assert prompt.kind == "decision"
    assert prompt.prompt_text.count("Conversation history:") == 1
    assert prompt.prompt_text.count("Current user request:") == 1
    assert prompt.prompt_text.count("Available tools:") == 1


def test_answer_prompt_can_include_notes(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [Message(role="user", content="hello", created_at="t1")]
    prompt = builder.build_answer_prompt(messages, prompt_mode="standard", notes_block="[note] important")

    assert "Working notes:" in prompt.prompt_text
    assert "[note] important" in prompt.prompt_text


def test_decision_prompt_omits_tool_section_when_no_tools(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [Message(role="user", content="hello", created_at="t1")]
    prompt = builder.build_decision_prompt(messages, [], prompt_mode="lean")

    assert prompt.kind == "decision"
    assert "Available tools:" not in prompt.prompt_text


def test_tool_input_prompt_uses_tool_input_kind(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [Message(role="user", content="edit sample.py", created_at="t1")]
    prompt = builder.build_tool_input_prompt(messages, tool_name="edit_text", prompt_mode="lean")

    assert prompt.kind == "tool_input"


def test_summary_prompt_contains_history(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [
        Message(role="summary", content="older summary", created_at="t1"),
        Message(role="user", content="current user", created_at="t2"),
    ]
    prompt = builder.build_summary_prompt(messages)

    assert "Summarize this transcript for future continuation:" in prompt.prompt_text
    assert "older summary" in prompt.prompt_text
    assert "current user" in prompt.prompt_text
from swaag.types import PromptComponent



def test_prompt_can_include_context_components(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [Message(role="user", content="hello", created_at="t1")]
    prompt = builder.build_answer_prompt(
        messages,
        prompt_mode="standard",
        context_components=[PromptComponent(name="plan", category="plan", text="Active plan:\n- step\n\n")],
    )

    assert "Active plan:" in prompt.prompt_text


def test_analysis_prompt_contains_current_user_request(make_config) -> None:
    builder = PromptBuilder(make_config())
    prompt = builder.build_analysis_prompt("make a game", prompt_mode="lean")

    assert prompt.kind == "analysis"
    assert "Current user request:" in prompt.prompt_text
    assert "make a game" in prompt.prompt_text


def test_task_decision_prompt_contains_analysis(make_config) -> None:
    builder = PromptBuilder(make_config())
    prompt = builder.build_task_decision_prompt(
        "build a tool",
        '{"task_type":"structured"}',
        prompt_mode="lean",
    )

    assert prompt.kind == "task_decision"
    assert "Prompt analysis:" in prompt.prompt_text
    assert '"task_type":"structured"' in prompt.prompt_text


def test_task_expansion_prompt_contains_decision(make_config) -> None:
    builder = PromptBuilder(make_config())
    prompt = builder.build_task_expansion_prompt(
        "make a game",
        '{"task_type":"vague"}',
        '{"expand_task":true}',
        prompt_mode="lean",
    )

    assert prompt.kind == "expansion"
    assert "Task decision:" in prompt.prompt_text
    assert '{"expand_task":true}' in prompt.prompt_text
