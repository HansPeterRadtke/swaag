from __future__ import annotations

from swaag.grammar import tool_result_projection_contract
from swaag.prompts import PromptBuilder
from swaag.types import Message


def test_projection_contract_is_closed_json_schema():
    contract = tool_result_projection_contract()
    assert contract.name == "tool_result_projection"
    assert contract.json_schema["additionalProperties"] is False
    assert contract.json_schema["required"] == ["projection"]


def test_prompt_builder_substitutes_projection_but_keeps_source_reference(make_config):
    builder = PromptBuilder(make_config())
    messages = [
        Message(role="user", content="find the answer", created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="RAW BULK " * 500,
            created_at="t1",
            metadata={"source_event_sequence": 77, "source_event_hash": "deadbeef"},
        ),
    ]
    assembly = builder.build_agent_action_prompt(
        messages,
        [],
        original_request="find the answer",
        pending_user_messages=[],
        prompt_mode="standard",
        tool_result_projections={77: "only the semantically relevant fact"},
    )
    assert "SOURCE EVENT sequence=77 hash=deadbeef" in assembly.prompt_text
    assert "SEMANTIC PROJECTION" in assembly.prompt_text
    assert "only the semantically relevant fact" in assembly.prompt_text
    assert "RAW BULK RAW BULK" not in assembly.prompt_text


def test_projection_prompt_contains_goal_source_and_target(make_config):
    builder = PromptBuilder(make_config())
    assembly = builder.build_tool_result_projection_prompt(
        original_request="locate the failing test",
        tool_name="shell_command",
        raw_tool_result="lots of output",
        source_event_sequence=12,
        source_event_hash="abc",
        target_tokens=222,
    )
    assert assembly.kind == "tool_result_projection"
    assert "locate the failing test" in assembly.prompt_text
    assert "sequence=12 hash=abc" in assembly.prompt_text
    assert "222 tokens" in assembly.prompt_text
