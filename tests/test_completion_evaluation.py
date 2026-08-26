from __future__ import annotations

from swaag.grammar import completion_evaluation_contract
from swaag.prompts import PromptBuilder


def test_completion_contract_requires_semantic_decision_and_remaining_work():
    schema = completion_evaluation_contract().json_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"complete", "reason", "remaining_work"}


def test_completion_prompt_contains_goal_candidate_and_evidence(make_config):
    builder = PromptBuilder(make_config())
    assembly = builder.build_completion_evaluation_prompt(
        original_request="make all tests pass",
        assistant_message="done",
        status_json='{"importance":"normal"}',
        tool_evidence='[{"tool_name":"run_tests","output":{"passed":false}}]',
    )
    assert assembly.kind == "completion_evaluation"
    assert "make all tests pass" in assembly.prompt_text
    assert "run_tests" in assembly.prompt_text
    assert "passed" in assembly.prompt_text


def test_runtime_defaults_enable_completion_evaluation(make_config):
    config = make_config(runtime__completion_evaluation_enabled=True)
    assert config.runtime.completion_evaluation_enabled is True
