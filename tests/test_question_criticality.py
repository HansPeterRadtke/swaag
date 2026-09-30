from __future__ import annotations

from swaag.action import action_from_payload
from swaag.grammar import agent_action_contract


def _payload(questions):
    return {
        "assistant_message": "Which target?", "tool_calls": [], "continue_loop": False, "silent_completion": False,
        "status": {"situation":"uncertain","action":"ask","reason":"needed","importance":"normal"},
        "questions": questions,
    }


def test_action_parses_semantic_question_criticality():
    action = action_from_payload(_payload([{
        "question":"Which target?", "criticality":"blocking", "reason":"two destructive targets", "assumption_if_unanswered":""
    }]), enabled_tool_names=[] )
    assert action.questions[0].criticality == "blocking"


def test_action_backwards_compatible_missing_questions():
    payload = _payload([]); payload.pop("questions")
    action = action_from_payload(payload, enabled_tool_names=[] )
    assert action.questions == []


def test_contract_requires_structured_questions():
    schema = agent_action_contract([]).json_schema
    assert "questions" in schema["required"]
    assert schema["properties"]["questions"]["items"]["properties"]["criticality"]["enum"] == ["optional", "blocking"]


def test_optional_question_requires_assumption_and_user_facing_disclosure():
    payload = _payload([{
        "question": "Which harmless format do you prefer?",
        "criticality": "optional",
        "reason": "A safe default exists.",
        "assumption_if_unanswered": "Use text format.",
    }])
    payload["assistant_message"] = (
        "Which harmless format do you prefer? Use text format. I can continue meanwhile."
    )
    action = action_from_payload(payload, enabled_tool_names=[])
    assert action.questions[0].assumption_if_unanswered == "Use text format."

    missing = _payload([{
        "question": "Which harmless format do you prefer?",
        "criticality": "optional",
        "reason": "A safe default exists.",
        "assumption_if_unanswered": "Use text format.",
    }])
    missing["assistant_message"] = "I can continue meanwhile."
    import pytest
    with pytest.raises(Exception, match="must disclose"):
        action_from_payload(missing, enabled_tool_names=[])


def test_blocking_question_forbids_provisional_assumption():
    payload = _payload([{
        "question": "Which target?",
        "criticality": "blocking",
        "reason": "Wrong target would be destructive.",
        "assumption_if_unanswered": "Guess production.",
    }])
    payload["assistant_message"] = "Which target?"
    import pytest
    with pytest.raises(Exception, match="must be empty for a blocking question"):
        action_from_payload(payload, enabled_tool_names=[])


def test_blocking_question_rejects_tool_calls_in_same_action():
    payload = _payload([{
        "question": "Which production target?",
        "criticality": "blocking",
        "reason": "Choosing the wrong target would be destructive.",
        "assumption_if_unanswered": "",
    }])
    payload["assistant_message"] = "Which production target?"
    payload["tool_calls"] = [{"tool_name": "read_file", "arguments": {"path": "README.md"}}]
    payload["continue_loop"] = True
    import pytest
    with pytest.raises(Exception, match="blocking questions cannot accompany tool calls"):
        action_from_payload(payload, enabled_tool_names=["read_file"])


def test_exact_word_count_response_constraint_is_mechanically_enforced():
    good = _payload([])
    good["assistant_message"] = "alpha beta gamma"
    good["response_constraints"] = {"exact_word_count": 3}
    action = action_from_payload(good, enabled_tool_names=[])
    assert action.response_constraints.exact_word_count == 3

    bad = dict(good)
    bad["response_constraints"] = {"exact_word_count": 4}
    import pytest
    with pytest.raises(Exception, match="3 whitespace-delimited words"):
        action_from_payload(bad, enabled_tool_names=[])


def test_action_contract_requires_response_constraints_for_new_model_calls():
    schema = agent_action_contract([]).json_schema
    assert "response_constraints" in schema["required"]
    exact = schema["properties"]["response_constraints"]["properties"]["exact_word_count"]
    assert {variant.get("type") for variant in exact["anyOf"]} == {"integer", "null"}
    assert all("minimum" not in variant for variant in exact["anyOf"])


def test_exact_word_sequence_contract_has_one_required_field_per_word():
    from swaag.grammar import exact_word_sequence_contract
    from swaag.schema_portability import assert_portable_json_schema

    contract = exact_word_sequence_contract(45)
    assert contract.name == "exact_word_sequence_45"
    assert len(contract.json_schema["properties"]) == 45
    assert len(contract.json_schema["required"]) == 45
    assert contract.json_schema["required"][0] == "word_001"
    assert contract.json_schema["required"][-1] == "word_045"
    assert_portable_json_schema(contract.json_schema, schema_name=contract.name)
