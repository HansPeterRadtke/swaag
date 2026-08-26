from __future__ import annotations

from swaag.action import action_from_payload
from swaag.grammar import agent_action_contract


def _payload(questions):
    return {
        "assistant_message": "Need input", "tool_calls": [], "continue_loop": False, "silent_completion": False,
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
