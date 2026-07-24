import pytest

from swaag.grammar import plan_contract, prompt_analysis_contract, task_decision_contract, tool_input_contract, task_decision_semantic_review_contract, verification_contract
from swaag.schema_portability import assert_portable_json_schema
from swaag.tools.base import ToolValidationError
from swaag.tools.builtin import EditTextTool


def test_edit_text_tool_input_contract_is_portable_closed_schema() -> None:
    contract = tool_input_contract("edit_text", EditTextTool.input_schema)
    schema = contract.json_schema
    assert schema is not None

    assert_portable_json_schema(schema, schema_name=contract.name)
    assert set(schema["required"]) == set(schema["properties"])
    assert "allOf" not in schema
    assert "if" not in schema
    assert "const" not in schema
    assert "replace_exact" in schema["properties"]["operation"]["enum"]
    assert "old_text" in schema["required"]
    assert "new_text" in schema["required"]


def test_edit_text_operation_requirements_are_validated_locally() -> None:
    tool = EditTextTool()

    with pytest.raises(ToolValidationError, match="requires pattern and replacement"):
        tool.validate(
            {
                "path": "sample.py",
                "operation": "replace_pattern_once",
                "dry_run": False,
                "old_text": None,
                "new_text": None,
                "start": None,
                "end": None,
                "position": None,
                "expected_text": None,
                "pattern": "return 0",
                "replacement": None,
                "insertion": None,
            }
        )

    validated = tool.validate(
        {
            "path": "sample.py",
            "operation": "replace_pattern_once",
                "dry_run": False,
                "old_text": None,
                "new_text": None,
                "start": None,
                "end": None,
            "position": None,
            "expected_text": None,
            "pattern": "return 0",
            "replacement": "return 1",
            "insertion": None,
        }
    )

    assert validated["replacement"] == "return 1"

    with pytest.raises(ToolValidationError, match="requires non-empty old_text"):
        tool.validate(
            {
                "path": "sample.py",
                "operation": "replace_exact",
                "dry_run": False,
                "old_text": "",
                "new_text": "return 1",
                "start": None,
                "end": None,
                "position": None,
                "expected_text": None,
                "pattern": None,
                "replacement": None,
                "insertion": None,
            }
        )


def test_verification_contract_supports_multiple_exact_candidate_excerpts() -> None:
    contract = verification_contract(["criterion_a"])
    assert contract.json_schema is not None
    item = contract.json_schema["properties"]["criteria"]["items"]
    assert item["properties"]["candidate_excerpts"] == {
        "type": "array", "items": {"type": "string"}
    }
    assert "candidate_excerpts" in item["required"]
    assert "candidate_excerpt" not in item["properties"]


def test_task_decision_semantic_review_contract_is_closed_and_portable() -> None:
    contract = task_decision_semantic_review_contract()
    assert contract.json_schema is not None
    assert contract.json_schema["additionalProperties"] is False
    assert set(contract.json_schema["required"]) == set(contract.json_schema["properties"])
    assert contract.json_schema["properties"]["required_evidence_sources"] == {
        "type": "array", "items": {"type": "string"}
    }
    assert contract.json_schema["properties"]["minimum_evidence_call_count"] == {"type": "integer"}


def test_prompt_analysis_and_task_decision_contracts_expose_evidence_fields() -> None:
    analysis = prompt_analysis_contract()
    decision = task_decision_contract(["read_file"])
    assert analysis.json_schema is not None and decision.json_schema is not None
    assert analysis.json_schema["properties"]["missing_required_information"] == {"type": "boolean"}
    assert "missing_required_information" in analysis.json_schema["required"]
    assert decision.json_schema["properties"]["evidence_required_before_response"] == {"type": "boolean"}
    assert "evidence_required_before_response" in decision.json_schema["required"]
    assert decision.json_schema["properties"]["evidence_call_count"] == {"type": "integer"}
    assert "evidence_call_count" in decision.json_schema["required"]


def test_plan_contract_exposes_all_payload_fields_consumed_by_verifier() -> None:
    contract = plan_contract(["read_file", "edit_text"])
    assert contract.json_schema is not None
    check_schema = contract.json_schema["properties"]["steps"]["items"]["properties"]["verification_checks"]["items"]
    for field in ("schema_json", "function_name", "symbol", "regex"):
        assert field in check_schema["properties"]
        assert field in check_schema["required"]


def test_plan_contract_exposes_required_tool_name_expected_field() -> None:
    contract = plan_contract(["read_file", "edit_text"])
    assert contract.json_schema is not None
    check_schema = contract.json_schema["properties"]["steps"]["items"]["properties"]["verification_checks"]["items"]
    assert check_schema["properties"]["expected"] == {"type": "string"}
    assert "expected" in check_schema["required"]


def test_plan_contract_includes_registered_tool_effect_check_type() -> None:
    contract = plan_contract(["edit_text"])
    assert contract.json_schema is not None
    check_type = contract.json_schema["properties"]["steps"]["items"]["properties"]["verification_checks"]["items"]["properties"]["check_type"]
    assert "tool_effect_verified" in check_type["enum"]


def test_verification_contract_constrains_candidate_excerpts_to_exact_options() -> None:
    contract = verification_contract(
        ["criterion_a"],
        candidate_excerpt_options=["exact evidence one", "exact evidence two"],
    )
    schema = contract.json_schema
    assert schema is not None
    item_schema = schema["properties"]["criteria"]["items"]
    assert item_schema["properties"]["candidate_excerpts"]["items"] == {
        "type": "string",
        "enum": ["exact evidence one", "exact evidence two"],
    }
