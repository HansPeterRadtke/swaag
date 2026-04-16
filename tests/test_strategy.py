from __future__ import annotations

import pytest

from swaag.failure import FailureClassification
from swaag.planner import plan_from_payload
from swaag.strategy import (
    StrategyValidationError,
    adapt_strategy,
    available_profiles,
    build_strategy_from_profile,
    reconcile_strategy_to_plan,
    select_strategy_emergency_default,
    strategy_from_payload,
    validate_plan_against_strategy,
)
from swaag.types import SessionMetrics


def test_select_strategy_emergency_default_returns_safe_generic_profile() -> None:
    strategy = select_strategy_emergency_default()

    assert strategy.task_profile == "generic"
    assert strategy.strategy_name == "conservative"
    assert strategy.expected_flow == ["respond"]


def test_build_strategy_from_profile_materialises_each_catalog_profile() -> None:
    for profile_name in available_profiles():
        strategy = build_strategy_from_profile(profile_name, reason="test")
        assert strategy.task_profile == profile_name
        assert strategy.expected_flow, f"profile {profile_name} must have a flow"
        assert strategy.allowed_tools, f"profile {profile_name} must have allowed tools"


def test_strategy_from_payload_materialises_coding_profile() -> None:
    payload = {
        "task_profile": "coding",
        "strategy_name": "exploratory",
        "explore_before_commit": True,
        "tool_chain_depth": 2,
        "verification_intensity": 0.95,
        "reason": "llm_chose_coding",
    }
    strategy = strategy_from_payload(payload)

    assert strategy.task_profile == "coding"
    assert strategy.strategy_name == "exploratory"
    assert strategy.mode == "exploratory"
    assert strategy.explore_before_commit is True
    assert "edit_text" in strategy.allowed_tools
    assert strategy.expected_flow == ["read", "write", "respond"]


def test_strategy_from_payload_rejects_unknown_profile() -> None:
    with pytest.raises(StrategyValidationError):
        strategy_from_payload({"task_profile": "definitely_not_a_profile"})


def test_strategy_from_payload_rejects_unknown_strategy_name() -> None:
    with pytest.raises(StrategyValidationError):
        strategy_from_payload({"task_profile": "coding", "strategy_name": "wild_guess"})


def test_strategy_switches_to_recovery_after_failure() -> None:
    strategy = build_strategy_from_profile("generic", reason="test")

    adapted = adapt_strategy(
        strategy,
        failure=FailureClassification(
            kind="tool_failure",
            retryable=True,
            requires_replan=False,
            suggested_strategy_mode="recovery",
            reason="tool timeout",
        ),
        metrics=SessionMetrics(),
        verification_failed=False,
    )

    assert adapted.mode == "recovery"
    assert adapted.verification_intensity == 1.0


def test_strategy_validation_rejects_plan_that_skips_required_flow() -> None:
    strategy = build_strategy_from_profile("coding", reason="test")
    plan = plan_from_payload(
        {
            "goal": "Read src/app.py, update the code, and add tests.",
            "success_criteria": "done",
            "fallback_strategy": "replan",
            "steps": [
                {
                    "step_id": "step_answer",
                    "title": "Answer",
                    "goal": "Answer",
                    "kind": "respond",
                    "expected_tool": "",
                    "input_text": "answer",
                    "expected_output": "done",
                    "expected_outputs": ["done"],
                    "done_condition": "assistant_response_nonempty",
                    "success_criteria": "done",
                    "verification_type": "llm_fallback",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                        {"name": "meets_success_criteria", "check_type": "criterion", "criterion": "done"},
                        {"name": "satisfies_done_condition", "check_type": "criterion", "criterion": "assistant_response_nonempty"},
                    ],
                    "required_conditions": ["dependencies_completed", "assistant_text_nonempty", "meets_success_criteria", "satisfies_done_condition"],
                    "optional_conditions": [],
                    "input_refs": [],
                    "output_refs": [],
                    "fallback_strategy": "replan",
                    "depends_on": [],
                }
            ],
        },
        available_tools=["read_text", "edit_text", "calculator", "notes", "echo", "time_now"],
    )

    with pytest.raises(StrategyValidationError):
        validate_plan_against_strategy(plan, strategy)


def test_strategy_validation_allows_replan_that_preserves_completed_required_flow() -> None:
    strategy = build_strategy_from_profile("multi_step", reason="test")
    plan = plan_from_payload(
        {
            "goal": "Read sample.txt, edit it, compute 6 * 7, and answer.",
            "success_criteria": "done",
            "fallback_strategy": "replan",
            "steps": [
                {
                    "step_id": "step_edit",
                    "title": "Edit",
                    "goal": "Edit",
                    "kind": "write",
                    "expected_tool": "edit_text",
                    "input_text": "edit",
                    "expected_output": "done",
                    "expected_outputs": ["done"],
                    "done_condition": "tool_result:edit_text",
                    "success_criteria": "done",
                    "verification_type": "composite",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                        {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
                        {"name": "output_schema_valid", "check_type": "tool_output_schema_valid"},
                    ],
                    "required_conditions": [
                        "dependencies_completed",
                        "tool_result_present",
                        "tool_name_matches",
                        "output_nonempty",
                        "output_schema_valid",
                    ],
                    "optional_conditions": [],
                    "input_refs": [],
                    "output_refs": [],
                    "fallback_strategy": "replan",
                    "depends_on": [],
                },
                {
                    "step_id": "step_answer",
                    "title": "Answer",
                    "goal": "Answer",
                    "kind": "respond",
                    "expected_tool": "",
                    "input_text": "answer",
                    "expected_output": "done",
                    "expected_outputs": ["done"],
                    "done_condition": "assistant_response_nonempty",
                    "success_criteria": "done",
                    "verification_type": "llm_fallback",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                        {"name": "meets_success_criteria", "check_type": "criterion", "criterion": "done"},
                        {"name": "satisfies_done_condition", "check_type": "criterion", "criterion": "assistant_response_nonempty"},
                    ],
                    "required_conditions": [
                        "dependencies_completed",
                        "assistant_text_nonempty",
                        "meets_success_criteria",
                        "satisfies_done_condition",
                    ],
                    "optional_conditions": [],
                    "input_refs": [],
                    "output_refs": [],
                    "fallback_strategy": "replan",
                    "depends_on": ["step_edit"],
                },
            ],
        },
        available_tools=["read_text", "edit_text", "calculator", "notes", "echo", "time_now"],
    )

    validate_plan_against_strategy(plan, strategy, completed_step_kinds=["read", "tool"])


def test_reconcile_strategy_to_plan_prefers_structurally_compatible_profile() -> None:
    strategy = strategy_from_payload(
        {
            "task_profile": "multi_step",
            "strategy_name": "conservative",
            "explore_before_commit": False,
            "tool_chain_depth": 1,
            "verification_intensity": 0.9,
            "reason": "llm picked a broad profile",
        }
    )
    plan = plan_from_payload(
        {
            "goal": "Use calculator to compute 6 * 7",
            "success_criteria": "Return 42",
            "fallback_strategy": "replan",
            "steps": [
                {
                    "step_id": "step_calc",
                    "title": "Compute",
                    "goal": "Compute 6 * 7",
                    "kind": "tool",
                    "expected_tool": "calculator",
                    "input_text": "6 * 7",
                    "expected_output": "42",
                    "expected_outputs": ["42"],
                    "done_condition": "tool_result:calculator",
                    "success_criteria": "calculator returns 42",
                    "verification_type": "composite",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                        {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
                    ],
                    "required_conditions": [
                        "dependencies_completed",
                        "tool_result_present",
                        "tool_name_matches",
                        "output_nonempty",
                    ],
                    "optional_conditions": [],
                    "input_refs": [],
                    "output_refs": [],
                    "fallback_strategy": "replan",
                    "depends_on": [],
                },
                {
                    "step_id": "step_answer",
                    "title": "Answer",
                    "goal": "Return the answer",
                    "kind": "respond",
                    "expected_tool": "",
                    "input_text": "answer",
                    "expected_output": "42",
                    "expected_outputs": ["42"],
                    "done_condition": "assistant_response_nonempty",
                    "success_criteria": "answer is 42",
                    "verification_type": "llm_fallback",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                    ],
                    "required_conditions": ["dependencies_completed", "assistant_text_nonempty"],
                    "optional_conditions": [],
                    "input_refs": [],
                    "output_refs": [],
                    "fallback_strategy": "replan",
                    "depends_on": ["step_calc"],
                },
            ],
        },
        available_tools=["calculator", "read_text", "respond", "echo"],
    )

    reconciled = reconcile_strategy_to_plan(strategy, plan)

    assert reconciled.task_profile == "reading"
    assert reconciled.strategy_name == "conservative"
    validate_plan_against_strategy(plan, reconciled)


def test_reconcile_strategy_to_plan_rejects_downgrade_that_drops_file_edit_commitment() -> None:
    strategy = strategy_from_payload(
        {
            "task_profile": "file_edit",
            "strategy_name": "conservative",
            "explore_before_commit": False,
            "tool_chain_depth": 1,
            "verification_intensity": 0.95,
            "reason": "llm picked file_edit",
        }
    )
    plan = plan_from_payload(
        {
            "goal": "Inspect src/app.py and answer what to change.",
            "success_criteria": "The next edit is identified.",
            "fallback_strategy": "replan",
            "steps": [
                {
                    "step_id": "step_read",
                    "title": "Read the file",
                    "goal": "Read src/app.py",
                    "kind": "read",
                    "expected_tool": "read_text",
                    "input_text": "src/app.py",
                    "expected_output": "File text",
                    "expected_outputs": ["File text"],
                    "done_condition": "tool_result:read_text",
                    "success_criteria": "The file is read.",
                },
                {
                    "step_id": "step_answer",
                    "title": "Answer",
                    "goal": "Describe the change",
                    "kind": "respond",
                    "expected_tool": "",
                    "input_text": "answer",
                    "expected_output": "Change description",
                    "expected_outputs": ["Change description"],
                    "done_condition": "assistant_response_nonempty",
                    "success_criteria": "The answer describes the missing edit.",
                    "depends_on": ["step_read"],
                },
            ],
        },
        available_tools=["read_text", "edit_text", "write_file", "echo"],
    )

    with pytest.raises(StrategyValidationError, match="semantic commitment"):
        reconcile_strategy_to_plan(strategy, plan)
