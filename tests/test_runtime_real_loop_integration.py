from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swaag.retrieval.embeddings import EmbeddingBackend, SemanticBackendProtocolError
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps

from tests.helpers import FakeModelClient, _normalize_scripted_plan_response, plan_response, plan_step


class _CodingSkillBackend(EmbeddingBackend):
    mode = "llm_scoring"
    degraded = False

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        del query
        scores: list[float] = []
        for text in texts:
            lowered = text.lower()
            if "repair broken code" in lowered or "exact file content change" in lowered:
                scores.append(0.95)
            elif "retrieve and ground answers in browser evidence" in lowered:
                scores.append(0.0)
            else:
                scores.append(0.1)
        return scores


def _tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    return json.dumps({"action": "call_tool", "response": "", "tool_name": tool_name, "tool_input": tool_input})


def _read_text_input(*, path: str | None = None, paths: list[str] | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "paths": paths,
        "note_id": None,
        "reader_id": None,
        "chunk_chars": 4096,
        "overlap_chars": 0,
        "start_offset": None,
        "end_offset": None,
    }


def _edit_replace_input(path: Path, pattern: str, replacement: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "operation": "replace_exact",
        "dry_run": False,
        "old_text": pattern,
        "new_text": replacement,
        "start": None,
        "end": None,
        "position": None,
        "expected_text": None,
        "replacement": None,
        "insertion": None,
        "pattern": None,
    }


def _edit_range_input(path: Path, *, start: int, end: int, expected_text: str, replacement: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "operation": "replace_range",
        "dry_run": False,
        "old_text": None,
        "new_text": None,
        "start": start,
        "end": end,
        "position": None,
        "expected_text": expected_text,
        "replacement": replacement,
        "insertion": None,
        "pattern": None,
    }


def _write_file_input(path: Path, content: str) -> dict[str, Any]:
    return {"path": str(path), "content": content, "create": True}


def _run_pytest_input(test_name: str) -> dict[str, Any]:
    return {"command": ["python", "-m", "pytest", "-q", test_name], "background": False}


def _exact_answer_step(step_id: str, answer: str, *, depends_on: list[str] | None = None) -> dict[str, Any]:
    return plan_step(
        step_id,
        "Answer",
        "respond",
        expected_output=answer,
        success_criteria=f"Return exactly {answer!r}.",
        depends_on=depends_on,
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": answer},
        ],
        required_conditions=["dependencies_completed", "answer_exact"],
        optional_conditions=[],
    )


def _exact_answer_with_file_check_step(
    step_id: str,
    answer: str,
    *,
    path: Path,
    pattern: str,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return plan_step(
        step_id,
        "Answer",
        "respond",
        expected_output=answer,
        success_criteria=f"Return exactly {answer!r} after the declared file state check passes.",
        depends_on=depends_on,
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "file_contains_expected_state", "check_type": "file_contains", "path": str(path), "pattern": pattern},
            {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": answer},
        ],
        required_conditions=["dependencies_completed", "file_contains_expected_state", "answer_exact"],
        optional_conditions=[],
    )


def _tool_names(events) -> list[str]:
    return [event.payload["tool_name"] for event in events if event.event_type == "tool_called"]


def _event_sequences(events, event_type: str, **payload_matches: Any) -> list[int]:
    sequences: list[int] = []
    for event in events:
        if event.event_type != event_type:
            continue
        if all(event.payload.get(key) == value for key, value in payload_matches.items()):
            sequences.append(event.sequence)
    return sequences


def test_real_loop_gathers_evidence_before_model_authored_clarification(make_config, tmp_path: Path) -> None:
    request = tmp_path / "request.txt"
    context = tmp_path / "context.txt"
    request.write_text("User request: make the release safer before tonight's rollout.\n", encoding="utf-8")
    context.write_text("Missing details: which service, what risk, and what success criterion.\n", encoding="utf-8")
    goal = "Read request.txt and context.txt, then ask the single most useful clarifying question before acting."
    question = 'Which service is being rolled out, which specific risk should be reduced, and what measurable success criterion defines a safer release?'
    read_request = plan_step(
        "read_request", "Read request", "read", expected_tool="read_file",
        expected_output="request text", success_criteria="The request text is observed.",
        verification_checks=[
            {"name": "tool_name", "check_type": "tool_name_equals", "expected": "read_file"},
            {"name": "output", "check_type": "tool_output_nonempty"},
        ], required_conditions=["tool_name", "output"], optional_conditions=[],
    )
    read_context = plan_step(
        "read_context", "Read context", "read", expected_tool="read_file",
        expected_output="context text", success_criteria="The missing details are observed.", depends_on=["read_request"],
        verification_checks=[
            {"name": "dependencies", "check_type": "dependencies_completed"},
            {"name": "tool_name", "check_type": "tool_name_equals", "expected": "read_file"},
            {"name": "output", "check_type": "tool_output_nonempty"},
        ], required_conditions=["dependencies", "tool_name", "output"], optional_conditions=[],
    )
    clarify = plan_step(
        "clarify", "Ask clarification", "respond", expected_output=question,
        success_criteria="Ask one grounded question covering service, risk, and success criterion.",
        depends_on=["read_context"],
        verification_checks=[
            {"name": "dependencies", "check_type": "dependencies_completed"},
            {"name": "answer", "check_type": "exact_match", "actual_source": "assistant_text", "expected": question},
        ], required_conditions=["dependencies", "answer"], optional_conditions=[],
    )
    client = FakeModelClient(contract_responses={
        "prompt_analysis": [json.dumps({
            "task_type": "incomplete", "completeness": "incomplete",
            "requires_expansion": False, "requires_decomposition": False,
            "missing_required_information": True,
            "confidence": 1.0, "detected_entities": ["request.txt", "context.txt"],
            "detected_goals": ["read evidence and ask clarification"],
        })],
        "task_decision": [json.dumps({
            "split_task": False, "expand_task": False, "ask_user": True,
            "assume_missing": False, "generate_ideas": False, "direct_response": False,
            "execution_mode": "full_plan", "preferred_tool_name": "",
            "evidence_required_before_response": True,
            "evidence_call_count": 2,
            "confidence": 1.0, "reason": "read both files before asking",
        })],
        "task_plan": [plan_response(goal=goal, steps=[read_request, read_context, clarify])],
        "tool_input:read_file": [json.dumps({"path": str(request)}), json.dumps({"path": str(context)})],
        "answer_response": [question],
    })
    runtime = AgentRuntime(make_config(model__context_limit=32_000, tools__allow_stateful_tools=True), model_client=client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    contracts = [item["contract"] for item in client.requests]

    assert result.assistant_text == question
    assert _tool_names(events) == ["read_file", "read_file"]
    assert contracts.count("clarification_response") == 0
    assert contracts.count("task_plan") == 1
    assert any(event.event_type == "filesystem_read" for event in events)
    assert not any(event.event_type in {"edit_applied", "file_write_applied"} for event in events)


def test_runtime_task_decision_selector_receives_complete_enabled_registry(make_config) -> None:
    """Difficulty: extremely_easy. Family: failure/semantic_authority."""
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    observed: dict[str, str] = {}

    def inspect_task_decision(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["prompt"] = prompt
        for fragment in [
            "- calculator",
            "Evaluate a basic arithmetic expression",
            '"expression":{"type":"string"}',
            "- edit_text",
            "Preview or apply a bounded text edit",
            '"operation":{"enum":["replace_exact","replace_range","insert_at","delete_range","replace_pattern_once","replace_pattern_all"],"type":"string"}',
            "- run_tests",
            "Run a test command inside the persistent workspace",
            '"command":{"items":{"type":"string"},"type":"array"}',
        ]:
            assert fragment in prompt
        return json.dumps(
            {
                "split_task": False,
                "expand_task": False,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": True,
                "execution_mode": "direct_response",
                "preferred_tool_name": "",
                "confidence": 0.9,
                "reason": "complete registry was available for this decision",
            }
        )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_decision": [inspect_task_decision],
                "task_plan": [
                    plan_response(
                        goal="Answer with the words registry observed.",
                        steps=[_exact_answer_step("answer", "registry observed")],
                    )
                ],
                "answer_response": ["registry observed"],
            }
        ),
    )

    result = runtime.run_turn("Answer with the words registry observed.")

    assert result.assistant_text == "registry observed"
    assert observed["prompt"].count("input_schema:") >= 6


def test_single_tool_decision_still_requires_model_plan_objective_verification(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_easy. Family: file_edit/failure/semantic_authority."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Edit {release} so status moves from draft to ready and nothing else changes; summarize final state."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_decision": [
                    json.dumps(
                        {
                            "split_task": False,
                            "expand_task": False,
                            "ask_user": False,
                            "assume_missing": False,
                            "generate_ideas": False,
                            "direct_response": False,
                            "execution_mode": "single_tool",
                            "preferred_tool_name": "edit_text",
                            "confidence": 0.9,
                            "reason": "one edit tool is appropriate, but the planner must declare objective verification",
                        }
                    )
                ],
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_status",
                                "Edit YAML status",
                                "write",
                                expected_tool="edit_text",
                                expected_output="status ready",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready_key", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_result_present", "tool_name_matches", "file_contains_ready_key"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "status ready", depends_on=["edit_status"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready")),
                ],
                "answer_response": ["status ready"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "status ready"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text"]
    assert plan_requests
    assert "execution_mode=single_tool" in plan_requests[0]["prompt"]
    assert "preferred_tool_name=edit_text" in plan_requests[0]["prompt"]
    assert "- calculator" in plan_requests[0]["prompt"]
    assert "- edit_text" in plan_requests[0]["prompt"]
    assert "- run_tests" in plan_requests[0]["prompt"]
    assert any(event.event_type == "subsystem_progress" and "preview_passed=False" in event.payload.get("progress", "") for event in events)
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "edit_status" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_real_loop_semantic_result_review_rejects_corrupt_partial_file_edit(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: file_edit/failure/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Read {release}, change the release status from draft to ready, and answer repaired."
    review_prompts: list[str] = []

    def semantic_review(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        review_prompts.append(prompt)
        assert "result_satisfies_step" in prompt
        assert "current_file" in prompt
        assert "diff" in prompt
        passed = "status: ready" in prompt and "+status: ready" in prompt
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "result_satisfies_step",
                        "passed": passed,
                        "evidence": "the current file either preserves the status key or it does not",
                    }
                ]
            }
        )

    def repaired_edit_decision(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        assert "semantic_result_review_failed" in prompt
        assert "name: report-62\\nready\\nowner: team-6\\n" in prompt
        return _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready"))

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_release",
                                "Read release",
                                "read",
                                expected_tool="read_text",
                                expected_output="release content",
                                success_criteria="The release file content is observed.",
                            ),
                            plan_step(
                                "edit_release",
                                "Edit release status",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready release file",
                                success_criteria="The release status is changed from draft to ready without corrupting the file.",
                                depends_on=["read_release"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["edit_release"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "repair_release",
                                "Repair release from current text",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["repair_release"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("read_text", _read_text_input(path=str(release))),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    repaired_edit_decision,
                ],
                "verification": [semantic_review, semantic_review],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_text", "edit_text", "edit_text"]
    assert len(review_prompts) == 2
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "semantic_result"
        and event.payload.get("passed") is False
        for event in events
    )
    assert any(
        event.event_type == "message_added"
        and "semantic_result_review_failed" in event.payload.get("message", {}).get("content", "")
        for event in events
    )
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="edit_release"))
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_release" for event in events)


def test_real_loop_model_disallowed_semantic_review_retry_replans_without_stale_mutation(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step/semantic_authority."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
        planner__max_replans=1,
    )
    goal = f"Edit {release} so status moves from draft to ready and answer status ready."
    observed: dict[str, str] = {}

    def contradictory_semantic_review(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["review_prompt"] = prompt
        assert "current_file" in prompt
        assert "status: ready" in prompt
        assert "+status: ready" in prompt
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "result_satisfies_step",
                        "passed": False,
                        "evidence": "The current artifact state satisfies the requested edit, but the boolean is false.",
                    }
                ]
            }
        )

    def no_retry_failure(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["failure_prompt"] = prompt
        assert "semantic_result_review_failed" in prompt
        assert "status: ready" in prompt
        return json.dumps(
            {
                "kind": "verification_failure",
                "retryable": False,
                "requires_replan": False,
                "suggested_strategy_mode": "conservative",
                "wait_seconds": 0.0,
                "reason": "The model judged the same mutation step should not be retried.",
            }
        )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_status",
                                "Edit YAML status",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready release file",
                                success_criteria="The release status is changed from draft to ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "status ready", depends_on=["edit_status"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            _exact_answer_with_file_check_step(
                                "answer",
                                "status ready",
                                path=release,
                                pattern="status: ready",
                            )
                        ],
                    ),
                ],
                "tool_input:edit_text": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "verification": [contradictory_semantic_review],
                "failure_classification": [no_retry_failure],
                "answer_response": ["status ready"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "status ready"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text"]
    assert observed["review_prompt"]
    assert observed["failure_prompt"]
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "semantic_result"
        and event.payload.get("passed") is False
        for event in events
    )
    assert any(
        event.event_type == "retry_suppressed"
        and event.payload.get("step_id") == "edit_status"
        and event.payload.get("reason") == "model_disallowed_same_step_retry"
        for event in events
    )
    assert not any(event.event_type == "retry_triggered" and event.payload.get("step_id") == "edit_status" for event in events)
    assert any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_status" for event in events)
    assert any(event.event_type == "replan_triggered" and event.payload.get("step_id") == "edit_status" for event in events)
    assert len(plan_requests) == 2
    assert "status: ready" in plan_requests[1]["prompt"]
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_pattern_not_found_after_corrupt_edit_feeds_current_text_and_repairs(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=5,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Read {release}, change only the status value from draft to ready, and answer repaired."
    observed: dict[str, str] = {}

    def repair_after_pattern_error(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["repair_prompt"] = prompt
        assert "tool_error:" in prompt
        assert "old_text not found" in prompt
        assert "match_count=0" in prompt
        assert "current_text" in prompt
        assert "name: report-62" in prompt
        assert "ready" in prompt
        assert "owner: team-6" in prompt
        return _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready"))

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_release",
                                "Read release",
                                "read",
                                expected_tool="read_text",
                                expected_output="release content",
                                success_criteria="The release file content is observed.",
                            ),
                            plan_step(
                                "edit_release",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                depends_on=["read_release"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["edit_release"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "repair_release",
                                "Repair release from current text",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["repair_release"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("read_text", _read_text_input(path=str(release))),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    repair_after_pattern_error,
                    repair_after_pattern_error,
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_text", "edit_text", "edit_text", "edit_text"]
    assert observed["repair_prompt"]
    assert any(
        event.event_type == "message_added"
        and "verification_preview_failed" in event.payload.get("message", {}).get("content", "")
        for event in events
    )
    assert any(
        event.event_type == "tool_error"
        and "old_text not found" in event.payload.get("error", "")
        and "match_count=0" in event.payload.get("error", "")
        and "current_text" in event.payload.get("error", "")
        for event in events
    )
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="edit_release"))
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "semantic_result"
        and event.payload.get("passed") is True
        for event in events
    )
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_release" for event in events)


def test_real_loop_replan_after_stale_source_errors_exposes_current_state_and_repairs(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=24,
        runtime__max_total_actions=40,
        runtime__max_repeated_action_occurrences=3,
        planner__max_replans=2,
    )
    goal = f"Edit {release} so the status moves from draft to ready and nothing else changes. Answer repaired."
    stale_edit = json.dumps(_edit_replace_input(release, "status: draft", "status: ready"))
    observed: dict[str, str] = {}

    def repair_after_replan(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["repair_prompt"] = prompt
        assert "Recent failed tool or verification evidence" in prompt
        assert "Latest observed file snapshots" in prompt
        assert "current_text" in prompt
        assert "old_text not found" in prompt
        assert "match_count=0" in prompt
        assert "name: report-62\\nready\\nowner: team-6" in prompt
        assert "do not repeat arguments that depend only on that stale target" in prompt
        assert "If the pattern is absent, choose an edit that matches the current file text" in prompt
        return json.dumps(_edit_replace_input(release, "ready", "status: ready"))

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_initial",
                                "Read initial release",
                                "read",
                                expected_tool="read_file",
                                expected_output="release file content",
                                success_criteria="The release file content is observed.",
                            ),
                            plan_step(
                                "edit_release",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="release file with ready status",
                                success_criteria="release.yaml contains the exact status line and unchanged surrounding lines.",
                                depends_on=["read_initial"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {
                                        "name": "exact_file_state",
                                        "check_type": "file_contains",
                                        "path": str(release),
                                        "pattern": "name: report-62\nstatus: ready\nowner: team-6",
                                    },
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "exact_file_state"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "repaired", depends_on=["edit_release"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_current",
                                "Read current release",
                                "read",
                                expected_tool="read_file",
                                expected_output="current release file content",
                                success_criteria="The current release file content is observed after the failed edit attempts.",
                            ),
                            plan_step(
                                "repair_from_current",
                                "Repair release from current state",
                                "write",
                                expected_tool="edit_text",
                                expected_output="release file with ready status",
                                success_criteria="release.yaml contains the exact status line and unchanged surrounding lines.",
                                depends_on=["read_current"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {
                                        "name": "exact_file_state",
                                        "check_type": "file_contains",
                                        "path": str(release),
                                        "pattern": "name: report-62\nstatus: ready\nowner: team-6",
                                    },
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "exact_file_state"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["repair_from_current"]),
                        ],
                    ),
                ],
                "tool_input:read_file": [
                    json.dumps({"path": str(release)}),
                    json.dumps({"path": str(release)}),
                ],
                "tool_input:edit_text": [
                        json.dumps(_edit_replace_input(release, "status: draft", "ready")),
                        stale_edit,
                        stale_edit,
                        repair_after_replan,
                        repair_after_replan,
                    ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the current artifact does not satisfy the exact requested final state",
                        }
                    )
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_file", "edit_text", "edit_text", "edit_text", "read_file", "edit_text"]
    assert observed["repair_prompt"]
    assert len(plan_requests) == 2
    assert "Recent failed tool or verification evidence" in plan_requests[1]["prompt"]
    assert "Latest observed file snapshots" in plan_requests[1]["prompt"]
    assert "do not plan another action that depends only on that stale target" in plan_requests[1]["prompt"]
    assert "executable tests for semantic correctness" in plan_requests[1]["prompt"]
    assert "do not add speculative file-content assertions" in plan_requests[1]["prompt"]
    assert any(
        event.event_type == "tool_error"
        and event.payload.get("tool_name") == "edit_text"
        and "old_text not found" in event.payload.get("error", "")
        and "match_count=0" in event.payload.get("error", "")
        and "current_text" in event.payload.get("error", "")
        for event in events
    )
    assert any(event.event_type == "verification_failed" and event.payload.get("step_id") == "edit_release" for event in events)
    assert any(event.event_type == "replan_triggered" and event.payload.get("step_id") == "edit_release" for event in events)
    assert max(_event_sequences(events, "edit_applied")) < max(
        _event_sequences(events, "verification_passed", step_id="repair_from_current")
    )
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "repair_from_current" for event in events)


def test_real_loop_model_owned_plan_review_rejects_weak_recovery_plan_before_execution(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: extremely_hard. Family: file_edit/failure/multi_step/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        model__max_retries=1,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=24,
        runtime__max_total_actions=40,
        runtime__max_repeated_action_occurrences=3,
        planner__max_replans=2,
    )
    goal = f"Edit {release} so the status moves from draft to ready and nothing else changes. Answer repaired."
    observed: dict[str, str] = {}

    def accept_plan_review(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        assert "Plan semantic adequacy review" in prompt
        criteria = [
            {"name": "plan_satisfies_original_request", "passed": True, "evidence": "candidate plan preserves the request"},
            {"name": "plan_uses_current_evidence", "passed": True, "evidence": "candidate plan uses available evidence"},
            {"name": "plan_verifies_exact_requested_state", "passed": True, "evidence": "candidate plan checks the exact artifact state"},
        ]
        return json.dumps({"criteria": criteria})

    def reject_weak_recovery_plan(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["weak_plan_review_prompt"] = prompt
        assert "Plan semantic adequacy review" in prompt
        assert "latest_observed_file_snapshots" in prompt
        assert "recent_failed_tool_or_verification_evidence" in prompt
        assert "name: report-62" in prompt
        assert "owner: team-6" in prompt
        assert "status: draft" in prompt
        assert '"pattern": "ready"' in prompt
        return json.dumps(
            {
                "criteria": [
                    {"name": "plan_satisfies_original_request", "passed": False, "evidence": "the plan narrows the requested final state"},
                    {"name": "plan_uses_current_evidence", "passed": False, "evidence": "the plan still targets a stale source pattern"},
                    {"name": "plan_verifies_exact_requested_state", "passed": False, "evidence": "the plan uses a broad value match that could accept corruption"},
                ]
            }
        )

    def accept_non_plan_verification(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        criteria = json.loads(prompt.split("Criteria:\n", 1)[1].split("\n\n", 1)[0])
        return json.dumps(
            {
                "criteria": [
                    {"name": item["name"], "passed": True, "evidence": "scripted verifier accepts final evidence"}
                    for item in criteria
                ]
            }
        )

    initial_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_initial",
                "Read initial release",
                "read",
                expected_tool="read_file",
                expected_output="release file content",
                success_criteria="The release file content is observed.",
            ),
            plan_step(
                "edit_release",
                "Edit release",
                "write",
                expected_tool="edit_text",
                expected_output="release file with ready status",
                success_criteria="release.yaml contains the exact status line and unchanged surrounding lines.",
                depends_on=["read_initial"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {
                        "name": "exact_file_state",
                        "check_type": "file_contains",
                        "path": str(release),
                        "pattern": "name: report-62\nstatus: ready\nowner: team-6",
                    },
                ],
                required_conditions=["dependencies_completed", "tool_name_matches", "exact_file_state"],
                optional_conditions=[],
            ),
            _exact_answer_step("unreachable_answer", "repaired", depends_on=["edit_release"]),
        ],
    )
    weak_recovery_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_current_weak",
                "Read current release weakly",
                "read",
                expected_tool="read_file",
                expected_output="current release file content",
                success_criteria="The current file contains ready.",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "ready"},
                ],
                required_conditions=["dependencies_completed", "file_contains_ready"],
                optional_conditions=[],
            ),
            plan_step(
                "stale_edit_after_weak_read",
                "Stale edit after weak read",
                "write",
                expected_tool="edit_text",
                expected_output="file contains ready",
                success_criteria="The file contains ready.",
                depends_on=["read_current_weak"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "ready"},
                ],
                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                optional_conditions=[],
            ),
            _exact_answer_step("weak_answer", "repaired", depends_on=["stale_edit_after_weak_read"]),
        ],
    )
    corrected_recovery_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_current",
                "Read current release",
                "read",
                expected_tool="read_file",
                expected_output="current release file content",
                success_criteria="The current release file content is observed after failed attempts.",
            ),
            plan_step(
                "repair_from_current",
                "Repair release from current state",
                "write",
                expected_tool="edit_text",
                expected_output="release file with ready status",
                success_criteria="release.yaml contains the exact status line and unchanged surrounding lines.",
                depends_on=["read_current"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {
                        "name": "exact_file_state",
                        "check_type": "file_contains",
                        "path": str(release),
                        "pattern": "name: report-62\nstatus: ready\nowner: team-6",
                    },
                ],
                required_conditions=["dependencies_completed", "tool_name_matches", "exact_file_state"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["repair_from_current"]),
        ],
    )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [initial_plan, weak_recovery_plan, corrected_recovery_plan],
                "tool_input:read_file": [
                    json.dumps({"path": str(release)}),
                    json.dumps({"path": str(release)}),
                ],
                "tool_input:edit_text": [
                    json.dumps(_edit_replace_input(release, "status: draft", "ready")),
                    json.dumps(_edit_replace_input(release, "status: draft", "status: ready")),
                    json.dumps(_edit_replace_input(release, "status: draft", "status: ready")),
                    json.dumps(_edit_replace_input(release, "ready", "status: ready")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the current artifact does not satisfy the exact requested final state",
                        }
                    )
                ],
                "plan_semantic_verification": [
                    accept_plan_review,
                    reject_weak_recovery_plan,
                    accept_plan_review,
                ],
                "verification": [
                    accept_non_plan_verification,
                    accept_non_plan_verification,
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert observed["weak_plan_review_prompt"]
    assert _tool_names(events) == ["read_file", "edit_text", "edit_text", "edit_text", "read_file", "edit_text"]
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "plan_semantic"
        and event.payload.get("passed") is False
        and event.payload.get("target_id")
        for event in events
    )
    assert not any(event.event_type == "step_started" and event.payload.get("step_id") == "stale_edit_after_weak_read" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "repair_from_current" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_retries_plan_when_semantic_plan_review_protocol_fails(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: file_edit/failure/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        planner__max_plan_steps=4,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Edit {release} so status moves from draft to ready and answer verified."
    plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_release",
                "Read release",
                "read",
                expected_tool="read_file",
                expected_output="release file content",
                success_criteria="The release file content is observed.",
            ),
            plan_step(
                "edit_release",
                "Edit release",
                "write",
                expected_tool="edit_text",
                expected_output="release ready",
                success_criteria="release.yaml contains status: ready.",
                depends_on=["read_release"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                ],
                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "verified", depends_on=["edit_release"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [plan, plan],
                "tool_input:read_file": [json.dumps({"path": str(release)})],
                "tool_input:edit_text": [json.dumps(_edit_replace_input(release, "status: draft", "status: ready"))],
                "answer_response": ["verified"],
            }
        ),
    )
    original_run_llm_verification = runtime._run_llm_verification
    plan_review_calls = 0

    def _flaky_plan_review(*args, **kwargs):
        nonlocal plan_review_calls
        if kwargs.get("contract_name") == "plan_semantic_verification":
            plan_review_calls += 1
            if plan_review_calls == 1:
                raise SemanticBackendProtocolError("structured relevance response violated schema")
        return original_run_llm_verification(*args, **kwargs)

    runtime._run_llm_verification = _flaky_plan_review  # type: ignore[method-assign]

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "verified"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_file", "edit_text"]
    assert plan_review_calls >= 2
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "plan_semantic"
        and event.payload.get("passed") is False
        and event.payload.get("reason") == "plan_semantic_review_protocol_error"
        for event in events
    )
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "plan_validation"
        and "plan_semantic_review_protocol_error" in event.payload.get("error", "")
        for event in events
    )
    assert any(event.event_type == "model_retry_scheduled" and event.payload.get("kind") == "plan" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "edit_release" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_retries_tool_validation_error_inside_same_step(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: file_edit/failure/multi_step."""
    target = tmp_path / "record.txt"
    target.write_text("alpha beta gamma\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
        planner__max_replans=1,
    )
    goal = f"Read {target}, change beta to delta, and answer finished."
    observed: dict[str, str] = {}

    def first_edit_input(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        assert "alpha beta gamma" in prompt
        return json.dumps(_edit_range_input(target, start=0, end=5, expected_text="beta", replacement="delta"))

    def recovered_edit_input(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["recovery_prompt"] = prompt
        assert "tool_error:" in prompt
        assert "selected='alpha'" in prompt
        assert "expected_text_matching_ranges" in prompt
        assert '\\"end\\":10' in prompt
        assert '\\"start\\":6' in prompt
        return json.dumps(_edit_range_input(target, start=6, end=10, expected_text="beta", replacement="delta"))

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_record",
                                "Read record",
                                "read",
                                expected_tool="read_text",
                                expected_output="record content",
                                success_criteria="record content observed",
                            ),
                            plan_step(
                                "edit_record",
                                "Edit record",
                                "write",
                                expected_tool="edit_text",
                                expected_output="record edited",
                                success_criteria="record.txt contains alpha delta gamma.",
                                depends_on=["read_record"],
                                verification_checks=[
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_delta", "check_type": "file_contains", "path": str(target), "pattern": "alpha delta gamma"},
                                ],
                                required_conditions=["tool_result_present", "tool_name_matches", "file_contains_delta"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "finished", depends_on=["edit_record"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("read_text", {}),
                    _tool_call("edit_text", {}),
                    _tool_call("edit_text", {}),
                ],
                "tool_input:read_text": [json.dumps(_read_text_input(path=str(target)))],
                "tool_input:edit_text": [first_edit_input, recovered_edit_input],
                "answer_response": ["finished"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "finished"
    assert target.read_text(encoding="utf-8") == "alpha delta gamma\n"
    assert _tool_names(events) == ["read_text", "edit_text", "edit_text"]
    assert observed["recovery_prompt"]
    assert any(event.event_type == "tool_error" and 'expected_text_matching_ranges=[{"end":10,"start":6}]' in event.payload.get("error", "") for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_record" for event in events)
    assert not any(event.event_type == "replan_triggered" for event in events)
    assert max(_event_sequences(events, "tool_error")) < min(_event_sequences(events, "edit_applied"))
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="edit_record"))


def test_real_loop_accepts_edit_text_plan_with_registered_mechanical_check(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: file_edit/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        model__max_retries=1,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=16,
    )
    goal = f"Move {release} from draft to ready and answer repaired."
    plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "edit_status",
                "Edit status",
                "write",
                expected_tool="edit_text",
                expected_output="edited_file_content",
                success_criteria="release.yaml is edited.",
                output_refs=["edited_file_content"],
                verification_checks=[
                    {"name": "edited_file_content", "check_type": "artifact_present", "artifact": "edited_file_content"},
                ],
                required_conditions=["edited_file_content"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["edit_status"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [plan],
                "tool_decision": [_tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready"))],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text"]
    assert not any(event.event_type == "error" and event.payload.get("operation") == "plan_validation" for event in events)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 1
    verification_events = [
        event for event in events if event.event_type == "verification_completed" and event.payload["step_id"] == "edit_status"
    ]
    assert verification_events
    assert "registered_tool_effect_verified" in verification_events[-1].payload["conditions_met"]


def test_real_loop_uses_intrinsic_success_criteria_for_response(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_easy. Family: file_edit/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Edit {release} so status moves from draft to ready and answer status ready."
    edit_step = plan_step(
        "edit_status",
        "Edit status",
        "write",
        expected_tool="edit_text",
        expected_output="status ready",
        success_criteria="release.yaml contains status: ready.",
        verification_checks=[
            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
            {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
        ],
        required_conditions=["tool_result_present", "tool_name_matches", "file_contains_ready"],
        optional_conditions=[],
    )
    answer_step = plan_step(
        "answer",
        "Answer",
        "respond",
        expected_output="status ready",
        success_criteria="The answer states that the status is ready.",
        depends_on=["edit_status"],
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
        ],
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [plan_response(goal=goal, steps=[edit_step, answer_step])],
                "tool_input:edit_text": [json.dumps(_edit_replace_input(release, "status: draft", "status: ready"))],
                "answer_response": ["status ready"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    contracts = [request["contract"] for request in runtime.client.requests]
    verification_prompts = [request["prompt"] for request in runtime.client.requests if request["contract"] == "verification"]

    assert result.assistant_text == "status ready"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert not release.with_name(release.name + ".bak").exists()
    assert contracts.count("task_plan") == 1
    assert not any(contract == "tool_decision" for contract in contracts)
    assert _tool_names(events) == ["edit_text"]
    assert any("__contract_success_criteria__" in prompt for prompt in verification_prompts)
    assert any("The answer states that the status is ready." in prompt for prompt in verification_prompts)
    assert not any(event.event_type == "error" and event.payload.get("operation") == "plan_validation" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "edit_status" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_retry_feedback_repairs_redundant_mutation_step(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: file_edit/failure/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        model__max_retries=1,
        planner__max_replans=2,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=5,
        runtime__max_reasoning_steps=16,
        runtime__max_total_actions=24,
    )
    goal = f"Move {release} from draft to ready and answer repaired."
    initial_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_file_content",
                "Read file",
                "read",
                expected_tool="read_file",
                expected_output="file_content",
                success_criteria="File content observed.",
                output_refs=["file_content"],
            ),
            plan_step(
                "edit_file_content",
                "Edit file",
                "write",
                expected_tool="edit_text",
                expected_output="edited_file_content",
                success_criteria="release.yaml contains status: ready.",
                depends_on=["read_file_content"],
                input_refs=["file_content"],
                output_refs=["edited_file_content"],
            ),
            plan_step(
                "write_file_content",
                "Redundantly write the same file",
                "write",
                expected_tool="write_file",
                expected_output="written_file",
                success_criteria="release.yaml remains in the requested ready state.",
                depends_on=["edit_file_content"],
                input_refs=["edited_file_content"],
                output_refs=["written_file"],
            ),
            _exact_answer_step("unreachable_answer", "repaired", depends_on=["write_file_content"]),
        ],
    )
    recovery_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_current",
                "Read the already repaired file",
                "read",
                expected_tool="read_file",
                expected_output="current_file_content",
                success_criteria="The current ready file is observed.",
                depends_on=["edit_file_content"],
                output_refs=["current_file_content"],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["read_current"]),
        ],
    )
    ready_text = "name: report-62\nstatus: ready\nowner: team-6\n"
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [initial_plan, recovery_plan],
                "tool_input:read_file": [
                    json.dumps({"path": str(release)}),
                    json.dumps({"path": str(release)}),
                ],
                "tool_input:edit_text": [
                    json.dumps(_edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "tool_input:write_file": [
                    json.dumps(_write_file_input(release, ready_text)),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the redundant full-file write was a no-op and must not be repeated",
                        }
                    )
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == ready_text
    assert _tool_names(events) == ["read_file", "edit_text", "write_file", "read_file"]
    assert any(
        event.event_type == "verification_failed"
        and event.payload.get("step_id") == "write_file_content"
        and "registered_tool_effect_verified" in event.payload.get("conditions_failed", [])
        for event in events
    )
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 2
    retry_prompt = plan_requests[1]["prompt"]
    assert "Completed prior step ids that may be referenced as already satisfied dependencies" in retry_prompt
    assert "edit_file_content" in retry_prompt
    assert "dependencies on omitted completed prior steps are already satisfied" in retry_prompt
    assert "do not add speculative file-content assertions" in retry_prompt


def test_real_loop_supersedes_legacy_unrequired_edit_effect_with_registered_check(make_config, tmp_path: Path) -> None:
    """Difficulty: easy. Family: file_edit/compatibility."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=12,
    )
    goal = f"Edit {release} so status becomes ready and answer repaired."
    legacy_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "edit_status",
                "Edit status",
                "write",
                expected_tool="edit_text",
                expected_output="edited_file_content",
                success_criteria="release.yaml contains status: ready.",
                output_refs=["edited_file_content"],
                verification_checks=[
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "tool_effect", "check_type": "tool_effect_verified"},
                ],
                required_conditions=["tool_name_matches"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["edit_status"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [legacy_plan],
                "tool_decision": [_tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready"))],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text"]
    assert not any(event.event_type == "plan_repaired" for event in events)
    assert not any(event.event_type == "error" and event.payload.get("operation") == "plan_validation" for event in events)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 1
    verification_events = [
        event for event in events if event.event_type == "verification_completed" and event.payload["step_id"] == "edit_status"
    ]
    assert verification_events
    assert "registered_tool_effect_verified" in verification_events[-1].payload["conditions_met"]


def test_real_loop_rejects_malformed_file_contains_check_then_accepts_corrected_plan(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: failure/file_edit/quality."""
    report = tmp_path / "capacity_report.txt"
    config = make_config(
        model__context_limit=32_000,
        model__max_retries=1,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=10,
        runtime__max_total_actions=14,
    )
    goal = f"Write the capacity report to {report} and answer repaired."
    invalid_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "write_report",
                "Write report",
                "write",
                expected_tool="write_file",
                expected_output="report written",
                success_criteria="capacity_report.txt contains the final report.",
                output_refs=["report"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {
                        "name": "file_written",
                        "check_type": "file_contains",
                        "expected_json": "headroom=74",
                    },
                ],
                required_conditions=["dependencies_completed", "file_written"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["write_report"]),
        ],
    )
    valid_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "write_report",
                "Write report",
                "write",
                expected_tool="write_file",
                expected_output="report written",
                success_criteria="capacity_report.txt contains the final report.",
                output_refs=["report"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                    {"name": "file_written", "check_type": "file_contains", "path": str(report), "pattern": "headroom=74"},
                ],
                required_conditions=["dependencies_completed", "tool_name_matches", "file_written"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["write_report"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [invalid_plan, valid_plan],
                "tool_decision": [_tool_call("write_file", _write_file_input(report, "service=svc-02\nheadroom=74\nworkers=6\n"))],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert report.read_text(encoding="utf-8") == "service=svc-02\nheadroom=74\nworkers=6\n"
    assert _tool_names(events) == ["write_file"]
    assert any(
        event.event_type == "error"
        and "file_contains check file_written must declare a non-empty pattern or textual expected_json" in event.payload.get("error", "")
        for event in events
    )
    assert any(event.event_type == "model_retry_scheduled" and event.payload.get("kind") == "plan" for event in events)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 2
    assert "file_contains check file_written must declare a non-empty pattern or textual expected_json" in plan_requests[1]["prompt"]
    assert "The live plan wire does not support file_contains" in plan_requests[1]["prompt"]
    assert "rely on the registered mechanical effect check" in plan_requests[1]["prompt"]
    assert "Every step must also fill objective_verification_check" not in plan_requests[1]["prompt"]
    write_verifications = [
        event for event in events if event.event_type == "verification_completed" and event.payload["step_id"] == "write_report"
    ]
    assert write_verifications
    assert write_verifications[-1].payload["verification_passed"] is True
    assert "file_written" in write_verifications[-1].payload["conditions_met"]


def test_real_loop_rejects_unsupported_read_tool_effect_before_execution(make_config, tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("status=ready\n", encoding="utf-8")
    goal = f"Read {target} and answer observed."

    invalid_read = plan_step(
        "read_target",
        "Read target",
        "read",
        expected_tool="read_file",
        expected_output="file contents",
        success_criteria="The file is read successfully.",
        output_refs=["file_contents"],
        verification_checks=[
            {"name": "tool_name", "check_type": "tool_name_equals", "expected": "read_file"},
            {"name": "file_contents", "check_type": "tool_effect_verified"},
        ],
        required_conditions=["tool_name", "file_contents"],
        optional_conditions=[],
    )
    valid_read = plan_step(
        "read_target",
        "Read target",
        "read",
        expected_tool="read_file",
        expected_output="file contents",
        success_criteria="The file is read successfully.",
        output_refs=["file_contents"],
        verification_checks=[
            {"name": "tool_name", "check_type": "tool_name_equals", "expected": "read_file"},
            {"name": "file_contents", "check_type": "tool_output_nonempty"},
        ],
        required_conditions=["tool_name", "file_contents"],
        optional_conditions=[],
    )
    answer = _exact_answer_step("answer", "observed", depends_on=["read_target"])
    client = FakeModelClient(
        contract_responses={
            "task_plan": [
                plan_response(goal=goal, steps=[invalid_read, answer]),
                plan_response(goal=goal, steps=[valid_read, answer]),
            ],
            "tool_input:read_file": [json.dumps({"path": str(target)})],
            "answer_response": ["observed"],
        }
    )
    runtime = AgentRuntime(
        make_config(
            model__context_limit=32_000,
            model__max_retries=1,
            tools__allow_stateful_tools=True,
        ),
        model_client=client,
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "observed"
    assert [event.payload.get("tool_name") for event in events if event.event_type == "tool_called"] == ["read_file"]
    assert len(plan_requests) == 2
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "plan_validation"
        and "tool_effect_verified is not supported by that tool" in event.payload.get("error", "")
        for event in events
    )
    assert "tool_effect_verified is not supported by that tool" in plan_requests[1]["prompt"]
    assert "Do not emit tool_effect_verified or file_contains; neither is part of the live plan wire" in plan_requests[1]["prompt"]
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "read_target" for event in events)


def test_real_loop_repairs_missing_tool_name_expected_before_execution(make_config, tmp_path: Path) -> None:
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    goal = f"Edit {release} so status moves from draft to ready and nothing else changes, then summarize the final state."

    invalid_read = plan_step(
        "read_file",
        "Read release file",
        "read",
        expected_tool="read_file",
        expected_output="file contents",
        success_criteria="The file is read.",
        verification_checks=[
            {"name": "tool_name_matches", "check_type": "tool_name_equals"},
            {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
        ],
        required_conditions=["tool_name_matches", "output_nonempty"],
        optional_conditions=[],
    )
    invalid_edit = plan_step(
        "edit_status",
        "Edit status",
        "write",
        expected_tool="edit_text",
        expected_output="status ready",
        success_criteria="The status is ready and nothing else changed.",
        depends_on=["read_file"],
        verification_checks=[
            {"name": "tool_name_matches", "check_type": "tool_name_equals"},
            {"name": "tool_effect", "check_type": "tool_effect_verified"},
        ],
        required_conditions=["tool_name_matches", "tool_effect"],
        optional_conditions=[],
    )
    answer = plan_step(
        "answer",
        "Summarize",
        "respond",
        expected_output="final state",
        success_criteria="The answer accurately summarizes the final file state.",
        depends_on=["edit_status"],
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
        ],
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
    )

    valid_read = dict(invalid_read)
    valid_read["verification_checks"] = [
        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "read_file"},
        {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
    ]
    valid_edit = dict(invalid_edit)
    valid_edit["verification_checks"] = [
        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
        {"name": "tool_effect", "check_type": "tool_effect_verified"},
    ]

    client = FakeModelClient(
        contract_responses={
            "task_plan": [
                plan_response(goal=goal, steps=[invalid_read, invalid_edit, answer]),
                plan_response(goal=goal, steps=[valid_read, valid_edit, answer]),
            ],
            "tool_input:read_file": [json.dumps({"path": str(release)})],
            "tool_input:edit_text": [json.dumps(_edit_replace_input(release, "status: draft", "status: ready"))],
            "answer_response": ["name: report-62\nstatus: ready\nowner: team-6"],
        }
    )
    runtime = AgentRuntime(
        make_config(
            model__context_limit=32_000,
            tools__allow_stateful_tools=True,
            tools__allow_side_effect_tools=True,
            editor__allow_writes=True,
        ),
        model_client=client,
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "name: report-62\nstatus: ready\nowner: team-6"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert len(plan_requests) == 2
    assert "must declare a non-empty expected tool name" in plan_requests[1]["prompt"]
    assert "Previous rejected plan JSON:" in plan_requests[1]["prompt"]
    assert not any(event.event_type == "tool_called" for event in events if event.sequence < next(
        event.sequence for event in events if event.event_type == "plan_created"
    ))
    assert [event.payload.get("tool_name") for event in events if event.event_type == "tool_called"] == ["read_file", "edit_text"]
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "edit_status" for event in events)


def test_real_loop_retries_schema_valid_but_locally_invalid_plan(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: failure/file_edit."""
    target = tmp_path / "state.txt"
    target.write_text("state=draft\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        model__max_retries=1,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=10,
        runtime__max_total_actions=14,
    )
    goal = f"Change {target} from draft to ready and answer repaired."
    invalid_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "bad_read",
                "Read with invalid verification",
                "read",
                expected_tool="read_text",
                expected_output="file text",
                success_criteria="File text is read.",
                verification_checks=[
                    {"name": "read_done", "check_type": "tool_output_nonempty"},
                ],
                required_conditions=["missing_condition"],
                optional_conditions=[],
            ),
            _exact_answer_step("bad_answer", "repaired", depends_on=["bad_read"]),
        ],
    )
    valid_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "edit_state",
                "Edit state",
                "write",
                expected_tool="edit_text",
                expected_output="ready state",
                success_criteria="state.txt contains state=ready.",
                verification_checks=[
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(target), "pattern": "state=ready"},
                ],
                required_conditions=["tool_result_present", "tool_name_matches", "file_contains_ready"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["edit_state"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [invalid_plan, invalid_plan, valid_plan],
                "tool_decision": [_tool_call("edit_text", _edit_replace_input(target, "state=draft", "state=ready"))],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert target.read_text(encoding="utf-8") == "state=ready\n"
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 3
    assert "never artifact/input/output labels" in plan_requests[1]["prompt"]
    assert "runtime derives the structural done condition" in plan_requests[1]["prompt"]
    assert 'actual_source must be exactly "assistant_text"' in plan_requests[1]["prompt"]
    assert "runtime-derived respond completion condition" in plan_requests[1]["prompt"]
    assert "string_nonempty is only a presence check" in plan_requests[1]["prompt"]
    assert "success_criteria field is the authoritative semantic criterion" in plan_requests[1]["prompt"]
    assert "Do not duplicate success_criteria as a criterion check" in plan_requests[1]["prompt"]
    assert any(event.event_type == "error" and event.payload.get("operation") == "plan_validation" for event in events)
    assert any(event.event_type == "model_retry_scheduled" and event.payload.get("kind") == "plan" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_real_loop_continues_after_multiple_distinct_plan_contract_repairs(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: failure/multi_step/quality."""
    inputs = tmp_path / "inputs.json"
    report = tmp_path / "capacity_report.txt"
    test_file = tmp_path / "test_capacity.py"
    inputs.write_text(json.dumps({"service": "svc-02", "capacity": 120, "used": 46, "workers": 6}), encoding="utf-8")
    report.write_text("service=pending\nheadroom=pending\nworkers=pending\n", encoding="utf-8")
    test_file.write_text(
        "\n".join(
            [
                "import pathlib",
                "import unittest",
                "",
                "class CapacityTests(unittest.TestCase):",
                "    def test_capacity_report(self):",
                "        text = pathlib.Path('capacity_report.txt').read_text(encoding='utf-8')",
                "        self.assertIn('service=svc-02', text)",
                "        self.assertIn('headroom=74', text)",
                "        self.assertIn('workers=6', text)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = make_config(
        model__max_retries=1,
        model__context_limit=32_000,
        planner__max_plan_steps=6,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=6,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=24,
    )
    goal = "Read inputs.json, compute capacity headroom, write capacity_report.txt, run unittest, and answer verified."

    invalid_file_contains_repr = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_inputs",
                "Read inputs",
                "read",
                expected_tool="read_text",
                expected_output="inputs",
                success_criteria="inputs observed",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {
                        "name": "file_content",
                        "check_type": "file_contains",
                        "path": str(inputs),
                        "expected_json": "{'service': 'svc-02'}",
                    },
                ],
                required_conditions=["dependencies_completed"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "verified", depends_on=["read_inputs"]),
        ],
    )
    invalid_empty_file_contains = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_inputs",
                "Read inputs",
                "read",
                expected_tool="read_text",
                expected_output="inputs",
                success_criteria="inputs observed",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "file_contains", "check_type": "file_contains", "path": str(inputs), "pattern": ""},
                ],
                required_conditions=["dependencies_completed", "file_contains"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "verified", depends_on=["read_inputs"]),
        ],
    )
    invalid_empty_reasoning_criterion = plan_response(
        goal=goal,
        steps=[
            plan_step("read_inputs", "Read inputs", "read", expected_tool="read_text", expected_output="inputs", success_criteria="inputs observed"),
            plan_step(
                "compute_headroom",
                "Compute headroom",
                "reasoning",
                expected_output="headroom",
                success_criteria="headroom computed",
                depends_on=["read_inputs"],
                verification_checks=[
                    {
                        "name": "headroom_quality",
                        "check_type": "criterion",
                        "actual_source": "assistant_text",
                        "criterion": "",
                    }
                ],
                required_conditions=["headroom_quality"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "verified", depends_on=["compute_headroom"]),
        ],
    )
    valid_plan = plan_response(
        goal=goal,
        steps=[
            plan_step("read_inputs", "Read inputs", "read", expected_tool="read_text", expected_output="inputs", success_criteria="inputs observed"),
            plan_step(
                "write_report",
                "Write report",
                "write",
                expected_tool="write_file",
                expected_output="report",
                success_criteria="capacity_report.txt contains the computed capacity report.",
                depends_on=["read_inputs"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                    {"name": "file_contains_report", "check_type": "file_contains", "path": str(report), "pattern": "headroom=74"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_report"],
                optional_conditions=[],
            ),
            plan_step(
                "run_unittest",
                "Run unittest",
                "tool",
                expected_tool="run_tests",
                expected_output="tests pass",
                success_criteria="The unittest command succeeds.",
                depends_on=["write_report"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
                    {"name": "tests_pass", "check_type": "command_success", "command": ["python", "-m", "unittest", "-q", "test_capacity.py"], "cwd": str(tmp_path)},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "verified", depends_on=["run_unittest"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    invalid_file_contains_repr,
                    invalid_empty_file_contains,
                    invalid_empty_reasoning_criterion,
                    invalid_empty_reasoning_criterion,
                    valid_plan,
                ],
                "tool_input:read_text": [json.dumps(_read_text_input(path=str(inputs)))],
                "tool_input:write_file": [json.dumps(_write_file_input(report, "service=svc-02\nheadroom=74\nworkers=6\n"))],
                "tool_input:run_tests": [
                    json.dumps({"command": ["python", "-m", "unittest", "-q", "test_capacity.py"], "background": False})
                ],
                "answer_response": ["verified"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "verified"
    assert report.read_text(encoding="utf-8") == "service=svc-02\nheadroom=74\nworkers=6\n"
    assert _tool_names(events) == ["read_text", "write_file", "run_tests"]
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 5
    assert "Keep dependencies_completed mechanical" in plan_requests[-1]["prompt"]
    assert "success_criteria field is the authoritative semantic criterion" in plan_requests[-1]["prompt"]
    assert "Do not duplicate success_criteria as a criterion check" in plan_requests[-1]["prompt"]
    assert "Conditions must name declared checks" in plan_requests[-1]["prompt"]
    assert "Do not emit tool_effect_verified or file_contains" in plan_requests[-1]["prompt"]
    assert "Every step, including respond steps, must declare non-empty expected_outputs labels" in plan_requests[-1]["prompt"]
    assert "For read/list/note context steps" in plan_requests[-1]["prompt"]
    assert "prefer dependencies_completed, tool_name_equals, tool_output_nonempty, or tool_output_schema_valid" in plan_requests[-1]["prompt"]
    assert "A read step verifies that trustworthy context was gathered" in plan_requests[-1]["prompt"]
    assert "allow the registered persisted-effect check and later whole-goal review" in plan_requests[-1]["prompt"]
    assert "add command_success only when there is a distinct executable correctness test" in plan_requests[-1]["prompt"]
    assert "Plan correction evidence from this generation cycle" in plan_requests[-1]["prompt"]
    assert "Previous rejected plan JSON:" in plan_requests[1]["prompt"]
    assert "Correct this model-authored plan rather than regenerating unrelated fields" in plan_requests[1]["prompt"]
    assert stable_json_dumps(json.loads(_normalize_scripted_plan_response(invalid_file_contains_repr))) in plan_requests[1]["prompt"]
    assert stable_json_dumps(json.loads(_normalize_scripted_plan_response(invalid_empty_reasoning_criterion))) in plan_requests[-1]["prompt"]
    assert "attempt 1 validation:" in plan_requests[-1]["prompt"]
    assert "attempt 2 validation:" in plan_requests[-1]["prompt"]
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "plan_validation"
        and "file_contains check file_content expected_json must be JSON text" in event.payload.get("error", "")
        for event in events
    )
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "plan_validation"
        and "check headroom_quality is missing criterion text" in event.payload.get("error", "")
        for event in events
    )
    assert not any(event.event_type == "fatal_system_error" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "run_unittest" for event in events)


def test_real_loop_plan_retry_feedback_rejects_self_dependency(make_config) -> None:
    """Difficulty: normal. Family: failure/multi_step."""
    config = make_config(
        model__max_retries=1,
        tools__allow_stateful_tools=True,
        runtime__max_tool_steps=2,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=10,
    )
    goal = "Answer repaired."
    invalid_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_file",
                "Read",
                "read",
                expected_tool="read_file",
                expected_output="file text",
                success_criteria="File text is read.",
                depends_on=["read_file"],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["read_file"]),
        ],
    )
    valid_plan = plan_response(goal=goal, steps=[_exact_answer_step("answer", "repaired")])
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [invalid_plan, valid_plan],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "repaired"
    assert len(plan_requests) == 2
    assert "depends_on names only earlier step_id values" in plan_requests[1]["prompt"]
    assert "Conditions must name declared checks" in plan_requests[1]["prompt"]
    assert any(event.event_type == "error" and "Circular dependency detected" in event.payload.get("error", "") for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_real_loop_treats_plan_input_text_as_instruction_not_executable_json(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: file_edit/failure."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__max_retries=1,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=16,
    )
    goal = f"Move {release} from draft to ready and answer repaired."
    plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "edit_status",
                "Edit status",
                "write",
                expected_tool="edit_text",
                input_text=json.dumps({"instruction": "Use the selected edit tool; do not copy {{edited_file_content}}."}),
                expected_output="ready file",
                success_criteria="release.yaml contains status: ready.",
                verification_checks=[
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "expected_json": "\"status: ready\""},
                ],
                required_conditions=["tool_name_matches", "file_contains_ready"],
                optional_conditions=[],
            ),
            _exact_answer_step("answer", "repaired", depends_on=["edit_status"]),
        ],
    )
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [plan],
                "tool_decision": [_tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready"))],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text"]
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 1
    assert "input_text is instruction context, not executable tool JSON" in plan_requests[0]["prompt"]
    tool_input_requests = [request for request in runtime.client.requests if request["contract"] == "tool_input:edit_text"]
    assert len(tool_input_requests) == 1
    assert "{{edited_file_content}}" in tool_input_requests[0]["prompt"]
    assert "Step instructions are model-authored context, not executable arguments" in tool_input_requests[0]["prompt"]
    assert not any(event.event_type == "error" and event.payload.get("operation") == "plan_validation" for event in events)
    assert not any(event.event_type == "tool_called" and event.payload.get("tool_name") == "write_file" for event in events)


def test_real_loop_verifies_model_declared_tool_output_ref_alias(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_easy. Family: reading/failure."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=2,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=12,
    )
    goal = f"Read {release} and answer read."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_release",
                                "Read release",
                                "read",
                                expected_tool="read_file",
                                expected_output="file_content",
                                success_criteria="The release file is read.",
                                output_refs=["file_content"],
                                verification_checks=[
                                    {"name": "file_content", "check_type": "artifact_present", "artifact": "file_content"},
                                ],
                                required_conditions=["file_content"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "read", depends_on=["read_release"]),
                        ],
                    )
                ],
                "tool_decision": [_tool_call("read_file", {"path": str(release)})],
                "answer_response": ["read"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "read"
    assert _tool_names(events) == ["read_file"]
    read_verification = next(
        event
        for event in events
        if event.event_type == "verification_completed" and event.payload.get("step_id") == "read_release"
    )
    assert read_verification.payload["verification_passed"] is True
    assert "file_content" in read_verification.payload["conditions_met"]


def test_real_loop_rejects_unresolved_placeholder_tool_input_and_recovers(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: failure/file_edit."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        planner__max_replans=2,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=16,
        runtime__max_total_actions=24,
    )
    goal = f"Move {release} from draft to ready and answer repaired."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_release",
                                "Read release",
                                "read",
                                expected_tool="read_file",
                                expected_output="edited_file_content",
                                success_criteria="release.yaml is observed.",
                                output_refs=["edited_file_content"],
                            ),
                            plan_step(
                                "write_release",
                                "Write release",
                                "write",
                                expected_tool="write_file",
                                input_text="Write concrete file content from observed evidence.",
                                expected_output="file_written",
                                success_criteria="release.yaml contains status: ready.",
                                depends_on=["read_release"],
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "expected_json": "\"status: ready\""},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["write_release"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("read_file", {"path": str(release)}),
                    _tool_call("write_file", {"path": str(release), "content": "{{edited_file_content}}", "create": False}),
                    _tool_call("write_file", _write_file_input(release, "name: report-62\nstatus: ready\nowner: team-6\n")),
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_file", "write_file", "write_file"]
    assert any(event.event_type == "tool_error" and "unresolved artifact placeholder" in event.payload.get("error", "") for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "write_release" for event in events)
    assert not any(event.event_type == "replan_triggered" for event in events)
    assert not any(event.event_type == "file_write_applied" and event.payload.get("content") == "{{edited_file_content}}" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "write_release" for event in events)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)
    assert not any(
        item.metadata.get("source_event_type") in {"plan_created", "plan_updated"}
        for item in rebuilt.semantic_memory
    )


def test_real_loop_retry_after_failed_edit_includes_observed_file_text(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: failure/file_edit."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        planner__max_replans=2,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=26,
    )
    goal = f"Move {release} from draft to ready and answer repaired."
    first_plan = plan_response(
        goal=goal,
        steps=[
            plan_step(
                "read_release",
                "Read release",
                "read",
                expected_tool="read_file",
                expected_output="release text",
                success_criteria="release.yaml text is observed.",
            ),
            plan_step(
                "bad_edit",
                "Try edit",
                "write",
                expected_tool="edit_text",
                expected_output="ready file",
                success_criteria="release.yaml contains status: ready.",
                depends_on=["read_release"],
                verification_checks=[
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                ],
                required_conditions=["tool_name_matches", "file_contains_ready"],
                optional_conditions=[],
            ),
            _exact_answer_step("unreachable_answer", "repaired", depends_on=["bad_edit"]),
        ],
    )

    def repaired_edit_input(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        assert "read_file result:" in prompt
        assert "status: draft" in prompt
        assert "owner: team-6" in prompt
        assert "old_text not found" in prompt
        assert "match_count=0" in prompt
        return json.dumps(_edit_replace_input(release, "status: draft", "status: ready"))

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    first_plan,
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "repair_edit",
                                "Repair edit from observed text",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["repair_edit"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("read_file", {"path": str(release)}),
                    _tool_call("edit_text", {}),
                    _tool_call("edit_text", {}),
                ],
                    "tool_input:edit_text": [
                        json.dumps(_edit_replace_input(release, r"release:\s*draft", "release: ready")),
                        repaired_edit_input,
                        repaired_edit_input,
                    ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_file", "edit_text", "edit_text"]
    assert any(
        event.event_type == "tool_error"
        and "old_text not found" in event.payload.get("error", "")
        and "match_count=0" in event.payload.get("error", "")
        for event in events
    )
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "bad_edit" for event in events)
    assert not any(event.event_type == "replan_triggered" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "bad_edit" for event in events)


def test_real_loop_pathless_file_contains_binds_latest_edit_path_and_refines(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Change the release status in {release} to ready and answer repaired."

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_file",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "expected_json": "\"status: ready\""},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["edit_file"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready")),
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    failed_previews = [
        event
        for event in events
        if event.event_type == "subsystem_progress"
        and event.payload.get("step_id") == "edit_file"
        and "preview_passed=False" in str(event.payload.get("progress", ""))
    ]

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text"]
    assert failed_previews
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "edit_file" for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_file" for event in events)
    assert not any("Is a directory" in str(event.payload) for event in events)


def test_real_loop_expected_tool_step_refines_without_reselecting_tool(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Change the release status in {release} to ready and answer repaired."

    def refinement_input(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        assert "verification_preview_failed" in prompt
        assert "file_contains_ready" in prompt
        assert '"matched": false' in prompt
        return json.dumps(_edit_replace_input(release, "ready", "status: ready"))

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_file",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["edit_file"]),
                        ],
                    )
                ],
                "tool_input:edit_text": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    refinement_input,
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    edit_applied_sequences = _event_sequences(events, "edit_applied")
    edit_verified_sequences = _event_sequences(events, "verification_passed", step_id="edit_file")

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text"]
    assert any(
        event.event_type == "subsystem_progress"
        and event.payload.get("step_id") == "edit_file"
        and "preview_passed=False" in str(event.payload.get("progress", ""))
        for event in events
    )
    assert any(
        event.event_type == "message_added"
        and "verification_preview_failed" in event.payload.get("message", {}).get("content", "")
        for event in events
    )
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert sum(1 for request in runtime.client.requests if request["contract"] == "tool_input:edit_text") == 2
    assert not any(event.event_type.startswith("tool_graph") for event in events)
    assert edit_applied_sequences
    assert edit_verified_sequences
    assert max(edit_applied_sequences) < max(edit_verified_sequences)
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "answer" for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_file" for event in events)


def test_real_loop_expected_tool_step_rejects_bad_arguments_without_wrong_tool_execution(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=8,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=24,
        runtime__max_repeated_action_occurrences=2,
        planner__max_replans=1,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    invalid_edit_input = json.dumps({"path": str(release), "content": "invalid edit payload", "create": True})
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_file",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited release file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_current_file",
                                "Read current release file",
                                "read",
                                expected_tool="read_file",
                                expected_output="current release file",
                                success_criteria="release.yaml content is observed.",
                            ),
                            plan_step(
                                "edit_file_after_recovery",
                                "Edit release after recovery",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited release file",
                                success_criteria="release.yaml contains status: ready.",
                                depends_on=["read_current_file"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "recovered", depends_on=["edit_file_after_recovery"]),
                        ],
                    ),
                ],
                "tool_input:edit_text": [
                    invalid_edit_input,
                    invalid_edit_input,
                    invalid_edit_input,
                    json.dumps(_edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "tool_input:read_file": [
                    json.dumps({"path": str(release)}),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "tool_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the required edit tool received invalid repeated arguments",
                        }
                    )
                ],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text", "read_file", "edit_text"]
    assert any(event.event_type == "duplicate_action_detected" and "edit_file" in event.payload.get("action_key", "") for event in events)
    assert not any(event.event_type == "duplicate_action_detected" and "read_current_file" in event.payload.get("action_key", "") for event in events)
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert any(
        event.event_type == "tool_chain_completed"
        and event.payload.get("step_id") == "read_current_file"
        and event.payload.get("attempts") == 1
        and event.payload.get("success") is True
        for event in events
    )
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "read_current_file" for event in events)
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="edit_file_after_recovery"))


def test_real_loop_failure_classifier_uses_latest_verification_signal_after_prior_tool_error(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=1,
        runtime__max_reasoning_steps=24,
        runtime__max_total_actions=36,
        planner__max_replans=2,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    observed: dict[str, str] = {}

    def classify_current_verification_failure(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["current_verification_failure_prompt"] = prompt
        assert "Current failure signal to classify" in prompt
        assert '"step_id":"verify_initial_state"' in prompt
        assert '"reported_reason": "verification:file_contains_ready"' in prompt
        assert '"verification_passed": false' in prompt
        assert '"conditions_failed"' in prompt
        assert '"file_contains_ready"' in prompt
        assert '"matched": false' in prompt
        return json.dumps(
            {
                "kind": "verification_failure",
                "retryable": False,
                "requires_replan": True,
                "suggested_strategy_mode": "recovery",
                "wait_seconds": 0.0,
                "reason": "current read verification failed because status ready is not present",
            }
        )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_file",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited release file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "verify_initial_state",
                                "Verify initial state",
                                "read",
                                expected_tool="read_file",
                                expected_output="current release file",
                                success_criteria="release.yaml should be inspected before another edit.",
                                verification_checks=[
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_after_verify", "recovered", depends_on=["verify_initial_state"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_after_current_failure",
                                "Edit release after current failure",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited release file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "recovered", depends_on=["edit_after_current_failure"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_range_input(release, start=22, end=27, expected_text="draft", replacement="ready")),
                    _tool_call("read_file", {"path": str(release)}),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "tool_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "offset mismatch needs replan",
                        }
                    ),
                    classify_current_verification_failure,
                ],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "read_file", "edit_text"]
    assert observed["current_verification_failure_prompt"]
    assert len(plan_requests) == 3
    assert "current read verification failed because status ready is not present" in plan_requests[2]["prompt"]
    assert any(
        event.event_type == "tool_error"
        and "expected_text_matching_ranges" in event.payload.get("error", "")
        for event in events
    )
    assert any(
        event.event_type == "verification_failed"
        and event.payload.get("step_id") == "verify_initial_state"
        and "file_contains_ready" in event.payload.get("conditions_failed", [])
        for event in events
    )
    assert any(
        event.event_type == "replan_triggered"
        and event.payload.get("step_id") == "verify_initial_state"
        and event.payload.get("reason") == "current read verification failed because status ready is not present"
        for event in events
    )
    assert max(_event_sequences(events, "edit_applied")) < max(
        _event_sequences(events, "verification_passed", step_id="edit_after_current_failure")
    )
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_stale_refinement_after_failed_preview_hands_off_for_replan(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=8,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=24,
        planner__max_replans=1,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_yaml_file",
                                "Edit YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited file",
                                success_criteria="The YAML file should contain status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "wrong_ready_check", "check_type": "file_contains", "path": str(release), "pattern": "release: ready"},
                                ],
                                required_conditions=["tool_name_matches", "wrong_ready_check"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_yaml_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            _exact_answer_with_file_check_step(
                                "answer",
                                "recovered",
                                path=release,
                                pattern="status: ready",
                            )
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": True,
                            "requires_replan": False,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the edit was applied but the verification step failed to confirm the expected state",
                        }
                    ),
                ],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    chain_events = [event for event in events if event.event_type == "tool_chain_completed"]
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text"]
    assert any(
        event.event_type == "subsystem_progress"
        and event.payload.get("step_id") == "edit_yaml_file"
        and "handoff_to_verification_after_preview_failure" in str(event.payload.get("progress", ""))
        for event in events
    )
    assert any(event.payload.get("attempts") == 3 and event.payload.get("handoff_to_verification") is True for event in chain_events)
    assert not any(event.payload.get("attempts") == config.runtime.max_tool_steps for event in chain_events)
    assert any(
        event.event_type == "verification_failed"
        and event.payload.get("step_id") == "edit_yaml_file"
        and "wrong_ready_check" in event.payload.get("conditions_failed", [])
        for event in events
    )
    assert any(
        event.event_type == "retry_suppressed"
        and event.payload.get("step_id") == "edit_yaml_file"
        and event.payload.get("reason") == "subsystem_disallowed_same_step_retry"
        for event in events
    )
    assert not any(
        event.event_type == "retry_triggered"
        and event.payload.get("step_id") == "edit_yaml_file"
        and event.payload.get("reason") == "the edit was applied but the verification step failed to confirm the expected state"
        for event in events
    )
    assert any(event.event_type == "replan_triggered" and event.payload.get("step_id") == "edit_yaml_file" for event in events)
    assert len(plan_requests) == 2
    assert "Replan from current observations, history, and environment state" in plan_requests[1]["prompt"]
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_hands_off_after_exact_repeated_successful_write_before_max_tool_steps(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: extremely_hard. Family: failure/multi_step/quality."""
    report = tmp_path / "capacity_report.txt"
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=6,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=24,
        runtime__max_repeated_action_occurrences=1,
        planner__max_replans=1,
    )
    goal = f"Write {report} with computed capacity fields and answer verified."
    wrong_content = "Capacity headroom: 5 workers available out of 6 total workers."
    correct_content = "service=svc-02\nheadroom=74\nworkers=6\n"
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "write_bad_report",
                                "Write report",
                                "write",
                                expected_tool="write_file",
                                expected_output="capacity report",
                                success_criteria="capacity_report.txt contains the expected service, headroom, and worker fields.",
                                verification_checks=[
                                    {"name": "headroom_line", "check_type": "file_contains", "path": str(report), "pattern": "headroom=74"},
                                ],
                                required_conditions=["headroom_line"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "verified", depends_on=["write_bad_report"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "write_correct_report",
                                "Write corrected report",
                                "write",
                                expected_tool="write_file",
                                expected_output="correct capacity report",
                                success_criteria="capacity_report.txt contains the expected service, headroom, and worker fields.",
                                verification_checks=[
                                    {"name": "service_line", "check_type": "file_contains", "path": str(report), "pattern": "service=svc-02"},
                                    {"name": "headroom_line", "check_type": "file_contains", "path": str(report), "pattern": "headroom=74"},
                                    {"name": "workers_line", "check_type": "file_contains", "path": str(report), "pattern": "workers=6"},
                                ],
                                required_conditions=["service_line", "headroom_line", "workers_line"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "verified", depends_on=["write_correct_report"]),
                        ],
                    ),
                ],
                "tool_input:write_file": [
                    json.dumps(_write_file_input(report, wrong_content)),
                    json.dumps(_write_file_input(report, wrong_content)),
                    json.dumps(_write_file_input(report, correct_content)),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": True,
                            "requires_replan": False,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the report write repeated without satisfying verification",
                        }
                    ),
                ],
                "answer_response": ["verified"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    first_chain = next(
        event
        for event in events
        if event.event_type == "tool_chain_completed" and event.payload.get("step_id") == "write_bad_report"
    )

    assert result.assistant_text == "verified"
    assert report.read_text(encoding="utf-8") == correct_content
    assert _tool_names(events) == ["write_file", "write_file"]
    assert any(
        event.event_type == "duplicate_action_detected"
        and event.payload.get("scope") == "current_step_exact_action"
        and "write_bad_report" in event.payload.get("action_key", "")
        for event in events
    )
    assert first_chain.payload.get("handoff_to_verification") is True
    assert first_chain.payload.get("attempts") == 2
    assert first_chain.payload.get("attempts") < config.runtime.max_tool_steps
    assert any(event.event_type == "verification_failed" and event.payload.get("step_id") == "write_bad_report" for event in events)
    assert any(event.event_type == "replan_triggered" and event.payload.get("step_id") == "write_bad_report" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "write_correct_report" for event in events)
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_replan_defers_unresolved_objective_to_final_proof_when_state_satisfies_goal(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: extremely_hard. Family: file_edit/failure/multi_step/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=1,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=26,
        planner__max_replans=1,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    final_objective_prompts: list[str] = []

    def verify_final_objective(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        criteria_section = prompt.split("Criteria:\n", 1)[1].split("\n\n", 1)[0]
        evidence_section = prompt.split("Deterministic evidence:\n", 1)[1].split("\n\n", 1)[0]
        assert '"name":"final_objective_satisfied"' in criteria_section
        final_objective_prompts.append(prompt)
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "final_objective_satisfied",
                        "passed": (
                            "known_files" in evidence_section
                            and "name: report-62\\nstatus: ready\\nowner: team-6\\n" in evidence_section
                            and "recovered" in evidence_section
                        ),
                        "evidence": "the final proof inspected the current file state and candidate answer",
                    }
                ]
            }
        )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_yaml_file",
                                "Edit YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited file",
                                success_criteria="The YAML file should contain status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "release: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_yaml_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "reread_current_file",
                                "Read current YAML",
                                "read",
                                expected_tool="read_file",
                                expected_output="current file",
                                success_criteria="The current YAML file is observed before answering.",
                            ),
                            _exact_answer_step("answer", "recovered", depends_on=["reread_current_file"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "status: ready")),
                    _tool_call("read_file", {"path": str(release)}),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the file state is correct but the failed verification check was wrong",
                        }
                    ),
                ],
                "verification": [verify_final_objective],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "read_file"]
    assert len(plan_requests) == 2
    assert len(final_objective_prompts) == 1
    assert any(
        event.event_type == "verification_failed"
        and event.payload.get("step_id") == "edit_yaml_file"
        and "file_contains_ready" in event.payload.get("conditions_failed", [])
        for event in events
    )
    assert any(
        event.event_type == "unresolved_objective_verification_deferred"
        and event.payload.get("final_step_id") == "answer"
        and event.payload.get("reason") == "mandatory_final_objective_verification"
        for event in events
    )
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer:final_objective" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)
    assert not any(
        event.event_type == "error"
        and "cannot abandon unresolved objective verification" in event.payload.get("error", "")
        for event in events
    )


def test_real_loop_final_objective_verification_replans_after_weak_read_only_success(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_hard. Family: file_edit/failure/multi_step/quality."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=1,
        runtime__max_reasoning_steps=20,
        runtime__max_total_actions=28,
        planner__max_replans=2,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    final_objective_prompts: list[str] = []

    def verify_semantically(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        criteria_section = prompt.split("Criteria:\n", 1)[1].split("\n\n", 1)[0]
        evidence_section = prompt.split("Deterministic evidence:\n", 1)[1].split("\n\n", 1)[0]
        if '"name":"result_satisfies_step"' in criteria_section:
            passed = "+status: ready" in evidence_section and "name: report-62\\nstatus: ready\\nowner: team-6\\n" in evidence_section
            return json.dumps(
                {
                    "criteria": [
                        {
                            "name": "result_satisfies_step",
                            "passed": passed,
                            "evidence": "the model-owned semantic result check inspected the concrete diff and current file text",
                        }
                    ]
                }
            )
        assert '"name":"final_objective_satisfied"' in criteria_section
        final_objective_prompts.append(prompt)
        current_file_is_correct = (
            "known_files" in evidence_section
            and "name: report-62\\nstatus: ready\\nowner: team-6\\n" in evidence_section
        )
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "final_objective_satisfied",
                        "passed": current_file_is_correct,
                        "evidence": "the final objective check used the current workspace evidence",
                    }
                ]
            }
        )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_yaml_file",
                                "Edit YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited file",
                                success_criteria="The YAML file should contain status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_yaml_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "verify_weaker_state",
                                "Verify weaker state",
                                "read",
                                expected_tool="read_file",
                                expected_output="current file",
                                success_criteria="The file contains ready.",
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "file_contains_ready_word", "check_type": "file_contains", "path": str(release), "pattern": "ready"},
                                ],
                                required_conditions=["dependencies_completed", "file_contains_ready_word"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer_without_repair", "recovered", depends_on=["verify_weaker_state"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "repair_yaml_file",
                                "Repair YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited file",
                                success_criteria="The YAML file should contain status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "recovered", depends_on=["repair_yaml_file"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    _tool_call("read_file", {"path": str(release)}),
                    _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the file state is not yet objectively verified",
                        }
                    ),
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the final objective verifier rejected the weakened current state",
                        }
                    ),
                ],
                "verification": [verify_semantically, verify_semantically, verify_semantically],
                "answer_response": ["recovered", "recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    final_failure = _event_sequences(events, "verification_failed", step_id="answer_without_repair:final_objective")
    final_success = _event_sequences(events, "verification_passed", step_id="answer:final_objective")
    edit_applied = _event_sequences(events, "edit_applied")

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "read_file", "edit_text"]
    assert len(plan_requests) == 3
    assert len(final_objective_prompts) == 2
    assert "name: report-62\\nready\\nowner: team-6\\n" in final_objective_prompts[0]
    assert "name: report-62\\nstatus: ready\\nowner: team-6\\n" in final_objective_prompts[1]
    assert final_failure
    assert final_success
    assert len(edit_applied) == 2
    assert final_failure[-1] < edit_applied[-1] < final_success[-1]
    assert any(event.event_type == "step_failed" and event.payload.get("step_id") == "answer_without_repair" for event in events)
    assert not any(event.event_type == "step_completed" and event.payload.get("step_id") == "answer_without_repair" for event in events)
    assert any(
        event.event_type == "replan_triggered"
        and event.payload.get("reason") == "the final objective verifier rejected the weakened current state"
        for event in events
    )
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "repair_yaml_file" for event in events)


def test_real_loop_rejects_wrong_tool_inside_required_tool_step_and_refines(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Change the release status in {release} to ready and answer repaired."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_file",
                                "Edit release",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready file",
                                success_criteria="release.yaml contains status: ready.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "repaired", depends_on=["edit_file"]),
                        ],
                    )
                ],
                "tool_input:edit_text": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    _tool_call("write_file", _write_file_input(release, "name: report-62\nstatus: ready\nowner: team-6\n")),
                    _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready")),
                ],
                "answer_response": ["repaired"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    edit_applied_sequences = _event_sequences(events, "edit_applied")
    edit_verified_sequences = _event_sequences(events, "verification_passed", step_id="edit_file")

    assert result.assistant_text == "repaired"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text", "edit_text"]
    assert any(
        event.event_type == "subsystem_progress"
        and event.payload.get("step_id") == "edit_file"
        and "preview_passed=False" in str(event.payload.get("progress", ""))
        for event in events
    )
    assert any(event.event_type == "tool_error" and event.payload.get("tool_name") == "edit_text" for event in events)
    assert not any(event.event_type.startswith("tool_graph") for event in events)
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert not any(event.event_type == "tool_called" and event.payload.get("tool_name") == "write_file" for event in events)
    assert edit_applied_sequences
    assert edit_verified_sequences
    assert max(edit_applied_sequences) < max(edit_verified_sequences)
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "answer" for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "edit_file" for event in events)


def test_real_loop_extremely_easy_exact_edit_reread_and_verified_final_state(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_easy. Family: file_edit."""
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha old omega\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=6,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Change old to new in {sample}, reread the file, and answer done."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_before", "Read before edit", "read", expected_tool="read_text", expected_output="original text", success_criteria="The original text is observed."),
                            plan_step(
                                "edit_file",
                                "Edit file",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited text",
                                success_criteria="The file contains alpha new omega.",
                                depends_on=["read_before"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "tool_effect", "check_type": "tool_effect_verified"},
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tool_effect"],
                                optional_conditions=[],
                            ),
                            plan_step("read_after", "Read after edit", "read", expected_tool="read_text", expected_output="edited text reread", success_criteria="The edited text is observed.", depends_on=["edit_file"]),
                            _exact_answer_step("answer", "done", depends_on=["read_after"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("read_text", _read_text_input(path=str(sample))),
                    _tool_call("edit_text", _edit_replace_input(sample, "old", "new")),
                    _tool_call("read_text", _read_text_input(path=str(sample))),
                ],
                "answer_response": ["done"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "done"
    assert sample.read_text(encoding="utf-8") == "alpha new omega\n"
    assert _tool_names(events) == ["read_text", "edit_text", "read_text"]
    assert _event_sequences(events, "edit_applied")
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="edit_file"))
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "read_after" for event in events)


def test_real_loop_status_ready_exact_replacement_reread_and_final_objective_proof(
    make_config,
    tmp_path: Path,
) -> None:
    """Difficulty: extremely_easy. Family: file_edit/quality/semantic_authority."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=6,
        runtime__max_reasoning_steps=12,
        runtime__max_total_actions=18,
    )
    goal = f"Change {release} from status: draft to status: ready, reread it, and answer done."
    final_objective_prompts: list[str] = []

    def verify(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        if "final_objective_satisfied" in prompt:
            final_objective_prompts.append(prompt)
            assert "status: ready" in prompt
            assert "owner: team-6" in prompt
            criterion_name = "final_objective_satisfied"
        else:
            assert "result_satisfies_step" in prompt
            criterion_name = "result_satisfies_step"
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": criterion_name,
                        "passed": True,
                        "evidence": "the current file evidence proves the requested state",
                    }
                ]
            }
        )

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_before", "Read before edit", "read", expected_tool="read_text", expected_output="draft release", success_criteria="The original release text is observed."),
                            plan_step(
                                "edit_status",
                                "Edit status",
                                "write",
                                expected_tool="edit_text",
                                expected_output="ready release",
                                success_criteria="The file contains status: ready and preserves the rest of the file.",
                                depends_on=["read_before"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "tool_effect", "check_type": "tool_effect_verified"},
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tool_effect"],
                                optional_conditions=[],
                            ),
                            plan_step("read_after", "Reread after edit", "read", expected_tool="read_text", expected_output="ready release text", success_criteria="The ready release text is observed.", depends_on=["edit_status"]),
                            plan_step(
                                "answer",
                                "Answer",
                                "respond",
                                expected_output="done",
                                success_criteria="Answer done after the final file state is proven.",
                                depends_on=["read_after"],
                                verification_type="composite",
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                    {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": "done"},
                                ],
                                required_conditions=["dependencies_completed", "file_contains_ready", "answer_exact"],
                                optional_conditions=[],
                            ),
                        ],
                    )
                ],
                "tool_input:read_text": [
                    json.dumps(_read_text_input(path=str(release))),
                    json.dumps(_read_text_input(path=str(release))),
                ],
                "tool_input:edit_text": [
                    json.dumps(_edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "verification": [verify, verify],
                "answer_response": ["done"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    edit_event = next(event for event in events if event.event_type == "tool_result" and event.payload.get("tool_name") == "edit_text")
    edit_applied = _event_sequences(events, "edit_applied")
    file_reads = _event_sequences(events, "file_chunk_read")
    final_proof = _event_sequences(events, "verification_passed", step_id="answer:final_objective")

    assert result.assistant_text == "done"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_text", "edit_text", "read_text"]
    assert edit_event.payload["validated_input"]["operation"] == "replace_exact"
    assert any(
        event.event_type == "verification_passed"
        and event.payload.get("step_id") == "edit_status"
        and "tool_effect" in event.payload.get("conditions_met", [])
        for event in events
    )
    assert "start" not in edit_event.payload["validated_input"]
    assert "end" not in edit_event.payload["validated_input"]
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert len(file_reads) >= 2
    assert edit_applied and max(edit_applied) < max(file_reads)
    assert final_proof and max(file_reads) < max(final_proof)
    assert final_objective_prompts


def test_real_loop_easy_structured_multi_file_reading_uses_observations(make_config, tmp_path: Path) -> None:
    """Difficulty: easy. Family: reading."""
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("left: cobalt\n", encoding="utf-8")
    right.write_text("right: amber\n", encoding="utf-8")
    config = make_config(tools__allow_stateful_tools=True, runtime__max_reasoning_steps=10, runtime__max_total_actions=14)
    goal = "Read the two files and answer with cobalt+amber."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_left", "Read left", "read", expected_tool="read_text", expected_output="left content", success_criteria="left content observed"),
                            plan_step("read_right", "Read right", "read", expected_tool="read_text", expected_output="right content", success_criteria="right content observed", depends_on=["read_left"]),
                            _exact_answer_step("answer", "cobalt+amber", depends_on=["read_right"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("read_text", _read_text_input(path=str(left))),
                    _tool_call("read_text", _read_text_input(path=str(right))),
                ],
                "answer_response": ["cobalt+amber"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    state = runtime.history.rebuild_from_history(result.session_id)

    assert result.assistant_text == "cobalt+amber"
    assert _tool_names(events) == ["read_text", "read_text"]
    assert str(left) in state.reader_states[next(iter(state.reader_states))].source_ref or state.file_views
    assert any("cobalt" in event.payload.get("text", "") for event in events if event.event_type == "file_chunk_read")
    assert any("amber" in event.payload.get("text", "") for event in events if event.event_type == "file_chunk_read")


def test_real_loop_normal_multistep_read_compute_write_verify_with_distractor_tools(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: multi_step."""
    numbers = tmp_path / "numbers.txt"
    output = tmp_path / "answer.txt"
    numbers.write_text("6\n7\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_reasoning_steps=16,
        runtime__max_total_actions=24,
    )
    goal = "Read the two numbers, multiply them, write result=42, verify the file, and answer written."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_numbers", "Read numbers", "read", expected_tool="read_text", expected_output="numbers", success_criteria="numbers observed"),
                            plan_step(
                                "compute_product",
                                "Compute product",
                                "tool",
                                expected_tool="calculator",
                                expected_output="42",
                                success_criteria="The calculator returns 42.",
                                depends_on=["read_numbers"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                                    {"name": "exact_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 42},
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "exact_result"],
                                optional_conditions=[],
                            ),
                            plan_step(
                                "write_answer",
                                "Write answer",
                                "write",
                                expected_tool="write_file",
                                expected_output="result file",
                                success_criteria="answer.txt contains result=42",
                                depends_on=["compute_product"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                                    {"name": "file_contains_result", "check_type": "file_contains", "path": str(output), "pattern": "result=42"},
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_result"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "written", depends_on=["write_answer"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("read_text", _read_text_input(path=str(numbers))),
                    _tool_call("calculator", {"expression": "6 * 7"}),
                    _tool_call("write_file", _write_file_input(output, "result=42\n")),
                ],
                "answer_response": ["written"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "written"
    assert output.read_text(encoding="utf-8") == "result=42\n"
    assert _tool_names(events) == ["read_text", "calculator", "write_file"]
    planning_prompts = [request["prompt"] for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert planning_prompts and all("- calculator" in prompt and "- write_file" in prompt and "- run_tests" in prompt for prompt in planning_prompts)
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert any(request["contract"] == "tool_input:calculator" for request in runtime.client.requests)
    assert any(request["contract"] == "tool_input:write_file" for request in runtime.client.requests)
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="write_answer"))


def test_real_loop_hard_coding_failed_tests_replans_repairs_and_verifies(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: coding/quality/failure."""
    module = tmp_path / "calc.py"
    test_file = tmp_path / "test_calc.py"
    module.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    test_file.write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        planner__max_replans=3,
        runtime__max_tool_steps=6,
        runtime__max_reasoning_steps=20,
        runtime__max_total_actions=30,
    )
    goal = "Run the tests, fix the add implementation if they fail, rerun tests, and answer fixed."
    run_test_check = {
        "name": "tests_pass",
        "check_type": "command_success",
        "command": ["python", "-m", "pytest", "-q", "test_calc.py"],
        "cwd": str(tmp_path),
        "framework": "pytest",
        "timeout_seconds": 15,
    }
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "run_failing_tests",
                                "Run initial tests",
                                "tool",
                                expected_tool="run_tests",
                                expected_output="failing test output",
                                success_criteria="Tests pass before editing.",
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
                                    run_test_check,
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "fixed", depends_on=["run_failing_tests"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_module", "Read module", "read", expected_tool="read_text", expected_output="module source", success_criteria="source observed"),
                            plan_step(
                                "repair_module",
                                "Repair module",
                                "write",
                                expected_tool="edit_text",
                                expected_output="fixed module",
                                success_criteria="calc.add uses addition.",
                                depends_on=["read_module"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_add", "check_type": "file_contains", "path": str(module), "pattern": "return a + b"},
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_add"],
                                optional_conditions=[],
                            ),
                            plan_step(
                                "run_passing_tests",
                                "Run repaired tests",
                                "tool",
                                expected_tool="run_tests",
                                expected_output="passing tests",
                                success_criteria="Tests pass after repair.",
                                depends_on=["repair_module"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
                                    run_test_check,
                                ],
                                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "fixed", depends_on=["run_passing_tests"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("run_tests", _run_pytest_input("test_calc.py")),
                    _tool_call("read_text", _read_text_input(path=str(module))),
                    _tool_call("edit_text", _edit_replace_input(module, "return a - b", "return a + b")),
                    _tool_call("run_tests", _run_pytest_input("test_calc.py")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "initial test run failed and needs a repair plan",
                        }
                    )
                ],
                "answer_response": ["fixed"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "fixed"
    assert module.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    process_events = [event for event in events if event.event_type == "process_completed"]
    assert any(event.payload.get("return_code") not in {0, None} for event in process_events)
    assert any(event.payload.get("return_code") == 0 for event in process_events)
    assert any(event.event_type == "replan_triggered" for event in events)
    assert _tool_names(events) == ["run_tests", "read_text", "edit_text", "run_tests"]
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="run_passing_tests"))


def test_real_loop_failure_recovery_after_failed_tool_call_continues_from_history(make_config, tmp_path: Path) -> None:
    """Difficulty: normal. Family: failure/reading."""
    target = tmp_path / "target.txt"
    target.write_text("recoverable content\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        planner__max_replans=2,
        runtime__max_tool_steps=3,
        runtime__max_reasoning_steps=14,
        runtime__max_total_actions=20,
    )
    goal = "Read target.txt and answer recovered."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_missing", "Read missing file", "read", expected_tool="read_text", expected_output="file text", success_criteria="file text observed"),
                            _exact_answer_step("unreachable", "recovered", depends_on=["read_missing"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_target", "Read target file", "read", expected_tool="read_text", expected_output="file text", success_criteria="target text observed"),
                            _exact_answer_step("answer", "recovered", depends_on=["read_target"]),
                        ],
                    ),
                ],
                "tool_decision": [
                    _tool_call("read_text", _read_text_input(path=str(tmp_path / "missing.txt"))),
                    _tool_call("read_text", _read_text_input(path=str(target))),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "tool_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the observed path was missing; use the available target path",
                        }
                    )
                ],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "recovered"
    assert _tool_names(events) == ["read_text", "read_text"]
    assert any(event.event_type == "tool_error" and "missing.txt" in event.payload.get("error", "") for event in events)
    assert any(event.event_type == "file_chunk_read" and "recoverable content" in event.payload.get("text", "") for event in events)
    assert not any(event.event_type == "step_failed" and event.payload.get("step_id") == "read_missing" for event in events)
    assert not any(event.event_type == "replan_triggered" for event in events)


def test_real_loop_replan_reused_step_ids_do_not_trigger_repeated_action_abort(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=1,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=24,
        planner__max_replans=1,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_yaml_file", "Read YAML", "read", expected_tool="read_text", expected_output="file text", success_criteria="file content observed"),
                            plan_step(
                                "edit_yaml_file",
                                "Edit YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited file",
                                success_criteria="The YAML file should contain status: ready.",
                                depends_on=["read_yaml_file"],
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "wrong_ready_check", "check_type": "file_contains", "path": str(release), "pattern": "release: ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "wrong_ready_check"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_yaml_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "read_yaml_file",
                                "Read YAML",
                                "read",
                                expected_tool="read_text",
                                expected_output="file text",
                                success_criteria="file content observed and contains the expected state.",
                                verification_checks=[
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "read_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "recovered", depends_on=["read_yaml_file"]),
                        ],
                    ),
                ],
                "tool_input:read_text": [
                    json.dumps(_read_text_input(path=str(release))),
                    json.dumps(_read_text_input(path=str(release))),
                ],
                "tool_input:edit_text": [
                    json.dumps(_edit_replace_input(release, "status: draft", "status: ready")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": False,
                            "requires_replan": True,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the edit was applied but the plan verifier checked the wrong text",
                        }
                    ),
                ],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["read_text", "edit_text", "read_text"]
    plan_requests = [request for request in runtime.client.requests if request["contract"] == "task_plan"]
    assert len(plan_requests) == 2
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert sum(1 for request in runtime.client.requests if request["contract"] == "tool_input:read_text") == 2
    replan_prompt = plan_requests[1]["prompt"]
    assert "Replan from current observations, history, and environment state" in replan_prompt
    assert "Failed steps do not undo prior tool side effects" in replan_prompt
    assert "plan verification and final response rather than repeating the mutation" in replan_prompt
    assert any(event.event_type == "replan_triggered" and event.payload.get("step_id") == "edit_yaml_file" for event in events)
    action_events = [event for event in events if event.event_type == "action_selected"]
    assert not any(
        event.payload.get("selected_action") == "replan"
        and event.payload.get("step_id") == "read_yaml_file"
        and any(score.get("reason") == "repeated_action_limit_exceeded" for score in event.payload.get("scores", []))
        for event in action_events
    )
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="read_yaml_file"))
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_invalid_arguments_after_failed_preview_hands_off_for_replan(make_config, tmp_path: Path) -> None:
    """Difficulty: hard. Family: file_edit/failure/multi_step."""
    release = tmp_path / "release.yaml"
    release.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    config = make_config(
        model__context_limit=32_000,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=8,
        runtime__max_reasoning_steps=18,
        runtime__max_total_actions=24,
        runtime__max_repeated_action_occurrences=2,
        planner__max_replans=1,
    )
    goal = f"Edit {release} so the status moves from draft to ready and answer recovered."
    invalid_edit_input = json.dumps({"path": str(release), "content": "invalid edit payload", "create": True})
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "edit_yaml_file",
                                "Edit YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="edited file",
                                success_criteria="The YAML file should contain status: ready.",
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("unreachable_answer", "recovered", depends_on=["edit_yaml_file"]),
                        ],
                    ),
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "repair_yaml_file",
                                "Repair YAML",
                                "write",
                                expected_tool="edit_text",
                                expected_output="repaired file",
                                success_criteria="The YAML file should contain status: ready.",
                                verification_checks=[
                                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                    {"name": "file_contains_ready", "check_type": "file_contains", "path": str(release), "pattern": "status: ready"},
                                ],
                                required_conditions=["dependencies_completed", "tool_name_matches", "file_contains_ready"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "recovered", depends_on=["repair_yaml_file"]),
                        ],
                    ),
                ],
                "tool_input:edit_text": [
                    _tool_call("edit_text", _edit_replace_input(release, "status: draft", "ready")),
                    invalid_edit_input,
                    invalid_edit_input,
                    _tool_call("edit_text", _edit_replace_input(release, "ready", "status: ready")),
                ],
                "failure_classification": [
                    json.dumps(
                        {
                            "kind": "verification_failure",
                            "retryable": True,
                            "requires_replan": False,
                            "suggested_strategy_mode": "recovery",
                            "wait_seconds": 0.0,
                            "reason": "the edit mutated the file but failed objective verification",
                        }
                    )
                ],
                "answer_response": ["recovered"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "recovered"
    assert release.read_text(encoding="utf-8") == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert _tool_names(events) == ["edit_text", "edit_text", "edit_text", "edit_text"]
    assert not any(request["contract"] == "tool_decision" for request in runtime.client.requests)
    assert any(
        event.event_type == "subsystem_progress"
        and event.payload.get("step_id") == "edit_yaml_file"
        and "preview_passed=False" in str(event.payload.get("progress", ""))
        for event in events
    )
    assert sum(1 for event in events if event.event_type == "tool_error" and event.payload.get("tool_name") == "edit_text") == 2
    assert not any(event.event_type == "duplicate_action_detected" for event in events)
    assert any(
        event.event_type == "tool_chain_completed"
        and event.payload.get("step_id") == "edit_yaml_file"
        and event.payload.get("success") is True
        and event.payload.get("handoff_to_verification") is True
        for event in events
    )
    assert any(event.event_type == "retry_suppressed" and event.payload.get("step_id") == "edit_yaml_file" for event in events)
    assert any(event.event_type == "replan_triggered" and event.payload.get("step_id") == "edit_yaml_file" for event in events)
    assert not any(event.event_type == "retry_triggered" and event.payload.get("step_id") == "edit_yaml_file" for event in events)
    assert max(_event_sequences(events, "edit_applied")) < max(_event_sequences(events, "verification_passed", step_id="repair_yaml_file"))
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "answer" for event in events)


def test_real_loop_answer_prompt_exposes_model_authored_step_contract(make_config) -> None:
    """Difficulty: normal. Family: quality/multi_step."""
    expected = "The file content is now: name: report-62\nstatus: ready\nowner: team-6"
    goal = "Summarize the verified final state exactly as planned."
    observed: dict[str, str] = {}
    config = make_config(
        model__context_limit=32_000,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=12,
    )

    def exact_answer_from_visible_contract(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        observed["prompt"] = prompt
        assert "Current answer step contract:" in prompt
        assert "expected_output: " + expected in prompt
        assert '"check_type":"exact_match"' in prompt
        assert expected in prompt
        return expected

    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "summarize_final_state",
                                "Summarize final state",
                                "respond",
                                expected_output=expected,
                                success_criteria="The answer exactly matches the model-authored expected summary.",
                                verification_type="composite",
                                verification_checks=[
                                    {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": expected}
                                ],
                                required_conditions=["answer_exact"],
                                optional_conditions=[],
                            )
                        ],
                    )
                ],
                "answer_response": [exact_answer_from_visible_contract],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == expected
    assert observed["prompt"]
    assert any(event.event_type == "verification_passed" and event.payload.get("step_id") == "summarize_final_state" for event in events)
    assert not any(event.event_type == "verification_failed" for event in events)


def test_real_loop_extremely_hard_iterative_refinement_is_not_duplicate_abort(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_hard. Family: file_edit/quality."""
    target = tmp_path / "value.txt"
    target.write_text("value = 0\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=10,
        runtime__max_total_actions=16,
        runtime__max_repeated_action_occurrences=1,
    )
    goal = "Refine value.txt until it contains value = 2 and answer refined."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step(
                                "refine_value",
                                "Refine value",
                                "write",
                                expected_tool="edit_text",
                                expected_output="value = 2",
                                success_criteria="The file contains value = 2.",
                                verification_checks=[
                                    {"name": "file_contains_final", "check_type": "file_contains", "path": str(target), "pattern": "value = 2"},
                                ],
                                required_conditions=["file_contains_final"],
                                optional_conditions=[],
                            ),
                            _exact_answer_step("answer", "refined", depends_on=["refine_value"]),
                        ],
                    )
                ],
                "tool_decision": [
                    _tool_call("edit_text", _edit_replace_input(target, "value = 0", "value = 1")),
                    _tool_call("edit_text", _edit_replace_input(target, "value = 1", "value = 2")),
                ],
                "answer_response": ["refined"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "refined"
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert _tool_names(events) == ["edit_text", "edit_text"]
    assert not any(event.event_type == "duplicate_action_detected" for event in events)
    assert any(event.event_type == "subsystem_progress" and "preview_passed=False" in event.payload.get("progress", "") for event in events)
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "refine_value" for event in events)


def test_real_loop_prevents_repeated_action_loop_without_false_success(make_config, tmp_path: Path) -> None:
    """Difficulty: extremely_hard. Family: failure."""
    target = tmp_path / "loop.txt"
    target.write_text("loop\n", encoding="utf-8")
    config = make_config(
        tools__allow_stateful_tools=True,
        runtime__max_tool_steps=4,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=10,
        runtime__max_repeated_action_occurrences=1,
        runtime__no_progress_failure_limit=2,
        planner__max_replans=0,
    )
    goal = "Read loop.txt and answer done."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [
                    plan_response(
                        goal=goal,
                        steps=[
                            plan_step("read_loop", "Read loop file", "read", expected_tool="read_text", expected_output="file text", success_criteria="file text observed"),
                            _exact_answer_step("answer", "done", depends_on=["read_loop"]),
                        ],
                    )
                ],
                "tool_decision": [
                    json.dumps({"action": "respond", "response": "not yet", "tool_name": "none", "tool_input": {}}),
                    json.dumps({"action": "respond", "response": "not yet", "tool_name": "none", "tool_input": {}}),
                ],
                "answer_response": ["done"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text.startswith("Task incomplete:")
    assert "Verified success was not reached" in result.assistant_text
    assert not any(event.event_type == "tool_result" for event in events)
    assert any(event.event_type == "duplicate_action_detected" for event in events)
    assert any(event.event_type == "reasoning_completed" and event.payload.get("status") != "completed" for event in events)
    assert not any(request["contract"] == "answer_response" for request in runtime.client.requests)


def test_real_loop_skill_selection_adds_instructions_without_hiding_tools(make_config, monkeypatch) -> None:
    """Difficulty: normal. Family: quality/semantic_authority."""
    monkeypatch.setattr("swaag.skills.selector.build_backend", lambda *args, **kwargs: _CodingSkillBackend())
    monkeypatch.setattr("swaag.retrieval.retriever.build_backend", lambda *args, **kwargs: _CodingSkillBackend())
    monkeypatch.setattr("swaag.guidance.resolver.build_backend", lambda *args, **kwargs: _CodingSkillBackend())
    config = make_config(
        model__context_limit=32_000,
        retrieval__backend="llm_scoring",
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=12,
    )
    goal = "Repair broken code behavior and answer inspected."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_plan": [plan_response(goal=goal, steps=[_exact_answer_step("answer", "inspected")])],
                "answer_response": ["inspected"],
            }
        ),
    )

    result = runtime.run_turn(goal)
    plan_prompts = [event.payload["prompt"] for event in runtime.history.read_history(result.session_id) if event.event_type == "prompt_built" and event.payload.get("kind") == "plan"]

    assert result.assistant_text == "inspected"
    assert plan_prompts
    assert "Selected skill instructions:" in plan_prompts[0]
    assert "Coding Patch" in plan_prompts[0]
    assert "- read_text" in plan_prompts[0]
    assert "- edit_text" in plan_prompts[0]
    assert "- run_tests" in plan_prompts[0]
