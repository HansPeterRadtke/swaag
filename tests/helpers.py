from __future__ import annotations

import json
import re
from typing import Any

from swaag.expander import expand_task
from swaag.failure import classify_failure_from_payload
from swaag.model import CompletionRequestPolicy
from swaag.strategy import strategy_from_payload
from swaag.types import CompletionResult, ContractSpec, DecisionOutcome, PromptAnalysis


class FakeModelClient:
    def __init__(self, responses: list[Any] | None = None, *, contract_responses: dict[str, list[Any]] | None = None):
        self._responses = list(responses or [])
        self._contract_responses = {key: list(value) for key, value in (contract_responses or {}).items()}
        self.requests: list[dict[str, Any]] = []
        self.tokenize_requests: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tokenize(self, text: str) -> int:
        self.tokenize_requests.append(text)
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def build_completion_request(self, prompt: str, *, max_tokens: int, contract, temperature: float | None = None) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "contract": contract.name,
        }
        if contract.grammar:
            payload["grammar"] = contract.grammar
        if contract.json_schema:
            payload["json_schema"] = contract.json_schema
        return payload

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> CompletionRequestPolicy:
        return CompletionRequestPolicy(
            profile_name="test",
            structured_output_mode="server_schema",
            effective_contract_mode=contract.mode,
            effective_timeout_seconds=30,
            progress_poll_seconds=0.05,
        )

    def resolve_contract(
        self,
        contract: ContractSpec,
        *,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> tuple[ContractSpec, CompletionRequestPolicy]:
        return contract, self.select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> CompletionResult:
        self.requests.append(payload)
        contract_name = str(payload.get("contract", ""))
        response = None
        contract_queue = self._contract_responses.get(contract_name)
        if contract_queue:
            response = contract_queue.pop(0)
        elif contract_name in {
            "prompt_analysis",
            "task_decision",
            "task_expansion",
            "active_session_control",
            "verification",
            "strategy_selection",
            "failure_classification",
            "action_selection",
            "subagent_selection",
            "generation_decomposition",
            "overflow_recovery",
        }:
            response = self._auto_frontend_response(payload)
        elif self._responses:
            response = self._responses.pop(0)
        else:
            raise AssertionError("No fake model responses left")
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(payload=payload)
        if isinstance(response, CompletionResult):
            return response
        if not isinstance(response, str):
            raise TypeError(f"Unsupported fake response: {response!r}")
        if contract_name.startswith("tool_input:"):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("action") == "call_tool" and isinstance(parsed.get("tool_input"), dict):
                response = json.dumps(parsed["tool_input"])
        if contract_name == "plain_text" and self._responses:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("action") == "call_tool" and isinstance(parsed.get("tool_input"), dict):
                response = self._responses.pop(0)
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )

    def complete(self, prompt: str, *, max_tokens: int, contract, temperature: float | None = None) -> CompletionResult:
        return self.send_completion(self.build_completion_request(prompt, max_tokens=max_tokens, contract=contract, temperature=temperature))

    def _auto_frontend_response(self, payload: dict[str, Any]) -> str:
        contract_name = str(payload.get("contract", ""))
        prompt = str(payload.get("prompt", ""))
        current_request = _extract_section(prompt, "Current user request:")
        if contract_name == "prompt_analysis":
            analysis = _default_prompt_analysis(current_request)
            return json.dumps(
                {
                    "task_type": analysis.task_type,
                    "completeness": analysis.completeness,
                    "requires_expansion": analysis.requires_expansion,
                    "requires_decomposition": analysis.requires_decomposition,
                    "confidence": analysis.confidence,
                    "detected_entities": analysis.detected_entities,
                    "detected_goals": analysis.detected_goals,
                }
            )
        if contract_name == "task_decision":
            analysis_payload = json.loads(_extract_section(prompt, "Prompt analysis:"))
            analysis = _analysis_from_payload(current_request, analysis_payload)
            analysis.task_type = analysis_payload.get("task_type", analysis.task_type)
            analysis.completeness = analysis_payload.get("completeness", analysis.completeness)
            analysis.requires_expansion = bool(analysis_payload.get("requires_expansion", analysis.requires_expansion))
            analysis.requires_decomposition = bool(analysis_payload.get("requires_decomposition", analysis.requires_decomposition))
            analysis.confidence = float(analysis_payload.get("confidence", analysis.confidence))
            analysis.detected_entities = list(analysis_payload.get("detected_entities", analysis.detected_entities))
            analysis.detected_goals = list(analysis_payload.get("detected_goals", analysis.detected_goals))
            decision = _decision_from_analysis(analysis)
            return json.dumps(
                {
                    "split_task": decision.split_task,
                    "expand_task": decision.expand_task,
                    "ask_user": decision.ask_user,
                    "assume_missing": decision.assume_missing,
                    "generate_ideas": decision.generate_ideas,
                    "direct_response": decision.direct_response,
                    "execution_mode": decision.execution_mode,
                    "preferred_tool_name": decision.preferred_tool_name,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                }
            )
        if contract_name == "task_expansion":
            analysis_payload = json.loads(_extract_section(prompt, "Prompt analysis:"))
            decision_payload = json.loads(_extract_section(prompt, "Task decision:"))
            analysis = _analysis_from_payload(current_request, analysis_payload)
            analysis.task_type = analysis_payload.get("task_type", analysis.task_type)
            analysis.completeness = analysis_payload.get("completeness", analysis.completeness)
            analysis.requires_expansion = bool(analysis_payload.get("requires_expansion", analysis.requires_expansion))
            analysis.requires_decomposition = bool(analysis_payload.get("requires_decomposition", analysis.requires_decomposition))
            analysis.confidence = float(analysis_payload.get("confidence", analysis.confidence))
            analysis.detected_entities = list(analysis_payload.get("detected_entities", analysis.detected_entities))
            analysis.detected_goals = list(analysis_payload.get("detected_goals", analysis.detected_goals))
            decision = _decision_from_analysis(analysis)
            decision.split_task = bool(decision_payload.get("split_task", decision.split_task))
            decision.expand_task = bool(decision_payload.get("expand_task", decision.expand_task))
            decision.ask_user = bool(decision_payload.get("ask_user", decision.ask_user))
            decision.assume_missing = bool(decision_payload.get("assume_missing", decision.assume_missing))
            decision.generate_ideas = bool(decision_payload.get("generate_ideas", decision.generate_ideas))
            decision.direct_response = bool(decision_payload.get("direct_response", decision.direct_response))
            decision.execution_mode = str(decision_payload.get("execution_mode", decision.execution_mode))
            decision.preferred_tool_name = str(decision_payload.get("preferred_tool_name", decision.preferred_tool_name))
            decision.confidence = float(decision_payload.get("confidence", decision.confidence))
            decision.reason = str(decision_payload.get("reason", decision.reason))
            expanded = expand_task(current_request, analysis, decision)
            return json.dumps(
                {
                    "original_goal": expanded.original_goal,
                    "expanded_goal": expanded.expanded_goal,
                    "scope": expanded.scope,
                    "constraints": expanded.constraints,
                    "expected_outputs": expanded.expected_outputs,
                    "assumptions": expanded.assumptions,
                }
            )
        if contract_name == "active_session_control":
            return json.dumps(
                {
                    "action": "continue_with_note",
                    "reason": "default control handling continues current work",
                    "response_text": "",
                    "added_context": current_request.strip(),
                    "replacement_goal": "",
                    "queued_task": "",
                    "clarification_question": "",
                }
            )
        if contract_name == "verification":
            criteria = json.loads(_extract_section(prompt, "Criteria:"))
            candidate = _extract_section(prompt, "Candidate result:")
            return json.dumps(
                {
                    "criteria": [
                        {
                            "name": criterion["name"] if isinstance(criterion, dict) else criterion,
                            "passed": bool(candidate.strip()),
                            "evidence": "candidate result is non-empty",
                        }
                        for criterion in criteria
                    ]
                }
            )
        if contract_name == "strategy_selection":
            # Test fixture simulating an LLM strategy_selection response.
            # Defaults to "generic" so test plans aren't rejected by profile-
            # specific required_step_kinds. The real LLM uses the full goal
            # context to pick a tighter profile when appropriate.
            payload = {
                "task_profile": "generic",
                "strategy_name": "conservative",
                "explore_before_commit": False,
                "tool_chain_depth": 1,
                "verification_intensity": 1.0,
                "reason": "default strategy",
            }
            strategy_from_payload(payload)
            return json.dumps(payload)
        if contract_name == "failure_classification":
            payload = {
                "kind": "reasoning_failure",
                "retryable": True,
                "requires_replan": False,
                "suggested_strategy_mode": "recovery",
                "wait_seconds": 0.0,
                "reason": "generic failure",
            }
            classify_failure_from_payload(payload)
            return json.dumps(payload)
        if contract_name == "action_selection":
            match = re.search(r"Default deterministic choice:\s*([a-z_]+)", prompt)
            action = match.group(1) if match is not None else "execute_step"
            return json.dumps({"action": action, "reason": "test scaffold action choice"})
        if contract_name == "subagent_selection":
            match = re.search(r"Available subagents:\s*([^\n]+)", prompt)
            available = []
            if match is not None:
                available = [item.strip().rstrip(".") for item in match.group(1).split(",") if item.strip()]
            chosen = next((item for item in available if item != "none"), "none")
            spawn = chosen != "none"
            return json.dumps(
                {
                    "spawn": spawn,
                    "subagent_type": chosen,
                    "reason": "test scaffold subagent choice",
                    "focus": "use the scoped specialist view",
                }
            )
        if contract_name == "generation_decomposition":
            return json.dumps(
                {
                    "output_class": "open_ended",
                    "reason": "single answer unit is sufficient for the test scaffold",
                    "units": [
                        {
                            "unit_id": "answer_unit_01",
                            "title": "Final answer",
                            "instruction": "Provide the final answer for the user request.",
                        }
                    ],
                }
            )
        if contract_name == "overflow_recovery":
            return json.dumps(
                {
                    "keep_partial": True,
                    "reason": "keep the current partial output",
                    "next_units": [],
                }
            )
        raise AssertionError(f"Unsupported automatic frontend contract: {contract_name}")


_SECTION_RE = re.compile(r"^(?P<label>[A-Za-z ]+):\n(?P<body>.*?)(?:\n\n|\Z)", re.DOTALL | re.MULTILINE)


def _extract_section(prompt: str, label: str) -> str:
    for match in _SECTION_RE.finditer(prompt):
        if match.group("label").strip() == label.rstrip(":"):
            return match.group("body").strip()
    return ""


def _default_prompt_analysis(current_request: str) -> PromptAnalysis:
    stripped = current_request.strip()
    if not stripped:
        return PromptAnalysis(
            task_type="incomplete",
            completeness="incomplete",
            requires_expansion=False,
            requires_decomposition=False,
            confidence=0.8,
            detected_entities=[],
            detected_goals=[],
        )
    return PromptAnalysis(
        task_type="structured",
        completeness="complete",
        requires_expansion=False,
        requires_decomposition=False,
        confidence=0.8,
        detected_entities=[],
        detected_goals=[],
    )


def _analysis_from_payload(current_request: str, payload: dict[str, Any]) -> PromptAnalysis:
    base = _default_prompt_analysis(current_request)
    return PromptAnalysis(
        task_type=str(payload.get("task_type", base.task_type)),
        completeness=str(payload.get("completeness", base.completeness)),
        requires_expansion=bool(payload.get("requires_expansion", base.requires_expansion)),
        requires_decomposition=bool(payload.get("requires_decomposition", base.requires_decomposition)),
        confidence=float(payload.get("confidence", base.confidence)),
        detected_entities=[str(item) for item in payload.get("detected_entities", base.detected_entities)],
        detected_goals=[str(item) for item in payload.get("detected_goals", base.detected_goals)],
    )


def _decision_from_analysis(analysis: PromptAnalysis) -> DecisionOutcome:
    return DecisionOutcome(
        split_task=analysis.requires_decomposition,
        expand_task=analysis.requires_expansion,
        ask_user=analysis.completeness == "incomplete" and analysis.task_type != "vague",
        assume_missing=analysis.task_type == "vague",
        generate_ideas=analysis.task_type in {"vague", "unstructured"},
        confidence=analysis.confidence,
        reason=f"test_scaffold task_type={analysis.task_type} completeness={analysis.completeness}",
        direct_response=False,
        execution_mode="full_plan",
        preferred_tool_name="",
    )


def plan_step(
    step_id: str,
    title: str,
    kind: str,
    *,
    expected_tool: str = "",
    goal: str | None = None,
    input_text: str | None = None,
    expected_output: str,
    done_condition: str | None = None,
    success_criteria: str,
    input_refs: list[str] | None = None,
    output_refs: list[str] | None = None,
    fallback_strategy: str = "",
    depends_on: list[str] | None = None,
    verification_type: str | None = None,
    verification_checks: list[dict[str, Any]] | None = None,
    required_conditions: list[str] | None = None,
    optional_conditions: list[str] | None = None,
) -> dict[str, Any]:
    if done_condition is None:
        if kind == "respond":
            done_condition = "assistant_response_nonempty"
        elif kind == "reasoning":
            done_condition = "reasoning_result_nonempty"
        elif expected_tool:
            done_condition = f"tool_result:{expected_tool}"
        else:
            done_condition = "reasoning_result_nonempty"
    if verification_type is None and verification_checks is None and required_conditions is None and optional_conditions is None:
        if kind in {"tool", "read", "write", "note"}:
            verification_type = "composite"
            verification_checks = [
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": expected_tool},
                {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
                {"name": "output_schema_valid", "check_type": "tool_output_schema_valid"},
            ]
            required_conditions = [item["name"] for item in verification_checks]
            optional_conditions = []
        else:
            verification_type = "llm_fallback"
            verification_checks = [
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {
                    "name": "assistant_text_nonempty" if kind == "respond" else "reasoning_text_nonempty",
                    "check_type": "string_nonempty",
                    "actual_source": "assistant_text",
                },
                {"name": "meets_success_criteria", "check_type": "criterion", "criterion": success_criteria},
                {"name": "satisfies_done_condition", "check_type": "criterion", "criterion": done_condition},
            ]
            required_conditions = [item["name"] for item in verification_checks]
            optional_conditions = []
    if verification_type is None:
        verification_type = "composite" if kind in {"tool", "read", "write", "note"} else "llm_fallback"
    if verification_checks is None:
        verification_checks = []
    if required_conditions is None:
        required_conditions = [item["name"] for item in verification_checks]
    if optional_conditions is None:
        optional_conditions = []
    return {
        "step_id": step_id,
        "title": title,
        "goal": goal or title,
        "kind": kind,
        "expected_tool": expected_tool,
        "input_text": input_text or "Use the available context.",
        "expected_output": expected_output,
        "expected_outputs": [expected_output],
        "done_condition": done_condition,
        "success_criteria": success_criteria,
        "verification_type": verification_type,
        "verification_checks": verification_checks,
        "required_conditions": required_conditions,
        "optional_conditions": optional_conditions,
        "input_refs": [] if input_refs is None else input_refs,
        "output_refs": [] if output_refs is None else output_refs,
        "fallback_strategy": fallback_strategy,
        "depends_on": [] if depends_on is None else depends_on,
    }


def plan_response(
    *,
    goal: str,
    steps: list[dict[str, Any]],
    success_criteria: str = "Complete the task safely and correctly.",
    fallback_strategy: str = "Replan from the latest valid state.",
) -> str:
    return json.dumps(
        {
            "goal": goal,
            "success_criteria": success_criteria,
            "fallback_strategy": fallback_strategy,
            "steps": steps,
        }
    )
