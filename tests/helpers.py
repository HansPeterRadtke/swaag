from __future__ import annotations

import json
import re
from typing import Any

from swaag.failure import classify_failure_from_payload
from swaag.model import CompletionRequestPolicy
from swaag.strategy import strategy_from_payload
from swaag.types import CompletionResult, ContractSpec


class FakeModelClient:
    is_deterministic_test_client = True

    def __init__(self, responses: list[Any] | None = None, *, contract_responses: dict[str, list[Any]] | None = None):
        self._responses = list(responses or [])
        self._contract_responses = {key: list(value) for key, value in (contract_responses or {}).items()}
        self._pending_tool_inputs: dict[str, list[str]] = {}
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

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
    ) -> CompletionResult:
        self.requests.append(payload)
        contract_name = str(payload.get("contract", ""))
        response = None
        contract_queue = self._contract_responses.get(contract_name)
        if contract_queue:
            response = contract_queue.pop(0)
        elif contract_name.startswith("tool_input:"):
            tool_name = contract_name.split(":", 1)[1]
            pending = self._pending_tool_inputs.get(tool_name, [])
            if pending:
                response = pending.pop(0)
            elif self._contract_responses.get("tool_decision"):
                response = self._contract_responses["tool_decision"].pop(0)
            elif self._responses:
                response = self._responses.pop(0)
        elif contract_name in {
            "prompt_analysis",
            "task_decision",
            "task_expansion",
            "active_session_control",
            "verification",
            "plan_semantic_verification",
            "task_decision_semantic_verification",
            "task_decision_semantic_review",
            "strategy_selection",
            "failure_classification",
            "action_selection",
            "subagent_selection",
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
        if contract_name == "task_plan":
            response = _normalize_scripted_plan_response(response)
        if contract_name in {"verification", "plan_semantic_verification", "task_decision_semantic_verification"}:
            response = _normalize_scripted_verification_response(response, prompt=str(payload.get("prompt", "")))
        if contract_name == "yes_no":
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None
            if not isinstance(parsed, dict):
                response = json.dumps({"answer": response.strip()})
        if contract_name in {"answer_response", "clarification_response"}:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None
            if not isinstance(parsed, dict) or "text" not in parsed:
                response = json.dumps({"text": response})
        if contract_name == "tool_decision":
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None
            if (
                isinstance(parsed, dict)
                and parsed.get("action") == "call_tool"
                and isinstance(parsed.get("tool_name"), str)
                and isinstance(parsed.get("tool_input"), dict)
                and parsed["tool_input"]
            ):
                tool_name = str(parsed["tool_name"])
                self._pending_tool_inputs.setdefault(tool_name, []).append(json.dumps(parsed["tool_input"]))
                parsed["tool_input"] = {}
                response = json.dumps(parsed)
        if contract_name.startswith("tool_input:"):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("action") == "call_tool" and isinstance(parsed.get("tool_input"), dict):
                response = json.dumps(parsed["tool_input"])
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
        if contract_name == "prompt_analysis":
            return json.dumps(
                {
                    "task_type": "structured",
                    "completeness": "complete",
                    "requires_expansion": False,
                    "requires_decomposition": False,
                    "missing_required_information": False,
                    "confidence": 0.8,
                    "detected_entities": [],
                    "detected_goals": [],
                }
            )
        if contract_name == "task_decision":
            return json.dumps(
                {
                    "split_task": False,
                    "expand_task": False,
                    "ask_user": False,
                    "assume_missing": False,
                    "generate_ideas": False,
                    "direct_response": False,
                    "execution_mode": "full_plan",
                    "preferred_tool_name": "",
                    "evidence_required_before_response": False,
                    "evidence_call_count": 0,
                    "confidence": 0.8,
                    "reason": "test scaffold default full-plan decision",
                }
            )
        if contract_name == "task_decision_semantic_review":
            prompt = str(payload.get("prompt", ""))
            decision: dict[str, Any] = {}
            marker = "Candidate task decision:\n"
            if marker in prompt:
                candidate_text = prompt.split(marker, 1)[1]
                try:
                    decision = json.JSONDecoder().raw_decode(candidate_text.lstrip())[0]
                except (json.JSONDecodeError, TypeError, ValueError):
                    decision = {}
            count = int(decision.get("evidence_call_count", 0) or 0)
            sources = [f"declared_evidence_source_{index + 1}" for index in range(count)]
            return json.dumps(
                {
                    "decision_matches_request": True,
                    "decision_is_internally_consistent": True,
                    "required_evidence_sources": sources,
                    "minimum_evidence_call_count": count,
                    "selected_mode_and_tool_can_cover_declared_count": True,
                    "feedback": "default semantic decision review passed",
                }
            )
        if contract_name == "task_expansion":
            return json.dumps(
                {
                    "original_goal": "test fixture goal",
                    "expanded_goal": "test fixture goal",
                    "scope": ["model fixture scope"],
                    "constraints": ["model fixture constraint"],
                    "expected_outputs": ["model fixture output"],
                    "assumptions": [],
                }
            )
        if contract_name == "active_session_control":
            return json.dumps(
                {
                    "action": "continue_with_note",
                    "reason": "default control handling continues current work",
                    "response_text": "",
                    "added_context": "",
                    "replacement_goal": "",
                    "queued_task": "",
                    "clarification_question": "",
                }
            )
        if contract_name in {"verification", "plan_semantic_verification", "task_decision_semantic_verification"}:
            criteria = json.loads(_extract_section(prompt, "Criteria:"))
            candidate = _extract_section(prompt, "Candidate result:")
            excerpt_id = _allowed_candidate_excerpt_id(prompt, candidate)
            return json.dumps(
                {
                    "criteria": [
                        {
                            "name": criterion["name"] if isinstance(criterion, dict) else criterion,
                            "passed": bool(candidate.strip()),
                            "evidence": "The quoted candidate excerpt provides concrete result evidence.",
                            "candidate_excerpt_id_1": excerpt_id,
                            "candidate_excerpt_id_2": "",
                            "candidate_excerpt_id_3": "",
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
            return json.dumps({"action": "execute_step", "reason": "test scaffold neutral action choice"})
        if contract_name == "subagent_selection":
            return json.dumps(
                {
                    "spawn": False,
                    "subagent_type": "none",
                    "reason": "test scaffold neutral subagent choice",
                    "focus": "",
                }
            )
        raise AssertionError(f"Unsupported automatic frontend contract: {contract_name}")


_SECTION_RE = re.compile(r"^(?P<label>[A-Za-z ]+):\n(?P<body>.*?)(?:\n\n|\Z)", re.DOTALL | re.MULTILINE)


def _extract_section(prompt: str, label: str) -> str:
    for match in _SECTION_RE.finditer(prompt):
        if match.group("label").strip() == label.rstrip(":"):
            return match.group("body").strip()
    return ""


def _empty_objective_check() -> dict[str, Any]:
    return {
        "name": "",
        "check_type": "none",
        "path": "",
        "pattern": "",
        "command": [],
        "cwd": "",
    }


def _objective_check_from_scripted_check(check: dict[str, Any]) -> dict[str, Any]:
    objective = _empty_objective_check()
    check_type = str(check.get("check_type", ""))
    objective["name"] = str(check.get("name", ""))
    objective["check_type"] = check_type
    if check_type == "file_contains":
        objective["path"] = str(check.get("path", ""))
        pattern = str(check.get("pattern", ""))
        if not pattern:
            expected_json = str(check.get("expected_json", ""))
            if expected_json:
                try:
                    decoded = json.loads(expected_json)
                except json.JSONDecodeError:
                    decoded = ""
                if isinstance(decoded, str):
                    pattern = decoded
        if not pattern:
            pattern = str(check.get("expected", ""))
        objective["pattern"] = pattern
    elif check_type == "command_success":
        command = check.get("command", [])
        objective["command"] = command if isinstance(command, list) else []
        objective["cwd"] = str(check.get("cwd", ""))
    return objective


def _normalize_scripted_plan_response(response: str) -> str:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        return response
    changed = False
    objective_types = {"tool_effect_verified", "file_contains", "command_success"}
    for step in payload["steps"]:
        if not isinstance(step, dict) or not isinstance(step.get("verification_checks"), list):
            continue
        if "done_condition" in step:
            step.pop("done_condition", None)
            changed = True
        had_legacy_lists = "required_conditions" in step or "optional_conditions" in step
        required = {str(item) for item in step.pop("required_conditions", [])}
        optional = {str(item) for item in step.pop("optional_conditions", [])}
        changed = changed or had_legacy_lists
        for check in step["verification_checks"]:
            if not isinstance(check, dict) or "condition" in check:
                continue
            name = str(check.get("name", ""))
            check["condition"] = "required" if name in required else "optional"
            changed = True
        if "objective_verification_check" not in step:
            objective_index = next(
                (
                    index
                    for index, check in enumerate(step["verification_checks"])
                    if isinstance(check, dict)
                    and check.get("condition") == "required"
                    and str(check.get("check_type", "")) in objective_types
                ),
                None,
            )
            if objective_index is None:
                step["objective_verification_check"] = _empty_objective_check()
            else:
                objective = dict(step["verification_checks"].pop(objective_index))
                step["objective_verification_check"] = _objective_check_from_scripted_check(objective)
            changed = True
    return json.dumps(payload) if changed else response


def _candidate_excerpt_catalog(prompt: str) -> dict[str, str]:
    label = "Candidate excerpt ID catalog:"
    raw_catalog = _extract_section(prompt, label)
    if not raw_catalog:
        return {}
    try:
        catalog = json.loads(raw_catalog)
    except json.JSONDecodeError:
        return {}
    if not isinstance(catalog, dict):
        return {}
    return {
        str(key): value
        for key, value in catalog.items()
        if isinstance(key, str) and isinstance(value, str) and key and value
    }


def _allowed_candidate_excerpt_id(prompt: str, candidate: str) -> str:
    for excerpt_id, excerpt in _candidate_excerpt_catalog(prompt).items():
        if excerpt in candidate:
            return excerpt_id
    return ""


def _normalize_scripted_verification_response(response: str, *, prompt: str) -> str:
    """Convert normalized scripted excerpts into the bounded production wire IDs."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(payload, dict) or not isinstance(payload.get("criteria"), list):
        return response
    candidate = _extract_section(prompt, "Candidate result:").strip()
    catalog = _candidate_excerpt_catalog(prompt)
    default_id = next((key for key, excerpt in catalog.items() if excerpt in candidate), "")
    reverse_catalog = {excerpt: key for key, excerpt in catalog.items()}
    changed = False
    for item in payload["criteria"]:
        if not isinstance(item, dict):
            continue
        if all(field in item for field in ("candidate_excerpt_id_1", "candidate_excerpt_id_2", "candidate_excerpt_id_3")):
            continue
        raw_excerpts = item.pop("candidate_excerpts", None)
        if raw_excerpts is None:
            selected_ids = [default_id] if default_id else []
        elif isinstance(raw_excerpts, list):
            selected_ids = []
            for excerpt in raw_excerpts:
                if not isinstance(excerpt, str):
                    selected_ids.append("INVALID_EXCERPT_ID")
                    continue
                excerpt_id = reverse_catalog.get(excerpt)
                if excerpt_id is None and excerpt in candidate:
                    excerpt_id = next(
                        (
                            key
                            for key, catalog_excerpt in catalog.items()
                            if excerpt in catalog_excerpt or catalog_excerpt in excerpt
                        ),
                        None,
                    )
                selected_ids.append(excerpt_id or "INVALID_EXCERPT_ID")
        else:
            selected_ids = ["INVALID_EXCERPT_ID"]
        selected_ids = list(dict.fromkeys(selected_ids))[:3]
        item["candidate_excerpt_id_1"] = selected_ids[0] if len(selected_ids) > 0 else ""
        item["candidate_excerpt_id_2"] = selected_ids[1] if len(selected_ids) > 1 else ""
        item["candidate_excerpt_id_3"] = selected_ids[2] if len(selected_ids) > 2 else ""
        changed = True
    return json.dumps(payload) if changed else response


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
    fallback_strategy: str = "Use the model-declared fallback for this step.",
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
            verification_type = "composite"
            verification_checks = [
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {
                    "name": "assistant_text_nonempty" if kind == "respond" else "reasoning_text_nonempty",
                    "check_type": "string_nonempty",
                    "actual_source": "assistant_text",
                },
                {
                    "name": "meets_success_criteria",
                    "check_type": "criterion",
                    "actual_source": "assistant_text",
                    "criterion": success_criteria,
                },
                {
                    "name": "satisfies_done_condition",
                    "check_type": "criterion",
                    "actual_source": "assistant_text",
                    "criterion": done_condition,
                },
            ]
            required_conditions = [item["name"] for item in verification_checks]
            optional_conditions = []
    if verification_type is None:
        verification_type = "composite"
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
