from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from swaag.expander import expand_task
from swaag.failure import classify_failure_from_payload
from swaag.model import CompletionRequestPolicy
from swaag.strategy import strategy_from_payload
from swaag.types import CompletionResult, ContractSpec, DecisionOutcome, PromptAnalysis
from swaag.utils import stable_json_dumps

BenchmarkTaskType = Literal["coding", "file_edit", "reading", "multi_step", "failure", "quality"]
BenchmarkDifficulty = Literal["easy", "medium", "hard"]
ExpectedOutcome = Literal["success", "expected_failure"]


@dataclass(slots=True)
class BenchmarkVerificationContract:
    task_type: BenchmarkTaskType
    expected_answer: str | None = None
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_answer_regex: str | None = None
    expected_json: dict[str, Any] | None = None
    expected_json_schema: dict[str, Any] | None = None
    expected_files: dict[str, str] = field(default_factory=dict)
    expected_file_patterns: dict[str, list[str]] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    command_cwd: str | None = None
    command_framework: str | None = None
    required_history_events: list[str] = field(default_factory=list)
    forbidden_history_events: list[str] = field(default_factory=list)
    required_event_counts: dict[str, int] = field(default_factory=dict)
    required_tools_used: list[str] = field(default_factory=list)
    forbidden_tools_used: list[str] = field(default_factory=list)
    min_tool_calls: int | None = None
    max_tool_calls: int | None = None
    expected_stop_reason: str | None = None
    allowed_modified_files: list[str] = field(default_factory=list)
    forbid_unexpected_workspace_changes: bool = False


@dataclass(slots=True)
class PromptUnderstandingOracle:
    task_type: str | None = None
    completeness: str | None = None
    requires_expansion: bool | None = None
    requires_decomposition: bool | None = None
    expand_task: bool | None = None
    split_task: bool | None = None
    ask_user: bool | None = None
    assume_missing: bool | None = None
    generate_ideas: bool | None = None
    strategy_profile: str | None = None
    detected_goals_contains: list[str] = field(default_factory=list)
    detected_entities_contains: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskScenario:
    prompt: str
    workspace: Path
    model_client: Any | None
    verification_contract: BenchmarkVerificationContract
    expected_outcome: ExpectedOutcome = "success"
    expected_failure_category: str | None = None
    oracle: PromptUnderstandingOracle | None = None

    def __post_init__(self) -> None:
        if self.oracle is not None and self.model_client is not None:
            attach = getattr(self.model_client, "attach_oracle", None)
            if callable(attach):
                attach(self.oracle)


@dataclass(slots=True)
class BenchmarkTaskDefinition:
    task_id: str
    task_type: BenchmarkTaskType
    description: str
    build: Callable[[Path], TaskScenario]
    build_live: Callable[[Path], TaskScenario] | None = None
    difficulty: BenchmarkDifficulty = "medium"
    tags: list[str] = field(default_factory=list)
    setup_instructions: list[str] = field(default_factory=list)
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def create(self, output_root: Path, *, live_mode: bool = False) -> TaskScenario:
        workspace = output_root / self.task_id
        os.makedirs(workspace, exist_ok=True)
        if live_mode and self.build_live is not None:
            return self.build_live(workspace)
        return self.build(workspace)


class ScriptedBenchmarkClient:
    is_scripted_benchmark_client = True

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        contract_responses: dict[str, list[Any]] | None = None,
        oracle: "PromptUnderstandingOracle | None" = None,
    ):
        self._responses = list(responses or [])
        self._contract_responses = {key: list(value) for key, value in (contract_responses or {}).items()}
        self._oracle = oracle
        self.requests: list[dict[str, Any]] = []
        self.tokenize_requests: list[str] = []

    def attach_oracle(self, oracle: "PromptUnderstandingOracle | None") -> None:
        self._oracle = oracle

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tokenize(self, text: str) -> int:
        self.tokenize_requests.append(text)
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
            profile_name="benchmark-scripted",
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
            raise AssertionError(f"No scripted benchmark responses left for contract={contract_name!r}")
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(payload=payload)
        if isinstance(response, CompletionResult):
            return response
        if not isinstance(response, str):
            raise TypeError(f"Unsupported scripted response: {response!r}")
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
            analysis = self._oracle_analysis(current_request)
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
            analysis = self._oracle_analysis(current_request)
            analysis.task_type = analysis_payload.get("task_type", analysis.task_type)
            analysis.completeness = analysis_payload.get("completeness", analysis.completeness)
            analysis.requires_expansion = bool(analysis_payload.get("requires_expansion", analysis.requires_expansion))
            analysis.requires_decomposition = bool(analysis_payload.get("requires_decomposition", analysis.requires_decomposition))
            analysis.confidence = float(analysis_payload.get("confidence", analysis.confidence))
            analysis.detected_entities = list(analysis_payload.get("detected_entities", analysis.detected_entities))
            analysis.detected_goals = list(analysis_payload.get("detected_goals", analysis.detected_goals))
            decision = self._oracle_decision(analysis)
            return json.dumps(
                {
                    "split_task": decision.split_task,
                    "expand_task": decision.expand_task,
                    "ask_user": decision.ask_user,
                    "assume_missing": decision.assume_missing,
                    "generate_ideas": decision.generate_ideas,
                    "direct_response": decision.direct_response,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                }
            )
        if contract_name == "task_expansion":
            analysis_payload = json.loads(_extract_section(prompt, "Prompt analysis:"))
            decision_payload = json.loads(_extract_section(prompt, "Task decision:"))
            analysis = self._oracle_analysis(current_request)
            analysis.task_type = analysis_payload.get("task_type", analysis.task_type)
            analysis.completeness = analysis_payload.get("completeness", analysis.completeness)
            analysis.requires_expansion = bool(analysis_payload.get("requires_expansion", analysis.requires_expansion))
            analysis.requires_decomposition = bool(analysis_payload.get("requires_decomposition", analysis.requires_decomposition))
            analysis.confidence = float(analysis_payload.get("confidence", analysis.confidence))
            analysis.detected_entities = list(analysis_payload.get("detected_entities", analysis.detected_entities))
            analysis.detected_goals = list(analysis_payload.get("detected_goals", analysis.detected_goals))
            decision = self._oracle_decision(analysis)
            decision.split_task = bool(decision_payload.get("split_task", decision.split_task))
            decision.expand_task = bool(decision_payload.get("expand_task", decision.expand_task))
            decision.ask_user = bool(decision_payload.get("ask_user", decision.ask_user))
            decision.assume_missing = bool(decision_payload.get("assume_missing", decision.assume_missing))
            decision.generate_ideas = bool(decision_payload.get("generate_ideas", decision.generate_ideas))
            decision.direct_response = bool(decision_payload.get("direct_response", decision.direct_response))
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
        if contract_name == "verification":
            criteria = json.loads(_extract_section(prompt, "Criteria:"))
            candidate = _extract_section(prompt, "Candidate result:")
            return json.dumps(
                {
                    "criteria": [
                        {
                            "name": criterion,
                            "passed": bool(candidate.strip()),
                            "evidence": "candidate result is non-empty",
                        }
                        for criterion in criteria
                    ]
                }
            )
        if contract_name == "strategy_selection":
            profile = "generic"
            if self._oracle is not None and self._oracle.strategy_profile:
                profile = self._oracle.strategy_profile
            payload_obj = {
                "task_profile": profile,
                "strategy_name": "conservative",
                "explore_before_commit": False,
                "tool_chain_depth": 1,
                "verification_intensity": 1.0,
                "reason": f"oracle profile={profile}" if self._oracle else "default strategy",
            }
            strategy_from_payload(payload_obj)
            return json.dumps(payload_obj)
        if contract_name == "failure_classification":
            payload_obj = {
                "kind": "reasoning_failure",
                "retryable": True,
                "requires_replan": False,
                "suggested_strategy_mode": "recovery",
                "wait_seconds": 0.0,
                "reason": "generic failure",
            }
            classify_failure_from_payload(payload_obj)
            return json.dumps(payload_obj)
        if contract_name == "action_selection":
            match = re.search(r"Default deterministic choice:\s*([a-z_]+)", prompt)
            action = match.group(1) if match is not None else "execute_step"
            return json.dumps({"action": action, "reason": "scripted action choice"})
        if contract_name == "subagent_selection":
            match = re.search(r"Available subagents:\s*([^\n]+)", prompt)
            available = []
            if match is not None:
                available = [item.strip().rstrip(".") for item in match.group(1).split(",") if item.strip()]
            chosen = next((item for item in available if item != "none"), "none")
            return json.dumps(
                {
                    "spawn": chosen != "none",
                    "subagent_type": chosen,
                    "reason": "scripted subagent choice",
                    "focus": "use scoped specialist evidence",
                }
            )
        if contract_name == "generation_decomposition":
            return json.dumps(
                {
                    "output_class": "open_ended",
                    "reason": "one benchmark answer unit is sufficient",
                    "units": [
                        {
                            "unit_id": "answer_unit_01",
                            "title": "Final answer",
                            "instruction": "Produce the final answer for the benchmark task.",
                        }
                    ],
                }
            )
        if contract_name == "overflow_recovery":
            return json.dumps(
                {
                    "keep_partial": True,
                    "reason": "keep the current partial benchmark output",
                    "next_units": [],
                }
            )
        raise AssertionError(f"Unsupported automatic benchmark contract: {contract_name}")

    def _oracle_analysis(self, current_request: str) -> "PromptAnalysis":
        oracle = self._oracle
        if oracle is None:
            return _default_prompt_analysis(current_request)
        detected_entities: list[str] = []
        for needle in oracle.detected_entities_contains:
            if needle and needle in current_request and needle not in detected_entities:
                detected_entities.append(needle)
        detected_goals: list[str] = []
        for needle in oracle.detected_goals_contains:
            if needle and needle not in detected_goals:
                detected_goals.append(needle)
        return PromptAnalysis(
            task_type=oracle.task_type or "unstructured",
            completeness=oracle.completeness or "partial",
            requires_expansion=bool(oracle.requires_expansion) if oracle.requires_expansion is not None else False,
            requires_decomposition=bool(oracle.requires_decomposition) if oracle.requires_decomposition is not None else False,
            confidence=0.85,
            detected_entities=detected_entities,
            detected_goals=detected_goals,
        )

    def _oracle_decision(self, analysis: "PromptAnalysis") -> "DecisionOutcome":
        oracle = self._oracle
        if oracle is None:
            return _decision_from_analysis(analysis)
        return DecisionOutcome(
            split_task=bool(oracle.split_task) if oracle.split_task is not None else analysis.requires_decomposition,
            expand_task=bool(oracle.expand_task) if oracle.expand_task is not None else analysis.requires_expansion,
            ask_user=bool(oracle.ask_user) if oracle.ask_user is not None else False,
            assume_missing=bool(oracle.assume_missing) if oracle.assume_missing is not None else False,
            generate_ideas=bool(oracle.generate_ideas) if oracle.generate_ideas is not None else False,
            confidence=0.85,
            reason=f"oracle task_type={analysis.task_type}",
            direct_response=False,
        )


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
            confidence=0.85,
            detected_entities=[],
            detected_goals=[],
        )
    return PromptAnalysis(
        task_type="structured",
        completeness="complete",
        requires_expansion=False,
        requires_decomposition=False,
        confidence=0.85,
        detected_entities=[],
        detected_goals=[],
    )


def _decision_from_analysis(analysis: PromptAnalysis) -> DecisionOutcome:
    return DecisionOutcome(
        split_task=analysis.requires_decomposition,
        expand_task=analysis.requires_expansion,
        ask_user=analysis.completeness == "incomplete" and analysis.task_type != "vague",
        assume_missing=analysis.task_type == "vague",
        generate_ideas=analysis.task_type in {"vague", "unstructured"},
        confidence=analysis.confidence,
        reason=f"oracle task_type={analysis.task_type}",
        direct_response=False,
    )


def _plan_step(
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


def _plan_response(*, goal: str, steps: list[dict[str, Any]], success_criteria: str = "Complete the task safely and correctly.", fallback_strategy: str = "Replan from the latest valid state.") -> str:
    return json.dumps(
        {
            "goal": goal,
            "success_criteria": success_criteria,
            "fallback_strategy": fallback_strategy,
            "steps": steps,
        }
    )


def _write(path: Path, content: str) -> str:
    os.makedirs(path.parent, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    return str(path)


def _tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    return json.dumps({"action": "call_tool", "response": "", "tool_name": tool_name, "tool_input": tool_input})


def _contract_scripted_client(
    *,
    plan: str | None = None,
    tool_calls: list[str] | None = None,
    answer: str | None = None,
    responses: list[Any] | None = None,
    contract_responses: dict[str, list[Any]] | None = None,
) -> ScriptedBenchmarkClient:
    scoped_contract_responses = {
        name: list(items) for name, items in (contract_responses or {}).items()
    }
    if plan is not None:
        scoped_contract_responses.setdefault("task_plan", []).append(plan)
    if tool_calls:
        scoped_contract_responses.setdefault("tool_decision", []).extend(tool_calls)
    if answer is not None:
        scoped_contract_responses.setdefault("plain_text", []).append(answer)
    return ScriptedBenchmarkClient(
        responses=list(responses or []),
        contract_responses=scoped_contract_responses,
    )


def _coding_function_task(workspace: Path) -> TaskScenario:
    source = _write(
        workspace / "math_utils.py",
        "def multiply(a: int, b: int) -> int:\n    return 0\n",
    )
    _write(
        workspace / "test_math_utils.py",
        "import unittest\n\nfrom math_utils import multiply\n\n\nclass MultiplyTests(unittest.TestCase):\n    def test_multiply(self) -> None:\n        self.assertEqual(multiply(6, 7), 42)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    prompt = f"Read {source}, implement multiply correctly, verify with tests, and reply implemented."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_impl", "Read implementation", "read", expected_tool="read_text", expected_output="Implementation source", success_criteria="Implementation source is loaded."),
            _plan_step(
                "step_write_impl",
                "Implement multiply",
                "write",
                expected_tool="edit_text",
                expected_output="Updated multiply implementation",
                success_criteria="The implementation returns the product and the tests pass.",
                depends_on=["step_read_impl"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_product", "check_type": "file_contains", "path": source, "pattern": "return a * b"},
                    {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_math_utils.py"], "cwd": str(workspace), "framework": "unittest"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_product", "tests_pass"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="implemented", success_criteria="The assistant replies implemented.", depends_on=["step_write_impl"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": source}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": source, "operation": "replace_pattern_once", "pattern": "return 0", "replacement": "return a * b"}}),
            "implemented",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_answer="implemented",
            expected_files={source: "def multiply(a: int, b: int) -> int:\n    return a * b\n"},
            command=["python3", "-m", "unittest", "-q", "test_math_utils.py"],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_history_events=["tool_called", "tool_result", "verification_passed"],
        ),
    )


def _coding_multifile_task(workspace: Path) -> TaskScenario:
    core = _write(workspace / "core.py", "def base_value() -> int:\n    return 20\n")
    service = _write(
        workspace / "service.py",
        "from core import base_value\n\n\ndef total() -> int:\n    return base_value() + 1\n\n\ndef describe() -> str:\n    return f'total={total() - 2}'\n",
    )
    _write(
        workspace / "test_service.py",
        "import unittest\n\nfrom service import describe, total\n\n\nclass ServiceTests(unittest.TestCase):\n    def test_total(self) -> None:\n        self.assertEqual(total(), 42)\n\n    def test_describe(self) -> None:\n        self.assertEqual(describe(), 'total=42')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    prompt = f"Read {core} and {service}, fix the bug across both files, verify with tests, and reply fixed."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_core", "Read core", "read", expected_tool="read_text", expected_output="core source", success_criteria="Core source is loaded."),
            _plan_step("step_read_service", "Read service", "read", expected_tool="read_text", expected_output="service source", success_criteria="Service source is loaded.", depends_on=["step_read_core"]),
            _plan_step("step_write_core", "Fix base value", "write", expected_tool="edit_text", expected_output="Updated core.py", success_criteria="core.py returns 41.", depends_on=["step_read_service"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_41", "check_type": "file_contains", "path": core, "pattern": "return 41"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_41"], optional_conditions=[]),
            _plan_step("step_write_service", "Fix describe", "write", expected_tool="edit_text", expected_output="Updated service.py", success_criteria="describe and total are both correct.", depends_on=["step_write_core"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_total", "check_type": "file_contains", "path": service, "pattern": "return f'total={total()}'"},
                {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_service.py"], "cwd": str(workspace), "framework": "unittest"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_total", "tests_pass"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="fixed", success_criteria="The assistant replies fixed.", depends_on=["step_write_service"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": core}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": service}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": core, "operation": "replace_pattern_once", "pattern": "return 20", "replacement": "return 41"}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": service, "operation": "replace_pattern_once", "pattern": "return f'total={total() - 2}'", "replacement": "return f'total={total()}'"}}),
            "fixed",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_answer="fixed",
            expected_files={
                core: "def base_value() -> int:\n    return 41\n",
                service: "from core import base_value\n\n\ndef total() -> int:\n    return base_value() + 1\n\n\ndef describe() -> str:\n    return f'total={total()}'\n",
            },
            command=["python3", "-m", "unittest", "-q", "test_service.py"],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_history_events=["tool_called", "tool_result", "verification_passed"],
        ),
    )


def _file_edit_task(workspace: Path) -> TaskScenario:
    document = _write(workspace / "document.txt", "alpha\nbeta\n")
    prompt = f"Read {document}, replace beta with gamma, and reply file updated."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_document", "Read document", "read", expected_tool="read_text", expected_output="Document text", success_criteria="The document is read."),
            _plan_step("step_write_document", "Update document", "write", expected_tool="edit_text", expected_output="Updated document", success_criteria="The document contains gamma instead of beta.", depends_on=["step_read_document"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_gamma", "check_type": "file_contains", "path": document, "pattern": "gamma"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_gamma"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="file updated", success_criteria="The assistant replies file updated.", depends_on=["step_write_document"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": document}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": document, "operation": "replace_pattern_once", "pattern": "beta", "replacement": "gamma"}}),
            "file updated",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="file_edit",
            expected_answer="file updated",
            expected_files={document: "alpha\ngamma\n"},
            required_history_events=["file_read_requested", "edit_applied", "file_write_applied", "verification_passed"],
        ),
    )


def _reading_task(workspace: Path) -> TaskScenario:
    config_path = _write(workspace / "config.json", '{"api_url": "http://localhost", "timeout": 30}\n')
    notes_path = _write(workspace / "mode.txt", "mode=debug\n")
    expected_object = {"api_url": "http://localhost", "mode": "debug", "timeout": 30}
    expected_answer = stable_json_dumps(expected_object)
    prompt = f"Read {config_path} and {notes_path}. Return exact JSON with the API url, timeout, and mode."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_config", "Read config", "read", expected_tool="read_text", expected_output="Config JSON", success_criteria="The config file is read."),
            _plan_step("step_read_notes", "Read notes", "read", expected_tool="read_text", expected_output="Notes text", success_criteria="The notes file is read.", depends_on=["step_read_config"]),
            _plan_step("step_answer", "Return extracted JSON", "respond", expected_output=expected_answer, success_criteria="The assistant returns the requested JSON exactly.", depends_on=["step_read_notes"]),
        ],
    )
    client = _contract_scripted_client(
        plan=plan,
        tool_calls=[
            _tool_call("read_text", {"path": config_path}),
            _tool_call("read_text", {"path": notes_path}),
        ],
        answer=expected_answer,
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="reading",
            expected_answer=expected_answer,
            expected_json=expected_object,
            expected_json_schema={
                "type": "object",
                "properties": {
                    "api_url": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "mode": {"type": "string"},
                },
                "required": ["api_url", "timeout", "mode"],
                "additionalProperties": False,
            },
            required_history_events=["file_read_requested", "reader_chunk_read", "verification_passed"],
        ),
    )


def _multi_step_task(workspace: Path) -> TaskScenario:
    version = _write(workspace / "version.txt", "1.2.3\n")
    app = _write(workspace / "app.py", 'VERSION = "0.0.0"\n')
    _write(
        workspace / "test_app.py",
        "import unittest\n\nfrom app import VERSION\n\n\nclass AppTests(unittest.TestCase):\n    def test_version(self) -> None:\n        self.assertEqual(VERSION, '1.2.3')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    prompt = f"Read {version}, update {app} so VERSION matches it, verify the tests pass, and reply synced."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_version", "Read version file", "read", expected_tool="read_text", expected_output="Version text", success_criteria="The version file is read."),
            _plan_step("step_read_app", "Read app file", "read", expected_tool="read_text", expected_output="App source", success_criteria="The app file is read.", depends_on=["step_read_version"]),
            _plan_step("step_write_app", "Update app version", "write", expected_tool="edit_text", expected_output="Updated app.py", success_criteria="The app version is synchronized and the tests pass.", depends_on=["step_read_app"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_version", "check_type": "file_contains", "path": app, "pattern": 'VERSION = "1.2.3"'},
                {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_app.py"], "cwd": str(workspace), "framework": "unittest"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_version", "tests_pass"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="synced", success_criteria="The assistant replies synced.", depends_on=["step_write_app"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": version}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": app}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": app, "operation": "replace_pattern_once", "pattern": 'VERSION = "0.0.0"', "replacement": 'VERSION = "1.2.3"'}}),
            "synced",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer="synced",
            expected_files={app: 'VERSION = "1.2.3"\n'},
            command=["python3", "-m", "unittest", "-q", "test_app.py"],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_history_events=["file_read_requested", "edit_applied", "file_write_applied", "verification_passed"],
        ),
    )


def _failure_wrong_tool_task(workspace: Path) -> TaskScenario:
    info = _write(workspace / "info.txt", "payload=42\n")
    prompt = f"Read {info} and reply with the payload value."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_info", "Read info file", "read", expected_tool="read_text", expected_output="Info text", success_criteria="The info file is read."),
            _plan_step("step_answer", "Answer user", "respond", expected_output="42", success_criteria="The assistant replies with 42.", depends_on=["step_read_info"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "40 + 2"}}),
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="failure",
            required_history_events=["tool_graph_rejected", "verification_failed"],
        ),
        expected_outcome="expected_failure",
        expected_failure_category="wrong_tool_usage",
    )


def _failure_bad_planning_task(workspace: Path) -> TaskScenario:
    prompt = "Refactor the project and verify it thoroughly."
    invalid_plan = json.dumps({
        "goal": prompt,
        "success_criteria": "Do the task.",
        "fallback_strategy": "Try again.",
        "steps": [{"step_id": "broken"}],
    })
    client = ScriptedBenchmarkClient(responses=[invalid_plan])
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(task_type="failure", required_history_events=["error"]),
        expected_outcome="expected_failure",
        expected_failure_category="bad_planning",
    )


def _coding_refactor_task(workspace: Path) -> TaskScenario:
    source = _write(
        workspace / "normalize.py",
        "def normalize(text: str) -> str:\n    cleaned = text.strip()\n    lowered = cleaned.lower()\n    return lowered\n",
    )
    _write(
        workspace / "test_normalize.py",
        "import unittest\n\nfrom normalize import normalize\n\n\nclass NormalizeTests(unittest.TestCase):\n    def test_normalize(self) -> None:\n        self.assertEqual(normalize('  HeLLo  '), 'hello')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    prompt = f"Read {source}, refactor normalize to a shorter implementation without changing behavior, verify the tests, and reply refactored."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_normalize", "Read normalize implementation", "read", expected_tool="read_text", expected_output="normalize source", success_criteria="normalize.py is loaded."),
            _plan_step(
                "step_refactor_normalize",
                "Refactor normalize",
                "write",
                expected_tool="edit_text",
                expected_output="Shorter normalize implementation",
                success_criteria="normalize uses one return statement and the tests pass.",
                depends_on=["step_read_normalize"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_refactor", "check_type": "file_contains", "path": source, "pattern": "return text.strip().lower()"},
                    {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_normalize.py"], "cwd": str(workspace), "framework": "unittest"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_refactor", "tests_pass"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="refactored", success_criteria="The assistant replies refactored.", depends_on=["step_refactor_normalize"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("read_text", {"path": source}),
            _tool_call(
                "edit_text",
                {
                    "path": source,
                    "operation": "replace_pattern_once",
                    "pattern": "def normalize(text: str) -> str:\n    cleaned = text.strip()\n    lowered = cleaned.lower()\n    return lowered\n",
                    "replacement": "def normalize(text: str) -> str:\n    return text.strip().lower()\n",
                },
            ),
            "refactored",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_answer="refactored",
            expected_files={source: "def normalize(text: str) -> str:\n    return text.strip().lower()\n"},
            command=["python3", "-m", "unittest", "-q", "test_normalize.py"],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_history_events=["tool_called", "tool_result", "verification_passed"],
            required_tools_used=["read_text", "edit_text"],
            forbidden_tools_used=["calculator"],
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="coding", detected_goals_contains=["read"]),
    )


def _coding_optional_calculator_task(workspace: Path) -> TaskScenario:
    source = _write(workspace / "area.py", "def rectangle_area(width: int, height: int) -> int:\n    return 0\n")
    _write(
        workspace / "test_area.py",
        "import unittest\n\nfrom area import rectangle_area\n\n\nclass AreaTests(unittest.TestCase):\n    def test_area(self) -> None:\n        self.assertEqual(rectangle_area(4, 7), 28)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    prompt = f"Read {source}, use the calculator if helpful, implement rectangle_area correctly, verify the tests, and reply implemented."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_area", "Read area implementation", "read", expected_tool="read_text", expected_output="area source", success_criteria="area.py is loaded."),
            _plan_step(
                "step_write_area",
                "Implement rectangle_area",
                "write",
                expected_tool="edit_text",
                expected_output="Updated area implementation",
                success_criteria="rectangle_area returns width * height and tests pass.",
                depends_on=["step_read_area"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_formula", "check_type": "file_contains", "path": source, "pattern": "return width * height"},
                    {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_area.py"], "cwd": str(workspace), "framework": "unittest"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_formula", "tests_pass"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="implemented", success_criteria="The assistant replies implemented.", depends_on=["step_write_area"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("read_text", {"path": source}),
            _tool_call("edit_text", {"path": source, "operation": "replace_pattern_once", "pattern": "return 0", "replacement": "return width * height"}),
            "implemented",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_answer="implemented",
            expected_files={source: "def rectangle_area(width: int, height: int) -> int:\n    return width * height\n"},
            command=["python3", "-m", "unittest", "-q", "test_area.py"],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_history_events=["tool_called", "tool_result", "verification_passed"],
            required_tools_used=["read_text", "edit_text"],
            forbidden_tools_used=["calculator"],
            min_tool_calls=2,
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="coding"),
    )


def _coding_no_unnecessary_tool_task(workspace: Path) -> TaskScenario:
    source = _write(workspace / "negate.py", "def negate(value: int) -> int:\n    return value\n")
    _write(
        workspace / "test_negate.py",
        "import unittest\n\nfrom negate import negate\n\n\nclass NegateTests(unittest.TestCase):\n    def test_negate(self) -> None:\n        self.assertEqual(negate(7), -7)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    prompt = f"Fix negate in {source}, verify the tests, and reply fixed without using the calculator."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_negate", "Read negate", "read", expected_tool="read_text", expected_output="negate source", success_criteria="negate.py is loaded."),
            _plan_step(
                "step_write_negate",
                "Fix negate",
                "write",
                expected_tool="edit_text",
                expected_output="Updated negate implementation",
                success_criteria="negate returns -value and tests pass.",
                depends_on=["step_read_negate"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "file_contains_fix", "check_type": "file_contains", "path": source, "pattern": "return -value"},
                    {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_negate.py"], "cwd": str(workspace), "framework": "unittest"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_fix", "tests_pass"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="fixed", success_criteria="The assistant replies fixed.", depends_on=["step_write_negate"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[plan, _tool_call("read_text", {"path": source}), _tool_call("edit_text", {"path": source, "operation": "replace_pattern_once", "pattern": "return value", "replacement": "return -value"}), "fixed"]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_answer="fixed",
            expected_files={source: "def negate(value: int) -> int:\n    return -value\n"},
            command=["python3", "-m", "unittest", "-q", "test_negate.py"],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_tools_used=["read_text", "edit_text"],
            forbidden_tools_used=["calculator"],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _file_edit_multilocation_task(workspace: Path) -> TaskScenario:
    document = _write(workspace / "settings.txt", "ENV=dev\nmode=dev\n")
    prompt = f"Edit {document} so every dev value becomes prod, then reply updated."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_write_settings", "Update all dev values", "write", expected_tool="edit_text", expected_output="Updated settings", success_criteria="Both dev values become prod.", verification_checks=[
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_prod", "check_type": "file_contains", "path": document, "pattern": "prod"},
            ], required_conditions=["tool_result_present", "tool_name_matches", "file_contains_prod"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="updated", success_criteria="The assistant replies updated.", depends_on=["step_write_settings"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[plan, _tool_call("edit_text", {"path": document, "operation": "replace_pattern_all", "pattern": "dev", "replacement": "prod"}), "updated"]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="file_edit",
            expected_answer="updated",
            expected_files={document: "ENV=prod\nmode=prod\n"},
            required_tools_used=["edit_text"],
            max_tool_calls=1,
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _file_edit_noop_task(workspace: Path) -> TaskScenario:
    document = _write(workspace / "document.txt", "alpha\ngamma\n")
    prompt = f"Check {document} and reply already correct if no edit is required."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step(
                "step_check_document",
                "Check whether an edit is needed",
                "write",
                expected_tool="edit_text",
                expected_output="No-op edit preview",
                success_criteria="The file stays unchanged when no edit is needed.",
                verification_checks=[
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    {"name": "tool_output_nonempty", "check_type": "tool_output_nonempty"},
                ],
                required_conditions=["tool_result_present", "tool_name_matches", "tool_output_nonempty"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="already correct", success_criteria="The assistant replies already correct.", depends_on=["step_check_document"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call(
                "edit_text",
                {
                    "path": document,
                    "operation": "replace_pattern_once",
                    "pattern": "gamma",
                    "replacement": "gamma",
                    "dry_run": True,
                },
            ),
            "already correct",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="file_edit",
            expected_answer="already correct",
            expected_files={document: "alpha\ngamma\n"},
            required_history_events=["file_read_requested", "edit_previewed", "verification_passed"],
            forbidden_history_events=["edit_applied", "file_write_applied"],
            required_tools_used=["edit_text"],
            max_tool_calls=1,
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _file_edit_reread_task(workspace: Path) -> TaskScenario:
    version = _write(workspace / "version.txt", "release=2\n")
    summary = _write(workspace / "summary.txt", "release=1\n")
    prompt = f"Read {version}, update {summary} to match it, reread {summary} to confirm, and reply verified."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_version", "Read version", "read", expected_tool="read_text", expected_output="Version text", success_criteria="The version file is read."),
            _plan_step("step_write_summary", "Update summary", "write", expected_tool="edit_text", expected_output="Updated summary", success_criteria="summary.txt matches version.txt.", depends_on=["step_read_version"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_release", "check_type": "file_contains", "path": summary, "pattern": "release=2"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_release"], optional_conditions=[]),
            _plan_step("step_reread_summary", "Reread summary", "read", expected_tool="read_text", expected_output="Updated summary text", success_criteria="The updated summary is reread.", depends_on=["step_write_summary"]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="verified", success_criteria="The assistant replies verified.", depends_on=["step_reread_summary"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("read_text", {"path": version}),
            _tool_call("edit_text", {"path": summary, "operation": "replace_pattern_once", "pattern": "release=1", "replacement": "release=2"}),
            _tool_call("read_text", {"path": summary}),
            "verified",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="file_edit",
            expected_answer="verified",
            expected_files={summary: "release=2\n"},
            required_event_counts={"file_read_requested": 2},
            required_tools_used=["read_text", "edit_text"],
            min_tool_calls=3,
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _reading_debug_log_task(workspace: Path) -> TaskScenario:
    debug_log = _write(workspace / "debug.log", "ERROR missing config\nWARN retrying\n")
    expected = {"error_count": 1, "first_error": "missing config"}
    answer = stable_json_dumps(expected)
    prompt = f"Read {debug_log} and return exact JSON with error_count and first_error."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_debug_log", "Read debug log", "read", expected_tool="read_text", expected_output="debug.log contents", success_criteria="The debug log is loaded."),
            _plan_step("step_answer", "Return summary", "respond", expected_output=answer, success_criteria="The assistant returns the requested JSON exactly.", depends_on=["step_read_debug_log"]),
        ],
    )
    client = _contract_scripted_client(
        plan=plan,
        tool_calls=[_tool_call("read_text", {"path": debug_log})],
        answer=answer,
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="reading",
            expected_answer=answer,
            expected_json=expected,
            expected_json_schema={
                "type": "object",
                "properties": {"error_count": {"type": "integer"}, "first_error": {"type": "string"}},
                "required": ["error_count", "first_error"],
                "additionalProperties": False,
            },
            required_tools_used=["read_text"],
            max_tool_calls=1,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="reading", detected_entities_contains=["debug.log"]),
    )


def _reading_contradiction_task(workspace: Path) -> TaskScenario:
    left = _write(workspace / "left.txt", "version=1\n")
    right = _write(workspace / "right.txt", "version=2\n")
    expected = {"contradiction": True, "left_version": 1, "right_version": 2}
    answer = stable_json_dumps(expected)
    prompt = f"Read {left} and {right}. Return exact JSON describing whether the versions contradict each other."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_left", "Read left", "read", expected_tool="read_text", expected_output="left facts", success_criteria="The left file is read."),
            _plan_step("step_read_right", "Read right", "read", expected_tool="read_text", expected_output="right facts", success_criteria="The right file is read.", depends_on=["step_read_left"]),
            _plan_step("step_answer", "Return contradiction JSON", "respond", expected_output=answer, success_criteria="The assistant returns the requested JSON exactly.", depends_on=["step_read_right"]),
        ],
    )
    client = _contract_scripted_client(
        plan=plan,
        tool_calls=[
            _tool_call("read_text", {"path": left}),
            _tool_call("read_text", {"path": right}),
        ],
        answer=answer,
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="reading",
            expected_answer=answer,
            expected_json=expected,
            expected_json_schema={
                "type": "object",
                "properties": {
                    "contradiction": {"type": "boolean"},
                    "left_version": {"type": "integer"},
                    "right_version": {"type": "integer"},
                },
                "required": ["contradiction", "left_version", "right_version"],
                "additionalProperties": False,
            },
            required_tools_used=["read_text"],
            min_tool_calls=2,
            max_tool_calls=2,
        ),
    )


def _reading_hallucination_guard_task(workspace: Path) -> TaskScenario:
    profile = _write(workspace / "profile.txt", "name=alice\n")
    expected = {"name": "alice", "city": "", "unsupported": ["city"]}
    answer = stable_json_dumps(expected)
    prompt = f"Read {profile} and return exact JSON with the name and city. If the city is missing, set city to an empty string and list it under unsupported."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_profile", "Read profile", "read", expected_tool="read_text", expected_output="Profile facts", success_criteria="The profile file is read."),
            _plan_step("step_answer", "Return guarded JSON", "respond", expected_output=answer, success_criteria="The assistant returns the exact JSON without hallucinating extra facts.", depends_on=["step_read_profile"]),
        ],
    )
    client = _contract_scripted_client(
        plan=plan,
        tool_calls=[_tool_call("read_text", {"path": profile})],
        answer=answer,
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="reading",
            expected_answer=answer,
            expected_json=expected,
            expected_json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                    "unsupported": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "city", "unsupported"],
                "additionalProperties": False,
            },
            required_tools_used=["read_text"],
            max_tool_calls=1,
        ),
    )


def _multi_step_compute_write_task(workspace: Path) -> TaskScenario:
    numbers = _write(workspace / "numbers.txt", "6 7\n")
    result_file = _write(workspace / "result.txt", "0\n")
    prompt = f"Read {numbers}, compute the product, write it into {result_file}, and reply written."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_numbers", "Read numbers", "read", expected_tool="read_text", expected_output="numbers", success_criteria="The numbers file is read."),
            _plan_step("step_calculate_product", "Compute product", "tool", expected_tool="calculator", expected_output="42", success_criteria="The product is computed.", depends_on=["step_read_numbers"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                {"name": "exact_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 42},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "exact_result"], optional_conditions=[]),
            _plan_step("step_write_result", "Write result", "write", expected_tool="edit_text", expected_output="Updated result.txt", success_criteria="result.txt contains 42.", depends_on=["step_calculate_product"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_answer", "check_type": "file_contains", "path": result_file, "pattern": "42"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_answer"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="written", success_criteria="The assistant replies written.", depends_on=["step_write_result"]),
        ],
    )
    client = _contract_scripted_client(
        plan=plan,
        tool_calls=[
            _tool_call("read_text", {"path": numbers}),
            _tool_call("calculator", {"expression": "6 * 7"}),
            _tool_call(
                "edit_text",
                {
                    "path": result_file,
                    "operation": "replace_pattern_once",
                    "pattern": "0",
                    "replacement": "42",
                },
            ),
        ],
        answer="written",
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer="written",
            expected_files={result_file: "42\n"},
            required_tools_used=["read_text", "calculator", "edit_text"],
            min_tool_calls=3,
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="multi_step"),
    )


def _multi_step_mixed_task(workspace: Path) -> TaskScenario:
    brief = _write(workspace / "brief.txt", "title=Weekly report\nscore_a=6\nscore_b=7\n")
    report_path = _write(workspace / "report.md", "# TODO\n")
    prompt = f"Read {brief}, take a note with the title, compute the combined score, write it into {report_path}, and reply completed."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_brief", "Read brief", "read", expected_tool="read_text", expected_output="brief text", success_criteria="The brief is read."),
            _plan_step("step_note_brief", "Store a working note", "note", expected_tool="notes", expected_output="Stored note", success_criteria="A durable note is stored.", depends_on=["step_read_brief"]),
            _plan_step("step_compute_total", "Compute total score", "tool", expected_tool="calculator", expected_output="42", success_criteria="The total score is computed.", depends_on=["step_note_brief"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                {"name": "exact_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 42},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "exact_result"], optional_conditions=[]),
            _plan_step("step_write_report", "Write report", "write", expected_tool="edit_text", expected_output="Updated report", success_criteria="The report contains the title and score.", depends_on=["step_compute_total"], verification_checks=[
                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_title", "check_type": "file_contains", "path": report_path, "pattern": "Weekly report"},
                {"name": "file_contains_score", "check_type": "file_contains", "path": report_path, "pattern": "42"},
            ], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_title", "file_contains_score"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="completed", success_criteria="The assistant replies completed.", depends_on=["step_write_report"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("read_text", {"path": brief}),
            _tool_call("notes", {"action": "add", "title": "weekly-report", "content": "Weekly report title captured"}),
            _tool_call("calculator", {"expression": "6 * 7"}),
            _tool_call("edit_text", {"path": report_path, "operation": "replace_range", "start": 0, "end": 7, "replacement": "# Weekly report\n\nTotal score: 42\n"}),
            "completed",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer="completed",
            expected_file_patterns={report_path: ["Weekly report", "42"]},
            required_history_events=["note_added", "verification_passed"],
            required_tools_used=["read_text", "notes", "calculator", "edit_text"],
            min_tool_calls=4,
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="multi_step"),
    )


def _environment_shell_persistence_task(workspace: Path) -> TaskScenario:
    shell_dir = workspace / "shell"
    shell_file = shell_dir / "data.txt"
    final_output = f"alpha|{shell_dir}"
    prompt = (
        "Use shell commands to create shell/data.txt containing alpha, keep GREETING=alpha and the shell cwd in shell/, "
        f"then print GREETING|PWD and reply with the exact string {final_output}."
    )
    command_one = "mkdir -p shell && cd shell && export GREETING=alpha && printf 'alpha\\n' > data.txt"
    command_two = "printf '%s|%s' \"$GREETING\" \"$PWD\""
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step(
                "step_prepare_shell",
                "Prepare the shell workspace",
                "tool",
                expected_tool="shell_command",
                expected_output="shell prepared",
                success_criteria="The shell session creates data.txt and preserves cwd plus GREETING.",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "shell_command"},
                    {"name": "shell_exit_zero", "check_type": "exact_match", "actual_source": "tool_output.exit_code", "expected": 0},
                    {"name": "file_exists", "check_type": "file_exists", "path": str(shell_file)},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "shell_exit_zero", "file_exists"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_check_shell_state",
                "Confirm the persistent shell state",
                "tool",
                expected_tool="shell_command",
                expected_output=final_output,
                success_criteria="The second command sees the persisted environment variable and cwd.",
                depends_on=["step_prepare_shell"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "shell_command"},
                    {"name": "shell_exit_zero", "check_type": "exact_match", "actual_source": "tool_output.exit_code", "expected": 0},
                    {"name": "stdout_matches", "check_type": "exact_match", "actual_source": "tool_output.stdout", "expected": final_output},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "shell_exit_zero", "stdout_matches"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output=final_output, success_criteria="The assistant returns the exact shell output.", depends_on=["step_check_shell_state"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("shell_command", {"command": command_one}),
            _tool_call("shell_command", {"command": command_two}),
            final_output,
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer=final_output,
            expected_files={str(shell_file): "alpha\n"},
            required_tools_used=["shell_command"],
            required_event_counts={"shell_command_completed": 2, "workspace_snapshot": 2, "process_completed": 2},
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="generic"),
    )


def _environment_list_read_write_task(workspace: Path) -> TaskScenario:
    template = _write(workspace / "template.txt", "status=ready\n")
    summary = workspace / "summary.txt"
    prompt = f"List the files, read {template}, write exactly the same content into {summary}, reread it, and reply saved."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_list_workspace", "List files", "tool", expected_tool="list_files", expected_output="workspace file list", success_criteria="The workspace files are listed."),
            _plan_step("step_read_template", "Read the template", "read", expected_tool="read_file", expected_output="template contents", success_criteria="The template file is read.", depends_on=["step_list_workspace"]),
            _plan_step(
                "step_write_summary",
                "Write the summary file",
                "write",
                expected_tool="write_file",
                expected_output="summary written",
                success_criteria="summary.txt contains the template content.",
                depends_on=["step_read_template"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                    {"name": "file_contains_status", "check_type": "file_contains", "path": str(summary), "pattern": "status=ready"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_status"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_confirm_summary",
                "Reread the summary",
                "read",
                expected_tool="read_file",
                expected_output="summary contents",
                success_criteria="The summary file is reread.",
                depends_on=["step_write_summary"],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="saved", success_criteria="The assistant replies saved.", depends_on=["step_confirm_summary"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("list_files", {"path": "."}),
            _tool_call("read_file", {"path": template}),
            _tool_call("write_file", {"path": str(summary), "content": "status=ready\n"}),
            _tool_call("read_file", {"path": str(summary)}),
            "saved",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer="saved",
            expected_files={str(summary): "status=ready\n"},
            required_tools_used=["list_files", "read_file", "write_file"],
            required_event_counts={"filesystem_listed": 1, "filesystem_read": 2},
            min_tool_calls=4,
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="multi_step"),
    )


def _coding_run_tests_environment_task(workspace: Path) -> TaskScenario:
    source = _write(
        workspace / "classify.py",
        "def classify(n: int) -> str:\n    if n >= 0:\n        return 'positive'\n    return 'negative'\n",
    )
    _write(
        workspace / "test_classify.py",
        "from classify import classify\n\n\ndef test_negative() -> None:\n    assert classify(-1) == 'negative'\n\n\ndef test_zero() -> None:\n    assert classify(0) == 'zero'\n\n\ndef test_positive() -> None:\n    assert classify(3) == 'positive'\n",
    )
    fixed = "def classify(n: int) -> str:\n    if n == 0:\n        return 'zero'\n    if n > 0:\n        return 'positive'\n    return 'negative'\n"
    prompt = f"Read {source}, fix the zero-case bug, run tests, and reply fixed."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_source", "Read classify.py", "read", expected_tool="read_file", expected_output="Source file", success_criteria="The source file is read."),
            _plan_step(
                "step_write_source",
                "Write the fixed classify.py",
                "write",
                expected_tool="write_file",
                expected_output="Fixed source file",
                success_criteria="classify.py handles zero correctly.",
                depends_on=["step_read_source"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                    {"name": "file_contains_zero_case", "check_type": "file_contains", "path": source, "pattern": "if n == 0"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_zero_case"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_run_tests",
                "Run the tests",
                "tool",
                expected_tool="run_tests",
                expected_output="Passing pytest result",
                success_criteria="The test command exits successfully.",
                depends_on=["step_write_source"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
                    {"name": "tests_pass", "check_type": "exact_match", "actual_source": "tool_output.passed", "expected": True},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="fixed", success_criteria="The assistant replies fixed.", depends_on=["step_run_tests"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("read_file", {"path": source}),
            _tool_call("write_file", {"path": source, "content": fixed}),
            _tool_call("run_tests", {"command": ["python", "-m", "pytest", "-q"]}),
            "fixed",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_answer="fixed",
            expected_files={source: fixed},
            command=["python", "-m", "pytest", "-q"],
            command_cwd=str(workspace),
            command_framework="pytest",
            required_tools_used=["read_file", "write_file", "run_tests"],
            required_event_counts={"process_completed": 1},
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="coding"),
    )


def _multi_step_iterative_write_task(workspace: Path) -> TaskScenario:
    target = _write(workspace / "report.txt", "draft\n")
    prompt = f"Write the final content into {target} and then reply saved."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step(
                "step_write_report",
                "Write the final report",
                "write",
                expected_tool="write_file",
                expected_output="Final report written",
                success_criteria="report.txt contains the exact final content.",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                    {"name": "file_contains_final_content", "check_type": "file_contains", "path": target, "pattern": "final content"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_final_content"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer user", "respond", expected_output="saved", success_criteria="The assistant replies saved.", depends_on=["step_write_report"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("write_file", {"path": target, "content": "wrong content\n"}),
            _tool_call("write_file", {"path": target, "content": "final content\n"}),
            "saved",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer="saved",
            expected_files={target: "final content\n"},
            required_tools_used=["write_file"],
            required_event_counts={"tool_called": 2, "tool_result": 2},
            min_tool_calls=2,
            forbid_unexpected_workspace_changes=True,
        ),
        oracle=PromptUnderstandingOracle(task_type="structured", completeness="complete", split_task=True, expand_task=False, strategy_profile="file_edit"),
    )


def _failure_repeated_action_task(workspace: Path) -> TaskScenario:
    document = _write(workspace / "loop.txt", "alpha\nbeta\n")
    prompt = f"Read {document}, then edit it so beta becomes gamma, and reply updated."
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_document", "Read the document", "read", expected_tool="read_text", expected_output="Document text", success_criteria="The document is read."),
            _plan_step("step_fix_document", "Update the document", "write", expected_tool="edit_text", expected_output="Updated document", success_criteria="The document is updated to gamma.", depends_on=["step_read_document"], verification_checks=[
                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                {"name": "file_contains_gamma", "check_type": "file_contains", "path": document, "pattern": "gamma"},
            ], required_conditions=["tool_result_present", "tool_name_matches", "file_contains_gamma"], optional_conditions=[]),
            _plan_step("step_answer", "Answer user", "respond", expected_output="updated", success_criteria="The assistant replies updated.", depends_on=["step_fix_document"]),
        ],
    )
    client = ScriptedBenchmarkClient(
        responses=[
            plan,
            _tool_call("read_text", {"path": document}),
            _tool_call("read_text", {"path": document}),
            _tool_call("read_text", {"path": document}),
            "not done",
        ]
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="failure",
            required_history_events=["duplicate_action_detected"],
            required_event_counts={"tool_called": 2},
        ),
        expected_outcome="expected_failure",
        expected_failure_category="loop_no_progress",
    )


def _quality_vague_expansion_task(workspace: Path) -> TaskScenario:
    prompt = "make a game"
    expanded_goal = "make a game Build a small arcade prototype with one core mechanic and a playable loop."
    plan = _plan_response(goal=expanded_goal, steps=[_plan_step("step_answer", "Answer the user", "respond", expected_output="scoped", success_criteria="The assistant replies scoped.")])
    client = _contract_scripted_client(plan=plan, answer="scoped")
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="quality",
            expected_answer="scoped",
            required_history_events=["task_expanded", "plan_created", "verification_passed"],
        ),
        oracle=PromptUnderstandingOracle(task_type="vague", completeness="incomplete", requires_expansion=True, requires_decomposition=False, expand_task=True, split_task=False, ask_user=False, assume_missing=True, generate_ideas=True, strategy_profile="generic", detected_goals_contains=["make"]),
    )


def _quality_already_decomposed_task(workspace: Path) -> TaskScenario:
    facts = _write(workspace / "facts.txt", "owner=alice\nstatus=ready\n")
    answer = stable_json_dumps({"owner": "alice", "status": "ready"})
    prompt = f"1. Read {facts}\n2. Return exact JSON summary"
    plan = _plan_response(
        goal=prompt,
        steps=[
            _plan_step("step_read_facts", "Read facts", "read", expected_tool="read_text", expected_output="Facts content", success_criteria="facts.txt is read."),
            _plan_step("step_answer", "Answer user", "respond", expected_output=answer, success_criteria="The assistant returns the expected JSON.", depends_on=["step_read_facts"]),
        ],
    )
    client = _contract_scripted_client(
        plan=plan,
        tool_calls=[_tool_call("read_text", {"path": facts})],
        answer=answer,
    )
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="quality",
            expected_answer=answer,
            expected_json={"owner": "alice", "status": "ready"},
            expected_json_schema={
                "type": "object",
                "properties": {"owner": {"type": "string"}, "status": {"type": "string"}},
                "required": ["owner", "status"],
                "additionalProperties": False,
            },
            required_history_events=["prompt_analyzed", "decision_made", "verification_passed"],
        ),
        oracle=PromptUnderstandingOracle(task_type="already_decomposed", completeness="complete", requires_expansion=False, requires_decomposition=True, expand_task=False, split_task=True, ask_user=False, strategy_profile="reading", detected_entities_contains=["facts.txt"]),
    )


def _quality_incomplete_clarification_task(workspace: Path) -> TaskScenario:
    prompt = "Can you help?"
    client = ScriptedBenchmarkClient(responses=[])
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=client,
        verification_contract=BenchmarkVerificationContract(
            task_type="quality",
            expected_answer_contains=["I need clarification before I can continue."],
            required_history_events=["prompt_analyzed", "decision_made", "reasoning_completed"],
            expected_stop_reason="prompt_incomplete",
        ),
        oracle=PromptUnderstandingOracle(task_type="incomplete", completeness="incomplete", requires_expansion=True, requires_decomposition=False, expand_task=True, split_task=False, ask_user=True, assume_missing=False, generate_ideas=False, strategy_profile="generic"),
    )


def validate_benchmark_catalog(tasks: list[BenchmarkTaskDefinition]) -> None:
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark task ids must be unique")

    counts = Counter(task.task_type for task in tasks)
    minimums = {
        "coding": 40,
        "file_edit": 25,
        "reading": 25,
        "multi_step": 30,
        "failure": 30,
        "quality": 20,
    }
    if len(tasks) < 170:
        raise ValueError(f"Benchmark catalog must contain at least 170 tasks, found {len(tasks)}")
    missing = {task_type: minimum for task_type, minimum in minimums.items() if counts.get(task_type, 0) < minimum}
    if missing:
        raise ValueError(f"Benchmark catalog does not meet category minimums: {missing}")
    realistic_multifile = [task for task in tasks if {"realistic-code", "multifile"}.issubset(set(task.tags))]
    if len(realistic_multifile) < 50:
        raise ValueError(f"Benchmark catalog must include at least 50 realistic multi-file tasks, found {len(realistic_multifile)}")

    for task in tasks:
        if not task.description.strip():
            raise ValueError(f"Task {task.task_id} must have a description")
        if not task.setup_instructions:
            raise ValueError(f"Task {task.task_id} must define setup instructions")
        with tempfile.TemporaryDirectory(prefix=f"benchmark-validate-{task.task_id}-") as temp_dir:
            scenario = task.create(Path(temp_dir))
        contract = scenario.verification_contract
        if contract.task_type != task.task_type:
            raise ValueError(f"Task {task.task_id} contract type {contract.task_type} does not match task type {task.task_type}")
        if task.task_type in {"coding", "file_edit", "reading", "multi_step", "quality"} and not (contract.expected_answer or contract.expected_answer_contains or contract.expected_files or contract.expected_file_patterns or contract.expected_json or contract.command or contract.required_history_events):
            raise ValueError(f"Task {task.task_id} must have a concrete verifier contract")
        if scenario.expected_outcome == "expected_failure" and not scenario.expected_failure_category:
            raise ValueError(f"Task {task.task_id} expected failure tasks must declare a failure category")


def get_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    tasks = [
        BenchmarkTaskDefinition(
            task_id="coding_implement_function",
            task_type="coding",
            description="Implement a missing function and verify it with unittest.",
            build=_coding_function_task,
            difficulty="easy",
            tags=["coding", "bugfix", "unit-test"],
            setup_instructions=["Create math_utils.py with a broken multiply function.", "Create test_math_utils.py with a failing unittest."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="coding_multifile_fix",
            task_type="coding",
            description="Fix a bug across multiple files and verify consistency with unittest.",
            build=_coding_multifile_task,
            difficulty="medium",
            tags=["coding", "multifile", "unit-test"],
            setup_instructions=["Create core.py and service.py with inconsistent behavior.", "Create test_service.py to verify both files stay consistent."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="coding_refactor_keep_tests_green",
            task_type="coding",
            description="Refactor existing code without breaking the tests.",
            build=_coding_refactor_task,
            difficulty="medium",
            tags=["coding", "refactor", "quality"],
            setup_instructions=["Create normalize.py with verbose implementation.", "Create test_normalize.py to preserve behavior."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="coding_optional_calculator",
            task_type="coding",
            description="Implement code correctly when a calculator step is useful but not sufficient.",
            build=_coding_optional_calculator_task,
            difficulty="medium",
            tags=["coding", "calculator", "verification"],
            setup_instructions=["Create area.py with a broken implementation.", "Create test_area.py expecting the correct area."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="coding_no_unnecessary_tool",
            task_type="coding",
            description="Fix code without using unnecessary tools.",
            build=_coding_no_unnecessary_tool_task,
            difficulty="easy",
            tags=["coding", "tool-discipline"],
            setup_instructions=["Create negate.py with a simple sign bug.", "Create test_negate.py for the expected behavior."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="coding_run_tests_environment",
            task_type="coding",
            description="Fix code, run tests through the environment layer, and verify the final project state.",
            build=_coding_run_tests_environment_task,
            difficulty="medium",
            tags=["coding", "environment", "run-tests"],
            setup_instructions=["Create classify.py with a zero-case bug.", "Create test_classify.py that requires zero to map to 'zero'."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="file_edit_exact",
            task_type="file_edit",
            description="Apply an exact text edit and verify exact file contents.",
            build=_file_edit_task,
            difficulty="easy",
            tags=["file-edit", "exact"],
            setup_instructions=["Create document.txt with one incorrect line to replace."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="file_edit_multi_location",
            task_type="file_edit",
            description="Apply a multi-location replacement and verify all occurrences.",
            build=_file_edit_multilocation_task,
            difficulty="easy",
            tags=["file-edit", "replace-all"],
            setup_instructions=["Create settings.txt with repeated values that must all be updated."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="file_edit_noop_detection",
            task_type="file_edit",
            description="Detect that no edit is required and avoid mutating the file.",
            build=_file_edit_noop_task,
            difficulty="medium",
            tags=["file-edit", "no-op", "quality"],
            setup_instructions=["Create document.txt already in the desired state."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="file_edit_reread_after_modification",
            task_type="file_edit",
            description="Edit a file and reread it to confirm the final contents.",
            build=_file_edit_reread_task,
            difficulty="medium",
            tags=["file-edit", "reread", "verification"],
            setup_instructions=["Create version.txt with the desired release value.", "Create summary.txt with a stale release value."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="reading_extract_structured",
            task_type="reading",
            description="Read multiple files and return exact structured JSON.",
            build=_reading_task,
            difficulty="easy",
            tags=["reading", "json"],
            setup_instructions=["Create config.json and mode.txt with complementary facts."],
            config_overrides={},
        ),
        BenchmarkTaskDefinition(
            task_id="reading_debug_log_summary",
            task_type="reading",
            description="Read a debug log and extract exact structured facts without misclassifying it as coding.",
            build=_reading_debug_log_task,
            difficulty="easy",
            tags=["reading", "quality", "misleading-wording"],
            setup_instructions=["Create debug.log with one error line and one warning line."],
            config_overrides={},
        ),
        BenchmarkTaskDefinition(
            task_id="reading_identify_contradictions",
            task_type="reading",
            description="Read conflicting sources and report the contradiction exactly.",
            build=_reading_contradiction_task,
            difficulty="medium",
            tags=["reading", "contradiction"],
            setup_instructions=["Create two files with conflicting version values."],
            config_overrides={},
        ),
        BenchmarkTaskDefinition(
            task_id="reading_hallucination_guard",
            task_type="reading",
            description="Answer only from provided files and explicitly mark unsupported fields.",
            build=_reading_hallucination_guard_task,
            difficulty="medium",
            tags=["reading", "hallucination-guard", "quality"],
            setup_instructions=["Create profile.txt with a name but no city."],
            config_overrides={},
        ),
        BenchmarkTaskDefinition(
            task_id="multi_step_read_edit_verify",
            task_type="multi_step",
            description="Read input, edit a file, and verify the result with unittest.",
            build=_multi_step_task,
            difficulty="medium",
            tags=["multi-step", "edit", "verification"],
            setup_instructions=["Create version.txt, app.py, and a unittest that checks synchronization."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="multi_step_read_compute_write_verify",
            task_type="multi_step",
            description="Read facts, compute a derived value, write it, and verify the final file state.",
            build=_multi_step_compute_write_task,
            difficulty="medium",
            tags=["multi-step", "calculator", "write"],
            setup_instructions=["Create numbers.txt and result.txt with a stale placeholder value."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="multi_step_mixed_read_note_compute_write",
            task_type="multi_step",
            description="Mix reading, notes, computation, and editing in one bounded task.",
            build=_multi_step_mixed_task,
            difficulty="hard",
            tags=["multi-step", "notes", "calculator", "file-edit"],
            setup_instructions=["Create brief.txt with report inputs.", "Create report.md with placeholder content."],
            config_overrides={"tools_allow_side_effect_tools": True, "tools_allow_stateful_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="multi_step_environment_shell_persistence",
            task_type="multi_step",
            description="Use the persistent shell session across commands and verify the environment state.",
            build=_environment_shell_persistence_task,
            difficulty="medium",
            tags=["multi-step", "environment", "shell"],
            setup_instructions=["Use a clean workspace and create shell/data.txt only through shell commands."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="multi_step_environment_list_read_write",
            task_type="multi_step",
            description="List files, read a file, write a new file, reread it, and verify exact contents.",
            build=_environment_list_read_write_task,
            difficulty="medium",
            tags=["multi-step", "environment", "filesystem"],
            setup_instructions=["Create template.txt and require the agent to discover and copy it through environment tools."],
            config_overrides={"tools_allow_side_effect_tools": True},
        ),
        BenchmarkTaskDefinition(
            task_id="multi_step_iterative_write_refinement",
            task_type="multi_step",
            description="Refine a file write across multiple iterations until verification passes.",
            build=_multi_step_iterative_write_task,
            difficulty="hard",
            tags=["multi-step", "environment", "refinement"],
            setup_instructions=["Create report.txt with draft content and require an exact final write."],
            config_overrides={"tools_allow_side_effect_tools": True, "runtime_max_tool_steps": 3},
        ),
        BenchmarkTaskDefinition(
            task_id="failure_wrong_tool_usage",
            task_type="failure",
            description="Fail safely when the model selects the wrong tool.",
            build=_failure_wrong_tool_task,
            difficulty="medium",
            tags=["failure", "tooling"],
            setup_instructions=["Create info.txt with a simple payload that should be read, not calculated."],
            config_overrides={"planner_max_replans": 0},
        ),
        BenchmarkTaskDefinition(
            task_id="failure_bad_planning",
            task_type="failure",
            description="Fail safely when the planner returns an invalid plan.",
            build=_failure_bad_planning_task,
            difficulty="medium",
            tags=["failure", "planning"],
            setup_instructions=["Use a scripted invalid plan payload missing the required step fields."],
            config_overrides={"planner_max_replans": 0},
        ),
        BenchmarkTaskDefinition(
            task_id="failure_repeated_action_trap",
            task_type="failure",
            description="Detect repeated tool-helper behavior and stop instead of looping forever.",
            build=_failure_repeated_action_task,
            difficulty="hard",
            tags=["failure", "loop", "repeated-action"],
            setup_instructions=["Create loop.txt with a simple edit target.", "Script the model to repeat the same read helper action twice."],
            config_overrides={"tools_allow_side_effect_tools": True, "planner_max_replans": 0, "runtime_max_tool_steps": 3, "runtime_max_total_actions": 6},
        ),
        BenchmarkTaskDefinition(
            task_id="quality_vague_expansion",
            task_type="quality",
            description="Expand a vague prompt before execution instead of pretending it is already well-defined.",
            build=_quality_vague_expansion_task,
            difficulty="easy",
            tags=["quality", "expansion", "prompt-understanding"],
            setup_instructions=["Use a vague prompt with no files or concrete constraints."],
            config_overrides={},
        ),
        BenchmarkTaskDefinition(
            task_id="quality_already_decomposed_prompt",
            task_type="quality",
            description="Preserve an already-decomposed task instead of expanding or collapsing it incorrectly.",
            build=_quality_already_decomposed_task,
            difficulty="medium",
            tags=["quality", "decomposition", "prompt-understanding"],
            setup_instructions=["Create facts.txt with exact values.", "Use a prompt that already lists numbered steps."],
            config_overrides={},
        ),
        BenchmarkTaskDefinition(
            task_id="quality_incomplete_clarification",
            task_type="quality",
            description="Request clarification instead of claiming success on an incomplete task.",
            build=_quality_incomplete_clarification_task,
            difficulty="easy",
            tags=["quality", "clarification", "false-positive-killer"],
            setup_instructions=["Use an underspecified prompt with no actionable details."],
            config_overrides={},
        ),
    ]
    from swaag.benchmark.scaled_catalog import generated_benchmark_tasks

    tasks.extend(generated_benchmark_tasks())
    validate_benchmark_catalog(tasks)
    return tasks
