from __future__ import annotations

import json
from typing import Any

from swaag.benchmark.instruction_following import (
    _verify_case,
    run_instruction_following_benchmark,
    select_cases,
)
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec, PromptInstruction


def _final_action(message: str) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [],
            "continue_loop": False,
            "silent_completion": False,
            "status": {
                "situation": "Following standing response constraints.",
                "action": "Return the constrained response.",
                "reason": "The exact durable instruction is present in the system role.",
                "importance": "normal",
            },
            "questions": [],
        }
    )


class _InstructionFollowingClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def context_limit_resolution(self) -> tuple[int, str]:
        return 12_000, "test"

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy(
            "test", "server_schema", contract.mode, 10, 0.01
        )

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
        }

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        assert payload["contract"] == "agent_action"
        prompt = str(payload["prompt"])
        self.prompts.append(prompt)
        assert "Status exercise format" in prompt
        assert "OPERATIONS UPDATE" in prompt
        text = _final_action(
            "OPERATIONS UPDATE\nstatus: ready\nowner: field-team\nnext: verify"
        )
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_instruction_following_benchmark_uses_production_context_and_resumes(
    make_config,
    tmp_path,
) -> None:
    clients: list[_InstructionFollowingClient] = []

    def runtime_factory(config):
        client = _InstructionFollowingClient()
        clients.append(client)
        return AgentRuntime(config, model_client=client)

    output = tmp_path / "instruction-following"
    report = run_instruction_following_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["simultaneous_format_constraints"],
        clean=True,
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["instruction_success"] == report["total"] == 1
    assert report["constraint_success"] == report["total_constraints"] == 6
    assert report["instruction_success_rate"] == 1.0
    assert report["constraint_success_rate"] == 1.0
    assert clients[0].prompts
    result = report["results"][0]
    assert result["context_compilations"]
    workspace_components = [
        component
        for compilation in result["context_compilations"]
        for component in compilation.get("accounting", {}).get("components", [])
        if component.get("name") == "workspace_file_manifest"
    ]
    assert workspace_components
    assert max(int(item["tokens"]) for item in workspace_components) < 100
    assert result["instruction_event_hashes"]
    report_path = output / "instruction_following_results.json"
    assert report_path.exists()

    def forbidden_runtime(_config):
        raise AssertionError("completed checkpoint should not rerun model calls")

    resumed = run_instruction_following_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["simultaneous_format_constraints"],
        runtime_factory=forbidden_runtime,
        model_identity=report["model_identity"],
    )
    assert resumed["results"] == report["results"]


def test_instruction_following_case_verifiers_accept_complete_outputs() -> None:
    cases = {case.case_id: case for case in select_cases()}
    instruction = PromptInstruction(
        instruction_id="instruction_current",
        title="Current route marker",
        content="Answer exactly current-route.",
        scopes=["action"],
        created_at="2026-08-27T00:00:00+00:00",
        updated_at="2026-08-27T00:00:00+00:00",
    )
    examples = {
        "simultaneous_format_constraints": (
            "OPERATIONS UPDATE\nstatus: ready\nowner: field-team\nnext: verify",
            [],
            [],
        ),
        "conditional_tool_constraint": (
            "review-required",
            [{"tool_name": "calculator", "tool_input": {"expression": "17*19"}}],
            [],
        ),
        "user_report_relevance_constraints": (
            "Tests passed. Deployment remains blocked by authentication.",
            [],
            [],
        ),
        "dense_standing_policy": (
            '{"alpha":"A17","beta":"B29","gamma":["north","east"],"ready":true,"count":3}',
            [],
            [],
        ),
        "revised_instruction_applies": ("current-route", [], [instruction]),
    }
    for case_id, (assistant_text, tool_calls, instructions) in examples.items():
        verification = _verify_case(
            cases[case_id],
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            user_instructions=instructions,
        )
        assert verification["passed"] is True, case_id
        assert (
            verification["passed_constraints"]
            == verification["total_constraints"]
        )
