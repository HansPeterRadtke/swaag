from __future__ import annotations

import json
from typing import Any

from swaag.benchmark.prompt_instruction_behavior import (
    run_prompt_instruction_behavior_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


def _action(
    *,
    message: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    continue_loop: bool = False,
) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [
                {"tool_name": name, "arguments": arguments}
                for name, arguments in (tool_calls or [])
            ],
            "continue_loop": continue_loop,
            "silent_completion": False,
            "status": {
                "situation": "Managing durable instructions.",
                "action": "Apply the requested instruction operation.",
                "reason": "The user explicitly requested durable instruction state.",
                "importance": "normal",
            },
            "questions": [],
        }
    )


class _PromptInstructionBenchmarkClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.responses = [
            _action(
                tool_calls=[
                    (
                        "prompt_instructions",
                        {
                            "action": "list",
                            "instruction_store": "user",
                            "instruction_id": None,
                            "title": None,
                            "content": None,
                            "scopes": None,
                        },
                    )
                ],
                continue_loop=True,
            ),
            _action(
                tool_calls=[
                    (
                        "prompt_instructions",
                        {
                            "action": "add",
                            "instruction_store": "user",
                            "instruction_id": None,
                            "title": "Private operational identifiers",
                            "content": (
                                "Do not expose internal identifiers unless the user "
                                "explicitly requests them."
                            ),
                            "scopes": ["communication_status"],
                        },
                    )
                ],
                continue_loop=True,
            ),
            _action(message="The cross-session reporting instruction is stored."),
        ]

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
        if not self.responses:
            raise AssertionError("No deterministic benchmark response remains")
        text = self.responses.pop(0)
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_prompt_instruction_behavior_benchmark_uses_production_agent_loop(
    make_config,
    tmp_path,
) -> None:
    def runtime_factory(config):
        return AgentRuntime(
            config,
            model_client=_PromptInstructionBenchmarkClient(),
        )

    output = tmp_path / "prompt-instruction-benchmark"
    report = run_prompt_instruction_behavior_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["cross_session_scope"],
        clean=True,
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == report["total"] == 1
    result = report["results"][0]
    assert [item["tool_input"]["action"] for item in result["tool_calls"]] == [
        "list",
        "add",
    ]
    assert result["verification"]["passed"] is True
    assert result["context_compilations"]
    assert (
        output / "prompt_instruction_behavior_results.json"
    ).exists()

    def forbidden_runtime(_config):
        raise AssertionError("completed checkpoint should not rerun model calls")

    resumed = run_prompt_instruction_behavior_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["cross_session_scope"],
        runtime_factory=forbidden_runtime,
        model_identity=report["model_identity"],
    )
    assert resumed["results"] == report["results"]
