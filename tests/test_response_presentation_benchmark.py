from __future__ import annotations

import json
from typing import Any

from swaag.benchmark.response_presentation import (
    run_response_presentation_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


class _BenchmarkPresentationClient:
    is_deterministic_test_client = True

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
        contract = payload["contract"]
        if contract == "response_relevance":
            body = {
                "answer": (
                    "42 tests passed. The service is not deployed because polkit "
                    "blocks systemd authentication. Its intended localhost port is 13401."
                ),
                "omitted_as_irrelevant": ["routine command transcript"],
            }
        elif contract == "audio_rendering":
            body = {
                "audio_text": (
                    "Forty-two tests passed. The service is not deployed because polkit "
                    "blocks systemd authentication. Its intended localhost port is 13401."
                )
            }
        else:
            assert contract == "presentation_evaluation"
            body = {
                "acceptable": True,
                "reason": "All requested information is preserved without operational noise.",
                "missing_or_changed_information": [],
                "irrelevant_operational_details": [],
            }
        text = json.dumps(body)
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_response_presentation_benchmark_compares_all_strategies(
    make_config,
    tmp_path,
) -> None:
    def runtime_factory(config):
        return AgentRuntime(config, model_client=_BenchmarkPresentationClient())

    output = tmp_path / "presentation-benchmark"
    report = run_response_presentation_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        clean=True,
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == report["total"] == 3
    assert [item["strategy"] for item in report["results"]] == [
        "visual_only",
        "staged_audio",
        "single_call_audio",
    ]
    by_strategy = {item["strategy"]: item for item in report["results"]}
    assert by_strategy["visual_only"]["model_call_kinds"] == [
        "response_relevance",
        "presentation_evaluation",
    ]
    assert by_strategy["staged_audio"]["model_call_kinds"] == [
        "response_relevance",
        "presentation_evaluation",
        "audio_rendering",
        "presentation_evaluation",
    ]
    assert by_strategy["single_call_audio"]["model_call_kinds"] == [
        "audio_rendering",
        "presentation_evaluation",
    ]
    assert (
        output / "response_presentation_results.json"
    ).exists()


def test_response_presentation_benchmark_resumes_complete_checkpoint(
    make_config,
    tmp_path,
) -> None:
    output = tmp_path / "presentation-benchmark"

    def runtime_factory(config):
        return AgentRuntime(config, model_client=_BenchmarkPresentationClient())

    first = run_response_presentation_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        clean=True,
        runtime_factory=runtime_factory,
    )

    def forbidden_runtime(_config):
        raise AssertionError("completed checkpoint should not rerun model calls")

    resumed = run_response_presentation_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        runtime_factory=forbidden_runtime,
        model_identity=first["model_identity"],
    )

    assert resumed["results"] == first["results"]
    assert resumed["model_identity"] == first["model_identity"]
