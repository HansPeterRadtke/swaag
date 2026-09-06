from __future__ import annotations

import json
from typing import Any

from swaag.benchmark.response_length import run_response_length_benchmark
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


class _LengthClient:
    is_deterministic_test_client = True

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def context_limit_resolution(self) -> tuple[int, str]:
        return 12_000, "test"

    def cache_identity(self):
        return {"model": "length-test"}

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy("test", "server_schema", contract.mode, 10, 0.01)

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
        return {"prompt": prompt, "n_predict": max_tokens, "contract": contract.name}

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        prompt = payload["prompt"]
        if "exactly 45 words" in prompt:
            count = 45
        elif "short answer" in prompt:
            count = 40
        elif "medium-length" in prompt:
            count = 100
        else:
            count = 220
        answer = " ".join(f"word{i}" for i in range(count))
        text = json.dumps({"answer": answer})
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_response_length_benchmark_measures_exact_and_qualitative_instructions(
    make_config, tmp_path
) -> None:
    report = run_response_length_benchmark(
        output_dir=tmp_path / "length",
        config=make_config(model__context_limit=12_000),
        clean=True,
        runtime_factory=lambda config: AgentRuntime(config, model_client=_LengthClient()),
    )
    assert report["complete"] is True
    assert report["passed"] == report["total"] == 4
    rows = {item["case"]: item for item in report["results"]}
    assert rows["exact_words_45"]["word_count"] == 45
    assert rows["exact_words_45"]["absolute_target_error_words"] == 0
    assert rows["short"]["instruction_kind"] == "qualitative"
    assert rows["medium"]["word_count"] > rows["short"]["word_count"]
    assert rows["detailed"]["word_count"] > rows["medium"]["word_count"]
