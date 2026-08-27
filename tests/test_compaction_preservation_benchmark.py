from __future__ import annotations

import json

from swaag.benchmark.compaction_preservation import (
    PRESERVATION_FACTS,
    run_compaction_preservation_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.types import CompletionResult, ContractSpec


class _SummaryClient:
    is_deterministic_test_client = True

    def __init__(self):
        self.requests = []

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def cache_identity(self):
        return "compaction-test-client"

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> CompletionRequestPolicy:
        return CompletionRequestPolicy("test", "server_schema", contract.mode, 30, 0.01)

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict, **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        facts = "\n".join(PRESERVATION_FACTS.values())
        response = json.dumps(
            {"summary": facts, "preserve_recent_messages": 0}
        )
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_compaction_preservation_benchmark_repeats_and_checkpoints(
    make_config, tmp_path
) -> None:
    client = _SummaryClient()
    config = make_config(model__context_limit=12_000)
    config.sessions.root = tmp_path / "sessions"
    output = tmp_path / "compaction.json"

    report = run_compaction_preservation_benchmark(
        config=config,
        cycles=3,
        output_path=output,
        model_client=client,
    )

    assert report["complete"] is True
    assert report["passed"] == report["total"] == 3
    assert len(client.requests) == 3
    assert all(row["source_reference_count"] > 0 for row in report["results"])
    assert all(row["context_accounting"]["context_limit"] == 12_000 for row in report["results"])
    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint == report


def test_compaction_preservation_benchmark_resumes_completed_checkpoint(
    make_config, tmp_path
) -> None:
    client = _SummaryClient()
    config = make_config(model__context_limit=12_000)
    config.sessions.root = tmp_path / "sessions"
    output = tmp_path / "compaction.json"
    first = run_compaction_preservation_benchmark(
        config=config,
        cycles=2,
        output_path=output,
        model_client=client,
    )
    calls = len(client.requests)

    resumed = run_compaction_preservation_benchmark(
        config=config,
        cycles=2,
        output_path=output,
        model_client=client,
    )

    assert resumed == first
    assert len(client.requests) == calls
