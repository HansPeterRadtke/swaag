from __future__ import annotations

import json

from swaag.benchmark import benchmark_runner
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
        if payload["contract"] == "history_compaction_selection":
            response = json.dumps({"criticality": "compressible", "reason": "test window"})
        else:
            assert payload["contract"] == "summary"
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
    assert len(client.requests) >= 3
    assert all(row["source_reference_count"] > 0 for row in report["results"])
    assert all(row["required_recovery_tokens"] == 1 for row in report["results"])
    assert all(row["target_summary_tokens"] > 0 for row in report["results"])
    assert all(row["actual_recovered_tokens"] > 0 for row in report["results"])
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


def test_compaction_preservation_cli_passes_checkpoint_options(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "compaction.json"
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "complete": True,
            "passed": 4,
            "total": 4,
            "cycles_completed": 4,
        }

    monkeypatch.setattr(
        "swaag.benchmark.compaction_preservation.run_compaction_preservation_benchmark",
        fake_run,
    )

    exit_code = benchmark_runner.main(
        [
            "compaction-preservation",
            "--cycles",
            "4",
            "--output",
            str(output),
            "--no-resume",
        ]
    )

    assert exit_code == 0
    config = captured.pop("config")
    assert config.model.benchmark_timeout_seconds >= 900
    assert captured == {"cycles": 4, "output_path": output, "resume": False}


class _SpanSelectionClient(_SummaryClient):
    def send_completion(self, payload: dict, **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        assert payload["contract"] == "history_compaction_selection"
        prompt = str(payload["prompt"])
        if "CRITICAL_OLD" in prompt:
            response = json.dumps({"criticality": "protect", "reason": "contains exact user constraint"})
        else:
            response = json.dumps({"criticality": "compressible", "reason": "routine progress"})
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_history_compaction_span_selection_is_semantic_not_oldest_first(make_config, tmp_path) -> None:
    from swaag.runtime import AgentRuntime
    from swaag.types import Message
    from swaag.utils import utc_now_iso

    client = _SpanSelectionClient()
    config = make_config(model__context_limit=12_000)
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    for content in (
        "CRITICAL_OLD exact user constraint must remain verbatim",
        "CRITICAL_OLD unresolved identifier asset-7391",
        "routine progress heartbeat one",
        "routine progress heartbeat two",
    ):
        runtime._record_message(
            state,
            Message(role="user", content=content, created_at=utc_now_iso()),
        )

    ranked = runtime._select_history_compaction_spans(state, list(state.messages))

    assert len(client.requests) == 4
    assert ranked[0]["source_message_start"] == 2
    assert ranked[0]["source_message_count"] == 2
    assert ranked[0]["criticality"] == "compressible"
    protected_starts = {
        row["source_message_start"] for row in ranked if row["criticality"] == "protect"
    }
    assert {0, 1}.issubset(protected_starts)
