from __future__ import annotations

import json
from typing import Any

from swaag.benchmark import benchmark_runner
from swaag.benchmark.compaction_preservation import (
    PRESERVATION_FACTS,
    run_compaction_preservation_benchmark,
)
from swaag.benchmark.long_horizon_context import run_long_horizon_context_benchmark
from swaag.model import CompletionRequestPolicy
from swaag.types import CompletionResult, ContractSpec


class _LongHorizonClient:
    is_deterministic_test_client = True

    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def cache_identity(self):
        return "long-horizon-test-client"

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy("test", "server_schema", contract.mode, 30, 0.01)

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(self, prompt: str, *, max_tokens: int, contract: ContractSpec, temperature=None, **_kwargs):
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict, **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        if payload["contract"] == "history_compaction_selection":
            text = json.dumps({"criticality": "compressible", "reason": "test window"})
        else:
            assert payload["contract"] == "summary"
            text = json.dumps({
                "summary": "\n".join(PRESERVATION_FACTS.values()),
                "preserve_recent_messages": 0,
            })
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )

    def complete(self, prompt: str, *, max_tokens: int, contract: ContractSpec, temperature: float, kind: str, live_mode: bool):
        assert contract.name == "long_horizon_authoritative_retrieval"
        payload = dict(PRESERVATION_FACTS)
        text = json.dumps(payload)
        return CompletionResult(
            text=text,
            raw_request={"prompt": prompt, "json_schema": contract.json_schema},
            raw_response={"content": text},
            prompt_tokens=100,
            completion_tokens=40,
            finish_reason="stop",
        )


def test_compaction_stress_mode_separates_preservation_retrieval_and_decoys(make_config, tmp_path):
    client = _LongHorizonClient()
    config = make_config(model__context_limit=12_000)
    config.sessions.root = tmp_path / "sessions"
    report = run_compaction_preservation_benchmark(
        config=config,
        cycles=3,
        output_path=tmp_path / "compaction.json",
        model_client=client,
        adversarial_conflicts=True,
        semantic_retrieval_probe=True,
    )
    assert report["complete"] is True
    assert report["passed"] == 3
    assert report["semantic_retrieval_passed"] == 3
    assert report["semantic_retrieval_attempted"] == 3
    assert report["cycles_with_decoy_values_retained"] >= 1
    assert all(row["adversarial_conflicts_present"] for row in report["results"])
    assert all(row["semantic_retrieval_passed"] for row in report["results"])
    # Decoys may remain visible as recent context; the benchmark scores whether
    # authoritative values survive and win semantic retrieval despite them.


def _overflow_report() -> dict[str, Any]:
    checks = {
        "candidate_overflow_measured": True,
        "semantic_projection_used": True,
        "projection_lineage_matches_source": True,
        "raw_source_recoverable": True,
        "required_facts_preserved": True,
        "final_request_fits": True,
    }
    return {
        "complete": True,
        "passed": 1,
        "total": 1,
        "results": [
            {
                "case_id": "measured_overflow_projection",
                "verification": {"passed": True, "checks": checks},
            }
        ],
    }


def test_long_horizon_aggregate_keeps_dimensions_separate(monkeypatch, make_config, tmp_path):
    compaction = {
        "complete": True,
        "results": [
            {
                "passed": True,
                "source_reference_count": 2,
                "required_recovery_tokens": 10,
                "actual_recovered_tokens": 10,
                "semantic_retrieval_passed": True,
            },
            {
                "passed": True,
                "source_reference_count": 2,
                "required_recovery_tokens": 10,
                "actual_recovered_tokens": 10,
                "semantic_retrieval_passed": True,
            },
        ],
    }
    overflow_calls: list[str] = []
    monkeypatch.setattr(
        "swaag.benchmark.long_horizon_context.run_compaction_preservation_benchmark",
        lambda **_kwargs: compaction,
    )
    monkeypatch.setattr(
        "swaag.benchmark.long_horizon_context.run_context_engineering_benchmark",
        lambda **kwargs: overflow_calls.append(str(kwargs["output_dir"])) or _overflow_report(),
    )
    report = run_long_horizon_context_benchmark(
        output_dir=tmp_path / "long",
        config=make_config(model__context_limit=4096),
        cycles=2,
        overflow_trials=3,
    )
    assert report["complete"] is True
    assert report["all_dimensions_passed"] is True
    assert report["dimensions"] == {
        "exact_preservation": {"passed": 2, "total": 2},
        "provenance_recoverability": {"passed": 2, "total": 2},
        "semantic_retrieval": {"passed": 2, "total": 2},
        "adversarial_conflict_resistance": {"passed": 2, "total": 2},
        "measured_overflow_projection": {"passed": 3, "total": 3},
    }
    assert len(overflow_calls) == 3
    checkpoint = json.loads((tmp_path / "long" / "long_horizon_context_results.json").read_text())
    assert checkpoint == report


def test_long_horizon_cli_passes_profile(monkeypatch, make_config, tmp_path):
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "complete": True,
            "all_dimensions_passed": True,
            "dimensions": {
                "exact_preservation": {"passed": 4, "total": 4},
            },
        }

    monkeypatch.setattr(
        "swaag.benchmark.long_horizon_context.run_long_horizon_context_benchmark",
        fake_run,
    )
    monkeypatch.setattr(benchmark_runner, "_live_experiment_config", lambda **_kwargs: make_config())
    output = tmp_path / "long"
    code = benchmark_runner.main([
        "long-horizon-context",
        "--output", str(output),
        "--cycles", "4",
        "--overflow-trials", "2",
        "--clean",
    ])
    assert code == 0
    config = captured.pop("config")
    assert config.model.structured_output_mode == "server_schema"
    assert captured == {
        "output_dir": output,
        "cycles": 4,
        "overflow_trials": 2,
        "clean": True,
    }
