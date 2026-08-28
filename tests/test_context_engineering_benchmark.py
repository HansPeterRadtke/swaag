from __future__ import annotations

import json
from typing import Any

from swaag.benchmark import benchmark_runner
from swaag.benchmark.context_engineering import (
    DISTRACTOR_MARKERS,
    REQUIRED_FACTS,
    run_context_engineering_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


class _ProjectionClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def cache_identity(self) -> str:
        return "context-engineering-test-client"

    def context_limit_resolution(self) -> tuple[int, str]:
        return 6_000, "deterministic-test"

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy(
            "test", "server_schema", contract.mode, 30, 0.01
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
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
        }

    def send_completion(
        self, payload: dict[str, Any], **_kwargs
    ) -> CompletionResult:
        self.requests.append(payload)
        assert payload["contract"] == "tool_result_projection"
        prompt = str(payload["prompt"])
        preserved = [fact for fact in REQUIRED_FACTS if fact in prompt]
        projection = (
            "\n".join(preserved)
            if preserved
            else "This exact fragment contains only routine healthy-record noise."
        )
        assert not any(marker in projection for marker in DISTRACTOR_MARKERS)
        text = json.dumps({"projection": projection})
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_context_engineering_benchmark_exercises_fit_and_overflow_paths(
    make_config,
    tmp_path,
) -> None:
    clients: list[_ProjectionClient] = []

    def runtime_factory(config):
        client = _ProjectionClient()
        clients.append(client)
        return AgentRuntime(config, model_client=client)

    output = tmp_path / "context-engineering"
    report = run_context_engineering_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=512),
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == report["total"] == 2
    by_case = {item["case_id"]: item for item in report["results"]}
    fitted = by_case["full_fidelity_fit"]
    projected = by_case["measured_overflow_projection"]
    assert fitted["verification"]["checks"]["no_preemptive_projection"] is True
    assert fitted["context_limit"] == 6_000
    assert projected["projection_events"]
    assert projected["verification"]["checks"]["candidate_overflow_measured"] is True
    assert projected["verification"]["checks"]["projection_lineage_matches_source"] is True
    assert sum(len(client.requests) for client in clients) >= 1
    assert json.loads(
        (output / "context_engineering_results.json").read_text(encoding="utf-8")
    ) == report

    def forbidden_runtime(_config):
        raise AssertionError("completed checkpoint should not run model calls")

    resumed = run_context_engineering_benchmark(
        output_dir=output,
        config=make_config(),
        runtime_factory=forbidden_runtime,
        model_identity=report["model_identity"],
    )
    assert resumed == report


def test_context_engineering_cli_passes_checkpoint_options(
    make_config,
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "context-engineering"
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "complete": True,
            "passed": 1,
            "total": 1,
            "results": [
                {
                    "case_id": "full_fidelity_fit",
                    "verification": {"passed": True},
                }
            ],
        }

    monkeypatch.setattr(
        "swaag.benchmark.context_engineering.run_context_engineering_benchmark",
        fake_run,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_live_experiment_config",
        lambda **_kwargs: make_config(),
    )

    exit_code = benchmark_runner.main(
        [
            "context-engineering",
            "--case",
            "full_fidelity_fit",
            "--output",
            str(output),
            "--clean",
        ]
    )

    assert exit_code == 0
    config = captured.pop("config")
    assert config.model.base_url == "http://127.0.0.1:9999"
    assert captured == {
        "output_dir": output,
        "case_ids": ["full_fidelity_fit"],
        "clean": True,
    }
