from __future__ import annotations

import json
import re
from typing import Any

from swaag.benchmark.communication_routing import (
    run_communication_routing_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


class _RoutingBenchmarkClient:
    is_deterministic_test_client = True
    identity_generation = 0

    def __init__(self, label: str) -> None:
        self.label = label

    def cache_identity(self) -> dict[str, str]:
        type(self).identity_generation += 1
        return {
            "base_url": f"http://{self.label}",
            "model_alias": self.label,
            "server_properties_sha256": str(type(self).identity_generation),
        }

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
        prompt = str(payload["prompt"])
        cited = [int(value) for value in re.findall(r"SOURCE EVENT sequence=(\d+)", prompt)]
        if "Is the worker alive" in prompt:
            answer = "The worker is alive and currently awaiting input."
            escalate = False
        elif "Have the tests passed" in prompt:
            answer = "The test result is not established because no completion was recorded."
            escalate = False
        elif "Reconcile the rollout evidence" in prompt:
            answer = (
                "Deployment cannot be claimed complete: the listener is absent and the live "
                "unit differs from its source."
            )
            escalate = self.label == "assistant"
        else:
            assert "deleting the old production backup" in prompt
            answer = (
                "Do not delete the backup; restore verification of the replacement archive "
                "is still needed."
            )
            escalate = self.label == "assistant"
        body = {
            "answer": answer,
            "situation": "The supplied durable evidence was interpreted.",
            "action": "Return the evidence-backed answer.",
            "reason": "The cited source events support the answer.",
            "importance": "major" if escalate else "normal",
            "evidence_sequences": cited,
            "uncertainty": "The result is bounded by the supplied snapshot.",
            "escalate_to_stronger_model": escalate,
            "escalation_reason": (
                "Conflicting, high-impact evidence needs stronger interpretation."
                if escalate
                else ""
            ),
        }
        text = json.dumps(body)
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=100,
            completion_tokens=20,
            finish_reason="stop",
        )


def test_communication_routing_benchmark_uses_production_escalation_and_resumes(
    make_config,
    tmp_path,
) -> None:
    runtime_configs = []

    def runtime_factory(config):
        runtime_configs.append(config)
        label = "assistant" if config.model.base_url.endswith(":14830") else "strong"
        return AgentRuntime(config, model_client=_RoutingBenchmarkClient(label))

    output = tmp_path / "communication-routing"
    config = make_config(model__base_url="http://127.0.0.1:14829")
    report = run_communication_routing_benchmark(
        output_dir=output,
        config=config,
        assistant_model_base_url="http://127.0.0.1:14830",
        clean=True,
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == report["routing_correct"] == report["total"] == 4
    assert report["answer_quality_passed"] == report["total"]
    assert report["escalation_recall"] == 1.0
    assert report["non_escalation_specificity"] == 1.0
    assert report["routing_pair_is_distinct"] is True
    assert report["routing_policy_selection_supported"] is True
    assert [item["model_call_count"] for item in report["results"]] == [1, 1, 2, 2]
    assert report["prompt_tokens"] == 600
    assert report["completion_tokens"] == 120
    assert report["model_identities"]["distinct_endpoints"] is True
    assert runtime_configs
    assert all(
        config.tools.read_roots[0].name == "workspace"
        and config.tools.read_roots[0].is_dir()
        and not any(config.tools.read_roots[0].iterdir())
        for config in runtime_configs
    )
    assert (output / "communication_routing_results.json").exists()

    def forbidden_runtime(_config):
        raise AssertionError("completed checkpoint should not rerun model calls")

    resumed = run_communication_routing_benchmark(
        output_dir=output,
        config=config,
        assistant_model_base_url="http://127.0.0.1:14830",
        runtime_factory=forbidden_runtime,
        model_identities=report["model_identities"],
    )
    assert resumed["results"] == report["results"]
