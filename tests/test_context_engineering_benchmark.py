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
        prompt = str(payload["prompt"])
        if payload["contract"] == "tool_result_verbatim_selection":
            selected = [
                line.split()[0]
                for line in prompt.splitlines()
                if line.startswith("L")
                and any(fact in line for fact in REQUIRED_FACTS)
                and not any(marker in line for marker in DISTRACTOR_MARKERS)
            ]
            text = json.dumps({"has": bool(selected), "ids": selected})
        else:
            assert payload["contract"] == "tool_result_projection"
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



def test_semantic_reduction_working_set_cap_fragments_before_inference(
    make_config,
    tmp_path,
) -> None:
    clients: list[_ProjectionClient] = []
    runtimes: list[AgentRuntime] = []

    class _NoExactProjectionClient(_ProjectionClient):
        def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
            if payload["contract"] == "tool_result_verbatim_selection":
                self.requests.append(payload)
                text = json.dumps({"has": False, "ids": []})
                return CompletionResult(
                    text=text,
                    raw_request=payload,
                    raw_response={"content": text},
                    prompt_tokens=None,
                    completion_tokens=None,
                    finish_reason="stop",
                )
            return super().send_completion(payload, **_kwargs)

    def runtime_factory(config):
        client = _NoExactProjectionClient()
        runtime = AgentRuntime(config, model_client=client)
        clients.append(client)
        runtimes.append(runtime)
        return runtime

    cap = 1200
    report = run_context_engineering_benchmark(
        output_dir=tmp_path / "working-set-cap",
        config=make_config(
            model__context_limit=4096,
            context__semantic_reduction_max_input_tokens=cap,
        ),
        case_ids=["measured_overflow_projection"],
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == 1
    assert clients and clients[0].requests
    # Fake client tokenization is one token per whitespace-delimited token.
    assert all(
        len(str(request["prompt"]).split()) <= cap
        for request in clients[0].requests
    )
    events = runtimes[0].history.read_history(report["results"][0]["session_id"])
    exceeded = [
        event for event in events
        if event.event_type == "semantic_reduction_working_set_exceeded"
    ]
    assert exceeded
    assert all(event.payload["working_set_limit_tokens"] == cap for event in exceeded)
    assert any(event.payload["input_tokens"] > cap for event in exceeded)



def test_context_engineering_verifier_preserves_semantic_fact_with_formatting_changes(
    make_config,
    tmp_path,
) -> None:
    class _FormattingProjectionClient(_ProjectionClient):
        def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
            if payload["contract"] == "tool_result_verbatim_selection":
                return super().send_completion(payload, **_kwargs)
            self.requests.append(payload)
            prompt = str(payload["prompt"])
            if any(fact in prompt for fact in REQUIRED_FACTS):
                projection = (
                    "Change ticket: CHG-7419-Z\n"
                    "Deadline: 2042-06-19T15:40:00Z\n"
                    "Negative constraint: never delete the source archive\n"
                    "Checksum causality: The checksum failed because source row 812 was absent."
                )
            else:
                projection = "Routine healthy-record noise only."
            text = json.dumps({"projection": projection})
            return CompletionResult(
                text=text, raw_request=payload, raw_response={"content": text},
                prompt_tokens=None, completion_tokens=None, finish_reason="stop",
            )

    report = run_context_engineering_benchmark(
        output_dir=tmp_path / "formatting-preservation",
        config=make_config(model__context_limit=512),
        case_ids=["measured_overflow_projection"],
        runtime_factory=lambda config: AgentRuntime(
            config, model_client=_FormattingProjectionClient()
        ),
    )
    assert report["passed"] == 1
    result = report["results"][0]
    assert result["verification"]["checks"]["required_facts_preserved"] is True

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


def test_context_engineering_model_unavailable_is_resumable_interruption(
    make_config,
    tmp_path,
) -> None:
    from swaag.model import ModelClientError

    class _UnavailableProjectionClient(_ProjectionClient):
        def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
            self.requests.append(payload)
            raise ModelClientError("model_unavailable")

    output = tmp_path / "context-engineering-interrupted"

    def unavailable_runtime(config):
        runtime = AgentRuntime(config, model_client=_UnavailableProjectionClient())
        runtime._max_model_unavailable_attempts = 0
        runtime._sleep = lambda _seconds: None
        return runtime

    interrupted = run_context_engineering_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=512),
        case_ids=["measured_overflow_projection"],
        runtime_factory=unavailable_runtime,
    )

    assert interrupted["complete"] is False
    assert interrupted["completed"] == 0
    assert interrupted["passed"] == 0
    assert interrupted["results"] == []
    assert len(interrupted["interrupted_attempts"]) == 1
    assert interrupted["interrupted_attempts"][0]["error"] == {
        "error_type": "ModelClientError",
        "reason": "model_unavailable",
    }

    def healthy_runtime(config):
        return AgentRuntime(config, model_client=_ProjectionClient())

    resumed = run_context_engineering_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=512),
        case_ids=["measured_overflow_projection"],
        runtime_factory=healthy_runtime,
    )

    assert resumed["complete"] is True
    assert resumed["completed"] == resumed["passed"] == 1
    assert len(resumed["interrupted_attempts"]) == 1
    assert resumed["results"][0]["verification"]["passed"] is True
    archived = output / "runs" / "01-measured_overflow_projection-interrupted-001"
    assert archived.exists()


def test_context_engineering_raw_endpoint_connection_error_is_resumable_interruption(
    make_config,
    tmp_path,
) -> None:
    import requests

    class _ConnectionErrorRuntime(AgentRuntime):
        def _prepare_action_call(self, *args, **kwargs):
            raise requests.ConnectionError("endpoint disappeared during context preparation")

    output = tmp_path / "context-engineering-connection-interrupted"

    def unavailable_runtime(config):
        return _ConnectionErrorRuntime(config, model_client=_ProjectionClient())

    interrupted = run_context_engineering_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=512),
        case_ids=["measured_overflow_projection"],
        runtime_factory=unavailable_runtime,
    )

    assert interrupted["complete"] is False
    assert interrupted["completed"] == 0
    assert interrupted["passed"] == 0
    assert interrupted["results"] == []
    assert len(interrupted["interrupted_attempts"]) == 1
    assert interrupted["interrupted_attempts"][0]["error"]["error_type"] == "ConnectionError"

    def healthy_runtime(config):
        return AgentRuntime(config, model_client=_ProjectionClient())

    resumed = run_context_engineering_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=512),
        case_ids=["measured_overflow_projection"],
        runtime_factory=healthy_runtime,
    )

    assert resumed["complete"] is True
    assert resumed["completed"] == resumed["passed"] == 1
    assert len(resumed["interrupted_attempts"]) == 1


def test_projection_prompts_require_verbatim_objective_facts_and_omit_decoys():
    from pathlib import Path

    root = Path("src/swaag/assets/prompts")
    system = (root / "tool_result_projection_system.txt").read_text()
    user = (root / "tool_result_projection_user.txt").read_text()
    assert "Copy relevant source wording verbatim" in system
    assert "negative condition" in system
    assert "causal claim" in system
    assert "Omit irrelevant bulk and decoy labels entirely" in system
    assert "Preserve relevant source phrases verbatim" in user
    assert "Do not insert punctuation inside retained phrases" in user
    assert "Omit irrelevant or decoy labels" in user


def test_semantic_reduction_output_exhaustion_fragments_deeper(make_config, tmp_path) -> None:
    class _OutputExhaustingProjectionClient(_ProjectionClient):
        def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
            self.requests.append(payload)
            if payload["contract"] == "tool_result_verbatim_selection":
                text = json.dumps({"has": False, "ids": []})
                return CompletionResult(
                    text=text,
                    raw_request=payload,
                    raw_response={"content": text},
                    prompt_tokens=None,
                    completion_tokens=None,
                    finish_reason="stop",
                )
            assert payload["contract"] == "tool_result_projection"
            prompt = str(payload["prompt"])
            if len(prompt.split()) > 700:
                text = json.dumps({"projection": "incomplete projection"})
                return CompletionResult(
                    text=text,
                    raw_request=payload,
                    raw_response={"content": text},
                    prompt_tokens=None,
                    completion_tokens=int(payload.get("n_predict", 1)),
                    finish_reason="length",
                )
            preserved = [fact for fact in REQUIRED_FACTS if fact in prompt]
            projection = "\n".join(preserved) if preserved else "Routine noise omitted."
            text = json.dumps({"projection": projection})
            return CompletionResult(
                text=text,
                raw_request=payload,
                raw_response={"content": text},
                prompt_tokens=None,
                completion_tokens=None,
                finish_reason="stop",
            )

    runtimes: list[AgentRuntime] = []

    def runtime_factory(config):
        runtime = AgentRuntime(config, model_client=_OutputExhaustingProjectionClient())
        runtimes.append(runtime)
        return runtime

    report = run_context_engineering_benchmark(
        output_dir=tmp_path / "output-exhaustion-fragmentation",
        config=make_config(
            model__context_limit=4096,
            context__semantic_reduction_max_input_tokens=1800,
        ),
        case_ids=["measured_overflow_projection"],
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == 1
    result = report["results"][0]
    assert result["verification"]["checks"]["required_facts_preserved"] is True
    events = runtimes[0].history.read_history(result["session_id"])
    exhausted = [
        event
        for event in events
        if event.event_type == "semantic_reduction_output_exhausted"
    ]
    assert exhausted
    assert all(event.payload["finish_reason"] == "length" for event in exhausted)
    assert any(event.payload["hierarchical_depth"] >= 1 for event in exhausted)


def test_semantic_reduction_call_budget_scales_with_source_and_is_hard_capped(make_config):
    from swaag.runtime import AgentRuntime

    config = make_config(
        model__context_limit=2048,
        context__semantic_reduction_max_input_tokens=8192,
        context__semantic_reduction_max_calls=256,
    )
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    small = runtime._semantic_reduction_call_budget(
        state, source_tokens=1000, context_limit_resolution=(2048, "test")
    )
    medium = runtime._semantic_reduction_call_budget(
        state, source_tokens=18_000, context_limit_resolution=(2048, "test")
    )
    huge = runtime._semantic_reduction_call_budget(
        state, source_tokens=1_000_000, context_limit_resolution=(2048, "test")
    )
    assert small == 16
    assert 64 < medium <= 256
    assert huge == 256


def test_tiny_evidence_reduction_uses_target_aware_response_budget(make_config, monkeypatch):
    from swaag.grammar import evidence_projection_contract
    from swaag.runtime import AgentRuntime
    from swaag.tokens import ExactTokenCounter

    class _Client:
        @staticmethod
        def tokenize(text: str) -> int:
            return len(text.split()) if text.strip() else 0

        @staticmethod
        def cache_identity():
            return {"status": "test"}

    config = make_config(
        model__context_limit=2048,
        runtime__completion_evaluation_enabled=False,
    )
    runtime = AgentRuntime(
        config,
        model_client=_Client(),
        token_counter=ExactTokenCounter(_Client.tokenize),
    )
    state = runtime.create_or_load_session()
    observed = {}

    def fake_execute(_state, prepared, **kwargs):
        observed["reserve"] = prepared.report.reserved_response_tokens
        observed["minimum"] = kwargs["minimum_output_tokens"]
        observed["desired"] = kwargs["desired_output_tokens"]
        return ({"projection": "key fact"}, prepared)

    monkeypatch.setattr(runtime, "_execute_with_output_recovery", fake_execute)
    projection, report = runtime._reduce_text_hierarchically(
        state,
        source_text="small evidence with one key fact",
        source_label="test evidence",
        target_tokens=64,
        contract=evidence_projection_contract(),
        output_key="projection",
        build_assembly=lambda text, label, target: runtime.prompts.build_evidence_projection_prompt(
            purpose="retain the key fact",
            source_label=label,
            raw_evidence=text,
            target_tokens=target,
        ),
        remaining_calls=[16],
        context_limit_resolution=(2048, "test"),
    )
    assert projection == "key fact"
    assert report.fits is True
    assert observed["minimum"] == 32
    assert observed["desired"] == 88
    assert observed["reserve"] == 88
