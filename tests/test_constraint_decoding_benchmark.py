from __future__ import annotations

import json

from swaag.benchmark import benchmark_runner
from swaag.benchmark.constraint_decoding import run_constraint_decoding_benchmark
from swaag.types import CompletionResult


class _ConstrainedClient:
    def __init__(self, config, *, semantic_wrong: bool = False):
        self.config = config
        self.semantic_wrong = semantic_wrong
        self.calls = 0

    def cache_identity(self):
        return "constraint-decoding-test-client"

    def complete(self, prompt, *, max_tokens, contract, temperature, kind, live_mode):
        self.calls += 1
        name = contract.name
        if name == "yes_no":
            payload = {"answer": "no" if self.semantic_wrong else "yes"}
        elif name == "agent_tool_call":
            payload = {"tool_calls": []}
        else:
            raise AssertionError(f"unexpected contract {name}")
        text = json.dumps(payload)
        return CompletionResult(
            text=text,
            raw_request={"json_schema": contract.json_schema, "seed": self.config.model.seed},
            raw_response={"content": text},
            prompt_tokens=12,
            completion_tokens=4,
            finish_reason="stop",
            elapsed_seconds=0.1,
            tokens_per_second=40.0,
        )


def test_constraint_decoding_reports_structure_separately_from_semantics(make_config, tmp_path):
    config = make_config()
    client = _ConstrainedClient(config, semantic_wrong=True)
    output = tmp_path / "constraint.json"
    report = run_constraint_decoding_benchmark(
        config=config,
        output_path=output,
        seeds=[17, 42],
        repetitions_per_seed=2,
        case_ids=["yes_no_enum"],
        client=client,
    )
    assert report["complete"] is True
    assert report["planned_calls"] == report["completed_calls"] == 4
    assert report["structurally_valid"] == 4
    assert report["semantic_passed"] == 0
    assert report["constraint_present_in_request"] == 4
    assert json.loads(output.read_text()) == report

    calls = client.calls
    resumed = run_constraint_decoding_benchmark(
        config=config,
        output_path=output,
        seeds=[17, 42],
        repetitions_per_seed=2,
        case_ids=["yes_no_enum"],
        client=client,
    )
    assert resumed == report
    assert client.calls == calls


def test_constraint_decoding_zero_tool_schema_remains_structurally_valid(make_config, tmp_path):
    config = make_config()
    client = _ConstrainedClient(config)
    report = run_constraint_decoding_benchmark(
        config=config,
        output_path=tmp_path / "zero.json",
        seeds=[17],
        repetitions_per_seed=1,
        case_ids=["tool_call_zero_tool_state"],
        client=client,
    )
    assert report["structurally_valid"] == 1
    assert report["semantic_passed"] == 1
    assert report["transport_or_grammar_failures"] == 0


def test_constraint_decoding_cli_passes_reproducibility_options(monkeypatch, make_config, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "complete": True,
            "planned_calls": 2,
            "completed_calls": 2,
            "structurally_valid": 2,
            "structural_valid_percent": 100.0,
            "semantic_passed": 1,
            "semantic_pass_percent": 50.0,
        }

    monkeypatch.setattr(
        "swaag.benchmark.constraint_decoding.run_constraint_decoding_benchmark", fake_run
    )
    monkeypatch.setattr(benchmark_runner, "_live_experiment_config", lambda **_kwargs: make_config())
    output = tmp_path / "constraint.json"
    code = benchmark_runner.main(
        [
            "constraint-decoding",
            "--output", str(output),
            "--case", "yes_no_enum",
            "--seeds", "17,42",
            "--repetitions-per-seed", "3",
            "--no-resume",
        ]
    )
    assert code == 0
    config = captured.pop("config")
    assert config.model.structured_output_mode == "server_schema"
    assert captured == {
        "output_path": output,
        "seeds": [17, 42],
        "repetitions_per_seed": 3,
        "case_ids": ["yes_no_enum"],
        "resume": False,
    }
