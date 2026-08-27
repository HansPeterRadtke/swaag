from __future__ import annotations

import json

from swaag.benchmark import context_order
from swaag.benchmark.benchmark_runner import _build_parser, _live_experiment_config
from swaag.benchmark.context_order import (
    BENCHMARK_VERSION,
    POSITIONS,
    _verification_output_reserve,
    answer_contract,
    build_case,
    build_matrix,
)
from swaag.types import CompletionResult


def test_context_order_matrix_varies_only_position_for_each_utilization():
    cases = build_matrix(context_limit=32000, utilizations=[0.10, 0.25], seed=8)
    assert len(cases) == 6
    assert {case.position for case in cases} == set(POSITIONS)
    for utilization in (0.10, 0.25):
        group = [case for case in cases if case.requested_utilization == utilization]
        assert len({case.expected_code for case in group}) == 1
        fractions = {case.position: case.marker_char_fraction for case in group}
        assert fractions["early"] < 0.15
        assert 0.40 < fractions["middle"] < 0.60
        assert fractions["late"] > 0.85


def test_context_order_matrix_can_select_positions() -> None:
    cases = build_matrix(
        context_limit=8_000,
        utilizations=[0.10],
        positions=["early", "late"],
    )

    assert [case.position for case in cases] == ["early", "late"]


def test_context_order_case_contains_same_exact_fact_and_query():
    case = build_case(position="middle", requested_utilization=0.25, context_limit=16000, seed=44)
    assert case.expected_code in case.prompt
    assert case.prompt.count(case.expected_code) == 1
    assert "What is the exact retrieval code" in case.prompt


def test_context_order_contract_is_closed():
    schema = answer_contract().json_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["answer"]


def test_context_benchmark_cli_accepts_bounded_subsets() -> None:
    parser = _build_parser()
    order = parser.parse_args(
        [
            "context-order",
            "--working-context-limit",
            "8192",
            "--position",
            "middle",
            "--timeout-seconds",
            "1800",
        ]
    )
    layout = parser.parse_args(
        [
            "context-layout",
            "--working-context-limit",
            "8192",
            "--rotation",
            "4",
            "--timeout-seconds",
            "1800",
        ]
    )

    assert order.working_context_limit == 8192
    assert order.timeout_seconds == 1800
    assert order.position == ["middle"]
    assert layout.working_context_limit == 8192
    assert layout.timeout_seconds == 1800
    assert layout.rotation == [4]


def test_bespoke_live_experiments_use_documented_timeout_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SWAAG_LIVE_TIMEOUT_SECONDS", raising=False)

    config = _live_experiment_config()

    assert config.model.timeout_seconds >= 900
    assert config.model.structured_timeout_seconds >= 900
    assert config.model.verification_timeout_seconds >= 900
    assert config.model.benchmark_timeout_seconds >= 900


def test_verification_output_headroom_is_soft_under_input_pressure(make_config):
    config = make_config(model__context_limit=10_000)

    assert _verification_output_reserve(
        config,
        context_limit=10_000,
        input_tokens=1_000,
    ) == 1_200
    assert _verification_output_reserve(
        config,
        context_limit=10_000,
        input_tokens=9_500,
    ) == 436


def test_context_order_matrix_uses_exact_counter_for_requested_utilization():
    cases = build_matrix(
        context_limit=10_000,
        utilizations=[0.50],
        seed=9,
        token_counter=len,
    )

    counts = [len(case.prompt) for case in cases]
    assert max(abs(count - 5_000) for count in counts) < 300
    assert len({case.prompt.replace(case.expected_code, "CODE") for case in cases}) == 3


def test_context_order_benchmark_checkpoints_each_completed_case(
    monkeypatch, make_config, tmp_path
):
    snapshots: list[dict] = []

    class FakeClient:
        def __init__(self, _config):
            pass

        def cache_identity(self):
            return "fake-model"

        def context_limit_resolution(self):
            return 10_000, "test"

        def tokenize(self, text):
            return len(text)

        def render_chat_prompt(self, messages):
            return {
                "prompt": "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>",
                "prompt_protocol_sha256": "a" * 64,
            }

        def verify_prompt_protocol(self, prompt_protocol_sha256):
            assert prompt_protocol_sha256 == "a" * 64

        def complete(self, prompt, **kwargs):
            assert kwargs["messages"][0]["role"] == "system"
            expected = next(code for code in ("SWAAG-0017-ORBIT",) if code in prompt)
            return CompletionResult(
                text=json.dumps({"answer": expected}),
                raw_request={},
                raw_response={},
                prompt_tokens=len(prompt),
                completion_tokens=4,
                finish_reason="stop",
            )

    original_replace = context_order.Path.replace

    def recording_replace(path, target):
        result = original_replace(path, target)
        snapshots.append(json.loads(target.read_text(encoding="utf-8")))
        return result

    monkeypatch.setattr(context_order, "LlamaCppClient", FakeClient)
    monkeypatch.setattr(context_order.Path, "replace", recording_replace)
    output = tmp_path / "context-order.json"

    report = context_order.run_context_order_benchmark(
        config=make_config(model__context_limit=10_000),
        utilizations=[0.10],
        output_path=output,
    )

    assert [snapshot["completed"] for snapshot in snapshots] == [1, 2, 3]
    assert snapshots[0]["complete"] is False
    assert report["complete"] is True
    assert report["benchmark"] == BENCHMARK_VERSION
    assert report["planned"] == 3
    assert {row["reserved_output_tokens"] for row in report["results"]} == {1_200}
    assert all(0 <= row["marker_token_fraction"] <= 1 for row in report["results"])
    assert all(len(row["serialized_prompt_sha256"]) == 64 for row in report["results"])


def test_context_order_benchmark_records_explicit_working_window(
    monkeypatch,
    make_config,
    tmp_path,
) -> None:
    class FakeClient:
        def __init__(self, _config):
            pass

        def cache_identity(self):
            return "working-window-model"

        def context_limit_resolution(self):
            return 10_000, "server_props:n_ctx"

        def tokenize(self, text):
            return len(text)

        def render_chat_prompt(self, messages):
            return {
                "prompt": messages[0]["content"] + messages[1]["content"],
                "prompt_protocol_sha256": "c" * 64,
            }

        def verify_prompt_protocol(self, _prompt_protocol_sha256):
            return None

        def complete(self, prompt, **_kwargs):
            return CompletionResult(
                text=json.dumps({"answer": "SWAAG-0017-ORBIT"}),
                raw_request={},
                raw_response={},
                prompt_tokens=len(prompt),
                completion_tokens=4,
                finish_reason="stop",
            )

    monkeypatch.setattr(context_order, "LlamaCppClient", FakeClient)
    report = context_order.run_context_order_benchmark(
        config=make_config(model__context_limit=10_000),
        utilizations=[0.10],
        positions=["middle"],
        working_context_limit=5_000,
        output_path=tmp_path / "working-window.json",
    )

    assert report["context_limit"] == 5_000
    assert report["context_limit_source"] == "explicit_benchmark_working_window"
    assert report["server_context_limit"] == 10_000
    assert report["server_context_limit_source"] == "server_props:n_ctx"
    assert report["requested_positions"] == ["middle"]
    assert report["planned"] == 1
    row = report["results"][0]
    assert row["context_limit"] == 5_000
    assert row["server_context_limit"] == 10_000
    assert row["server_input_utilization"] < row["actual_input_utilization"]


def test_context_order_benchmark_resumes_matching_partial_checkpoint(
    monkeypatch, make_config, tmp_path
):
    calls: list[str] = []
    properties_hash = ["first-props"]

    class FakeClient:
        def __init__(self, _config):
            pass

        def cache_identity(self):
            return {
                "base_url": "http://model",
                "model_alias": "stable-fake",
                "model_file": {"path": "/model", "size": 10, "mtime_ns": 1},
                "server_build_info": "build-1",
                "server_properties_sha256": properties_hash[0],
            }

        def context_limit_resolution(self):
            return 10_000, "test"

        def tokenize(self, text):
            return len(text)

        def render_chat_prompt(self, messages):
            return {
                "prompt": "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>",
                "prompt_protocol_sha256": "a" * 64,
            }

        def verify_prompt_protocol(self, prompt_protocol_sha256):
            assert prompt_protocol_sha256 == "a" * 64

        def complete(self, prompt, **_kwargs):
            calls.append(prompt)
            return CompletionResult(
                text=json.dumps({"answer": "SWAAG-0017-ORBIT"}),
                raw_request={},
                raw_response={},
                prompt_tokens=len(prompt),
                completion_tokens=4,
                finish_reason="stop",
            )

    monkeypatch.setattr(context_order, "LlamaCppClient", FakeClient)
    output = tmp_path / "context-order.json"
    complete = context_order.run_context_order_benchmark(
        config=make_config(model__context_limit=10_000),
        utilizations=[0.10],
        output_path=output,
    )
    partial = dict(complete)
    partial["results"] = complete["results"][:2]
    partial["completed"] = partial["passed"] = partial["total"] = 2
    partial["complete"] = False
    output.write_text(json.dumps(partial), encoding="utf-8")
    calls.clear()
    properties_hash[0] = "second-props"

    resumed = context_order.run_context_order_benchmark(
        config=make_config(model__context_limit=10_000),
        utilizations=[0.10],
        output_path=output,
    )

    assert len(calls) == 1
    assert resumed["completed"] == resumed["passed"] == 3
    assert resumed["complete"] is True
    assert len(resumed["model_identity_history"]) == 2
