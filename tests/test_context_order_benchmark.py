from __future__ import annotations

import json

from swaag.benchmark import context_order
from swaag.benchmark.context_order import POSITIONS, answer_contract, build_case, build_matrix
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


def test_context_order_case_contains_same_exact_fact_and_query():
    case = build_case(position="middle", requested_utilization=0.25, context_limit=16000, seed=44)
    assert case.expected_code in case.prompt
    assert case.prompt.count(case.expected_code) == 1
    assert "What is the exact retrieval code" in case.prompt


def test_context_order_contract_is_closed():
    schema = answer_contract().json_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["answer"]


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

        def complete(self, prompt, **_kwargs):
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
    assert report["planned"] == 3
    assert all(0 <= row["marker_token_fraction"] <= 1 for row in report["results"])
