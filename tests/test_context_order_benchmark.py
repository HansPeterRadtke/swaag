from __future__ import annotations

from swaag.benchmark.context_order import POSITIONS, answer_contract, build_case, build_matrix


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
