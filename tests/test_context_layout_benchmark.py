from __future__ import annotations

import json
import re

from swaag.benchmark.context_layout import (
    ALL_FIELDS,
    BENCHMARK_VERSION,
    USER_FIELDS,
    answer_contract,
    build_cases,
    run_context_layout_benchmark,
)
from swaag.types import CompletionResult


def test_layout_matrix_balances_each_user_section_across_every_position() -> None:
    cases = build_cases(context_limit=10_000, utilizations=[0.25], seed=31)

    assert len(cases) == len(USER_FIELDS)
    for position in range(len(USER_FIELDS)):
        assert {case.user_section_order[position] for case in cases} == set(USER_FIELDS)
    assert all(set(case.expected) == set(ALL_FIELDS) for case in cases)


def test_layout_contract_requires_every_semantic_section() -> None:
    schema = answer_contract().json_schema

    assert schema["additionalProperties"] is False
    assert schema["required"] == list(ALL_FIELDS)
    assert set(schema["properties"]) == set(ALL_FIELDS)


class FakeClient:
    def __init__(self, _config) -> None:
        self.completed_prompts: list[str] = []

    def cache_identity(self):
        return {"model_alias": "layout-fake", "server_build_info": "build"}

    def context_limit_resolution(self):
        return 10_000, "test"

    def render_chat_prompt(self, messages):
        return {
            "prompt": (
                "<system>"
                + messages[0]["content"]
                + "<user>"
                + messages[1]["content"]
                + "<assistant>"
            ),
            "prompt_protocol_sha256": "b" * 64,
        }

    def verify_prompt_protocol(self, prompt_protocol_sha256):
        assert prompt_protocol_sha256 == "b" * 64

    def tokenize(self, text):
        return len(text)

    def complete(self, prompt, **kwargs):
        self.completed_prompts.append(prompt)
        assert kwargs["messages"][0]["role"] == "system"
        codes = re.findall(r"SWAAG-LAYOUT-\d{4}-\d{2}", prompt)
        unique_codes = list(dict.fromkeys(codes))
        answer = {
            field: next(code for code in unique_codes if code.endswith(f"-{index:02d}"))
            for index, field in enumerate(ALL_FIELDS, start=1)
        }
        return CompletionResult(
            text=json.dumps(answer),
            raw_request={},
            raw_response={},
            prompt_tokens=len(prompt),
            completion_tokens=20,
            finish_reason="stop",
        )


def test_layout_runner_checkpoints_server_serialized_balanced_cases(
    make_config,
    tmp_path,
) -> None:
    client = FakeClient(None)
    output = tmp_path / "layout.json"

    report = run_context_layout_benchmark(
        config=make_config(model__context_limit=10_000),
        utilizations=[0.10],
        output_path=output,
        client_factory=lambda _config: client,
    )

    assert report["benchmark"] == BENCHMARK_VERSION
    assert report["complete"] is True
    assert report["passed"] == report["total"] == len(USER_FIELDS)
    assert {row["reserved_output_tokens"] for row in report["results"]} == {1_200}
    assert all(values["passed"] == len(USER_FIELDS) for values in report["by_field"].values())
    assert all(len(row["serialized_prompt_sha256"]) == 64 for row in report["results"])
    assert json.loads(output.read_text())["complete"] is True

    resumed = run_context_layout_benchmark(
        config=make_config(model__context_limit=10_000),
        utilizations=[0.10],
        output_path=output,
        client_factory=lambda _config: client,
    )
    assert resumed["total"] == len(USER_FIELDS)
    assert len(client.completed_prompts) == len(USER_FIELDS)
