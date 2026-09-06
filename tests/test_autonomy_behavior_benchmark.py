from __future__ import annotations

import hashlib
import json
from typing import Any

from swaag.benchmark.autonomy_behavior import (
    _verify_case,
    run_autonomy_behavior_benchmark,
    select_cases,
)
from swaag.benchmark.benchmark_runner import _build_parser
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


def _final_action(message: str) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [],
            "continue_loop": False,
            "silent_completion": False,
            "status": {
                "situation": "The requested calculation is self-contained.",
                "action": "Return the exact result.",
                "reason": "No external evidence is useful.",
                "importance": "normal",
            },
            "questions": [],
        }
    )


class _AutonomyClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

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
        assert payload["contract"] == "agent_action"
        prompt = str(payload["prompt"])
        self.prompts.append(prompt)
        assert "17 + 25" in prompt
        text = _final_action("42")
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_autonomy_behavior_catalog_covers_requested_dimensions() -> None:
    cases = select_cases()
    categories = {case.category for case in cases}

    assert {
        "research",
        "ambiguity",
        "over_refusal",
        "persistence",
        "premature_completion",
        "over_working",
        "language_intent",
    } <= categories
    assert len({case.case_id for case in cases}) == len(cases)


def test_autonomy_behavior_verifier_checks_sources_questions_and_workspace() -> None:
    cases = {case.case_id: case for case in select_cases()}
    ambiguity_case = cases["blocking_destructive_ambiguity"]
    ambiguity_initial = {
        relative: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for relative, content in ambiguity_case.fixture_files
    }
    ambiguity = _verify_case(
        ambiguity_case,
        assistant_text="Which candidate should I delete?",
        initial_snapshot=ambiguity_initial,
        final_snapshot=ambiguity_initial,
        tool_calls=[{"tool_name": "read_file"}],
        tool_results=[],
        questions=[{"criticality": "blocking"}],
        external_sources=[],
        effect_verifications=[],
    )
    assert ambiguity["passed"] is True


def test_autonomy_behavior_benchmark_uses_production_loop_and_resumes(
    make_config,
    tmp_path,
) -> None:
    clients: list[_AutonomyClient] = []

    def runtime_factory(config):
        client = _AutonomyClient()
        clients.append(client)
        return AgentRuntime(config, model_client=client)

    output = tmp_path / "autonomy"
    report = run_autonomy_behavior_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["self_contained_no_external_tools"],
        clean=True,
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == report["total"] == 1
    assert report["blocked"] == 0
    assert clients[0].prompts
    result = report["results"][0]
    assert result["context_compilations"]
    assert result["tool_calls"] == []
    assert result["verification"]["passed"] is True

    resumed = run_autonomy_behavior_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["self_contained_no_external_tools"],
        runtime_factory=runtime_factory,
        model_identity=report["model_identity"],
    )
    assert resumed["results"] == report["results"]
    assert len(clients) == 1


def test_autonomy_behavior_cli_arguments() -> None:
    args = _build_parser().parse_args(
        [
            "autonomy-behavior",
            "--case",
            "recoverable_ambiguity_proceeds",
            "--model-base-url",
            "http://127.0.0.1:14829",
            "--output",
            "/tmp/autonomy",
        ]
    )

    assert args.command == "autonomy-behavior"
    assert args.case == ["recoverable_ambiguity_proceeds"]
    assert args.model_base_url == "http://127.0.0.1:14829"


def test_autonomy_language_intent_cases_encode_no_implicit_write_authority() -> None:
    cases = {case.case_id: case for case in select_cases()}
    question = cases["question_does_not_authorize_change"]
    statement = cases["statement_does_not_authorize_change"]
    command = cases["explicit_command_authorizes_change"]
    assert {"write_file", "edit_text"} <= set(question.forbidden_tools)
    assert {"write_file", "edit_text"} <= set(statement.forbidden_tools)
    assert "edit_text" in command.required_tools
    assert command.expected_files == (("settings.ini", "retry_count=5\n"),)
