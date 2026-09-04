from __future__ import annotations

import json
import re
from typing import Any

from swaag.benchmark.note_behavior import (
    _verify_case,
    run_note_behavior_benchmark,
    select_cases,
)
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec, Note


def _action(message: str) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [],
            "continue_loop": False,
            "silent_completion": False,
            "status": {
                "situation": "Relevant durable state is available.",
                "action": "Answer from the selected exact note.",
                "reason": "The note directly governs the current parser work.",
                "importance": "normal",
            },
            "questions": [],
        }
    )


class _NoteBehaviorClient:
    is_deterministic_test_client = True

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

    def send_completion(
        self, payload: dict[str, Any], **_kwargs
    ) -> CompletionResult:
        if payload["contract"] == "note_selection":
            match = re.search(
                r'"note_id": "([^"]+)"(?:(?!"note_id").)*'
                r'"title": "Parser implementation marker"',
                str(payload["prompt"]),
                re.DOTALL,
            )
            assert match is not None
            text = json.dumps(
                {
                    "operation_categories": ["software implementation"],
                    "selected_note_ids": [match.group(1)],
                    "reason": "Only the parser implementation note governs this action.",
                }
            )
        else:
            assert payload["contract"] == "agent_action"
            text = _action("The governing marker is PARSER-ALPHA-731.")
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_note_behavior_benchmark_preserves_full_fidelity_when_notes_fit(
    make_config,
    tmp_path,
) -> None:
    def runtime_factory(config):
        return AgentRuntime(config, model_client=_NoteBehaviorClient())

    output = tmp_path / "note-behavior"
    report = run_note_behavior_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["semantic_application_isolation"],
        clean=True,
        runtime_factory=runtime_factory,
    )

    assert report["complete"] is True
    assert report["passed"] == report["total"] == 1
    result = report["results"][0]
    assert result["verification"]["passed"] is True
    assert result["note_selections"] == []
    workspace_components = [
        component
        for compilation in result["context_compilations"]
        for component in compilation.get("accounting", {}).get("components", [])
        if component.get("name") == "workspace_file_manifest"
    ]
    assert workspace_components
    assert max(int(item["tokens"]) for item in workspace_components) < 100
    assert result["source_event_references"]
    assert len(result["source_prompt_sha256"]) == 64
    assert (output / "note_behavior_results.json").exists()

    def forbidden_runtime(_config):
        raise AssertionError("completed checkpoint should not rerun model calls")

    resumed = run_note_behavior_benchmark(
        output_dir=output,
        config=make_config(model__context_limit=12_000),
        case_ids=["semantic_application_isolation"],
        runtime_factory=forbidden_runtime,
        model_identity=report["model_identity"],
    )
    assert resumed["results"] == report["results"]


def test_note_lifecycle_verifier_accepts_distinct_categories() -> None:
    case = next(
        item for item in select_cases() if item.case_id == "categorized_lifecycle"
    )
    now = "2026-08-28T00:00:00+00:00"
    notes = [
        Note(
            note_id="implementation",
            title="Implementation discipline",
            content="Reproduce the defect before editing and test every change.",
            created_at=now,
            updated_at=now,
            categories=["software implementation", "testing"],
        ),
        Note(
            note_id="research",
            title="Source discipline",
            content="Prefer primary sources for version research.",
            created_at=now,
            updated_at=now,
            categories=["research", "source verification"],
        ),
    ]

    verification = _verify_case(
        case,
        seeded_ids=[],
        notes=notes,
        tool_actions=["list", "add", "add"],
        note_mutations=["note_added", "note_added"],
        selection_payloads=[],
        assistant_text="Stored both notes.",
    )

    assert verification["passed"] is True
