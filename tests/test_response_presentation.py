from __future__ import annotations

import json
from typing import Any

import pytest

from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


class _PresentationClient:
    is_deterministic_test_client = True

    def __init__(self, *, reject_all: bool = False) -> None:
        self.reject_all = reject_all
        self.requests: list[dict[str, Any]] = []
        self.relevance_calls = 0
        self.evaluation_calls = 0

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
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        contract = payload["contract"]
        if contract == "response_relevance":
            self.relevance_calls += 1
            answer = (
                "Tests passed."
                if self.relevance_calls == 1
                else "Tests passed. Deployment remains blocked by polkit."
            )
            text = json.dumps(
                {
                    "answer": answer,
                    "omitted_as_irrelevant": ["routine commit hash"],
                }
            )
        elif contract == "audio_rendering":
            text = json.dumps(
                {
                    "audio_text": (
                        "The tests passed. Deployment remains blocked by polkit."
                    )
                }
            )
        else:
            assert contract == "presentation_evaluation"
            self.evaluation_calls += 1
            rejected_first_relevance = self.evaluation_calls == 1
            acceptable = not self.reject_all and not rejected_first_relevance
            text = json.dumps(
                {
                    "acceptable": acceptable,
                    "reason": (
                        "The deployment blocker was omitted."
                        if not acceptable
                        else "The source information is preserved."
                    ),
                    "missing_or_changed_information": (
                        ["deployment remains blocked by polkit"]
                        if not acceptable
                        else []
                    ),
                    "irrelevant_operational_details": [],
                }
            )
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_response_presentations_are_separate_verified_operations(make_config) -> None:
    client = _PresentationClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=12_000, model__max_retries=1),
        model_client=client,
    )
    state = runtime.create_or_load_session()
    raw = (
        "Tests passed. Deployment remains blocked by polkit. "
        "Commit 0123456789abcdef was pushed."
    )

    result = runtime.generate_response_presentations(
        state,
        original_request="Finish and deploy the work, then report the meaningful result.",
        assistant_message=raw,
        modes={"visual", "audio"},
    )

    assert result["raw"] == raw
    assert result["visual"] == "Tests passed. Deployment remains blocked by polkit."
    assert result["audio"] == (
        "The tests passed. Deployment remains blocked by polkit."
    )
    assert result["completed_modes"] == ["visual", "audio"]
    assert client.relevance_calls == 2
    assert "previous candidate was independently rejected" in client.requests[2][
        "prompt"
    ].lower()
    events = runtime.history.read_history(state.session_id)
    assert sum(
        event.event_type == "response_presentation_rejected" for event in events
    ) == 1
    generated = [
        event
        for event in events
        if event.event_type == "response_presentation_generated"
    ]
    assert [event.payload["mode"] for event in generated] == [
        "response_relevance",
        "audio_rendering",
    ]
    assert all(event.payload["evaluation"]["acceptable"] for event in generated)
    assert generated[0].payload["source_answer"] == raw
    assert generated[1].payload["source_event_references"][0][
        "event_type"
    ] == "response_presentation_generated"


def test_rejected_presentation_falls_back_without_claiming_completion(
    make_config,
) -> None:
    client = _PresentationClient(reject_all=True)
    runtime = AgentRuntime(
        make_config(model__context_limit=12_000, model__max_retries=0),
        model_client=client,
    )
    state = runtime.create_or_load_session()
    raw = "Tests passed. Deployment remains blocked by polkit."

    result = runtime.generate_response_presentations(
        state,
        original_request="Report the outcome.",
        assistant_message=raw,
        modes={"audio"},
    )

    assert result["raw"] == raw
    assert result["visual"] == raw
    assert result["audio"] is None
    assert result["completed_modes"] == []
    assert not any(
        request["contract"] == "audio_rendering" for request in client.requests
    )
    assert runtime.history.read_history(state.session_id)[-1].event_type == (
        "response_presentation_unavailable"
    )
    assert runtime.history.read_history(state.session_id)[-1].payload[
        "error_type"
    ] == "PresentationPrerequisiteUnavailable"


def test_response_presentation_rejects_unknown_mode(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()

    with pytest.raises(ValueError, match="visual and/or audio"):
        runtime.generate_response_presentations(
            state,
            original_request="Report.",
            assistant_message="Done.",
            modes={"telepathy"},
        )
