from __future__ import annotations

import json

from swaag.action import AgentAction, AgentStatus
from swaag.environment.artifacts import TextArtifactStore
from swaag.grammar import completion_evaluation_contract
from swaag.model import CompletionRequestPolicy
from swaag.prompts import PromptBuilder
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec, Message, SessionState, ToolExecutionResult


def test_completion_contract_requires_semantic_decision_and_remaining_work():
    schema = completion_evaluation_contract().json_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "complete",
        "reason",
        "remaining_work",
        "evidence_requests",
    }


def test_completion_prompt_contains_goal_candidate_and_evidence(make_config):
    builder = PromptBuilder(make_config())
    assembly = builder.build_completion_evaluation_prompt(
        original_request="make all tests pass",
        assistant_message="done",
        status_json='{"importance":"normal"}',
        tool_evidence='[{"tool_name":"run_tests","output":{"passed":false}}]',
    )
    assert assembly.kind == "completion_evaluation"
    assert "make all tests pass" in assembly.prompt_text
    assert "run_tests" in assembly.prompt_text
    assert "passed" in assembly.prompt_text
    assert {item.name for item in assembly.components}.issuperset(
        {
            "completion_objective",
            "completion_candidate",
            "completion_status",
            "completion_tool_evidence_legacy",
        }
    )


def test_completion_prompt_accounts_for_exact_or_projected_historical_evidence(
    make_config,
):
    builder = PromptBuilder(make_config())
    exact = builder.build_completion_evaluation_prompt(
        original_request="verify the complete task",
        assistant_message="done",
        status_json='{"importance":"normal"}',
        historical_evidence='[{"sequence":7,"payload":"prior-marker"}]',
    )
    projected = builder.build_completion_evaluation_prompt(
        original_request="verify the complete task",
        assistant_message="done",
        status_json='{"importance":"normal"}',
        historical_evidence_projection="prior-marker remained relevant",
    )

    exact_component = next(
        item
        for item in exact.components
        if item.name == "completion_historical_evidence"
    )
    projected_component = next(
        item
        for item in projected.components
        if item.name == "completion_historical_evidence"
    )
    assert "prior-marker" in exact_component.text
    assert "SEMANTIC PROJECTION" not in exact_component.text
    assert "SEMANTIC PROJECTION" in projected_component.text


def test_runtime_defaults_enable_completion_evaluation(make_config):
    config = make_config(runtime__completion_evaluation_enabled=True)
    assert config.runtime.completion_evaluation_enabled is True


def test_completion_evidence_includes_failed_tools_from_current_turn() -> None:
    state = SessionState(
        session_id="session-evidence",
        created_at="t",
        updated_at="t",
        config_fingerprint="cfg",
        model_base_url="http://model",
        messages=[
            Message(role="tool", name="old_tool", content="old", created_at="t"),
            Message(role="user", content="current objective", created_at="t"),
            Message(
                role="tool",
                name="read_file",
                content="tool_error: missing",
                created_at="t",
                metadata={
                    "source_event_sequence": 12,
                    "source_event_hash": "abc",
                    "source_event_type": "tool_error",
                    "source_event_session_id": "session-evidence",
                },
            ),
        ],
    )

    rows = AgentRuntime._completion_evidence_rows(state, [])

    assert len(rows) == 1
    assert rows[0]["tool_name"] == "read_file"
    assert rows[0]["success"] is False
    assert rows[0]["source_event_references"][0]["sequence"] == 12


class _CompletionClient:
    is_deterministic_test_client = True

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> CompletionRequestPolicy:
        return CompletionRequestPolicy("test", "server_schema", contract.mode, 30, 0.01)

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict, **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        response = self.responses.pop(0)
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


class _OutputLimitedCompletionClient(_CompletionClient):
    def send_completion(self, payload: dict, **kwargs) -> CompletionResult:
        if not self.requests:
            self.requests.append(payload)
            return CompletionResult(
                text="{",
                raw_request=payload,
                raw_response={"content": "{", "stop_type": "limit"},
                prompt_tokens=None,
                completion_tokens=payload["n_predict"],
                finish_reason="length",
            )
        return super().send_completion(payload, **kwargs)


class _HistoricalProjectionClient(_CompletionClient):
    def __init__(self):
        super().__init__([])

    def send_completion(self, payload: dict, **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        if payload["contract"] == "evidence_projection":
            response = json.dumps(
                {"projection": "historical-marker-91 remains completion evidence"}
            )
        else:
            response = json.dumps(
                {
                    "complete": True,
                    "reason": "Verified.",
                    "remaining_work": [],
                    "evidence_requests": [],
                }
            )
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


class _EvidenceRequestClient(_CompletionClient):
    def __init__(self, *, source_kind: str, source_id: str, marker: str):
        super().__init__([])
        self.source_kind = source_kind
        self.source_id = source_id
        self.marker = marker

    def send_completion(self, payload: dict, **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        if payload["contract"] == "evidence_projection":
            response = json.dumps(
                {"projection": f"The exact source verifies {self.marker}."}
            )
        elif self.marker not in payload["prompt"]:
            response = json.dumps(
                {
                    "complete": False,
                    "reason": "The bounded evidence is insufficient.",
                    "remaining_work": [],
                    "evidence_requests": [
                        {
                            "source_kind": self.source_kind,
                            "source_id": self.source_id,
                            "purpose": "Verify the hidden completion fact.",
                        }
                    ],
                }
            )
        else:
            response = json.dumps(
                {
                    "complete": True,
                    "reason": "The exact evidence verifies completion.",
                    "remaining_work": [],
                    "evidence_requests": [],
                }
            )
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def _completed_action() -> AgentAction:
    return AgentAction(
        assistant_message="The requested work is complete.",
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus(
            "Verification reported success.",
            "Finish.",
            "A bounded result is available.",
            "normal",
        ),
        questions=[],
    )


def test_completion_evaluator_semantically_reexpands_exact_artifact(
    make_config, tmp_path
) -> None:
    config = make_config(model__context_limit=12_000)
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=_CompletionClient([]))
    state = runtime.create_or_load_session()
    marker = "artifact-verification-marker-715"
    artifact = TextArtifactStore(config.sessions.root, state.session_id).create(
        f"Complete exact verifier output: {marker}\n", kind="test_stdout"
    )
    artifact_event = runtime.history.record_event(
        state,
        "artifact_created",
        {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "size_chars": artifact.size_chars,
            "sha256": artifact.sha256,
        },
    )
    output = {
        "stdout": "bounded preview without the decisive fact",
        "stdout_artifact_id": artifact.artifact_id,
        "stdout_sha256": artifact.sha256,
    }
    tool_event = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "run_tests",
            "raw_input": {"command": ["test"]},
            "validated_input": {"command": ["test"]},
            "output": output,
        },
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="run_tests",
            content="run_tests returned a bounded preview",
            created_at="t",
            metadata={
                "output": output,
                "source_event_sequence": tool_event.sequence,
                "source_event_hash": tool_event.hash,
                "source_event_type": tool_event.event_type,
                "source_event_session_id": state.session_id,
                "source_event_references": [],
            },
        ),
    )
    client = _EvidenceRequestClient(
        source_kind="text_artifact",
        source_id=artifact.artifact_id,
        marker=marker,
    )
    runtime.client = client

    result = runtime._evaluate_completion(
        state,
        original_request="Verify the complete test result before finishing.",
        selected_action=_completed_action(),
        tool_results=[],
    )

    assert result["complete"] is True
    assert len(client.requests) == 2
    assert marker not in client.requests[0]["prompt"]
    assert marker in client.requests[1]["prompt"]
    expanded = result["reexpanded_evidence_sources"]
    assert expanded[0]["source_id"] == artifact.artifact_id
    assert expanded[0]["source_event_references"] == [
        {
            "session_id": state.session_id,
            "sequence": artifact_event.sequence,
            "hash": artifact_event.hash,
            "event_type": "artifact_created",
        }
    ]
    reexpansion_event = next(
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "completion_evidence_reexpanded"
    )
    assert reexpansion_event.payload["sha256"] == artifact.sha256
    assert reexpansion_event.payload["exact_chars"] == artifact.size_chars


def test_completion_evaluator_reexpands_exact_raw_attachment(
    make_config, tmp_path
) -> None:
    config = make_config(model__context_limit=12_000)
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=_CompletionClient([]))
    state = runtime.create_or_load_session()
    marker = "attachment-verification-marker-284"
    reference = runtime.add_attachment(
        f"Authoritative attachment evidence: {marker}\n".encode(),
        original_name="verification.txt",
        session_id=state.session_id,
    )
    state = runtime.create_or_load_session(state.session_id)
    runtime._record_message(
        state,
        Message(
            role="user",
            content="Use the attached evidence to verify completion.",
            created_at="t",
        ),
    )
    client = _EvidenceRequestClient(
        source_kind="raw_attachment",
        source_id=reference.attachment_id,
        marker=marker,
    )
    runtime.client = client

    result = runtime._evaluate_completion(
        state,
        original_request="Use the attachment to verify completion.",
        selected_action=_completed_action(),
        tool_results=[],
    )

    assert result["complete"] is True
    assert len(client.requests) == 2
    assert marker not in client.requests[0]["prompt"]
    assert marker in client.requests[1]["prompt"]
    assert result["reexpanded_evidence_sources"][0]["source_id"] == (
        reference.attachment_id
    )


def test_reexpanded_completion_evidence_projects_only_after_overflow(
    make_config, tmp_path
) -> None:
    config = make_config(
        model__context_limit=900,
        context__max_compaction_rounds=3,
    )
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=_CompletionClient([]))
    state = runtime.create_or_load_session()
    marker = "overflow-evidence-marker-922"
    raw = ("bulk exact evidence without the decisive marker " * 500) + marker
    artifact = TextArtifactStore(config.sessions.root, state.session_id).create(
        raw, kind="large_test_stdout"
    )
    runtime.history.record_event(
        state,
        "artifact_created",
        {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "size_chars": artifact.size_chars,
            "sha256": artifact.sha256,
        },
    )
    output = {
        "stdout": "bounded preview",
        "stdout_artifact_id": artifact.artifact_id,
    }
    tool_event = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "run_tests",
            "raw_input": {"command": ["test"]},
            "validated_input": {"command": ["test"]},
            "output": output,
        },
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="run_tests",
            content="bounded preview",
            created_at="t",
            metadata={
                "output": output,
                "source_event_sequence": tool_event.sequence,
                "source_event_hash": tool_event.hash,
                "source_event_type": tool_event.event_type,
                "source_event_session_id": state.session_id,
            },
        ),
    )
    client = _EvidenceRequestClient(
        source_kind="text_artifact",
        source_id=artifact.artifact_id,
        marker=marker,
    )
    runtime.client = client

    result = runtime._evaluate_completion(
        state,
        original_request="Verify the exact complete result.",
        selected_action=_completed_action(),
        tool_results=[],
    )

    assert result["complete"] is True
    assert result["reexpanded_evidence_sources"][0]["projected"] is True
    contracts = [request["contract"] for request in client.requests]
    assert contracts[0] == "completion_evaluation"
    assert "evidence_projection" in contracts
    assert contracts[-1] == "completion_evaluation"
    final_prompt = client.requests[-1]["prompt"]
    assert "semantic_projection" in final_prompt
    assert raw not in final_prompt
    projection_event = next(
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "completion_evidence_projected"
    )
    assert projection_event.payload["source_id"] == artifact.artifact_id
    assert projection_event.payload["source_sha256"] == artifact.sha256


def test_completion_evaluation_keeps_prior_turn_history_exact_when_it_fits(
    make_config,
) -> None:
    client = _CompletionClient(
        [
            json.dumps(
                {
                    "complete": True,
                    "reason": "Verified.",
                    "remaining_work": [],
                    "evidence_requests": [],
                }
            )
        ]
    )
    runtime = AgentRuntime(
        make_config(model__context_limit=12_000),
        model_client=client,
    )
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(role="user", content="Earlier objective", created_at="t"),
    )
    runtime._record_message(
        state,
        Message(
            role="assistant",
            content="Verified historical-marker-83 with the user.",
            created_at="t",
        ),
    )
    runtime._record_message(
        state,
        Message(role="user", content="Finish the current objective", created_at="t"),
    )
    prior_events = runtime.history.read_history(state.session_id)[:-1]
    action = AgentAction(
        assistant_message="The current objective is complete.",
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus("Done.", "Finish.", "Evidence is sufficient.", "normal"),
        questions=[],
    )

    result = runtime._evaluate_completion(
        state,
        original_request="Finish the current objective",
        selected_action=action,
        tool_results=[],
    )

    assert result["complete"] is True
    assert result["historical_evidence_projected"] is False
    assert "historical-marker-83" in client.requests[-1]["prompt"]
    assert {
        reference["sequence"]
        for reference in result["historical_source_event_references"]
    } == {event.sequence for event in prior_events}


def test_completion_evaluation_projects_all_prior_history_only_after_overflow(
    make_config,
) -> None:
    client = _HistoricalProjectionClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=900, context__max_compaction_rounds=3),
        model_client=client,
    )
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(role="user", content="Earlier objective", created_at="t"),
    )
    runtime._record_message(
        state,
        Message(
            role="assistant",
            content=("historical-marker-91 " * 1_500),
            created_at="t",
        ),
    )
    runtime._record_message(
        state,
        Message(role="user", content="Finish now", created_at="t"),
    )
    action = AgentAction(
        assistant_message="Finished.",
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus("Done.", "Finish.", "Evidence exists.", "normal"),
        questions=[],
    )

    result = runtime._evaluate_completion(
        state,
        original_request="Finish now",
        selected_action=action,
        tool_results=[],
    )

    projection_requests = [
        request
        for request in client.requests
        if request["contract"] == "evidence_projection"
    ]
    completion_request = client.requests[-1]
    assert result["complete"] is True
    assert result["historical_evidence_projected"] is True
    assert projection_requests
    assert any("historical-marker-91" in request["prompt"] for request in projection_requests)
    assert "SEMANTIC PROJECTION" in completion_request["prompt"]
    assert ("historical-marker-91 " * 100) not in completion_request["prompt"]
    event = next(
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "completion_evaluated"
    )
    assert event.payload["historical_evidence_projection"]
    assert event.payload["historical_projection_budget_report"]


def test_completion_evaluation_recompiles_after_output_starvation(make_config) -> None:
    client = _OutputLimitedCompletionClient(
        [
            json.dumps(
                {
                    "complete": True,
                    "reason": "Verified.",
                    "remaining_work": [],
                    "evidence_requests": [],
                }
            )
        ]
    )
    runtime = AgentRuntime(
        make_config(model__context_limit=12_000, model__max_retries=1),
        model_client=client,
    )
    state = runtime.create_or_load_session()
    action = AgentAction(
        assistant_message="The work is verified.",
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus("Verification passed.", "Finish.", "Evidence is sufficient.", "normal"),
        questions=[],
    )

    result = runtime._evaluate_completion(
        state,
        original_request="Complete and verify the work.",
        selected_action=action,
        tool_results=[],
    )

    assert result["complete"] is True
    assert len(client.requests) == 2
    assert client.requests[1]["n_predict"] > client.requests[0]["n_predict"]
    repaired = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "budget_repaired"
    ]
    assert repaired[-1].payload["kind"] == "completion_evaluation"


def test_completion_evaluation_semantically_projects_only_after_measured_overflow(
    make_config, tmp_path
) -> None:
    config = make_config(model__context_limit=900, context__max_compaction_rounds=3)
    config.sessions.root = tmp_path / "sessions"
    oversized_projection = "still-too-large " * 700
    client = _CompletionClient(
        [
            json.dumps({"projection": oversized_projection}),
            json.dumps({"projection": "Tests passed with exact verifier evidence."}),
            json.dumps(
                {
                    "complete": True,
                    "reason": "Verified.",
                    "remaining_work": [],
                    "evidence_requests": [],
                }
            ),
        ]
    )
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    raw = "bulk-observation " * 300
    output = {"passed": True, "stdout": raw}
    source = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "run_tests",
            "raw_input": {"command": ["pytest"]},
            "validated_input": {"command": ["pytest"]},
            "output": output,
        },
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="run_tests",
            content=f"run_tests result: {raw}",
            created_at="t",
            metadata={
                "output": output,
                "source_event_sequence": source.sequence,
                "source_event_hash": source.hash,
                "source_event_type": source.event_type,
                "source_event_session_id": source.session_id,
                "source_event_references": [],
            },
        ),
    )
    action = AgentAction(
        assistant_message="All requested work is complete.",
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus("Tests passed.", "Finish.", "Evidence is green.", "normal"),
        questions=[],
    )

    result = runtime._evaluate_completion(
        state,
        original_request="Make the implementation correct and verify it.",
        selected_action=action,
        tool_results=[ToolExecutionResult("run_tests", output, f"run_tests result: {raw}")],
    )

    assert result["complete"] is True
    assert result["projected_source_event_sequences"] == [source.sequence]
    assert [request["contract"] for request in client.requests] == [
        "tool_result_projection",
        "tool_result_projection",
        "completion_evaluation",
    ]
    assert raw.strip() in client.requests[0]["prompt"]
    assert raw.strip() in client.requests[1]["prompt"]
    assert "SEMANTIC PROJECTION" in client.requests[2]["prompt"]
    assert raw.strip() not in client.requests[2]["prompt"]
    events = runtime.history.read_history(state.session_id)
    failed_compilations = [
        event for event in events
        if event.event_type == "context_compiled"
        and event.payload.get("kind") == "completion_evaluation"
        and event.payload.get("cap_error") == "context_limit_exceeded"
    ]
    assert len(failed_compilations) == 2
    assert sum(event.event_type == "tool_result_projected" for event in events) == 2
