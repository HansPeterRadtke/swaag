from __future__ import annotations

import json
from typing import Any

import pytest

from swaag.grammar import communication_status_contract, evidence_projection_contract
from swaag.model import CompletionRequestPolicy
from swaag.prompts import PromptBuilder
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec
from swaag.utils import stable_json_dumps, utc_now_iso


class _CharacterStatusClient:
    is_deterministic_test_client = True

    def __init__(
        self,
        *,
        markers: list[str],
        cited_sequence: int = 0,
        reject_first_citation: bool = False,
    ) -> None:
        self.markers = markers
        self.cited_sequence = cited_sequence
        self.reject_first_citation = reject_first_citation
        self.requests: list[dict[str, Any]] = []
        self.status_calls = 0

    def tokenize(self, text: str) -> int:
        return len(text)

    def tokenize_selection(self, text: str) -> int:
        return len(text)

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
        temperature=None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        if payload["contract"] == "evidence_projection":
            projection = " ".join(
                marker for marker in self.markers if marker in payload["prompt"]
            ) or "fragment retained"
            text = json.dumps({"projection": projection})
        else:
            self.status_calls += 1
            cited = (
                [999_999]
                if self.reject_first_citation and self.status_calls == 1
                else ([self.cited_sequence] if self.cited_sequence else [])
            )
            visible = [
                marker for marker in self.markers if marker in payload["prompt"]
            ]
            text = json.dumps(
                {
                    "answer": "Status includes " + " ".join(visible),
                    "situation": "Durable evidence is available.",
                    "action": "Report the current evidence-backed snapshot.",
                    "reason": "The cited event supports the status.",
                    "importance": "major",
                    "evidence_sequences": cited,
                    "uncertainty": "The snapshot may become stale after generation.",
                    "escalate_to_stronger_model": False,
                    "escalation_reason": "",
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


def _target_event(runtime: AgentRuntime, state, text: str):
    return runtime.history.record_event(
        state,
        "message_added",
        {
            "message": {
                "role": "user",
                "content": text,
                "created_at": utc_now_iso(),
                "name": None,
                "metadata": {"source": "status-test"},
            }
        },
    )


def test_communication_status_contracts_are_closed() -> None:
    status = communication_status_contract()
    projection = evidence_projection_contract()

    assert status.json_schema["additionalProperties"] is False
    assert status.json_schema["properties"]["importance"]["enum"] == [
        "minor",
        "normal",
        "major",
        "critical",
    ]
    assert "escalate_to_stronger_model" in status.json_schema["required"]
    assert "escalation_reason" in status.json_schema["required"]
    assert projection.json_schema["additionalProperties"] is False
    assert projection.json_schema["required"] == ["projection"]


def test_status_prompt_keeps_question_mechanical_state_and_exact_events(make_config) -> None:
    builder = PromptBuilder(make_config())
    assembly = builder.build_communication_status_prompt(
        question="What failed?",
        mechanical_status={"mechanical_phase": "inference"},
        evidence_rows=[
            {
                "session_id": "session_target",
                "sequence": 17,
                "hash": "abc123",
                "event_type": "tool_error",
                "timestamp": "now",
                "payload": {"error": "marker-17"},
                "metadata": {},
            }
        ],
    )

    assert assembly.kind == "communication_status"
    assert "What failed?" in assembly.prompt_text
    assert '"mechanical_phase":"inference"' in assembly.prompt_text
    assert "SOURCE EVENT sequence=17 hash=abc123" in assembly.prompt_text
    assert "marker-17" in assembly.prompt_text


def test_status_uses_full_fidelity_when_exact_snapshot_fits(make_config) -> None:
    marker = "full-fidelity-status-marker"
    config = make_config(model__context_limit=50_000)
    client = _CharacterStatusClient(markers=[marker])
    runtime = AgentRuntime(config, model_client=client)
    target = runtime.create_or_load_session()
    event = _target_event(runtime, target, marker)
    client.cited_sequence = event.sequence
    events_before = runtime.history.read_history(target.session_id)

    result = runtime.generate_communication_status(
        target_session_id=target.session_id,
        question="What is the evidence-backed status?",
        mechanical_status=runtime.session_status_payload(target),
        source_events=events_before,
    )

    assert result["evidence_projected"] is False
    assert result["importance"] == "major"
    assert result["importance_rank"] == 3
    assert result["evidence_sequences"] == [event.sequence]
    assert marker in result["answer"]
    assert [request["contract"] for request in client.requests] == [
        "communication_status"
    ]
    assert client.requests[0]["n_predict"] == 192
    assert marker in client.requests[0]["prompt"]
    assert runtime.history.read_history(target.session_id) == events_before
    assert runtime.resolve_session_ref(None, latest_if_none=True) == target.session_id
    assert all(
        not str(entry["session_id"]).startswith("operation_")
        for entry in runtime.history.list_session_entries()
    )
    latest = runtime.latest_semantic_status_payload(target)
    assert latest is not None
    assert latest["status_kind"] == "independent_communication_status"
    assert latest["answer"] == result["answer"]


def test_status_projects_only_after_measured_overflow_and_preserves_fragments(
    make_config,
) -> None:
    markers = [f"status-fragment-marker-{index}" for index in range(4)]
    source = "".join("A" * 4_000 + marker for marker in markers)
    config = make_config(
        model__context_limit=10_000,
        context__max_compaction_rounds=4,
        context__safety_margin_tokens=32,
    )
    client = _CharacterStatusClient(markers=markers)
    runtime = AgentRuntime(config, model_client=client)
    target = runtime.create_or_load_session()
    event = _target_event(runtime, target, source)
    client.cited_sequence = event.sequence
    events_before = runtime.history.read_history(target.session_id)

    result = runtime.generate_communication_status(
        target_session_id=target.session_id,
        question="Which status markers remain relevant?",
        mechanical_status=runtime.session_status_payload(target),
        source_events=events_before,
    )

    assert result["evidence_projected"] is True
    assert result["projection_target_tokens"] is not None
    assert all(marker in result["answer"] for marker in markers)
    contracts = [request["contract"] for request in client.requests]
    assert contracts[-1] == "communication_status"
    assert contracts.count("evidence_projection") > 1
    assert runtime.history.read_history(target.session_id) == events_before


def test_status_retries_with_mechanical_citation_feedback(make_config) -> None:
    config = make_config(model__context_limit=50_000, model__max_retries=1)
    client = _CharacterStatusClient(
        markers=[], reject_first_citation=True
    )
    runtime = AgentRuntime(config, model_client=client)
    target = runtime.create_or_load_session()
    event = _target_event(runtime, target, "citation evidence")
    client.cited_sequence = event.sequence

    result = runtime.generate_communication_status(
        target_session_id=target.session_id,
        question="What supports the status?",
        mechanical_status=runtime.session_status_payload(target),
        source_events=runtime.history.read_history(target.session_id),
    )

    assert result["evidence_sequences"] == [event.sequence]
    status_requests = [
        request
        for request in client.requests
        if request["contract"] == "communication_status"
    ]
    assert len(status_requests) == 2
    assert "cites unavailable target event sequences" in status_requests[1]["prompt"]


def test_failed_status_operation_does_not_replace_target_heartbeat(make_config) -> None:
    config = make_config(model__context_limit=50_000, model__max_retries=0)
    client = _CharacterStatusClient(markers=[], reject_first_citation=True)
    runtime = AgentRuntime(config, model_client=client)
    target = runtime.create_or_load_session()
    runtime.history.set_active_run(
        target.session_id,
        run_id="target-run",
        user_text="keep working",
    )
    before = stable_json_dumps(
        runtime.history.read_active_run(target.session_id), indent=None
    )

    with pytest.raises(ValueError, match="cites unavailable"):
        runtime.generate_communication_status(
            target_session_id=target.session_id,
            question="What is happening?",
            mechanical_status=runtime.session_status_payload(target),
            source_events=runtime.history.read_history(target.session_id),
        )

    assert stable_json_dumps(
        runtime.history.read_active_run(target.session_id), indent=None
    ) == before


def test_status_evidence_mechanically_references_oversized_event_values(make_config) -> None:
    from swaag.types import HistoryEvent
    from swaag.utils import sha256_text

    huge_prompt = "GIANT-PROMPT-MARKER " * 4_000
    huge_components = [
        {"name": f"component-{index}", "text": "x" * 500}
        for index in range(32)
    ]
    event = HistoryEvent(
        id="event-big",
        sequence=42,
        session_id="session_target",
        timestamp="now",
        type="prompt_built",
        version=1,
        payload={
            "kind": "action",
            "prompt": huge_prompt,
            "components": huge_components,
            "prompt_sha256": sha256_text(huge_prompt),
        },
        hash="event-hash-42",
    )
    row = AgentRuntime._communication_status_evidence_row(event)
    encoded_payload = stable_json_dumps(event.payload, indent=None)
    payload_ref = row["payload"]["exact_payload_reference"]
    assert payload_ref == {
        "chars": len(encoded_payload),
        "sha256": sha256_text(encoded_payload),
    }
    assert row["payload"]["kind"] == "action"
    assert row["payload"]["payload_view"] == "reference_only_operational_event"
    serialized = stable_json_dumps(row, indent=None)
    assert "GIANT-PROMPT-MARKER" not in serialized
    assert len(serialized) < 2_000


def test_status_large_operational_events_fit_via_mechanical_references_without_semantic_projection(make_config) -> None:
    from swaag.types import HistoryEvent

    marker = "small-status-semantic-marker"
    config = make_config(model__context_limit=32_768)
    client = _CharacterStatusClient(markers=[marker])
    runtime = AgentRuntime(config, model_client=client)
    target = runtime.create_or_load_session()
    huge_prompt = "very large serialized prompt " * 6_000
    huge_request = {
        "prompt": huge_prompt,
        "json_schema": {"type": "object", "properties": {str(i): {"type": "string"} for i in range(400)}},
        "components": [{"name": str(i), "text": "z" * 400} for i in range(40)],
    }
    events = [
        HistoryEvent(
            id="e40",
            sequence=40,
            session_id=target.session_id,
            timestamp="now",
            type="context_compiled",
            version=1,
            payload={"kind": "action", "accounting": {"input_tokens": 5_000, "fits": True}},
            hash="hash40",
        ),
        HistoryEvent(
            id="e42",
            sequence=42,
            session_id=target.session_id,
            timestamp="now",
            type="prompt_built",
            version=1,
            payload={"kind": "action", "prompt": huge_prompt, "components": huge_request["components"]},
            hash="hash42",
        ),
        HistoryEvent(
            id="e45",
            sequence=45,
            session_id=target.session_id,
            timestamp="now",
            type="model_request_sent",
            version=1,
            payload={"kind": "action", "request": huge_request, "call_id": "model_call_test"},
            hash="hash45",
        ),
        HistoryEvent(
            id="e46",
            sequence=46,
            session_id=target.session_id,
            timestamp="now",
            type="model_call_preempted",
            version=1,
            payload={
                "kind": "action",
                "request_sha256": "request-hash",
                "preemption_id": "preemption-test",
                "reason": marker,
            },
            hash="hash46",
        ),
    ]
    client.cited_sequence = 46
    result = runtime.generate_communication_status(
        target_session_id=target.session_id,
        question="What is the main agent doing right now?",
        mechanical_status=runtime.session_status_payload(target),
        source_events=events,
    )
    contracts = [request["contract"] for request in client.requests]
    assert contracts == ["communication_status"]
    assert result["evidence_projected"] is False
    assert marker in result["answer"]
    prompt = client.requests[0]["prompt"]
    assert "GIANT-PROMPT-MARKER" not in prompt
    assert "exact_payload_reference" in prompt
    assert len(prompt) < 32_000


def test_status_groups_repetitive_mechanical_events_but_keeps_semantic_events_individual() -> None:
    from swaag.types import HistoryEvent

    events = []
    for sequence in range(1, 41):
        events.append(
            HistoryEvent(
                id=f"tok-{sequence}",
                sequence=sequence,
                session_id="session_target",
                timestamp=f"t{sequence}",
                type=(
                    "model_tokenize_requested"
                    if sequence % 2
                    else "model_tokenize_result"
                ),
                version=1,
                payload={"chars": sequence * 100, "tokens": sequence},
                hash=f"hash-{sequence}",
            )
        )
    events.append(
        HistoryEvent(
            id="msg-41",
            sequence=41,
            session_id="session_target",
            timestamp="t41",
            type="message_added",
            version=1,
            payload={"message": {"role": "user", "content": "keep this semantic marker"}},
            hash="hash-41",
        )
    )
    rows = AgentRuntime._communication_evidence_rows(events)
    assert len(rows) == 2
    assert [row["event_type"] for row in rows] == [
        "mechanical_event_index",
        "message_added",
    ]
    grouped = rows[0]["payload"]
    assert grouped["total_count"] == 40
    assert grouped["first_sequence"] == 1
    assert grouped["last_sequence"] == 40
    assert grouped["counts_by_event_type"] == {
        "model_tokenize_requested": 20,
        "model_tokenize_result": 20,
    }
    assert len(grouped["sequence_hash_index_sha256"]) == 64
    assert rows[-1]["payload"]["message"]["content"] == "keep this semantic marker"


def test_status_operational_payloads_keep_accounting_and_exact_reference_without_prompt_body() -> None:
    from swaag.types import HistoryEvent
    from swaag.utils import sha256_text, stable_json_dumps

    request = {
        "prompt": "GIANT-OPERATIONAL-PROMPT " * 5000,
        "n_predict": 768,
        "stream": True,
        "json_schema": {"type": "object", "properties": {str(i): {"type": "string"} for i in range(300)}},
    }
    event = HistoryEvent(
        id="request-event",
        sequence=45,
        session_id="session_target",
        timestamp="now",
        type="model_request_sent",
        version=1,
        payload={
            "kind": "action",
            "call_id": "model-call-1",
            "request_sha256": "request-sha",
            "request": request,
        },
        hash="event-hash",
    )
    row = AgentRuntime._communication_status_evidence_row(event)
    encoded_payload = stable_json_dumps(event.payload, indent=None)
    assert row["payload"]["exact_payload_reference"] == {
        "chars": len(encoded_payload),
        "sha256": sha256_text(encoded_payload),
    }
    assert row["payload"]["kind"] == "action"
    assert row["payload"]["call_id"] == "model-call-1"
    assert row["payload"]["payload_view"] == "reference_only_operational_event"
    serialized = stable_json_dumps(row, indent=None)
    assert "GIANT-OPERATIONAL-PROMPT" not in serialized
    assert len(serialized) < 1600
