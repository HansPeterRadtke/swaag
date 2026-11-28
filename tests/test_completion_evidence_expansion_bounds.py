"""Bound repeated model-requested historical evidence without false success."""

import json

import pytest

from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient
from tests.test_completion_evaluation import _completed_action


@pytest.mark.parametrize("source_count,expected_calls,expected_expansions", [(4, 4, 3), (13, 1, 0)])
def test_completion_evidence_requests_stop_at_independent_bounds(
    make_config, monkeypatch, tmp_path, source_count, expected_calls, expected_expansions,
):
    config = make_config(model__context_limit=60000)
    config.sessions.root = tmp_path / "sessions"
    requests = []
    def ask_for_more(payload):
        assert payload["contract"] == "completion_evaluation"
        schema = payload["json_schema"]["properties"]
        assert "evidence_requests" in schema
        variants = schema["evidence_requests"]["items"]["anyOf"]
        choices = variants[0]["properties"]["source_id"]["enum"]
        # Each successive response requests a still-available immutable source.
        # A single oversized batch is also bounded before expansion.
        selected = choices if source_count > 12 else choices[:1]
        return json.dumps({
            "complete": False,
            "reason": "More source evidence is needed.",
            "remaining_work": ["Read source"],
            "evidence_requests": [
                {"source_kind": "history_event", "source_id": key,
                 "purpose": "Verify exact evidence", "literal_query": ""}
                for key in selected
            ],
        })
    client = FakeModelClient([ask_for_more] * 10)
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    inventory = [{"source_kind": "history_event", "source_id": f"event-{i}",
                  "sha256": "test-only", "source_event_references": []}
                 for i in range(source_count)]
    monkeypatch.setattr(runtime, "_completion_evidence_source_inventory", lambda *a, **k: inventory)
    def expand(_state, *, source, **kwargs):
        requests.append(source["source_id"])
        return {**source, "text": "Evidence retained exactly in durable history", "requested_purpose": "Verify"}
    monkeypatch.setattr(runtime, "_reexpand_completion_evidence_source", expand)
    result = runtime._evaluate_completion(
        state, original_request="Verify all actual sources first.",
        selected_action=_completed_action(), tool_results=[])
    assert result["complete"] is False
    assert result["evidence_expansion_budget_exhausted"] is True
    assert len(client.requests) == expected_calls
    assert len(requests) == expected_expansions
    assert len(set(requests)) == len(requests)
    events = runtime.history.read_history(state.session_id)
    assert sum(e.event_type == "completion_evidence_reexpanded" for e in events) == expected_expansions
    assert [e.payload["complete"] for e in events if e.event_type == "completion_evaluated"] == [False]
