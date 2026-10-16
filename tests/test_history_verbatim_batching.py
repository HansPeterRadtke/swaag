import json

from swaag.runtime import AgentRuntime
from swaag.tokens import ExactTokenCounter
from swaag.types import Message
from tests.test_agent_action_loop import FakeModelClient


def test_fitting_verbatim_source_uses_one_call_with_room_for_all_selected_ids(make_config):
    selected_batches = []
    def select(payload):
        ids = [line.split()[0] for line in payload["prompt"].splitlines() if line.startswith("M001L")]
        selected_batches.append(ids)
        response = json.dumps({"selected_line_ids": ids, "reason": "All exact constraints are required."})
        assert payload["n_predict"] >= len(json.dumps({"selected_line_ids": ids, "reason": ""}, separators=(",", ":")))
        return response
    client = FakeModelClient([select for _ in range(128)])
    runtime = AgentRuntime(make_config(model__context_limit=16000), model_client=client,
        token_counter=ExactTokenCounter(len))
    state = runtime.create_or_load_session()
    lines = [f"Exact constraint {index:03d}: preserve marker VALUE_{index:03d}." for index in range(64)]
    message = Message(role="user", content="\n".join(lines), created_at="now")
    runtime._record_message(state, message)
    before = runtime.history.read_authoritative_messages(state.session_id)
    result, semantic_ok = runtime._select_history_verbatim_lines(state, [message])
    assert semantic_ok
    assert result == lines
    assert len(selected_batches) == 1, "Fitting exact units must not be split into fixed tiny model batches"
    assert runtime.history.read_authoritative_messages(state.session_id) == before


def test_overflow_splits_verbatim_source_without_losing_any_unit(make_config):
    def select(payload):
        ids = [line.split()[0] for line in payload["prompt"].splitlines() if line.startswith("M001L")]
        return json.dumps({"selected_line_ids": ids, "reason": "Preserve exact constraints."})
    client = FakeModelClient([select for _ in range(128)])
    runtime = AgentRuntime(make_config(model__context_limit=2500), model_client=client,
        token_counter=ExactTokenCounter(len))
    state = runtime.create_or_load_session()
    lines = [f"Do not publish marker {index:03d}; keep the exact value Ä_{index:03d}." for index in range(40)]
    result, semantic_ok = runtime._select_history_verbatim_lines(state, [Message(role="user", content="\n".join(lines), created_at="now")])
    assert semantic_ok
    assert result == lines
    assert len(client.requests) > 1
    assert all(p["n_predict"] < 2500 for p in client.requests)


def test_selector_failure_retains_every_exact_unit(make_config, monkeypatch):
    runtime = AgentRuntime(make_config(model__context_limit=16000), model_client=object(),
        token_counter=ExactTokenCounter(len))
    state = runtime.create_or_load_session()
    lines = [f"Exact retained evidence {index:03d}." for index in range(64)]
    def unavailable(*args, **kwargs):
        raise RuntimeError("selector unavailable")
    monkeypatch.setattr(runtime, "_execute_tool_semantic_call", unavailable)
    result, semantic_ok = runtime._select_history_verbatim_lines(state, [Message(role="user", content="\n".join(lines), created_at="now")])
    assert not semantic_ok
    assert result == lines


def test_desired_all_id_output_does_not_split_otherwise_fitting_input(make_config):
    client = FakeModelClient([json.dumps({"selected_line_ids": [], "reason": "No exact facts in routine progress."}) for _ in range(128)])
    runtime = AgentRuntime(make_config(model__context_limit=3500), model_client=client,
        token_counter=ExactTokenCounter(len))
    state = runtime.create_or_load_session()
    lines = [f"Routine {i}." for i in range(64)]
    selected, semantic_ok = runtime._select_history_verbatim_lines(state, [Message(role="assistant", content="\n".join(lines), created_at="now")])
    assert semantic_ok and selected == []
    assert len(client.requests) == 1, "Desired all-ID headroom must not force needless input splitting"
