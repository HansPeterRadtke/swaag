from __future__ import annotations

from pathlib import Path

import pytest

from swaag.environment.environment import AgentEnvironment
from swaag.grammar import history_analysis_contract
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime, OutputBudgetExhaustedError
from swaag.tokens import ConservativeEstimator
from swaag.tools.base import (
    SemanticCallContextOverflow,
    SemanticCallRequest,
    ToolContext,
    ToolValidationError,
)
from swaag.tools.history import HistorySearchTool
from swaag.tools.registry import ToolRegistry
from swaag.types import BudgetComponentReport, BudgetReport, Message, PromptComponent
from swaag.utils import utc_now_iso


def _state_with_history(config, session_id: str = "session_history_tools"):
    store = HistoryStore(config.sessions.root)
    state = store.create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=session_id,
    )
    store.record_event(
        state,
        "message_added",
        {"message": {"role": "user", "content": "Deployment codename is Blue Heron.", "created_at": utc_now_iso(), "name": None, "metadata": {}}},
    )
    store.record_event(
        state,
        "tool_result",
        {"tool_name": "echo", "raw_input": {"text": "artifact-marker-73"}, "validated_input": {"text": "artifact-marker-73"}, "output": {"text": "artifact-marker-73"}, "display_text": "artifact-marker-73"},
    )
    store.record_event(
        state,
        "message_added",
        {"message": {"role": "assistant", "content": "Remembered Blue Heron.", "created_at": utc_now_iso(), "name": None, "metadata": {}}},
    )
    return store, state


def test_history_search_tool_finds_ranked_exact_history(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config)
    registry = ToolRegistry()

    invocation, result = registry.dispatch(
        "history_search",
        {"query": '"Blue Heron"', "max_results": 5},
        config,
        state,
    )

    assert invocation.validated_input["query"] == '"Blue Heron"'
    assert result.output["session_id"] == state.session_id
    assert result.output["match_count"] >= 1
    assert any("Blue Heron" in item["preview"] for item in result.output["matches"])
    assert all("payload" not in item for item in result.output["matches"])
    assert all(item["hash"] for item in result.output["matches"])
    assert all(item["hash"] for item in result.output["source_event_references"])
    assert [event.event_type for event in result.generated_events] == ["history_retrieved"]


def test_history_search_tool_defaults_to_current_session(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, "session_current")
    other = store.create(config_fingerprint="cfg", model_base_url="http://model", session_id="session_other")
    store.record_event(other, "message_added", {"message": {"role": "user", "content": "Blue Heron only in other", "created_at": utc_now_iso(), "name": None, "metadata": {}}})

    _, result = ToolRegistry().dispatch("history_search", {"query": "artifact-marker-73"}, config, state)

    assert result.output["session_id"] == "session_current"
    assert result.output["match_count"] == 1


def test_history_search_tool_caps_results_to_config(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.history_search.max_results = 2
    store, state = _state_with_history(config)
    for index in range(6):
        store.record_event(state, "message_added", {"message": {"role": "user", "content": f"needle {index}", "created_at": utc_now_iso(), "name": None, "metadata": {}}})

    _, result = ToolRegistry().dispatch("history_search", {"query": "needle", "max_results": 99}, config, state)

    assert result.output["match_count"] == 2


def test_history_search_validation_rejects_empty_query(make_config) -> None:
    tool = ToolRegistry().get("history_search")
    with pytest.raises(ToolValidationError):
        tool.validate({"query": "   "})


def test_history_window_returns_exact_events(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config)

    _, result = ToolRegistry().dispatch(
        "history_window",
        {"start_sequence": 2, "limit": 2},
        config,
        state,
    )

    events = result.output["events"]
    assert [event["sequence"] for event in events] == [2, 3]
    assert events[0]["payload"]["message"]["content"] == "Deployment codename is Blue Heron."
    assert events[1]["payload"]["output"]["text"] == "artifact-marker-73"
    assert all(item["hash"] for item in result.output["source_event_references"])
    assert [event.event_type for event in result.generated_events] == ["history_window_read"]


def test_history_window_can_read_named_other_session(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, "session_primary")
    other = store.create(config_fingerprint="cfg", model_base_url="http://model", session_id="session_secondary", session_name="secondary")
    store.record_event(other, "message_added", {"message": {"role": "user", "content": "secondary fact", "created_at": utc_now_iso(), "name": None, "metadata": {}}})

    _, result = ToolRegistry().dispatch(
        "history_window",
        {"session_ref": "secondary", "start_sequence": 1, "limit": 2},
        config,
        state,
    )

    assert result.output["session_id"] == "session_secondary"
    assert result.output["events"][-1]["payload"]["message"]["content"] == "secondary fact"


def test_history_window_validation_bounds(make_config) -> None:
    tool = ToolRegistry().get("history_window")
    with pytest.raises(ToolValidationError):
        tool.validate({"start_sequence": 0})
    with pytest.raises(ToolValidationError):
        tool.validate({"start_sequence": 1, "limit": 21})


def test_history_tools_are_enabled_by_default(make_config) -> None:
    config = make_config()
    names = set(ToolRegistry().tool_names(config))
    assert {"history_search", "history_window"} <= names


def test_sqlite_wal_fts_is_durable_index_for_exact_history(make_config, tmp_path: Path) -> None:
    import sqlite3

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, session_id="session_sqlite_index")
    assert store.sqlite_history_path().exists()
    with sqlite3.connect(store.sqlite_history_path()) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        count = connection.execute("SELECT COUNT(*) FROM events WHERE session_id=?", (state.session_id,)).fetchone()[0]
        assert count == len(store.read_history(state.session_id))
        fts_count = connection.execute("SELECT COUNT(*) FROM events_fts WHERE session_id=?", (state.session_id,)).fetchone()[0]
        assert fts_count == count
    details = store.query_history_details(state.session_id, "Blue Heron", max_results=4)
    assert details["search_backend"] == "sqlite_fts5"
    assert details["matches"]
    assert any("Blue Heron" in item["preview"] for item in details["matches"])


def test_sqlite_control_priority_and_processed_idempotency(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint=config.config_fingerprint(), model_base_url=config.model.base_url)
    ordinary = store.enqueue_control_message(state.session_id, "please inspect logs", control_id="control_normal")
    pause = store.enqueue_control_message(state.session_id, "pause", control_id="control_pause")
    stop = store.enqueue_control_message(state.session_id, "stop now", control_id="control_stop")
    pending = store.list_pending_control_messages(state.session_id)
    assert [item["control_id"] for item in pending] == [stop["control_id"], pause["control_id"], ordinary["control_id"]]
    store.mark_control_message_processed(state.session_id, stop["control_id"])
    store.enqueue_control_message(state.session_id, "stop now", control_id=stop["control_id"])
    assert stop["control_id"] not in {item["control_id"] for item in store.list_pending_control_messages(state.session_id)}


def test_history_analyze_is_grounded_in_complete_exact_history(
    make_config, tmp_path: Path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, session_id="session_history_analyze")
    source_sequence = next(event.sequence for event in store.read_history(state.session_id) if "artifact-marker-73" in str(event.payload))

    def fake_semantic_call(request):
        prompt = "".join(component.text for component in request.components)
        assert request.kind == "history_analysis"
        assert request.contract.name == "history_analysis"
        assert str(source_sequence) in prompt
        assert "artifact-marker-73" in prompt
        assert "Deployment codename is Blue Heron" in prompt
        return {
            "goal_constraints": ["Recover the exact prior artifact marker."],
            "failure_evidence": ["The marker exists in a prior tool result."],
            "candidate_root_causes": [
                "The previous strategy did not retrieve the exact historical tool result."
            ],
            "source_sequences": [source_sequence],
            "wrong_strategy": "Relying on incomplete current context.",
            "recommended_strategy": "Retrieve the exact history event and use its value.",
            "uncertainties": [],
        }

    registry = ToolRegistry()
    invocation, result = registry.dispatch(
        "history_analyze",
        {"query": "Why did we miss artifact-marker-73?", "session_ref": state.session_id, "max_events": 1},
        config,
        state,
        semantic_call=fake_semantic_call,
    )
    assert invocation.tool_name == "history_analyze"
    assert result.output["source_sequences"] == [source_sequence]
    assert result.output["recommended_strategy"].startswith("Retrieve the exact history event")
    assert result.output["source_event_references"][0]["sequence"] == source_sequence
    assert result.output["source_event_references"][0]["hash"]
    assert [event.event_type for event in result.generated_events] == ["history_analyzed"]


def test_runtime_injects_semantic_service_into_model_backed_tools(
    make_config, monkeypatch
) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    source = runtime.history.record_event(
        state,
        "assistant_progress",
        {"action_index": 1, "assistant_text": "runtime semantic marker"},
    )

    def fake_semantic(call_state, request):
        assert call_state.session_id == state.session_id
        assert request.kind == "history_analysis"
        assert "runtime semantic marker" in "".join(
            component.text for component in request.components
        )
        return {
            "goal_constraints": ["Use the runtime semantic service."],
            "failure_evidence": ["The exact marker is present."],
            "candidate_root_causes": ["The marker had not been analyzed."],
            "source_sequences": [source.sequence],
            "wrong_strategy": "Bypass the runtime.",
            "recommended_strategy": "Use the central compiled call.",
            "uncertainties": [],
        }

    monkeypatch.setattr(runtime, "_execute_tool_semantic_call", fake_semantic)
    run = runtime.execute_tool_once(
        "history_analyze",
        {
            "query": "runtime semantic marker",
            "session_ref": None,
            "max_events": 4,
        },
        session_id=state.session_id,
    )

    assert run.error is None
    assert run.tool_result is not None
    assert run.tool_result.output["source_sequences"] == [source.sequence]
    event_types = [
        event.event_type for event in runtime.history.read_history(state.session_id)
    ]
    assert "history_analyzed" in event_types
    assert event_types[-2:] == ["tool_result", "message_added"]


def test_history_tools_accept_active_session_aliases(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint=config.config_fingerprint(), model_base_url=config.model.base_url, session_name="seed_23", session_name_source="explicit")
    store.record_event(
        state,
        "message_added",
        {"message": {"role": "user", "content": "alias-marker-91", "created_at": "2026-01-01T00:00:00+00:00", "name": None, "metadata": {}}},
    )
    env = AgentEnvironment(config, state)
    context = ToolContext(config=config, session_state=state, environment=env)
    registry = ToolRegistry()
    for session_ref in ("current", "seed_23", state.session_id):
        _invocation, result = registry.dispatch(
            "history_search",
            {"query": "alias-marker-91", "topic_hint": None, "session_ref": session_ref, "max_results": 4},
            config,
            state,
        )
        assert result.output["session_id"] == state.session_id
        assert result.output["matches"]


def test_history_search_exposes_search_backend_and_current_session_schema(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, session_id="session_backend_output")
    registry = ToolRegistry()
    _invocation, result = registry.dispatch(
        "history_search",
        {"query": "Blue Heron", "topic_hint": None, "session_ref": None, "max_results": 4},
        config,
        state,
    )
    assert result.output["search_backend"] == "sqlite_fts5"
    from swaag.tools.history import HistorySearchTool
    assert "session_ref=null" in HistorySearchTool.usage_guidance
    assert "never invent a session label" in HistorySearchTool.usage_guidance


def test_history_analyze_preserves_large_exact_history_before_context_compilation(
    make_config, tmp_path: Path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint=config.config_fingerprint(), model_base_url=config.model.base_url)
    huge = "X" * 200_000 + " root-cause-marker-77"
    store.record_event(state, "assistant_progress", {"action_index": 1, "assistant_text": huge})
    captured: dict[str, str] = {}

    def fake_semantic_call(request):
        prompt = "".join(component.text for component in request.components)
        captured["prompt"] = prompt
        sequence = next(event.sequence for event in store.read_history(state.session_id) if event.event_type == "assistant_progress")
        return {
            "goal_constraints": ["diagnose"],
            "failure_evidence": ["exact evidence"],
            "candidate_root_causes": ["cause"],
            "source_sequences": [sequence],
            "wrong_strategy": "old",
            "recommended_strategy": "new",
            "uncertainties": [],
        }

    registry = ToolRegistry()
    _invocation, result = registry.dispatch(
        "history_analyze",
        {"query": "root-cause-marker-77", "session_ref": None, "max_events": 12},
        config,
        state,
        semantic_call=fake_semantic_call,
    )
    assert result.output["candidate_root_causes"] == ["cause"]
    assert huge in captured["prompt"]
    assert "Exact durable history event" in captured["prompt"]


def test_history_analyze_projects_only_after_measured_overflow(make_config) -> None:
    config = make_config()
    store, state = _state_with_history(config, session_id="session_history_projection")
    source_event = next(
        event
        for event in store.read_history(state.session_id)
        if "artifact-marker-73" in str(event.payload)
    )
    analysis_calls = 0

    def fake_semantic_call(request):
        nonlocal analysis_calls
        prompt = "".join(component.text for component in request.components)
        if request.kind == "tool_result_projection":
            assert "artifact-marker-73" in prompt
            return {"projection": "The exact source contains artifact-marker-73."}
        analysis_calls += 1
        if analysis_calls == 1:
            assert "artifact-marker-73" in prompt
            event_component = next(
                component
                for component in request.components
                if component.name == "history_candidate_events"
            )
            raise SemanticCallContextOverflow(
                BudgetReport(
                    context_limit=500,
                    input_tokens=800,
                    reserved_response_tokens=100,
                    safety_margin_tokens=10,
                    required_tokens=910,
                    non_context_tokens=0,
                    fits=False,
                    exact=True,
                    breakdown=[
                        BudgetComponentReport(
                            name=event_component.name,
                            category="history",
                            tokens=700,
                            exact=True,
                            include_in_context=True,
                            optional=False,
                        )
                    ],
                )
            )
        assert "SEMANTIC PROJECTION CREATED ONLY AFTER MEASURED OVERFLOW" in prompt
        return {
            "goal_constraints": ["Recover the exact marker."],
            "failure_evidence": ["The projected source preserves the marker."],
            "candidate_root_causes": ["Prior context omitted the source."],
            "source_sequences": [source_event.sequence],
            "wrong_strategy": "Ignore measured context pressure.",
            "recommended_strategy": "Use the grounded semantic projection.",
            "uncertainties": [],
        }

    _invocation, result = ToolRegistry().dispatch(
        "history_analyze",
        {
            "query": "Why was artifact-marker-73 missed?",
            "session_ref": None,
            "max_events": 8,
        },
        config,
        state,
        semantic_call=fake_semantic_call,
    )

    assert analysis_calls == 2
    generated = result.generated_events[0].payload
    projection = generated["semantic_projections"][0]
    assert projection["source_event_start_sequence"] == 1
    assert projection["source_event_end_sequence"] >= source_event.sequence
    assert projection["source_event_count"] == len(
        store.read_history(state.session_id)
    )
    assert projection["source_sha256"]


def test_history_projection_attempts_have_a_total_bound(make_config) -> None:
    config = make_config(context__max_compaction_rounds=1)
    store = HistoryStore(config.sessions.root)
    state = store.create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
    )
    store.record_event(
        state,
        "assistant_progress",
        {"action_index": 1, "assistant_text": "Y" * 200_000 + " projection-bound-marker"},
    )
    projection_calls = 0

    def fake_semantic_call(request):
        nonlocal projection_calls
        if request.kind == "tool_result_projection":
            projection_calls += 1
            raise SemanticCallContextOverflow(
                BudgetReport(
                    context_limit=100,
                    input_tokens=200,
                    reserved_response_tokens=32,
                    safety_margin_tokens=8,
                    required_tokens=240,
                    non_context_tokens=0,
                    fits=False,
                    exact=True,
                    breakdown=[],
                )
            )
        candidate = next(
            component
            for component in request.components
            if component.name == "history_candidate_events"
        )
        raise SemanticCallContextOverflow(
            BudgetReport(
                context_limit=500,
                input_tokens=2_000,
                reserved_response_tokens=100,
                safety_margin_tokens=10,
                required_tokens=2_110,
                non_context_tokens=0,
                fits=False,
                exact=True,
                breakdown=[
                    BudgetComponentReport(
                        name=candidate.name,
                        category="history",
                        tokens=1_800,
                        exact=True,
                        include_in_context=True,
                        optional=False,
                    )
                ],
            )
        )

    with pytest.raises(ToolValidationError, match="bounded semantic segmentation"):
        ToolRegistry().dispatch(
            "history_analyze",
            {"query": "projection-bound-marker", "max_events": 1},
            config,
            state,
            semantic_call=fake_semantic_call,
        )
    assert projection_calls == 17


def test_runtime_semantic_service_compiles_named_context_and_refuses_overflow(
    make_config, monkeypatch
) -> None:
    config = make_config(
        model__context_limit=2_000,
        context__safety_margin_tokens=10,
    )
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ConservativeEstimator(chars_per_token=1.0),
    )
    state = runtime.create_or_load_session()
    captured = {}

    def fake_execute(call_state, prepared, **_kwargs):
        assert call_state.session_id == state.session_id
        captured["prepared"] = prepared
        return {"source_sequences": []}

    monkeypatch.setattr(runtime, "_execute_structured_call", fake_execute)
    request = SemanticCallRequest(
        kind="history_analysis",
        system_instruction="Analyze exact history without inventing evidence.",
        components=[
            PromptComponent(
                name="history_candidate_event_7",
                category="history",
                text="exact candidate marker",
            )
        ],
        contract=history_analysis_contract(),
        minimum_output_tokens=128,
        desired_output_tokens=700,
    )

    assert runtime._execute_tool_semantic_call(state, request) == {
        "source_sequences": []
    }
    prepared = captured["prepared"]
    assert prepared.assembly.kind == "history_analysis"
    assert any(
        item.name == "history_candidate_event_7"
        for item in prepared.assembly.components
    )
    assert prepared.report.fits
    assert prepared.report.reserved_response_tokens == 700

    overflow = SemanticCallRequest(
        kind="history_analysis",
        system_instruction=request.system_instruction,
        components=[
            PromptComponent(name="history_candidate_event_8", text="X" * 4_000)
        ],
        contract=request.contract,
        minimum_output_tokens=128,
    )
    with pytest.raises(SemanticCallContextOverflow) as exc_info:
        runtime._execute_tool_semantic_call(state, overflow)
    assert exc_info.value.report.required_tokens > exc_info.value.report.context_limit


def test_runtime_semantic_service_rebuilds_after_output_starvation(
    make_config, monkeypatch
) -> None:
    config = make_config(
        model__context_limit=4_000,
        model__max_retries=1,
        context__safety_margin_tokens=10,
    )
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ConservativeEstimator(chars_per_token=1.0),
    )
    state = runtime.create_or_load_session()
    reserved: list[int] = []

    def fake_execute(_state, prepared, **_kwargs):
        reserved.append(prepared.report.reserved_response_tokens)
        if len(reserved) == 1:
            raise OutputBudgetExhaustedError("length", reserved[-1])
        return {"source_sequences": []}

    monkeypatch.setattr(runtime, "_execute_structured_call", fake_execute)
    result = runtime._execute_tool_semantic_call(
        state,
        SemanticCallRequest(
            kind="history_analysis",
            system_instruction="Analyze exact history.",
            components=[PromptComponent(name="candidate", text="exact evidence")],
            contract=history_analysis_contract(),
            minimum_output_tokens=128,
        ),
    )

    assert result == {"source_sequences": []}
    assert len(reserved) == 2
    assert reserved[1] > reserved[0]
    repaired = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "budget_repaired"
    ]
    assert repaired[-1].payload["reason"] == "model_output_budget_exhausted"


def test_history_search_excludes_its_entire_current_action(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    runtime.history.record_event(
        state,
        "message_added",
        {"message": {"role": "assistant", "content": "Durable indexed-history marker: cobalt-history-fts-531.", "created_at": "2026-01-01T00:00:00+00:00", "name": None, "metadata": {}}},
    )
    marker_sequence = runtime.history.read_history(state.session_id)[-1].sequence
    runtime.history.record_event(
        state,
        "agent_action_selected",
        {"action_index": 1, "action": {"assistant_message": "", "tool_calls": [], "continue_loop": True, "status": {"situation": "Need durable indexed-history marker", "action": "Search history", "reason": "Need marker", "importance": "normal"}}, "occurrence": 1},
    )
    action_sequence = runtime.history.read_history(state.session_id)[-1].sequence
    runtime.history.record_event(
        state,
        "message_added",
        {"message": {"role": "assistant", "content": "Searching for durable indexed-history marker", "created_at": "2026-01-01T00:00:01+00:00", "name": None, "metadata": {"action_index": 1, "internal_action": True}}},
    )
    runtime.history.record_event(
        state,
        "tool_called",
        {"tool_name": "history_search", "tool_input": {"query": "durable indexed-history marker"}},
    )
    tool = HistorySearchTool()
    validated = tool.validate({"query": "durable indexed-history marker", "topic_hint": "history marker", "session_ref": None, "max_results": 1})
    result = tool.execute(validated, ToolContext(config=config, session_state=state, environment=AgentEnvironment(config, state)))
    match = result.output["matches"][0]
    assert match["event_type"] == "message_added"
    assert match["sequence"] == marker_sequence
    assert match["sequence"] < action_sequence
    assert "cobalt-history-fts-531" in match["preview"]
