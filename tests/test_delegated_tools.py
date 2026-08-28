from __future__ import annotations

import sqlite3

import pytest

from swaag.delegated_tools import (
    DelegatedToolInputRequired,
    DelegatedToolResultInput,
    DelegatedToolStore,
    prepare_delegated_tool_spec,
)
from swaag.runtime import AgentRuntime
from swaag.workers import WorkerManager
from tests.test_agent_action_loop import FakeModelClient, _action


def _tool(name: str = "select_record"):
    return prepare_delegated_tool_spec(
        {
            "name": name,
            "description": "Select one record in the connected client.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
            "metadata": {"owner": "client"},
        }
    )


def test_delegated_tool_catalogs_are_exact_versioned_and_idempotent(tmp_path) -> None:
    store = DelegatedToolStore(tmp_path)
    first = store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    duplicate = store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    with pytest.raises(ValueError, match="different exact catalog"):
        store.bind_catalog(
            "session-1",
            source="ag_ui",
            external_context_id="thread-1",
            external_request_id="run-1",
            tools=[],
        )
    second = store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-2",
        tools=[],
    )

    assert duplicate == first
    assert first.revision == 1
    assert first.tools[0].metadata == {"owner": "client"}
    assert second.revision == 2
    assert second.tools == ()
    assert second.catalog_sha256 != first.catalog_sha256
    assert store.latest_catalog("session-1") == second


def test_delegated_tool_call_requires_schema_valid_result_and_exact_lineage(
    tmp_path,
) -> None:
    store = DelegatedToolStore(tmp_path)
    catalog = store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    with pytest.raises(ValueError, match="record_id"):
        store.request_call(
            "session-1",
            catalog_revision=catalog.revision,
            tool_name="select_record",
            arguments={},
        )

    call = store.request_call(
        "session-1",
        catalog_revision=catalog.revision,
        tool_name="select_record",
        arguments={"record_id": "record-7"},
    )
    assert store.pending_call("session-1") == call
    with pytest.raises(ValueError, match="already awaits"):
        store.request_call(
            "session-1",
            catalog_revision=catalog.revision,
            tool_name="select_record",
            arguments={"record_id": "record-8"},
        )

    result = DelegatedToolResultInput(
        message_id="result-message-1",
        call_id=call.call_id,
        content='{"selected":"record-7"}',
        error=None,
        metadata={"durationMs": 7},
    )
    resolved = store.resolve_call(
        "session-1",
        call.call_id,
        source="ag_ui",
        external_request_id="run-2",
        result=result,
    )
    duplicate = store.resolve_call(
        "session-1",
        call.call_id,
        source="ag_ui",
        external_request_id="run-2",
        result=result,
    )
    linked = store.link_history(
        call.call_id,
        event_type="tool_result",
        sequence=19,
        event_hash="history-hash-19",
    )

    assert resolved.status == "resolved"
    assert resolved.result_content == result.content
    assert resolved.result_metadata == result.metadata
    assert duplicate == resolved
    assert store.pending_call("session-1") is None
    assert linked.history_event_sequence == 19
    assert linked.history_event_hash == "history-hash-19"

    changed = DelegatedToolResultInput(
        message_id="result-message-1",
        call_id=call.call_id,
        content="different",
        error=None,
        metadata={"durationMs": 7},
    )
    with pytest.raises(ValueError, match="different exact result"):
        store.resolve_call(
            "session-1",
            call.call_id,
            source="ag_ui",
            external_request_id="run-2",
            result=changed,
        )
    assert store.verify_result_message("session-1", result) == linked
    with pytest.raises(ValueError, match="differs from durable exact result"):
        store.verify_result_message("session-1", changed)


def test_delegated_tool_store_records_and_checks_its_schema_version(tmp_path) -> None:
    store = DelegatedToolStore(tmp_path)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        connection.execute("PRAGMA user_version=2")

    with pytest.raises(RuntimeError, match="newer than supported"):
        DelegatedToolStore(tmp_path)


def test_delegated_tool_store_rejects_tampered_catalog_and_arguments(tmp_path) -> None:
    catalog_store = DelegatedToolStore(tmp_path / "catalog")
    catalog_store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    with sqlite3.connect(catalog_store.path) as connection:
        connection.execute(
            "UPDATE delegated_tool_catalogs SET tools_json='[]'"
        )
    with pytest.raises(RuntimeError, match="catalog hash verification failed"):
        catalog_store.latest_catalog("session-1")

    call_store = DelegatedToolStore(tmp_path / "call")
    catalog = call_store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    call = call_store.request_call(
        "session-1",
        catalog_revision=catalog.revision,
        tool_name="select_record",
        arguments={"record_id": "record-7"},
    )
    with sqlite3.connect(call_store.path) as connection:
        connection.execute(
            "UPDATE delegated_tool_calls SET arguments_json='{}' WHERE call_id=?",
            (call.call_id,),
        )
    with pytest.raises(RuntimeError, match="argument hash verification failed"):
        call_store.call(call.call_id)


def test_canceling_a_pending_delegated_call_is_durable(tmp_path) -> None:
    store = DelegatedToolStore(tmp_path)
    catalog = store.bind_catalog(
        "session-1",
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    call = store.request_call(
        "session-1",
        catalog_revision=catalog.revision,
        tool_name="select_record",
        arguments={"record_id": "record-7"},
    )

    canceled = store.cancel_pending("session-1", reason="worker canceled")

    assert canceled is not None
    assert canceled.call_id == call.call_id
    assert canceled.status == "canceled"
    assert canceled.result_error == "worker canceled"
    assert store.pending_call("session-1") is None


def test_delegated_request_and_partial_result_history_recover_idempotently(
    make_config, monkeypatch
) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    catalog = runtime.delegated_tools.bind_catalog(
        state.session_id,
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )

    with pytest.raises(DelegatedToolInputRequired, match="waiting for client result"):
        runtime._request_delegated_tool(
            state,
            catalog=catalog,
            spec=catalog.tools[0],
            arguments={"record_id": "record-7"},
        )
    call = runtime.delegated_tools.pending_call(state.session_id)
    assert call is not None
    with pytest.raises(DelegatedToolInputRequired, match="waiting for client result"):
        runtime._request_delegated_tool(
            state,
            catalog=catalog,
            spec=catalog.tools[0],
            arguments={"record_id": "record-7"},
        )
    assert runtime.delegated_tools.pending_call(state.session_id) == call

    original_record_message = runtime._record_message
    monkeypatch.setattr(
        runtime,
        "_record_message",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash gap")),
    )
    result = DelegatedToolResultInput(
        message_id="result-message-1",
        call_id=call.call_id,
        content='{"selected":"record-7"}',
        error=None,
        metadata={"durationMs": 7},
    )
    with pytest.raises(RuntimeError, match="crash gap"):
        runtime.accept_delegated_tool_result(
            state.session_id,
            call.call_id,
            source="ag_ui",
            external_request_id="run-2",
            result=result,
        )
    monkeypatch.setattr(runtime, "_record_message", original_record_message)

    recovered = runtime.accept_delegated_tool_result(
        state.session_id,
        call.call_id,
        source="ag_ui",
        external_request_id="run-2",
        result=result,
    )
    history = runtime.history.read_history(state.session_id)

    assert recovered.history_event_sequence is not None
    assert len(
        [
            event
            for event in history
            if event.event_type == "tool_result"
            and event.payload.get("call_id") == call.call_id
        ]
    ) == 1
    assert len(
        [
            event
            for event in history
            if event.event_type == "message_added"
            and event.payload["message"]["metadata"].get(
                "source_event_sequence"
            )
            == recovered.history_event_sequence
        ]
    ) == 1


def test_failed_delegated_result_is_exact_tool_error_evidence(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    catalog = runtime.delegated_tools.bind_catalog(
        state.session_id,
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    call = runtime.delegated_tools.request_call(
        state.session_id,
        catalog_revision=catalog.revision,
        tool_name="select_record",
        arguments={"record_id": "record-7"},
    )

    failed = runtime.accept_delegated_tool_result(
        state.session_id,
        call.call_id,
        source="ag_ui",
        external_request_id="run-2",
        result=DelegatedToolResultInput(
            message_id="result-message-1",
            call_id=call.call_id,
            content='{"visible":false}',
            error="client permission denied",
            metadata={"permission": "records.read"},
        ),
    )
    history = runtime.history.read_history(state.session_id)
    error = next(event for event in history if event.event_type == "tool_error")
    message = next(
        event.payload["message"]
        for event in history
        if event.event_type == "message_added"
        and event.payload["message"]["role"] == "tool"
    )

    assert failed.status == "failed"
    assert error.payload["call_id"] == call.call_id
    assert error.payload["error"] == "client permission denied"
    assert error.payload["evidence"]["content"] == '{"visible":false}'
    assert message["metadata"]["source_event_hash"] == error.hash
    assert "client permission denied" in message["content"]


def test_orphaned_delegated_wait_recovers_without_inference(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    manager = WorkerManager(runtime)
    worker = manager.create("Wait for a connected-client capability.")
    manager.store.transition(
        worker.worker_id,
        "queued",
        expected={"created"},
        event_type="worker_queued",
    )
    working = manager.store.transition(
        worker.worker_id,
        "working",
        expected={"queued"},
        increment_run_count=True,
        event_type="worker_started",
    )
    catalog = runtime.delegated_tools.bind_catalog(
        working.session_id,
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    call = runtime.delegated_tools.request_call(
        working.session_id,
        catalog_revision=catalog.revision,
        tool_name="select_record",
        arguments={"record_id": "record-7"},
    )

    recovered = manager.reconcile_orphans()
    history = runtime.history.read_history(working.session_id)
    manager.shutdown()

    assert recovered[0].status == "input_required"
    assert runtime.delegated_tools.pending_call(working.session_id) == call
    assert any(
        event.event_type == "delegated_tool_requested"
        and event.payload["call_id"] == call.call_id
        for event in history
    )
    assert any(
        event.event_type == "worker_delegated_tool_input_required"
        and event.payload.get("recovered_orphan") is True
        for event in manager.store.events(worker.worker_id)
    )


def test_worker_stages_delegated_tool_and_resumes_from_exact_client_result(
    make_config,
) -> None:
    exact_result = '{"selected":"record-7","visible":true}'
    observed = {}

    def finish(payload):
        observed["prompt"] = str(payload["prompt"])
        return _action(message="The client selected record 7.")

    client = FakeModelClient(
        [
            _action(
                tool_calls=[("load_tools", {"tool_names": ["select_record"]})],
                continue_loop=True,
            ),
            _action(
                tool_calls=[
                    ("select_record", {"record_id": "record-7"})
                ],
                continue_loop=True,
            ),
            finish,
        ]
    )
    config = make_config(model__context_limit=32_000)
    config.tools.staged_discovery = True
    runtime = AgentRuntime(config, model_client=client)
    manager = WorkerManager(runtime)
    worker = manager.create("Ask the connected client to select a record.")
    runtime.delegated_tools.bind_catalog(
        worker.session_id,
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="run-1",
        tools=[_tool()],
    )
    manager.start(worker.worker_id)
    waiting = manager.wait(worker.worker_id, timeout_seconds=10)
    call = runtime.delegated_tools.pending_call(worker.session_id)

    assert waiting.status == "input_required", waiting.error
    assert call is not None
    assert call.tool_name == "select_record"
    assert call.arguments == {"record_id": "record-7"}
    with pytest.raises(ValueError, match="awaits delegated tool result"):
        manager.message(worker.worker_id, "continue without the result")

    runtime.accept_delegated_tool_result(
        worker.session_id,
        call.call_id,
        source="ag_ui",
        external_request_id="run-2",
        result=DelegatedToolResultInput(
            message_id="client-result-1",
            call_id=call.call_id,
            content=exact_result,
            error=None,
            metadata={"durationMs": 9},
        ),
    )
    manager.message(
        worker.worker_id,
        "The connected client returned the delegated tool result.",
        source="ag_ui:run-2",
    )
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    history = runtime.history.read_history(worker.session_id)
    manager.shutdown()

    assert finished.status == "completed", finished.error
    assert finished.result == "The client selected record 7."
    assert "record-7" in observed["prompt"]
    assert "visible" in observed["prompt"]
    assert "delegated_tool_result" in observed["prompt"]
    called = [event for event in history if event.event_type == "tool_called"]
    results = [event for event in history if event.event_type == "tool_result"]
    delegated_call = next(event for event in called if event.payload.get("delegated"))
    delegated_result = next(
        event for event in results if event.payload.get("delegated")
    )
    assert delegated_call.payload["call_id"] == call.call_id
    assert delegated_result.payload["call_id"] == call.call_id
    assert delegated_result.payload["output"]["content"] == exact_result
    assert delegated_result.payload["result_message_id"] == "client-result-1"
    assert runtime.delegated_tools.call(call.call_id).history_event_hash == (
        delegated_result.hash
    )
