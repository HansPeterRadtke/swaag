from __future__ import annotations

import time

import pytest

from swaag.communication import CommunicationService
from swaag.protocol_adapters import AgUiProjectionAdapter
from swaag.runtime import AgentRuntime
from swaag.shared_state import shared_state_event_payload
from swaag.tools.base import ToolValidationError
from swaag.tools.shared_state import SharedStateTool


def test_shared_state_tool_is_session_bound_durable_and_projectable(
    make_config,
) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    service = CommunicationService(runtime)
    worker = service.workers.create("Update the connected interface state.")
    service.store.set_protocol_worker("ag_ui", "thread-1", worker.worker_id)
    baseline = service.store.bind_protocol_state(
        "ag_ui",
        "thread-1",
        "run-1",
        state={"selection": {"id": "record-1"}, "steps": []},
        client_supplied=True,
    )
    service.store.record_protocol_message(
        "ag_ui", "run-1", "thread-1", worker.worker_id
    )
    service._bind_ag_ui_shared_state(worker, "thread-1")

    unbound_state = runtime.create_or_load_session()
    assert "shared_state" not in {
        name
        for name, _description, _guidance in runtime.tools.capability_index(
            config,
            runtime_capabilities=runtime.tool_runtime_capabilities(
                unbound_state.session_id
            ),
        )
    }
    assert "shared_state" in {
        name
        for name, _description, _guidance in runtime.tools.capability_index(
            config,
            runtime_capabilities=runtime.tool_runtime_capabilities(worker.session_id),
        )
    }

    read = runtime.execute_tool_once(
        "shared_state",
        {
            "operation": "read",
            "base_revision": None,
            "base_state_sha256": None,
            "patch": None,
        },
        session_id=worker.session_id,
    )
    assert read.error is None
    assert read.tool_result is not None
    assert read.tool_result.output["revision"] == baseline.revision
    assert read.tool_result.output["state"] == baseline.state

    patched = runtime.execute_tool_once(
        "shared_state",
        {
            "operation": "patch",
            "base_revision": baseline.revision,
            "base_state_sha256": baseline.state_sha256,
            "patch": [
                {
                    "op": "replace",
                    "path": "/selection/id",
                    "value_json": '"record-2"',
                },
                {
                    "op": "add",
                    "path": "/steps/-",
                    "value_json": '{"done":true,"name":"verified"}',
                },
            ],
        },
        session_id=worker.session_id,
    )
    assert patched.error is None
    assert patched.tool_result is not None
    assert patched.tool_result.output["state"] == {
        "selection": {"id": "record-2"},
        "steps": [{"done": True, "name": "verified"}],
    }

    history = runtime.history.read_history(worker.session_id)
    state_event = next(
        event for event in history if event.event_type == "shared_state_updated"
    )
    latest = service.store.latest_protocol_state("ag_ui", "thread-1")
    assert latest is not None
    assert latest.source_call_id == state_event.payload["source_call_id"]
    assert latest.history_event_sequence == state_event.sequence
    assert latest.history_event_hash == state_event.hash

    projected = AgUiProjectionAdapter().events(
        worker,
        service.workers.events(worker.worker_id),
        thread_id="thread-1",
        run_id="run-1",
    )
    delta = next(event for event in projected if event["type"] == "STATE_DELTA")
    assert delta["delta"] == list(latest.patch or ())
    assert delta["metadata"]["swaagStateRevision"] == latest.revision
    assert delta["metadata"]["swaagHistoryHash"] == state_event.hash
    projected_after_new_baseline = AgUiProjectionAdapter().events(
        worker,
        service.workers.events(worker.worker_id),
        thread_id="thread-1",
        run_id="run-2",
        state_baseline_revision=latest.revision,
    )
    assert not any(
        event["type"] == "STATE_DELTA" for event in projected_after_new_baseline
    )

    stale = runtime.execute_tool_once(
        "shared_state",
        {
            "operation": "patch",
            "base_revision": baseline.revision,
            "base_state_sha256": baseline.state_sha256,
            "patch": [
                {"op": "add", "path": "/stale", "value_json": "true"}
            ],
        },
        session_id=worker.session_id,
    )
    assert stale.tool_result is None
    assert stale.error is not None
    assert stale.error["error_type"] == "SharedStateConflictError"
    assert stale.error["evidence"]["current"]["revision"] == latest.revision
    service.workers.shutdown()

    restored_runtime = AgentRuntime(config, model_client=object())
    restored = CommunicationService(restored_runtime)
    restored_read = restored_runtime.execute_tool_once(
        "shared_state",
        {
            "operation": "read",
            "base_revision": None,
            "base_state_sha256": None,
            "patch": None,
        },
        session_id=worker.session_id,
    )
    assert restored_read.error is None
    assert restored_read.tool_result is not None
    assert restored_read.tool_result.output["state"] == latest.state
    restored.workers.shutdown()


def test_shared_state_tool_rejects_non_strict_json_values() -> None:
    tool = SharedStateTool()
    with pytest.raises(ToolValidationError, match="not strict JSON"):
        tool.validate(
            {
                "operation": "patch",
                "base_revision": 1,
                "base_state_sha256": "a" * 64,
                "patch": [
                    {
                        "op": "add",
                        "path": "/value",
                        "value_json": "NaN",
                    }
                ],
            }
        )


def test_ag_ui_run_binds_state_channel_before_asynchronous_worker_execution(
    make_config,
) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    service = CommunicationService(runtime)

    def complete_with_state_update(worker_id: str) -> None:
        working = service.workers.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            increment_run_count=True,
            event_type="worker_started",
        )
        read = runtime.execute_tool_once(
            "shared_state",
            {
                "operation": "read",
                "base_revision": None,
                "base_state_sha256": None,
                "patch": None,
            },
            session_id=working.session_id,
        )
        assert read.error is None
        assert read.tool_result is not None
        baseline = read.tool_result.output
        update = runtime.execute_tool_once(
            "shared_state",
            {
                "operation": "patch",
                "base_revision": baseline["revision"],
                "base_state_sha256": baseline["state_sha256"],
                "patch": [
                    {
                        "op": "add",
                        "path": "/progress",
                        "value_json": '{"phase":"verified"}',
                    }
                ],
            },
            session_id=working.session_id,
        )
        assert update.error is None
        service.workers._sync_history_events(working)
        service.workers.store.transition(
            worker_id,
            "completed",
            expected={"working"},
            result="done",
            event_type="worker_completed",
        )

    service.workers._run_worker = complete_with_state_update  # type: ignore[method-assign]
    run = AgUiProjectionAdapter().user_run(
        {
            "threadId": "thread-async",
            "runId": "run-async",
            "state": {"selection": "record-1"},
            "messages": [
                {"id": "user-1", "role": "user", "content": "Update progress."}
            ],
            "tools": [],
            "context": [],
            "forwardedProps": {},
        }
    )
    record, start_sequence, _end, duplicate, snapshot = service._ag_ui_begin(run)

    deadline = time.monotonic() + 5
    while service.workers.store.get(record.worker_id).status != "completed":
        if time.monotonic() >= deadline:
            pytest.fail("deterministic AG-UI worker did not complete")
        time.sleep(0.01)
    events = AgUiProjectionAdapter().events(
        service.workers.store.get(record.worker_id),
        service.workers.events(
            record.worker_id, after_sequence=start_sequence
        ),
        thread_id=run.thread_id,
        run_id=run.run_id,
    )
    service.workers.shutdown()

    assert duplicate is False
    assert snapshot.state == {"selection": "record-1"}
    event_types = [event["type"] for event in events]
    assert "STATE_DELTA" in event_types
    assert event_types.index("STATE_DELTA") < event_types.index("RUN_FINISHED")


@pytest.mark.parametrize("event_recorded_before_restart", [False, True])
def test_service_restart_recovers_unlinked_shared_state_history(
    make_config,
    event_recorded_before_restart: bool,
) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    service = CommunicationService(runtime)
    worker = service.workers.create("Keep connected state durable.")
    service.store.set_protocol_worker("ag_ui", "thread-recovery", worker.worker_id)
    baseline = service.store.bind_protocol_state(
        "ag_ui",
        "thread-recovery",
        "run-recovery-1",
        state={"value": 1},
        client_supplied=True,
    )
    service.store.record_protocol_message(
        "ag_ui",
        "run-recovery-1",
        "thread-recovery",
        worker.worker_id,
    )
    update = service.store.apply_protocol_state_patch(
        "ag_ui",
        "thread-recovery",
        source_session_id=worker.session_id,
        source_call_id="call-recovery",
        base_revision=baseline.revision,
        base_state_sha256=baseline.state_sha256,
        patch=[{"op": "replace", "path": "/value", "value": 2}],
    )
    if event_recorded_before_restart:
        state = runtime.history.rebuild_from_history(
            worker.session_id, write_projections=False
        )
        existing_event = runtime.history.record_event(
            state,
            "shared_state_updated",
            shared_state_event_payload(update),
        )
    else:
        existing_event = None

    # A later accepted client baseline must not hide an older unlinked update.
    inherited = service.store.bind_protocol_state(
        "ag_ui",
        "thread-recovery",
        "run-recovery-2",
        state=None,
        client_supplied=False,
    )
    service.store.record_protocol_message(
        "ag_ui",
        "run-recovery-2",
        "thread-recovery",
        worker.worker_id,
    )
    assert inherited.state == update.state
    service.workers.shutdown()

    restored_runtime = AgentRuntime(config, model_client=object())
    restored = CommunicationService(restored_runtime)
    recovered = restored.store.protocol_state_for_agent_call(
        "ag_ui", "call-recovery"
    )
    assert recovered is not None
    assert recovered.history_event_sequence is not None
    assert recovered.history_event_hash is not None
    events = [
        event
        for event in restored_runtime.history.read_history(worker.session_id)
        if event.event_type == "shared_state_updated"
        and event.payload.get("source_call_id") == "call-recovery"
    ]
    assert len(events) == 1
    assert events[0].payload == shared_state_event_payload(recovered)
    assert recovered.history_event_sequence == events[0].sequence
    assert recovered.history_event_hash == events[0].hash
    if existing_event is not None:
        assert events[0].sequence == existing_event.sequence
        assert events[0].hash == existing_event.hash
    restored.workers.shutdown()

    # Reconciliation is idempotent across repeated process restarts.
    repeated_runtime = AgentRuntime(config, model_client=object())
    repeated = CommunicationService(repeated_runtime)
    repeated_events = [
        event
        for event in repeated_runtime.history.read_history(worker.session_id)
        if event.event_type == "shared_state_updated"
        and event.payload.get("source_call_id") == "call-recovery"
    ]
    assert len(repeated_events) == 1
    repeated.workers.shutdown()
