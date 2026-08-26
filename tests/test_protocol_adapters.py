from __future__ import annotations

from swaag.protocol_adapters import (
    A2AProjectionAdapter,
    AgUiProjectionAdapter,
    OpenWebUiProjectionAdapter,
)
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from swaag.task_api import TaskApi
from swaag.workers import WorkerEvent, WorkerManager, WorkerRecord


def _record(*, status: str = "completed", archived_at: str | None = None) -> WorkerRecord:
    return WorkerRecord(
        worker_id="worker_1",
        session_id="session_1",
        objective="produce result",
        status=status,
        created_at="2026-08-26T10:00:00+00:00",
        updated_at="2026-08-26T10:01:00+00:00",
        started_at="2026-08-26T10:00:01+00:00",
        completed_at="2026-08-26T10:01:00+00:00",
        archived_at=archived_at,
        result="exact result" if status in {"completed", "input_required"} else None,
        error="failed exactly" if status == "failed" else None,
        run_count=1,
    )


def _event(
    event_type: str, sequence: int = 2, payload: dict[str, object] | None = None
) -> WorkerEvent:
    return WorkerEvent(
        event_id=f"event_{sequence}",
        worker_id="worker_1",
        sequence=sequence,
        timestamp="2026-08-26T10:01:00+00:00",
        event_type=event_type,
        payload={"to_status": event_type.removeprefix("worker_"), **dict(payload or {})},
    )


def test_task_api_is_transport_neutral_and_cursor_based(make_config) -> None:
    manager = WorkerManager(AgentRuntime(make_config(), model_client=object()))
    api = TaskApi(manager)

    created = api.execute("create", {"objective": "inspect durable state", "name": "worker-api"})
    worker_id = created["worker"]["worker_id"]
    listed = api.execute("list")
    events = api.execute("events", {"worker_id": worker_id, "after_sequence": 0})
    canceled = api.execute("cancel", {"worker_id": worker_id, "reason": "not needed"})
    first_page = api.execute(
        "events", {"worker_id": worker_id, "after_sequence": 0, "limit": 1}
    )
    second_page = api.execute(
        "events",
        {
            "worker_id": worker_id,
            "after_sequence": first_page["next_sequence"],
            "limit": 100,
        },
    )
    manager.shutdown()

    assert created["version"] == "swaag.task.v1"
    assert listed["workers"][0]["worker_id"] == worker_id
    assert events["events"][0]["event_type"] == "worker_created"
    assert events["next_sequence"] == 1
    assert first_page["has_more"] is True
    assert second_page["events"]
    assert second_page["has_more"] is False
    assert canceled["worker"]["status"] == "canceled"


def test_task_api_event_wait_is_resumable_and_reports_timeout_or_terminal(make_config) -> None:
    manager = WorkerManager(AgentRuntime(make_config(), model_client=object()))
    api = TaskApi(manager)
    created = api.execute("create", {"objective": "wait for durable events"})
    worker_id = created["worker"]["worker_id"]
    cursor = api.execute("events", {"worker_id": worker_id})["next_sequence"]

    timed_out = api.execute(
        "events.wait",
        {
            "worker_id": worker_id,
            "after_sequence": cursor,
            "timeout_seconds": 0.01,
        },
    )
    api.execute("cancel", {"worker_id": worker_id})
    terminal = api.execute(
        "events.wait",
        {"worker_id": worker_id, "after_sequence": cursor, "timeout_seconds": 0},
    )
    manager.shutdown()

    assert timed_out["events"] == []
    assert timed_out["timed_out"] is True
    assert timed_out["terminal"] is False
    assert terminal["events"]
    assert terminal["timed_out"] is False
    assert terminal["terminal"] is True


def test_a2a_projection_preserves_internal_task_state_and_archive_metadata() -> None:
    adapter = A2AProjectionAdapter()
    task = adapter.task(_record(archived_at="2026-08-26T11:00:00+00:00"))
    waiting = adapter.task(_record(status="input_required"))

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["contextId"] == "session_1"
    assert task["artifacts"][0]["parts"] == [{"text": "exact result"}]
    assert task["metadata"]["archivedAt"]
    assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


def test_a2a_projects_durable_status_and_artifact_updates() -> None:
    updates = A2AProjectionAdapter().updates(
        _record(),
        [
            _event("worker_started", 2, {"to_status": "working"}),
            _event(
                "worker_completed",
                3,
                {"to_status": "completed", "result": "historical exact result"},
            ),
        ],
    )

    assert updates[0]["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING"
    assert updates[0]["statusUpdate"]["final"] is False
    assert updates[1]["artifactUpdate"]["artifact"]["parts"] == [
        {"text": "historical exact result"}
    ]
    assert updates[2]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert updates[2]["statusUpdate"]["final"] is True


def test_a2a_message_parser_preserves_text_data_and_raw_attachments() -> None:
    message = A2AProjectionAdapter().user_message(
        {
            "message": {
                "role": "ROLE_USER",
                "messageId": "message_1",
                "parts": [
                    {"text": "Inspect the attached facts."},
                    {"data": {"priority": "high"}},
                    {
                        "raw": "ZXhhY3QgYnl0ZXM=",
                        "filename": "facts.txt",
                        "mediaType": "text/plain",
                    },
                ],
            },
            "configuration": {"returnImmediately": True},
        }
    )

    assert message.text == 'Inspect the attached facts.\n\n{"priority":"high"}'
    assert message.attachments == (
        {
            "original_name": "facts.txt",
            "media_type": "text/plain",
            "content_base64": "ZXhhY3QgYnl0ZXM=",
        },
    )
    assert message.return_immediately is True


def test_a2a_send_creates_started_task_with_raw_attachment(
    make_config, monkeypatch
) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))

    def queue_without_executor(worker_id: str):
        return service.workers.store.transition(
            worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )

    monkeypatch.setattr(service.workers, "start", queue_without_executor)
    response = service.protocol_projection(
        "a2a",
        "send",
        {
            "message": {
                "role": "ROLE_USER",
                "messageId": "message_2",
                "parts": [
                    {"text": "Inspect the exact attachment."},
                    {
                        "raw": "ZXhhY3QgYnl0ZXM=",
                        "filename": "facts.txt",
                        "mediaType": "text/plain",
                    },
                ],
            },
            "configuration": {"returnImmediately": True},
        },
    )
    worker_id = response["task"]["id"]
    record = service.workers.store.get(worker_id)
    attachments = service.workers.attachments(worker_id)
    service.workers.shutdown()

    assert response["task"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    assert record.objective == "Inspect the exact attachment."
    assert attachments[0].original_name == "facts.txt"
    assert attachments[0].sha256


def test_a2a_send_blocks_by_default_until_worker_interrupt_or_terminal(
    make_config, monkeypatch
) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    observed: dict[str, object] = {}

    def queue_without_executor(worker_id: str):
        return service.workers.store.transition(
            worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )

    def complete_wait(worker_id: str, *, timeout_seconds):
        observed["timeout_seconds"] = timeout_seconds
        return service.workers.store.transition(
            worker_id,
            "completed",
            expected={"queued"},
            result="blocking exact result",
            event_type="worker_completed",
        )

    monkeypatch.setattr(service.workers, "start", queue_without_executor)
    monkeypatch.setattr(service.workers, "wait", complete_wait)
    response = service.protocol_projection(
        "a2a",
        "send",
        {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "Complete before returning."}],
            }
        },
    )
    service.workers.shutdown()

    assert observed["timeout_seconds"] is None
    assert response["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert response["task"]["artifacts"][0]["parts"] == [
        {"text": "blocking exact result"}
    ]


def test_a2a_followup_rejects_mismatched_task_and_context(make_config) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    worker = service.workers.create("existing task")

    try:
        service.protocol_projection(
            "a2a",
            "send",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": worker.worker_id,
                    "contextId": "different-context",
                    "parts": [{"text": "Continue."}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    except ValueError as exc:
        assert "different contexts" in str(exc)
    else:
        raise AssertionError("mismatched A2A context was accepted")
    finally:
        service.workers.shutdown()


def test_a2a_list_uses_stable_cursor_pagination_and_state_filter(make_config) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    workers = [service.workers.create(f"task {index}") for index in range(3)]
    canceled = service.workers.cancel(workers[0].worker_id)

    first = service.protocol_projection("a2a", "list", {"pageSize": 2})
    second = service.protocol_projection(
        "a2a",
        "list",
        {"pageSize": 2, "pageToken": first["nextPageToken"]},
    )
    filtered = service.protocol_projection(
        "a2a",
        "list",
        {"status": "TASK_STATE_CANCELED"},
    )
    service.workers.shutdown()

    first_ids = {task["id"] for task in first["tasks"]}
    second_ids = {task["id"] for task in second["tasks"]}
    assert first["totalSize"] == 3
    assert first["nextPageToken"]
    assert len(first_ids) == 2
    assert len(second_ids) == 1
    assert not first_ids & second_ids
    assert second["nextPageToken"] == ""
    assert [task["id"] for task in filtered["tasks"]] == [canceled.worker_id]


def test_ag_ui_projection_uses_stable_run_message_and_terminal_event_shapes() -> None:
    adapter = AgUiProjectionAdapter()
    completed = adapter.events(_record(), [_event("worker_started", 1), _event("worker_completed", 2)])
    canceled = adapter.events(_record(status="canceled"), [_event("worker_canceled")])

    assert [item["type"] for item in completed] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert completed[2]["delta"] == "exact result"
    assert completed[0]["runId"] == completed[-1]["runId"] == "worker_1-run-1"
    assert completed[-1]["outcome"] == {"type": "success"}
    assert [item["type"] for item in canceled] == ["CUSTOM", "RUN_ERROR"]
    assert canceled[-1]["code"] == "SWAAG_WORKER_CANCELED"

    historical = adapter.events(
        _record(),
        [_event("worker_completed", 8, {"run_count": 2, "result": "prior exact result"})],
    )
    assert historical[1]["delta"] == "prior exact result"
    assert historical[-1]["result"] == "prior exact result"
    assert historical[-1]["runId"] == "worker_1-run-2"


def test_ag_ui_projects_input_required_as_a_resumable_interrupt() -> None:
    waiting = AgUiProjectionAdapter().events(
        _record(status="input_required"), [_event("worker_input_required")]
    )

    assert [item["type"] for item in waiting] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "CUSTOM",
        "RUN_FINISHED",
    ]
    finished = waiting[-1]
    assert finished["runId"] == "worker_1-run-1"
    assert finished["outcome"]["type"] == "interrupt"
    assert finished["outcome"]["interrupts"] == [
        {
            "id": "worker_1-input-2",
            "reason": "human_input",
            "message": "exact result",
            "metadata": {
                "swaagWorkerId": "worker_1",
                "swaagEventId": "event_2",
            },
        }
    ]


def test_ag_ui_projects_canonical_tool_and_status_history_events() -> None:
    def linked(
        source_type: str,
        sequence: int,
        payload: dict[str, object],
    ) -> WorkerEvent:
        return _event(
            "worker_history_event",
            sequence,
            {
                "canonical_event": {
                    "id": f"history_{sequence}",
                    "sequence": sequence + 10,
                    "hash": f"hash_{sequence}",
                    "type": source_type,
                    "payload": payload,
                }
            },
        )

    events = AgUiProjectionAdapter().events(
        _record(status="working"),
        [
            linked(
                "tool_called",
                3,
                {"call_id": "call_1", "tool_name": "reader", "tool_input": {"path": "a.txt"}},
            ),
            linked(
                "tool_result",
                4,
                {"call_id": "call_1", "tool_name": "reader", "output": {"text": "exact"}},
            ),
            linked(
                "agent_status",
                5,
                {"situation": "Reading evidence", "importance": "normal"},
            ),
        ],
    )

    assert [item["type"] for item in events] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "ACTIVITY_SNAPSHOT",
    ]
    assert events[0]["toolCallId"] == "call_1"
    assert events[1]["delta"] == '{"path":"a.txt"}'
    assert events[3]["content"] == '{"text":"exact"}'
    assert events[4]["content"]["situation"] == "Reading evidence"
    assert events[4]["metadata"]["swaagHistoryHash"] == "hash_5"


def test_open_webui_projection_uses_persisted_status_and_final_return_channel() -> None:
    response = OpenWebUiProjectionAdapter().response(_record(status="input_required"))

    assert response["return"] == "exact result"
    assert response["events"] == [
        {
            "type": "status",
            "data": {
                "description": "exact result",
                "done": False,
                "hidden": False,
            },
        }
    ]
    assert all(event["type"] not in {"input", "confirmation"} for event in response["events"])


def test_critical_failure_remains_retrievable_by_every_projection_after_restart(
    make_config,
) -> None:
    config = make_config()
    manager = WorkerManager(AgentRuntime(config, model_client=object()))
    worker = manager.create("preserve a critical failure")
    manager.store.transition(
        worker.worker_id,
        "failed",
        expected={"created"},
        error="CriticalBackendError: exact durable failure",
        event_type="worker_failed",
    )
    manager.shutdown()

    restarted = WorkerManager(AgentRuntime(config, model_client=object()))
    record = restarted.store.get(worker.worker_id)
    inspection = TaskApi(restarted).execute("get", {"worker_id": worker.worker_id})
    events = restarted.events(worker.worker_id)
    a2a = A2AProjectionAdapter().task(record)
    ag_ui = AgUiProjectionAdapter().events(record, events)
    open_webui = OpenWebUiProjectionAdapter().response(record)
    restarted.shutdown()

    assert inspection["error"] == "CriticalBackendError: exact durable failure"
    assert a2a["status"]["message"]["parts"] == [
        {"text": "CriticalBackendError: exact durable failure"}
    ]
    assert next(item for item in ag_ui if item["type"] == "RUN_ERROR")[
        "message"
    ] == "CriticalBackendError: exact durable failure"
    assert open_webui["return"] == "CriticalBackendError: exact durable failure"
