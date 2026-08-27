from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from swaag.heartbeat import WORKER_PHASES, heartbeat_payload
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime
from swaag.scheduler import WakeupStore


def test_heartbeat_payload_has_mechanical_phase_and_timestamp():
    payload = heartbeat_payload(phase="inference", detail="running action", active_kind="model", active_id="call-1")
    assert payload["phase"] == "inference"
    assert payload["heartbeat_at"]
    assert "inference" in WORKER_PHASES
    assert "structured_output" in WORKER_PHASES
    assert "response_presentation" in WORKER_PHASES
    assert "semantic_status" in WORKER_PHASES


def test_active_run_update_preserves_identity_and_updates_phase(make_config):
    config = make_config()
    store = HistoryStore(config.sessions.root)
    state = store.create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_name="heartbeat-test",
        session_name_source="explicit",
    )
    store.set_active_run(state.session_id, run_id="run-1", user_text="goal")
    before = store.read_active_run(state.session_id)
    updated = store.update_active_run(state.session_id, run_id="run-1", phase="tool_execution", detail="shell", active_kind="tool", active_id="shell_command")
    assert updated is not None
    assert updated["run_id"] == "run-1"
    assert updated["started_at"] == before["started_at"]
    assert updated["phase"] == "tool_execution"
    assert updated["active_id"] == "shell_command"
    assert json.loads(store.active_run_path(state.session_id).read_text())["heartbeat_at"]


def test_periodic_operation_heartbeat_advances_during_blocking_tool(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    runtime.history.set_active_run(state.session_id, run_id="run-1", user_text="goal")
    before = runtime.history.read_active_run(state.session_id)

    with runtime._periodic_heartbeat(
        state,
        phase="tool_execution",
        detail="running wait_seconds",
        active_kind="tool",
        active_id="wait_seconds",
        interval_seconds=0.01,
    ):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = runtime.history.read_active_run(state.session_id)
            if current is not None and current["heartbeat_at"] != before["heartbeat_at"]:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("periodic operation heartbeat did not advance")

    assert current["phase"] == "tool_execution"
    assert current["active_kind"] == "tool"
    assert current["active_id"] == "wait_seconds"
    assert current["detail"] == "running wait_seconds"


def test_mechanical_status_projects_pending_scheduled_wakeups(make_config):
    config = make_config()
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    wakeup = WakeupStore(config.sessions.root).schedule(
        session_id=state.session_id,
        reason="recheck deployment",
        duration="1 hour",
        now=datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc),
    )

    status = runtime.session_status_payload(state)

    assert status["next_wakeup_at"] == wakeup.wake_at
    assert status["scheduled_wakeups"] == [
        {
            "wakeup_id": wakeup.wakeup_id,
            "session_id": state.session_id,
            "wake_at": wakeup.wake_at,
            "reason": "recheck deployment",
            "status": "scheduled",
            "created_at": wakeup.created_at,
            "claimed_at": None,
            "delivered_at": None,
            "cancelled_at": None,
        }
    ]
