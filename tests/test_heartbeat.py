from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from swaag.grammar import yes_no_contract
from swaag.heartbeat import WORKER_PHASES, WORKER_SUBSTATES, heartbeat_payload
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime
from swaag.scheduler import WakeupStore
from swaag.tokens import ConservativeEstimator
from swaag.types import PromptAssembly, PromptComponent, PromptMessageRange


def test_heartbeat_payload_has_mechanical_phase_and_timestamp():
    payload = heartbeat_payload(phase="inference", detail="running action", active_kind="model", active_id="call-1")
    assert payload["phase"] == "inference"
    assert payload["substate"] == "awaiting_result"
    assert payload["heartbeat_at"]
    assert "inference" in WORKER_PHASES
    assert "structured_output" in WORKER_PHASES
    assert "response_presentation" in WORKER_PHASES
    assert "semantic_status" in WORKER_PHASES
    assert "streaming" in WORKER_SUBSTATES["inference"]

    with pytest.raises(ValueError, match="unknown worker substate"):
        heartbeat_payload(
            phase="context_compilation",
            substate="streaming",
        )


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
    updated = store.update_active_run(
        state.session_id,
        run_id="run-1",
        phase="tool_execution",
        substate="preparing",
        detail="shell",
        active_kind="tool",
        active_id="shell_command",
        operation_kind="shell_command",
    )
    assert updated is not None
    assert updated["run_id"] == "run-1"
    assert updated["started_at"] == before["started_at"]
    assert updated["phase"] == "tool_execution"
    assert updated["substate"] == "preparing"
    assert updated["active_id"] == "shell_command"
    assert updated["operation_kind"] == "shell_command"
    assert updated["activity_sequence"] == 1
    pulse = store.update_active_run(
        state.session_id,
        run_id="run-1",
        phase="tool_execution",
        substate="preparing",
        detail="still preparing",
        operation_kind="shell_command",
    )
    assert pulse is not None
    assert pulse["activity_sequence"] == 1
    running = store.update_active_run(
        state.session_id,
        run_id="run-1",
        phase="tool_execution",
        substate="running",
        operation_kind="shell_command",
    )
    assert running is not None
    assert running["activity_sequence"] == 2
    assert json.loads(store.active_run_path(state.session_id).read_text())["heartbeat_at"]


def test_context_compiler_projects_exact_mechanical_substates(make_config):
    runtime = AgentRuntime(
        make_config(model__context_limit=8_000),
        model_client=object(),
        token_counter=ConservativeEstimator(),
    )
    state = runtime.create_or_load_session()
    runtime.history.set_active_run(
        state.session_id,
        run_id="run-1",
        user_text="compile a test context",
    )
    assembly = PromptAssembly(
        kind="test_semantic_operation",
        prompt_mode="lean",
        prompt_text="systemuser",
        components=[
            PromptComponent(name="system", text="system"),
            PromptComponent(name="user", text="user"),
        ],
        message_ranges=[
            PromptMessageRange(
                role="system",
                component_start=0,
                component_end=1,
            ),
            PromptMessageRange(
                role="user",
                component_start=1,
                component_end=2,
            ),
        ],
    )

    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(8_000, "test"),
    )

    active = runtime.history.read_active_run(state.session_id)
    assert compilation.report.fits is True
    assert active is not None
    assert active["phase"] == "context_compilation"
    assert active["substate"] == "context_fit"
    assert active["operation_kind"] == "test_semantic_operation"
    assert active["activity_sequence"] == 4


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
    assert current["substate"] == "running"
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
