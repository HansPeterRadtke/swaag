from __future__ import annotations

import json

from swaag.heartbeat import WORKER_PHASES, heartbeat_payload
from swaag.history import HistoryStore


def test_heartbeat_payload_has_mechanical_phase_and_timestamp():
    payload = heartbeat_payload(phase="inference", detail="running action", active_kind="model", active_id="call-1")
    assert payload["phase"] == "inference"
    assert payload["heartbeat_at"]
    assert "inference" in WORKER_PHASES


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
