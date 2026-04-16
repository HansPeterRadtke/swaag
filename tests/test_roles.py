from __future__ import annotations

from swaag.history import HistoryStore
from swaag.roles import build_agent_envelope, default_agent_roles


def test_default_agent_roles_cover_planning_execution_and_verification() -> None:
    roles = {role.role_name: role for role in default_agent_roles()}

    assert {"primary", "planner", "executor", "verifier"} <= set(roles)
    assert roles["planner"].can_plan is True
    assert roles["executor"].can_execute_tools is True
    assert roles["verifier"].can_verify is True


def test_agent_envelope_builds_protocol_message() -> None:
    envelope = build_agent_envelope("planner", "verifier", "review plan", "check dependencies")

    assert envelope.sender_role == "planner"
    assert envelope.recipient_role == "verifier"
    assert envelope.purpose == "review plan"
    assert envelope.content == "check dependencies"


def test_history_rebuild_restores_default_roles(tmp_path) -> None:
    store = HistoryStore(tmp_path, write_projections=False)
    state = store.create(config_fingerprint="cfg", model_base_url="http://x")
    rebuilt = store.rebuild_from_history(state.session_id, write_projections=False, prefer_checkpoint=False)

    assert rebuilt.agent_roles
    assert rebuilt.active_role == "primary"
