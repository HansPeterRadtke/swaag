import pytest

from swaag.runtime import AgentRuntime
from swaag.tools.base import ToolValidationError
from swaag.tools.control import AgentStatusLookupTool


def test_status_lookup_defaults_to_exact_current_status_and_retains_full_history(make_config):
    runtime = AgentRuntime(make_config(tools__enabled=["agent_status_lookup"]))
    source = runtime.create_or_load_session()
    target = runtime.history.create(config_fingerprint="test", model_base_url="http://model")
    statuses = []
    for index in range(12):
        payload = {"action_index": index, "situation": "previous details " * 100 if index < 11 else "All checks completed.",
            "action": "inspect", "reason": "Keep exact status evidence.", "importance": "normal", "importance_rank": 1}
        statuses.append(payload)
        runtime.history.record_event(target, "agent_status", payload)
    def lookup(**options):
        result = runtime.execute_tool_once("agent_status_lookup", {"session_ref": target.session_id, **options}, session_id=source.session_id)
        assert result.error is None
        return result.tool_result.output
    current = lookup()
    assert "status_history" not in current
    assert current["latest_status"] == statuses[-1]
    reference = current["status_history_reference"]
    assert reference["session_id"] == target.session_id
    assert reference["event_type"] == "agent_status"
    assert reference["count"] == 12
    assert reference["first_sequence"] < reference["last_sequence"]
    complete = lookup(include_history=True)
    assert complete["status_history"] == statuses
    assert complete["latest_status"] == statuses[-1]
    assert complete["status_history_reference"] == reference
    assert lookup(include_history=None) == current
    assert lookup(include_history=False) == current


def test_empty_status_history_remains_explicit(make_config):
    runtime = AgentRuntime(make_config(tools__enabled=["agent_status_lookup"]))
    state = runtime.create_or_load_session()
    result = runtime.execute_tool_once("agent_status_lookup", {"session_ref": state.session_id}, session_id=state.session_id)
    assert result.error is None
    assert result.tool_result.output["latest_status"] is None
    reference = result.tool_result.output["status_history_reference"]
    assert reference["count"] == 0
    assert reference["first_sequence"] is None and reference["last_sequence"] is None


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}])
def test_history_selection_requires_boolean(value):
    with pytest.raises(ToolValidationError):
        AgentStatusLookupTool().validate({"session_ref": "target", "include_history": value})
