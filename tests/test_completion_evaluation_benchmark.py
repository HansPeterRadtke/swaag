from __future__ import annotations

from swaag.benchmark.completion_evaluation import CASES, select_cases


def test_completion_evaluation_live_cases_cover_required_evidence_modes():
    by_id = {case.case_id: case for case in CASES}
    assert set(by_id) == {
        "current_turn_failure_blocks_completion",
        "exact_attachment_reexpanded_before_completion",
        "binary_attachment_requires_specialist",
        "preemption_replay_without_success_is_incomplete",
    }
    assert by_id["current_turn_failure_blocks_completion"].expected_complete is False
    assert by_id["exact_attachment_reexpanded_before_completion"].expected_complete is True
    assert by_id["binary_attachment_requires_specialist"].expected_complete is False
    assert by_id["preemption_replay_without_success_is_incomplete"].expected_complete is False
    assert len(select_cases([])) == 4


def test_completion_evaluation_preemption_fixture_uses_real_event_schema(tmp_path, make_config):
    from swaag.benchmark.completion_evaluation import _setup_case
    from swaag.runtime import AgentRuntime

    case = next(case for case in CASES if case.case_id == "preemption_replay_without_success_is_incomplete")
    config = make_config(sessions__root=tmp_path / "sessions")
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    evidence = _setup_case(runtime, state, case)
    events = runtime.history.read_history(state.session_id)
    preempted = next(event for event in events if event.event_type == "model_call_preempted")
    replayed = next(event for event in events if event.event_type == "model_call_replayed")
    assert preempted.payload["preemption_id"] == "preempt-live-1"
    assert preempted.payload["usage_evidence"]["backend_completion_tokens"] == 17
    assert replayed.payload["request"]["contract"] == "agent_action"
    assert evidence["preempted_sequence"] == preempted.sequence
    assert evidence["replayed_sequence"] == replayed.sequence
