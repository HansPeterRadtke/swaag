from __future__ import annotations

from swaag.context_builder import build_context
from swaag.project_state import build_project_state
from swaag.tokens import ConservativeEstimator
from swaag.types import FileView, Plan, PlanStep, ProjectState, SessionState


def test_project_state_tracks_multi_file_relationships() -> None:
    state = SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        file_views={
            "/repo/src/app.py": FileView(path="/repo/src/app.py", content="print(1)", last_operation="edit_applied", updated_at="t1"),
            "/repo/src/helpers.py": FileView(path="/repo/src/helpers.py", content="def helper(): pass", last_operation="file_read_for_edit", updated_at="t2"),
            "/repo/tests/test_app.py": FileView(path="/repo/tests/test_app.py", content="def test_app(): pass", last_operation="edit_applied", updated_at="t3"),
        },
    )

    project_state = build_project_state(state)

    assert project_state.files_seen == ["/repo/src/app.py", "/repo/src/helpers.py", "/repo/tests/test_app.py"]
    assert "/repo/src/app.py" in project_state.files_modified
    assert any(item.startswith("directory:/repo/src") for item in project_state.relationships)


def _make_plan(steps: list[PlanStep]) -> Plan:
    return Plan(
        plan_id="plan_test",
        goal="test goal",
        steps=steps,
        success_criteria="done",
        fallback_strategy="",
        status="active",
        created_at="t0",
        updated_at="t0",
    )


def _make_step(step_id: str, status: str, expected_output: str = "result") -> PlanStep:
    return PlanStep(
        step_id=step_id,
        title=f"Step {step_id}",
        goal=f"Do {step_id}",
        kind="tool",
        expected_tool="calculator",
        input_text="",
        expected_output=expected_output,
        done_condition="",
        success_criteria="done",
        status=status,
    )


def test_artifact_tracking_pending_steps_appear_in_expected_and_pending() -> None:
    """Pending plan steps must appear in both expected_artifacts and pending_artifacts."""
    state = SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        active_plan=_make_plan([
            _make_step("s1", "pending", "output_A"),
            _make_step("s2", "pending", "output_B"),
        ]),
    )

    ps = build_project_state(state)

    assert any("output_A" in item for item in ps.expected_artifacts)
    assert any("output_B" in item for item in ps.expected_artifacts)
    assert any("output_A" in item for item in ps.pending_artifacts)
    assert any("output_B" in item for item in ps.pending_artifacts)
    assert ps.completed_artifacts == []


def test_artifact_tracking_completed_steps_move_to_completed() -> None:
    """Completed plan steps must appear in expected_artifacts and completed_artifacts but not pending."""
    state = SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        active_plan=_make_plan([
            _make_step("s1", "completed", "result_X"),
            _make_step("s2", "pending", "result_Y"),
        ]),
    )

    ps = build_project_state(state)

    assert any("result_X" in item for item in ps.expected_artifacts)
    assert any("result_X" in item for item in ps.completed_artifacts)
    assert all("result_X" not in item for item in ps.pending_artifacts)
    assert any("result_Y" in item for item in ps.pending_artifacts)
    assert all("result_Y" not in item for item in ps.completed_artifacts)


def test_artifact_state_is_rendered_in_context(make_config) -> None:
    """Project state with pending artifacts must be included in the context bundle."""
    config = make_config()
    state = SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        active_plan=_make_plan([
            _make_step("s1", "completed", "written_file"),
            _make_step("s2", "pending", "test_report"),
        ]),
    )
    state.project_state = build_project_state(state)

    bundle = build_context(config, state, ConservativeEstimator(), goal="write and test")

    project_state_components = [c for c in bundle.components if c.name == "project_state"]
    assert project_state_components, "project_state component must be present when artifacts exist"
    combined_text = " ".join(c.text for c in project_state_components)
    assert "Pending artifacts" in combined_text
    assert "test_report" in combined_text
    assert "Completed artifacts" in combined_text
    assert "written_file" in combined_text
