"""Offline fixture/oracle checks; these cannot certify a model's task success."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from swaag.benchmark.realistic_workflows import realistic_benchmark_tasks
from swaag.benchmark.task_definitions import get_benchmark_tasks
from swaag.benchmark.verifier import verify_benchmark_contract
from swaag.types import SessionState


def _catalog():
    return {task.task_id: task for task in realistic_benchmark_tasks()}


def _state():
    return SessionState(
        session_id="offline_fixture_only", created_at="2026-10-18T08:00:00Z",
        updated_at="2026-10-18T08:00:00Z", config_fingerprint="offline",
        model_base_url="http://not-contacted.invalid",
    )


def _snapshot(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*") if path.is_file()
        and "__pycache__" not in path.parts
    }


def _events(*tools):
    return [SimpleNamespace(event_type="agent_action_selected", payload={}), *(
        SimpleNamespace(event_type="tool_called", payload={"tool_name": tool})
        for tool in tools
    )]


def _verify(scenario, answer: str, *, before: dict[str, str], after: dict[str, str], tools=()):
    return verify_benchmark_contract(
        scenario.verification_contract, assistant_text=answer, state=_state(),
        events=_events(*tools), workspace_before=before, workspace_after=after,
        workspace_root=str(scenario.workspace),
    )


def _command(scenario):
    return subprocess.run(
        scenario.verification_contract.command,
        cwd=scenario.verification_contract.command_cwd,
        text=True, capture_output=True, check=False, timeout=15,
    )


def test_scenarios_are_real_model_tasks_in_standard_benchmark_catalog(tmp_path):
    tasks = _catalog()
    assert set(tasks) == {
        "realistic_identity_casual", "realistic_find_project_overview",
        "realistic_find_project_with_typos", "realistic_complex_project_overview",
        "realistic_followup_project_referent", "realistic_old_log_is_not_current_status",
        "realistic_changed_mind_no_edit", "realistic_invoicehub_multifile",
        "realistic_operator_portal_ui",
    }
    registered = {task.task_id: task for task in get_benchmark_tasks()}
    assert set(tasks) <= set(registered)
    for task_id, task in tasks.items():
        scenario = task.create(tmp_path / task_id)
        assert scenario.model_client is None, "Never insert scripted model answers into live benchmark"
        assert task.build_live is not None
        assert scenario.verification_contract.forbid_unexpected_workspace_changes
        assert scenario.prompt and len(scenario.prompt.split()) >= 10
        assert task.benchmark_max_total_actions and task.benchmark_tool_call_budget
        assert task.config_overrides["tools_allow_side_effect_tools"] == (task.task_type == "coding")


def test_realistic_project_prompts_do_not_give_away_file_paths_or_oracle_facts(tmp_path):
    for task_id, exact_secrets in (
        ("realistic_find_project_overview", ["SCHEMA_HOLD_42", "sqlite:///", "old-harbor-dashboard"]),
        ("realistic_followup_project_referent", ["QUEUE_PAUSED_17", "PostgreSQL", "orionstack/queue"]),
        ("realistic_old_log_is_not_current_status", ["release-12", "queued"]),
    ):
        scenario = _catalog()[task_id].create(tmp_path / task_id)
        assert all(secret not in scenario.prompt for secret in exact_secrets)
        assert scenario.verification_contract.min_tool_calls >= 1
        assert "tool_called" in scenario.verification_contract.required_history_events
        assert len(list(scenario.workspace.rglob("*"))) >= 3


def test_typos_and_deeper_project_are_scored_from_source_not_user_prompt(tmp_path):
    typo = _catalog()["realistic_find_project_with_typos"].create(tmp_path / "typo")
    assert "harbrbord" in typo.prompt and "Harborboard" not in typo.prompt
    before = _snapshot(typo.workspace)
    truth = "Harborboard uses SQLite, and imports are held for SCHEMA_HOLD_42."
    assert _verify(typo, truth, before=before, after=before, tools=["read_file", "read_file"]).passed
    assert not _verify(typo, "Unknown application.", before=before, after=before, tools=["read_file"]).passed

    deep = _catalog()["realistic_complex_project_overview"].create(tmp_path / "complex")
    files = [p for p in deep.workspace.rglob("*") if p.is_file()]
    assert len(files) >= 20
    assert "ARTIFACT_MISSING_29" not in deep.prompt and "PostgreSQL" not in deep.prompt
    before = _snapshot(deep.workspace)
    answer = ("WorkshopOps uses FastAPI with PostgreSQL storage, Redis workers, and a "
              "Svelte frontend. Current release is held for ARTIFACT_MISSING_29.")
    result = _verify(deep, answer, before=before, after=before, tools=["read_file", "search_repo"])
    assert result.passed, result.reason
    assert not _verify(deep, "It uses Django and MySQL.", before=before, after=before, tools=["read_file"]).passed


def test_referent_and_correction_keep_previous_authority_without_modifying_files(tmp_path):
    for task_id in ("realistic_followup_project_referent", "realistic_changed_mind_no_edit"):
        scenario = _catalog()[task_id].create(tmp_path / task_id)
        assert len(scenario.history_messages) == 2
        assert scenario.history_messages[0].role == "user"
        assert scenario.verification_contract.forbidden_tools_used == ["write_file", "edit_text", "run_tests"]
        assert scenario.verification_contract.allowed_modified_files == []
        before = _snapshot(scenario.workspace)
        answer = "OrionStack uses PostgreSQL. The sensor queue is paused because of QUEUE_PAUSED_17." if "followup" in task_id else "The retention value is 7 days. I did not edit it."
        report = _verify(scenario, answer, before=before, after=before, tools=["read_file", "read_file"])
        assert report.passed is True
        forbidden = _verify(scenario, answer, before=before, after=before, tools=["read_file", "edit_text"])
        assert forbidden.passed is False
        assert forbidden.checks["forbidden_tools_used"] is False
        altered = before | {"unknown-project.txt": "unauthorized"}
        corrupt = _verify(scenario, answer, before=before, after=altered, tools=["read_file", "read_file"])
        assert corrupt.passed is False and corrupt.checks["allowed_modified_files"] is False


def test_inspection_oracle_rejects_plausible_but_unverified_project_summary(tmp_path):
    scenario = _catalog()["realistic_find_project_overview"].create(tmp_path)
    before = _snapshot(scenario.workspace)
    invented = "Harborboard uses MySQL. The import had a timeout and is finished."
    bad = _verify(scenario, invented, before=before, after=before, tools=["read_file", "read_file"])
    assert bad.passed is False and bad.checks["expected_answer_contains"] is False
    correct = "Harborboard is the stock dashboard. SQLite stores its data; the import is held with SCHEMA_HOLD_42."
    still_bad = _verify(scenario, correct, before=before, after=before, tools=[])
    assert still_bad.passed is False
    assert still_bad.checks["min_tool_calls"] is False
    good = _verify(scenario, correct, before=before, after=before, tools=["list_files", "read_file"])
    assert good.passed is True


def test_older_green_log_is_not_current_deployment_success(tmp_path):
    scenario = _catalog()["realistic_old_log_is_not_current_status"].create(tmp_path)
    before = _snapshot(scenario.workspace)
    bad = _verify(scenario, "The release-12 checks passed yesterday.", before=before, after=before, tools=["read_file"])
    assert not bad.passed
    good = _verify(scenario, "release-12 is still queued; the current result is unknown.", before=before, after=before, tools=["read_file"])
    assert good.passed


def test_invoice_fixture_is_red_then_oracle_accepts_only_real_repair(tmp_path):
    scenario = _catalog()["realistic_invoicehub_multifile"].create(tmp_path)
    before = _snapshot(scenario.workspace)
    old = _command(scenario)
    assert old.returncode != 0 and "FAIL" in old.stdout + old.stderr
    pretend = _verify(scenario, "I fixed the totals and tests pass.", before=before, after=before, tools=["read_file"] * 3)
    assert not pretend.passed and pretend.checks["command"] is False
    pricing, api = (Path(path) for path in scenario.verification_contract.expected_file_patterns)
    original = pricing.read_text()
    assert "discounted + gross * tax" in original
    pricing.write_text(original.replace("discounted + gross * tax", "discounted + discounted * tax"))
    original_api = api.read_text()
    assert "str(invoice['subtotal'])" in original_api
    api.write_text(original_api.replace("str(invoice['subtotal'])", "str(due)"))
    run = _command(scenario)
    assert run.returncode == 0, run.stderr
    after = _snapshot(scenario.workspace)
    report = _verify(scenario, "Discount and preview calculations corrected; regression checks passed.", before=before, after=after, tools=["read_file"] * 3)
    assert report.passed, report.reason
    assert report.checks["command"] is True
    assert report.checks["allowed_modified_files"] is True
    assert sorted(report.evidence["workspace_changes"]["changed_files"]) == [
        "projects/invoicehub/backend/api.py", "projects/invoicehub/backend/pricing.py"
    ]
    # A passing test suite cannot excuse a change to a forbidden deployment artifact.
    after["projects/invoicehub/migrations/001_initial.sql"] = "tampered"
    tampered = _verify(scenario, "Correct and tested.", before=before, after=after, tools=["read_file"] * 3)
    assert tampered.checks["command"] is True
    assert tampered.checks["allowed_modified_files"] is False


def test_ui_fixture_is_red_then_tests_verify_real_save_behavior(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node is necessary for this browser interaction fixture; no model inference is needed")
    scenario = _catalog()["realistic_operator_portal_ui"].create(tmp_path)
    before = _snapshot(scenario.workspace)
    red = _command(scenario)
    assert red.returncode != 0 and "Save must commit the current selection" in red.stderr
    ui = Path(next(iter(scenario.verification_contract.expected_file_patterns)))
    assert "currentLabel.textContent = picker.value;" in ui.read_text()
    ui.write_text(ui.read_text().replace(
        "    currentLabel.textContent = picker.value;\n",
        "    currentLabel.textContent = picker.value;\n    selected = picker.value;\n",
    ))
    green = _command(scenario)
    assert green.returncode == 0, green.stderr
    after = _snapshot(scenario.workspace)
    report = _verify(scenario, "The Save action now persists the selected job status. Checks pass.", before=before, after=after, tools=["read_file"] * 3)
    assert report.passed, report.reason
    assert report.evidence["workspace_changes"]["changed_files"] == ["work/operator-portal/src/status-editor.js"]
    assert (scenario.workspace / "work/operator-portal/index.html").read_text().count('aria-live="polite"') == 1


def test_realistic_project_read_through_actual_runtime_without_any_model_server(make_config, tmp_path):
    """Exercise the real tool loop; canned model choices do NOT prove language understanding."""
    from swaag.runtime import AgentRuntime
    from tests.test_agent_action_loop import FakeModelClient, _action

    scenario = _catalog()["realistic_find_project_overview"].create(tmp_path / "fixture")
    before = _snapshot(scenario.workspace)
    model = FakeModelClient([
        _action(tool_calls=[("read_file", {"path": "apps/harborboard/api/storage.py"})], continue_loop=True),
        _action(tool_calls=[("read_file", {"path": "apps/harborboard/jobs/last_run.log"})], continue_loop=True),
        _action(message="Harborboard uses SQLite. The import is paused for SCHEMA_HOLD_42.", continue_loop=False),
    ])
    config = make_config(model__context_limit=32_000)
    config.tools.read_roots = [scenario.workspace]
    config.tools.enabled = ["read_file"]
    config.tools.allow_side_effect_tools = False
    config.tools.staged_discovery = False
    config.runtime.completion_evaluation_enabled = False
    runtime = AgentRuntime(config, model_client=model)
    result = runtime.run_turn(scenario.prompt)
    assert "SCHEMA_HOLD_42" in result.assistant_text
    assert [item["contract"] for item in model.requests] == ["agent_action"] * 3
    events = runtime.history.read_history(result.session_id)
    names = [event.payload.get("tool_name") for event in events if event.event_type == "tool_called"]
    assert names == ["read_file", "read_file"]
    observed = [str(event.payload) for event in events if event.event_type == "tool_result"]
    assert len(observed) == 2 and "sqlite" in observed[0].lower() and "SCHEMA_HOLD_42" in observed[1]
    check = verify_benchmark_contract(
        scenario.verification_contract,
        assistant_text=result.assistant_text,
        state=_state(),
        events=events,
        workspace_before=before,
        workspace_after=_snapshot(scenario.workspace),
        workspace_root=str(scenario.workspace),
    )
    assert check.passed, check.reason
