from __future__ import annotations

import tempfile
from pathlib import Path

from swaag.benchmark.task_definitions import get_benchmark_tasks


FALSE_POSITIVE_TASKS = {
    "file_edit_noop_detection",
    "failure_wrong_tool_usage",
    "failure_repeated_action_trap",
    "failure_generated_quiet_policy_bypass",
    "failure_generated_credential_exfiltration_request",
    "quality_incomplete_clarification",
    "quality_generated_conflicting_hints_scope_choice",
}


def test_false_positive_killer_tasks_are_catalogued_without_model_fixtures() -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}

    assert set(FALSE_POSITIVE_TASKS) <= set(tasks)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for task_id in FALSE_POSITIVE_TASKS:
            task = tasks[task_id]
            assert "false-positive-killer" in task.tags or task.task_type in {"failure", "file_edit"}
            scenario = task.create(root)
            assert scenario.model_client is None
            assert scenario.verification_contract.task_type == task.task_type
            assert "marker" not in scenario.prompt.lower()
            assert any(path.is_file() for path in scenario.workspace.rglob("*"))
            if task.task_type == "failure":
                assert scenario.verification_contract.expected_files
                assert scenario.verification_contract.forbid_unexpected_workspace_changes is True
