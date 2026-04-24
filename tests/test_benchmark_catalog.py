from __future__ import annotations

from collections import Counter

from swaag.benchmark.task_definitions import get_benchmark_tasks, validate_benchmark_catalog


def test_benchmark_catalog_is_large_diverse_and_valid() -> None:
    tasks = get_benchmark_tasks()
    validate_benchmark_catalog(tasks)

    counts = Counter(task.task_type for task in tasks)
    assert len(tasks) >= 50
    assert counts["coding"] >= 8
    assert counts["file_edit"] >= 8
    assert counts["reading"] >= 8
    assert counts["multi_step"] >= 8
    assert counts["failure"] >= 8
    assert counts["quality"] >= 8
    assert sum(1 for task in tasks if {"realistic-code", "multifile"}.issubset(set(task.tags))) >= 5
    assert any("long-run" in task.tags for task in tasks)
    assert any("false-positive-killer" in task.tags for task in tasks)
    assert any("environment" in task.tags for task in tasks)

    ids = [task.task_id for task in tasks]
    assert len(ids) == len(set(ids))
    assert all(task.setup_instructions for task in tasks)
    assert all(task.tags for task in tasks)
    assert {task.difficulty for task in tasks} == {
        "extremely_easy",
        "easy",
        "normal",
        "hard",
        "extremely_hard",
    }


def test_benchmark_catalog_uses_programmatic_verification_and_anti_tamper_contracts(tmp_path) -> None:
    tasks = get_benchmark_tasks()

    for task in tasks:
        scenario = task.create(tmp_path / task.task_id)
        contract = scenario.verification_contract
        if task.task_type == "coding":
            assert contract.command
            assert contract.expected_file_patterns
            assert contract.allowed_modified_files
            assert contract.forbid_unexpected_workspace_changes is True
        elif task.task_type == "file_edit":
            assert contract.expected_files
            assert contract.allowed_modified_files or contract.forbid_unexpected_workspace_changes
        elif task.task_type == "reading":
            assert contract.expected_json is not None
            assert contract.expected_json_schema is not None
            assert contract.forbid_unexpected_workspace_changes is True
        elif task.task_type == "multi_step":
            assert contract.command
            assert contract.expected_files
            assert contract.allowed_modified_files
            assert contract.forbid_unexpected_workspace_changes is True
        elif task.task_type == "failure":
            assert contract.expected_files
            assert "false-positive-killer" in task.tags
            assert contract.forbid_unexpected_workspace_changes is True
        elif task.task_type == "quality":
            assert scenario.oracle is not None
            assert contract.expected_answer_contains
            assert contract.forbid_unexpected_workspace_changes is True


def test_benchmark_catalog_preserves_family_specific_task_shapes(tmp_path) -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}

    coding = tasks["coding_generated_multifile_02"].create(tmp_path / "coding")
    coding_files = {path.name for path in coding.workspace.rglob("*") if path.is_file()}
    assert "release_settings.json" in coding_files
    assert "release_notes.txt" in coding_files
    assert any(name.endswith("_unit.py") for name in coding_files)
    assert any(name.endswith("_compat.py") for name in coding_files)
    assert any(name.endswith("_artifacts.py") for name in coding_files)
    assert coding.verification_contract.required_tools_used == ["run_tests"]

    reading = tasks["reading_generated_hallucination_guard_04"].create(tmp_path / "reading")
    reading_files = {path.name for path in reading.workspace.rglob("*") if path.is_file()}
    assert {"facts.json", "roadmap.md", "stale_note.txt"} <= reading_files

    shell_flow = tasks["multi_step_environment_shell_persistence"].create(tmp_path / "shell")
    shell_files = {path.name for path in shell_flow.workspace.rglob("*") if path.is_file()}
    assert {"release.env", "capture_release.sh", "shell_release_summary.txt"} <= shell_files
    assert "shell_command" in shell_flow.verification_contract.required_tools_used

    failure = tasks["failure_generated_bad_plan_02"].create(tmp_path / "failure")
    failure_files = {path.name for path in failure.workspace.rglob("*") if path.is_file()}
    assert "requested_plan.md" in failure_files
    assert {"shell_command", "edit_text", "write_file"} <= set(failure.verification_contract.forbidden_tools_used)

    quality = tasks["quality_generated_incomplete_04"].create(tmp_path / "quality")
    quality_files = {path.name for path in quality.workspace.rglob("*") if path.is_file()}
    assert {"request.txt", "ticket.md"} <= quality_files
    assert {"write_file", "edit_text", "run_tests"} <= set(quality.verification_contract.forbidden_tools_used)


def test_extremely_hard_catalog_tasks_have_high_complexity_structure(tmp_path) -> None:
    tasks = [task for task in get_benchmark_tasks() if task.difficulty == "extremely_hard"]
    assert len(tasks) >= 10

    for task in tasks:
        scenario = task.create(tmp_path / task.task_id)
        files = [path for path in scenario.workspace.rglob("*") if path.is_file()]
        if task.task_type in {"coding", "multi_step", "failure"}:
            assert len(files) >= 2
            assert {"multifile", "long-run", "recovery", "repeated-action", "adversarial", "environment"} & set(task.tags)
        elif task.task_type == "reading":
            assert scenario.verification_contract.expected_json_schema is not None
        elif task.task_type == "quality":
            assert scenario.oracle is not None
