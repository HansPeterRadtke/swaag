from __future__ import annotations

from collections import Counter
from pathlib import Path

from swaag.benchmark.scaled_catalog import (
    LIVE_SUBSET_DIFFICULTY_MINIMUMS,
    LIVE_SUBSET_STRUCTURAL_MINIMUMS,
    LIVE_SUBSET_TASK_TYPE_MINIMUMS,
    generated_benchmark_tasks,
    generated_live_subset_tasks,
    validate_live_subset_catalog,
)
from swaag.benchmark.task_definitions import get_benchmark_tasks, validate_benchmark_catalog


def test_scaled_catalog_imports_and_meets_distribution_requirements() -> None:
    tasks = generated_benchmark_tasks()
    assert tasks

    counts = Counter(task.task_type for task in tasks)
    assert counts["coding"] >= 2
    assert counts["file_edit"] >= 4
    assert counts["reading"] >= 4
    assert counts["multi_step"] >= 4
    assert counts["failure"] >= 5
    assert counts["quality"] >= 5
    assert sum(1 for task in tasks if {"realistic-code", "multifile"}.issubset(set(task.tags))) >= 5
    assert any("long-run" in task.tags for task in tasks)
    assert any("false-positive-killer" in task.tags for task in tasks)
    assert any("environment" in task.tags for task in tasks)
    assert all(task.config_overrides is not None for task in tasks)
    assert all(task.task_id.startswith(("coding_generated_", "file_edit_generated_", "reading_generated_", "multi_step_", "failure_generated_", "quality_generated_")) for task in tasks)


def test_scaled_catalog_is_consumed_by_full_catalog_and_validation() -> None:
    tasks = get_benchmark_tasks()
    validate_benchmark_catalog(tasks)
    generated_ids = {task.task_id for task in generated_benchmark_tasks()}
    catalog_ids = {task.task_id for task in tasks}
    assert generated_ids <= catalog_ids


def test_scaled_catalog_definitions_do_not_embed_model_responses(tmp_path: Path) -> None:
    for task in generated_benchmark_tasks()[:12]:
        scenario = task.create(tmp_path / "catalog")
        assert scenario.model_client is None
        assert scenario.verification_contract.task_type == task.task_type


def test_scaled_catalog_sample_tasks_build_realistic_workspace_fixtures(tmp_path: Path) -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}
    sample_ids = [
        "coding_multifile_fix",
        "file_edit_reread_after_modification",
        "reading_identify_contradictions",
        "multi_step_read_compute_write_verify",
        "failure_wrong_tool_usage",
        "quality_vague_expansion",
    ]

    for task_id in sample_ids:
        scenario = tasks[task_id].create(tmp_path / task_id)
        files = sorted(path.relative_to(scenario.workspace) for path in scenario.workspace.rglob("*") if path.is_file())
        assert files
        assert files != [Path("task.txt")]
        assert "marker" not in scenario.prompt.lower()
        assert scenario.verification_contract.task_type == tasks[task_id].task_type

    coding = tasks["coding_multifile_fix"].create(tmp_path / "coding")
    assert coding.verification_contract.command[:3] == ["python3", "-m", "unittest"]
    assert coding.verification_contract.expected_file_patterns
    assert coding.verification_contract.allowed_modified_files
    assert coding.verification_contract.forbid_unexpected_workspace_changes is True

    reading = tasks["reading_identify_contradictions"].create(tmp_path / "reading")
    assert reading.verification_contract.expected_json is not None
    assert reading.verification_contract.expected_json_schema is not None

    multi_step = tasks["multi_step_mixed_read_note_compute_write"].create(tmp_path / "multi")
    assert multi_step.verification_contract.command[:3] == ["python3", "-m", "unittest"]
    assert len(multi_step.verification_contract.allowed_modified_files) >= 2
    assert {"notes", "calculator", "file-edit"} <= set(tasks["multi_step_mixed_read_note_compute_write"].tags)


def test_live_subset_catalog_imports_and_meets_distribution_requirements() -> None:
    tasks = generated_live_subset_tasks()
    counts = Counter(task.task_type for task in tasks)
    difficulty_counts = Counter(task.difficulty for task in tasks)

    assert len(tasks) >= 50
    for task_type, minimum in LIVE_SUBSET_TASK_TYPE_MINIMUMS.items():
        assert counts[task_type] >= minimum
    for difficulty, minimum in LIVE_SUBSET_DIFFICULTY_MINIMUMS.items():
        assert difficulty_counts[difficulty] >= minimum
    assert sum(1 for task in tasks if task.task_type == "coding" and "multifile" in task.tags) >= LIVE_SUBSET_STRUCTURAL_MINIMUMS["multifile_coding"]
    assert sum(1 for task in tasks if "long-run" in task.tags) >= LIVE_SUBSET_STRUCTURAL_MINIMUMS["long_run"]
    assert sum(1 for task in tasks if "false-positive-killer" in task.tags) >= LIVE_SUBSET_STRUCTURAL_MINIMUMS["false_positive_killer"]
    assert sum(1 for task in tasks if "verification-edge" in task.tags) >= LIVE_SUBSET_STRUCTURAL_MINIMUMS["verification_edge"]
    assert sum(1 for task in tasks if "prompt-understanding" in task.tags or "ambiguity" in task.tags) >= LIVE_SUBSET_STRUCTURAL_MINIMUMS["prompt_understanding"]
    assert sum(1 for task in tasks if {"environment", "shell", "filesystem"} & set(task.tags)) >= LIVE_SUBSET_STRUCTURAL_MINIMUMS["environment_or_shell"]
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert all(task.build_live is not None for task in tasks)
    validate_live_subset_catalog(tasks)


def test_live_subset_catalog_keeps_at_least_ten_tasks_per_difficulty_tier() -> None:
    tasks = generated_live_subset_tasks()
    difficulty_counts = Counter(task.difficulty for task in tasks)

    assert set(difficulty_counts) == set(LIVE_SUBSET_DIFFICULTY_MINIMUMS)
    assert all(count >= 10 for count in difficulty_counts.values())


def test_live_subset_definitions_do_not_embed_model_responses(tmp_path: Path) -> None:
    for task in generated_live_subset_tasks()[:12]:
        scenario = task.create(tmp_path / "validation", live_mode=True)
        assert scenario.model_client is None
        assert scenario.verification_contract.task_type == task.task_type


def test_live_multifile_coding_tasks_have_enough_iteration_budget() -> None:
    tasks = {task.task_id: task for task in generated_live_subset_tasks()}

    for task_id in ("live_coding_fix_03", "live_coding_fix_04", "live_coding_fix_05"):
        task = tasks[task_id]
        assert task.config_overrides is not None
        assert task.config_overrides["runtime_max_reasoning_steps"] >= 14
        assert task.config_overrides["runtime_max_total_actions"] >= 14
