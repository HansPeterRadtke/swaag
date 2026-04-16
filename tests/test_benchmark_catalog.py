from __future__ import annotations

from collections import Counter

from swaag.benchmark.task_definitions import get_benchmark_tasks, validate_benchmark_catalog


def test_benchmark_catalog_is_large_diverse_and_valid() -> None:
    tasks = get_benchmark_tasks()
    validate_benchmark_catalog(tasks)

    counts = Counter(task.task_type for task in tasks)
    assert len(tasks) >= 170
    assert counts["coding"] >= 40
    assert counts["file_edit"] >= 25
    assert counts["reading"] >= 25
    assert counts["multi_step"] >= 30
    assert counts["failure"] >= 30
    assert counts["quality"] >= 20
    assert sum(1 for task in tasks if {"realistic-code", "multifile"}.issubset(set(task.tags))) >= 50
    assert any("long-run" in task.tags for task in tasks)
    assert any("false-positive-killer" in task.tags for task in tasks)
    assert any("environment" in task.tags for task in tasks)

    ids = [task.task_id for task in tasks]
    assert len(ids) == len(set(ids))
    assert all(task.setup_instructions for task in tasks)
    assert all(task.tags for task in tasks)
    assert {task.difficulty for task in tasks} == {"easy", "medium", "hard"}
