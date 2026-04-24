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
