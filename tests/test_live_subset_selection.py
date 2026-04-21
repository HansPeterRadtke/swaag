from __future__ import annotations

from collections import Counter

from swaag.benchmark.scaled_catalog import generated_live_subset_tasks


def test_live_subset_selection_is_deterministic_and_representative() -> None:
    first = generated_live_subset_tasks()
    second = generated_live_subset_tasks()

    first_ids = [task.task_id for task in first]
    second_ids = [task.task_id for task in second]
    difficulty_counts = Counter(task.difficulty for task in first)

    assert first_ids == second_ids
    assert len(first_ids) >= 50
    assert difficulty_counts["extremely_easy"] >= 10
    assert difficulty_counts["easy"] >= 10
    assert difficulty_counts["normal"] >= 10
    assert difficulty_counts["hard"] >= 10
    assert difficulty_counts["extremely_hard"] >= 10
    assert sum(1 for task in first if task.task_type == "coding" and "multifile" in task.tags) >= 3
    assert sum(1 for task in first if "long-run" in task.tags) >= 3
    assert sum(1 for task in first if "false-positive-killer" in task.tags) >= 3
    assert sum(1 for task in first if "verification-edge" in task.tags) >= 3
    assert sum(1 for task in first if "prompt-understanding" in task.tags or "ambiguity" in task.tags) >= 3
