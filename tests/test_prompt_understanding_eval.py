from __future__ import annotations

import tempfile
from pathlib import Path

from swaag.benchmark.task_definitions import get_benchmark_tasks


UNDERSTANDING_TASKS = {
    "reading_debug_log_summary",
    "quality_vague_expansion",
    "quality_already_decomposed_prompt",
    "quality_incomplete_clarification",
}


def test_prompt_understanding_benchmark_tasks_define_oracles_without_model_fixtures() -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}

    assert set(UNDERSTANDING_TASKS) <= set(tasks)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for task_id in UNDERSTANDING_TASKS:
            task = tasks[task_id]
            scenario = task.create(root)
            assert scenario.model_client is None
            assert scenario.oracle is not None
            assert scenario.oracle.completeness == "complete"
            assert "marker" not in scenario.prompt.lower()
