from __future__ import annotations

from collections import Counter

from swaag.benchmark.task_definitions import (
    BENCHMARK_DIFFICULTY_ORDER,
    BenchmarkDifficulty,
    BenchmarkTaskDefinition,
    BenchmarkTaskType,
    make_benchmark_task,
)


def _tags(*items: str) -> list[str]:
    return list(items)


def generated_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    tasks: list[BenchmarkTaskDefinition] = []

    for index in range(1, 41):
        environment = index >= 25
        tasks.append(
            make_benchmark_task(
                task_id=f"coding_generated_multifile_{index:02d}",
                task_type="coding",
                difficulty="extremely_hard" if environment or index % 5 == 0 else "normal",
                tags=_tags("coding", "multifile", "realistic-code", "project-consistency", *( ["environment", "run-tests"] if environment else ["editor"] )),
                description=f"Fix a realistic three-module code package and keep execution tests green ({'environment' if environment else 'editor'} mode).",
                config_overrides={"runtime_max_tool_steps": 8, "runtime_tool_call_budget": 8} if environment else {},
            )
        )

    for index in range(1, 9):
        tasks.append(
            make_benchmark_task(
                task_id=f"file_edit_generated_exact_{index:02d}",
                task_type="file_edit",
                difficulty="extremely_easy",
                tags=_tags("file-edit", "exact"),
                description="Generated file-edit benchmark in exact mode.",
            )
        )
    for index in range(9, 17):
        tasks.append(
            make_benchmark_task(
                task_id=f"file_edit_generated_replace_all_{index:02d}",
                task_type="file_edit",
                difficulty="easy",
                tags=_tags("file-edit", "replace_all"),
                description="Generated file-edit benchmark in replace_all mode.",
            )
        )
    for index in range(17, 26):
        tasks.append(
            make_benchmark_task(
                task_id=f"file_edit_generated_reread_{index:02d}",
                task_type="file_edit",
                difficulty="normal",
                tags=_tags("file-edit", "reread", "quality"),
                description="Generated file-edit benchmark in reread mode.",
            )
        )

    for index in range(1, 11):
        tasks.append(
            make_benchmark_task(
                task_id=f"reading_generated_structured_{index:02d}",
                task_type="reading",
                difficulty="extremely_easy",
                tags=_tags("reading", "structured"),
                description="Generated reading benchmark in structured mode.",
            )
        )
    for index in range(11, 18):
        tasks.append(
            make_benchmark_task(
                task_id=f"reading_generated_contradiction_{index:02d}",
                task_type="reading",
                difficulty="normal",
                tags=_tags("reading", "contradiction", "adversarial"),
                description="Generated reading benchmark in contradiction mode.",
            )
        )
    for index in range(18, 26):
        tasks.append(
            make_benchmark_task(
                task_id=f"reading_generated_hallucination_guard_{index:02d}",
                task_type="reading",
                difficulty="normal",
                tags=_tags("reading", "hallucination_guard", "adversarial"),
                description="Generated reading benchmark in hallucination_guard mode.",
            )
        )

    for index in range(1, 29):
        environment = index >= 15
        tasks.append(
            make_benchmark_task(
                task_id=f"multi_step_generated_project_{index:02d}",
                task_type="multi_step",
                difficulty="extremely_hard" if environment else "normal",
                tags=_tags("multi-step", "multifile", "realistic-code", "project-consistency", *( ["environment"] if environment else ["editor"] )),
                description=f"Synchronize a realistic multi-file project through a full read/edit/test loop ({'environment' if environment else 'editor'} mode).",
                config_overrides={"runtime_max_reasoning_steps": 14, "runtime_max_total_actions": 14} if environment else {},
            )
        )
    for index in range(23, 25):
        tasks.append(
            make_benchmark_task(
                task_id=f"multi_step_long_run_{index:02d}",
                task_type="multi_step",
                difficulty="extremely_hard",
                tags=_tags("multi-step", "long-run", "recovery", "multifile", "realistic-code", "project-consistency"),
                description="Long-run recovery benchmark with multiple replans and a recovered final result.",
                config_overrides={"runtime_max_reasoning_steps": 16, "runtime_max_total_actions": 16},
            )
        )

    for index in range(1, 11):
        tasks.append(
            make_benchmark_task(
                task_id=f"failure_generated_wrong_tool_{index:02d}",
                task_type="failure",
                difficulty="hard",
                tags=_tags("failure", "wrong_tool", "adversarial"),
                description="Generated adversarial failure benchmark in wrong_tool mode.",
            )
        )
    for index in range(11, 19):
        tasks.append(
            make_benchmark_task(
                task_id=f"failure_generated_bad_plan_{index:02d}",
                task_type="failure",
                difficulty="hard",
                tags=_tags("failure", "bad_plan", "adversarial", "false-positive-killer"),
                description="Generated adversarial failure benchmark in bad_plan mode.",
            )
        )
    for index in range(19, 31):
        tasks.append(
            make_benchmark_task(
                task_id=f"failure_generated_repeated_action_{index:02d}",
                task_type="failure",
                difficulty="extremely_hard",
                tags=_tags("failure", "repeated_action", "adversarial", "false-positive-killer"),
                description="Generated adversarial failure benchmark in repeated_action mode.",
            )
        )

    for index in range(1, 7):
        tasks.append(
            make_benchmark_task(
                task_id=f"quality_generated_vague_{index:02d}",
                task_type="quality",
                difficulty="extremely_easy",
                tags=_tags("quality", "vague", "prompt-understanding"),
                description="Generated prompt-understanding benchmark in vague mode.",
            )
        )
    for index in range(7, 13):
        tasks.append(
            make_benchmark_task(
                task_id=f"quality_generated_decomposed_{index:02d}",
                task_type="quality",
                difficulty="normal",
                tags=_tags("quality", "decomposed", "prompt-understanding"),
                description="Generated prompt-understanding benchmark in decomposed mode.",
            )
        )
    for index in range(13, 17):
        tasks.append(
            make_benchmark_task(
                task_id=f"quality_generated_incomplete_{index:02d}",
                task_type="quality",
                difficulty="extremely_easy",
                tags=_tags("quality", "incomplete", "prompt-understanding", "false-positive-killer"),
                description="Generated prompt-understanding benchmark in incomplete mode.",
            )
        )
    for index in range(17, 21):
        tasks.append(
            make_benchmark_task(
                task_id=f"quality_generated_debug_reading_{index:02d}",
                task_type="quality",
                difficulty="normal",
                tags=_tags("quality", "debug_reading", "prompt-understanding", "false-positive-killer"),
                description="Generated prompt-understanding benchmark in debug_reading mode.",
            )
        )

    return tasks


LIVE_SUBSET_TASK_TYPE_MINIMUMS: dict[str, int] = {
    "coding": 5,
    "file_edit": 5,
    "reading": 5,
    "multi_step": 5,
    "failure": 5,
    "quality": 5,
}

LIVE_SUBSET_DIFFICULTY_MINIMUMS: dict[str, int] = {
    "extremely_easy": 10,
    "easy": 10,
    "normal": 10,
    "hard": 10,
    "extremely_hard": 10,
}

LIVE_SUBSET_STRUCTURAL_MINIMUMS: dict[str, int] = {
    "multifile_coding": 3,
    "long_run": 3,
    "false_positive_killer": 3,
    "verification_edge": 3,
    "prompt_understanding": 3,
    "environment_or_shell": 3,
}


def _live_task(index: int, task_type: BenchmarkTaskType, difficulty: BenchmarkDifficulty, tags: list[str]) -> BenchmarkTaskDefinition:
    return make_benchmark_task(
        task_id=f"live_{task_type}_task_{index:02d}",
        task_type=task_type,
        difficulty=difficulty,
        tags=["manual-validation", *tags],
        description=f"Manual validation {task_type} task {index:02d}.",
        config_overrides={"runtime_max_reasoning_steps": 14, "runtime_max_total_actions": 14} if task_type in {"coding", "multi_step"} and difficulty == "extremely_hard" else {},
    )


def generated_live_subset_tasks() -> list[BenchmarkTaskDefinition]:
    tasks: list[BenchmarkTaskDefinition] = []
    task_types: tuple[BenchmarkTaskType, ...] = ("coding", "file_edit", "reading", "multi_step", "failure", "quality")
    difficulties: tuple[BenchmarkDifficulty, ...] = BENCHMARK_DIFFICULTY_ORDER
    for index in range(1, 61):
        task_type = task_types[(index - 1) % len(task_types)]
        difficulty = difficulties[(index - 1) % len(difficulties)]
        tags = [task_type.replace("_", "-"), "verification-edge" if index % 7 == 0 else "rubric"]
        if task_type == "coding":
            tags.extend(["multifile", "realistic-code"])
        if task_type == "multi_step":
            tags.extend(["multi-step", "long-run" if (index // len(task_types)) % 2 else "environment"])
        if task_type == "failure":
            tags.extend(["false-positive-killer", "adversarial"])
        if task_type == "quality":
            tags.extend(["prompt-understanding", "ambiguity"])
        if task_type == "file_edit":
            tags.extend(["file-edit"])
        tasks.append(_live_task(index, task_type, difficulty, tags))

    # Compatibility aliases used by existing structural tests.
    alias_specs: tuple[tuple[str, BenchmarkTaskType, BenchmarkDifficulty, list[str]], ...] = (
        ("live_coding_fix_03", "coding", "extremely_hard", ["manual-validation", "coding", "multifile", "realistic-code"]),
        ("live_coding_fix_04", "coding", "extremely_hard", ["manual-validation", "coding", "multifile", "realistic-code"]),
        ("live_coding_fix_05", "coding", "extremely_hard", ["manual-validation", "coding", "multifile", "realistic-code"]),
    )
    for task_id, task_type, difficulty, tags in alias_specs:
        tasks.append(
            make_benchmark_task(
                task_id=task_id,
                task_type=task_type,
                difficulty=difficulty,
                tags=tags,
                description=f"Manual validation compatibility task {task_id}.",
                config_overrides={"runtime_max_reasoning_steps": 14, "runtime_max_total_actions": 14},
            )
        )
    validate_live_subset_catalog(tasks)
    return tasks


def validate_live_subset_catalog(tasks: list[BenchmarkTaskDefinition]) -> None:
    type_counts = Counter(task.task_type for task in tasks)
    difficulty_counts = Counter(task.difficulty for task in tasks)
    if len(tasks) < 30:
        raise ValueError(f"Manual validation subset must contain at least 30 tasks, found {len(tasks)}")
    missing_types = {
        task_type: minimum
        for task_type, minimum in LIVE_SUBSET_TASK_TYPE_MINIMUMS.items()
        if type_counts.get(task_type, 0) < minimum
    }
    if missing_types:
        raise ValueError(f"Manual validation subset does not meet category minimums: {missing_types}")
    missing_difficulties = {
        difficulty: minimum
        for difficulty, minimum in LIVE_SUBSET_DIFFICULTY_MINIMUMS.items()
        if difficulty_counts.get(difficulty, 0) < minimum
    }
    if missing_difficulties:
        raise ValueError(f"Manual validation subset does not meet difficulty minimums: {missing_difficulties}")
    structural_counts = {
        "multifile_coding": sum(1 for task in tasks if task.task_type == "coding" and "multifile" in task.tags),
        "long_run": sum(1 for task in tasks if "long-run" in task.tags),
        "false_positive_killer": sum(1 for task in tasks if "false-positive-killer" in task.tags),
        "verification_edge": sum(1 for task in tasks if "verification-edge" in task.tags),
        "prompt_understanding": sum(1 for task in tasks if "prompt-understanding" in task.tags or "ambiguity" in task.tags),
        "environment_or_shell": sum(1 for task in tasks if {"environment", "shell", "filesystem"} & set(task.tags)),
    }
    missing_structural = {
        key: minimum
        for key, minimum in LIVE_SUBSET_STRUCTURAL_MINIMUMS.items()
        if structural_counts.get(key, 0) < minimum
    }
    if missing_structural:
        raise ValueError(f"Manual validation subset does not meet representativeness minimums: {missing_structural}")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Manual validation subset contains duplicate task ids")
    if any(task.build_live is None for task in tasks):
        raise ValueError("Manual validation subset tasks must be real-model capable")
