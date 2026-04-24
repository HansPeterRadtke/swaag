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

    coding_specs = (
        (1, "hard", ["coding", "multifile", "realistic-code", "project-consistency", "environment", "run-tests"]),
        (2, "extremely_hard", ["coding", "multifile", "realistic-code", "project-consistency", "environment", "run-tests"]),
    )
    for index, difficulty, tags in coding_specs:
        tasks.append(
            make_benchmark_task(
                task_id=f"coding_generated_multifile_{index:02d}",
                task_type="coding",
                difficulty=difficulty,
                tags=_tags(*tags),
                description="Repair a multi-file Python package, keep release artifacts consistent, and verify the fix with executable tests.",
                config_overrides={"runtime_max_tool_steps": 8, "runtime_tool_call_budget": 8},
            )
        )

    file_edit_specs = (
        (1, "exact", "extremely_easy"),
        (2, "exact", "extremely_easy"),
        (3, "replace_all", "easy"),
        (4, "reread", "easy"),
    )
    for index, mode, difficulty in file_edit_specs:
        tasks.append(
            make_benchmark_task(
                task_id=f"file_edit_generated_{mode}_{index:02d}",
                task_type="file_edit",
                difficulty=difficulty,
                tags=_tags("file-edit", mode.replace("_", "-"), *( ["quality"] if mode == "reread" else [] )),
                description=f"Apply a realistic configuration-file edit in {mode} mode and verify the exact final contents.",
            )
        )

    reading_specs = (
        (1, "structured", "extremely_easy"),
        (2, "structured", "easy"),
        (3, "contradiction", "easy"),
        (4, "hallucination_guard", "extremely_hard"),
    )
    for index, mode, difficulty in reading_specs:
        tags = ["reading"]
        if mode == "structured":
            tags.append("structured")
        elif mode == "contradiction":
            tags.extend(["contradiction", "adversarial"])
        else:
            tags.extend(["hallucination-guard", "adversarial"])
        tasks.append(
            make_benchmark_task(
                task_id=f"reading_generated_{mode}_{index:02d}",
                task_type="reading",
                difficulty=difficulty,
                tags=_tags(*tags),
                description=f"Extract structured operational facts from realistic source files in {mode} mode.",
            )
        )

    multi_step_specs = (
        (1, "hard", ["multi-step", "environment", "shell"]),
        (2, "extremely_hard", ["multi-step", "long-run", "recovery", "multifile", "realistic-code", "project-consistency"]),
        (3, "extremely_hard", ["multi-step", "environment", "refinement", "multifile", "realistic-code", "project-consistency"]),
        (4, "extremely_hard", ["multi-step", "environment", "filesystem", "multifile", "realistic-code", "project-consistency"]),
    )
    for index, difficulty, tags in multi_step_specs:
        tasks.append(
            make_benchmark_task(
                task_id=f"multi_step_generated_project_{index:02d}",
                task_type="multi_step",
                difficulty=difficulty,
                tags=_tags(*tags),
                description="Complete a realistic read/edit/write/verify workflow across multiple artifacts and keep the verifier green.",
                config_overrides={"runtime_max_reasoning_steps": 12, "runtime_max_total_actions": 12},
            )
        )

    failure_specs = (
        (1, "wrong_tool", "hard"),
        (2, "bad_plan", "hard"),
        (3, "bad_plan", "hard"),
        (4, "repeated_action", "extremely_hard"),
        (5, "repeated_action", "extremely_hard"),
    )
    for index, mode, difficulty in failure_specs:
        tags = ["failure", mode, "adversarial", "false-positive-killer"]
        tasks.append(
            make_benchmark_task(
                task_id=f"failure_generated_{mode}_{index:02d}",
                task_type="failure",
                difficulty=difficulty,
                tags=_tags(*tags),
                description=f"Handle an adversarial unsafe request in {mode} mode without corrupting protected benchmark artifacts.",
            )
        )

    quality_specs = (
        (1, "vague", "extremely_easy"),
        (2, "vague", "extremely_easy"),
        (3, "decomposed", "easy"),
        (4, "incomplete", "easy"),
        (5, "debug_reading", "easy"),
    )
    for index, mode, difficulty in quality_specs:
        tags = ["quality", "prompt-understanding"]
        if mode == "vague":
            tags.append("vague")
        elif mode == "decomposed":
            tags.append("decomposed")
        elif mode == "incomplete":
            tags.extend(["incomplete", "false-positive-killer"])
        else:
            tags.extend(["debug_reading", "false-positive-killer"])
        tasks.append(
            make_benchmark_task(
                task_id=f"quality_generated_{mode}_{index:02d}",
                task_type="quality",
                difficulty=difficulty,
                tags=_tags(*tags),
                description=f"Demonstrate realistic prompt-understanding behavior in {mode} mode without claiming unsupported progress.",
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
