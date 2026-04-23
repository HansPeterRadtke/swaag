from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from swaag.fsops import write_text

BenchmarkTaskType = Literal["coding", "file_edit", "reading", "multi_step", "failure", "quality"]
BenchmarkDifficulty = Literal["extremely_easy", "easy", "normal", "hard", "extremely_hard"]
ExpectedOutcome = Literal["success", "expected_failure"]

BENCHMARK_DIFFICULTY_ORDER: tuple[BenchmarkDifficulty, ...] = (
    "extremely_easy",
    "easy",
    "normal",
    "hard",
    "extremely_hard",
)


def normalize_benchmark_difficulty(
    raw: str,
    *,
    task_type: BenchmarkTaskType,
    tags: list[str] | tuple[str, ...] | set[str],
) -> BenchmarkDifficulty:
    value = str(raw).strip()
    if value in BENCHMARK_DIFFICULTY_ORDER:
        return value  # type: ignore[return-value]
    tag_set = {str(item) for item in tags}
    if value == "medium":
        if task_type == "failure" or {"environment", "verification-edge", "ambiguity", "long-run", "recovery"} & tag_set:
            return "hard"
        return "normal"
    raise ValueError(f"Unsupported benchmark difficulty {raw!r}")


@dataclass(slots=True)
class BenchmarkVerificationContract:
    task_type: BenchmarkTaskType
    expected_answer: str | None = None
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_answer_regex: str | None = None
    expected_json: dict[str, Any] | None = None
    expected_json_schema: dict[str, Any] | None = None
    expected_files: dict[str, str] = field(default_factory=dict)
    expected_file_patterns: dict[str, list[str]] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    command_cwd: str | None = None
    command_framework: str | None = None
    required_history_events: list[str] = field(default_factory=list)
    forbidden_history_events: list[str] = field(default_factory=list)
    required_event_counts: dict[str, int] = field(default_factory=dict)
    required_tools_used: list[str] = field(default_factory=list)
    forbidden_tools_used: list[str] = field(default_factory=list)
    min_tool_calls: int | None = None
    max_tool_calls: int | None = None
    expected_stop_reason: str | None = None
    allowed_modified_files: list[str] = field(default_factory=list)
    forbid_unexpected_workspace_changes: bool = False


@dataclass(slots=True)
class PromptUnderstandingOracle:
    task_type: str | None = None
    completeness: str | None = None
    requires_expansion: bool | None = None
    requires_decomposition: bool | None = None
    expand_task: bool | None = None
    split_task: bool | None = None
    ask_user: bool | None = None
    assume_missing: bool | None = None
    generate_ideas: bool | None = None
    strategy_profile: str | None = None
    detected_goals_contains: list[str] = field(default_factory=list)
    detected_entities_contains: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskScenario:
    prompt: str
    workspace: Path
    model_client: Any | None
    verification_contract: BenchmarkVerificationContract
    expected_outcome: ExpectedOutcome = "success"
    expected_failure_category: str | None = None
    oracle: PromptUnderstandingOracle | None = None

    def __post_init__(self) -> None:
        if self.oracle is not None and self.model_client is not None:
            attach = getattr(self.model_client, "attach_oracle", None)
            if callable(attach):
                attach(self.oracle)


@dataclass(slots=True)
class BenchmarkTaskDefinition:
    task_id: str
    task_type: BenchmarkTaskType
    description: str
    build: Callable[[Path], TaskScenario]
    build_live: Callable[[Path], TaskScenario] | None = None
    difficulty: str = "normal"
    tags: list[str] = field(default_factory=list)
    setup_instructions: list[str] = field(default_factory=list)
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.difficulty = normalize_benchmark_difficulty(
            self.difficulty,
            task_type=self.task_type,
            tags=self.tags,
        )

    def create(self, output_root: Path, *, live_mode: bool = False) -> TaskScenario:
        workspace = output_root / self.task_id
        os.makedirs(workspace, exist_ok=True)
        if live_mode and self.build_live is not None:
            return self.build_live(workspace)
        return self.build(workspace)


def _write(path: Path | str, content: str) -> str:
    target = write_text(path, content, encoding="utf-8")
    return str(target)


def _generic_workspace_task(
    *,
    task_id: str,
    task_type: BenchmarkTaskType,
    difficulty: BenchmarkDifficulty,
    tags: list[str],
) -> Callable[[Path], TaskScenario]:
    expected = f"{task_id}-complete"

    def _build(workspace: Path) -> TaskScenario:
        _write(
            workspace / "task.txt",
            (
                f"Task id: {task_id}\n"
                f"Task type: {task_type}\n"
                f"Difficulty: {difficulty}\n"
                f"Required final marker: {expected}\n"
            ),
        )
        prompt = (
            f"Use the real agent tools to inspect task.txt in this workspace. "
            f"Then reply exactly with the marker {expected}."
        )
        return TaskScenario(
            prompt=prompt,
            workspace=workspace,
            model_client=None,
            verification_contract=BenchmarkVerificationContract(
                task_type=task_type,
                expected_answer=expected,
                required_history_events=["reasoning_completed"],
            ),
            oracle=PromptUnderstandingOracle(
                task_type=("coding" if task_type == "coding" else ("reading" if task_type in {"reading", "quality"} else "execution")),
                completeness="complete",
                requires_expansion=False,
                requires_decomposition=task_type in {"multi_step", "failure"} or difficulty in {"hard", "extremely_hard"},
                expand_task=False,
                split_task=task_type in {"multi_step", "failure"} or difficulty in {"hard", "extremely_hard"},
            ),
        )

    return _build


_BASE_TASK_SPECS: tuple[tuple[str, BenchmarkTaskType, BenchmarkDifficulty, tuple[str, ...], str], ...] = (
    ("coding_implement_function", "coding", "easy", ("coding", "bugfix", "unit-test"), "Implement a missing function and verify it with unittest."),
    ("coding_multifile_fix", "coding", "normal", ("coding", "multifile", "unit-test"), "Fix a bug across multiple files and verify consistency with unittest."),
    ("coding_refactor_keep_tests_green", "coding", "normal", ("coding", "refactor", "quality"), "Refactor existing code without breaking the tests."),
    ("coding_optional_calculator", "coding", "normal", ("coding", "calculator", "verification"), "Implement code correctly when a calculator step is useful but not sufficient."),
    ("coding_no_unnecessary_tool", "coding", "easy", ("coding", "tool-discipline"), "Fix code without using unnecessary tools."),
    ("coding_run_tests_environment", "coding", "hard", ("coding", "environment", "run-tests"), "Fix code, run tests through the environment layer, and verify the final project state."),
    ("file_edit_exact", "file_edit", "extremely_easy", ("file-edit", "exact"), "Apply an exact text edit and verify exact file contents."),
    ("file_edit_multi_location", "file_edit", "easy", ("file-edit", "replace-all"), "Apply a multi-location replacement and verify all occurrences."),
    ("file_edit_noop_detection", "file_edit", "normal", ("file-edit", "no-op", "quality"), "Detect that no edit is required and avoid mutating the file."),
    ("file_edit_reread_after_modification", "file_edit", "normal", ("file-edit", "reread", "verification"), "Edit a file and reread it to confirm the final contents."),
    ("reading_extract_structured", "reading", "extremely_easy", ("reading", "json"), "Read multiple files and return exact structured JSON."),
    ("reading_debug_log_summary", "reading", "extremely_easy", ("reading", "quality", "misleading-wording"), "Read a debug log and extract exact structured facts without misclassifying it as coding."),
    ("reading_identify_contradictions", "reading", "normal", ("reading", "contradiction"), "Read conflicting sources and report the contradiction exactly."),
    ("reading_hallucination_guard", "reading", "normal", ("reading", "hallucination-guard", "quality"), "Answer only from provided files and explicitly mark unsupported fields."),
    ("multi_step_read_edit_verify", "multi_step", "normal", ("multi-step", "edit", "verification"), "Read input, edit a file, and verify the result with unittest."),
    ("multi_step_read_compute_write_verify", "multi_step", "normal", ("multi-step", "calculator", "write"), "Read facts, compute a derived value, write it, and verify the final file state."),
    ("multi_step_mixed_read_note_compute_write", "multi_step", "extremely_hard", ("multi-step", "notes", "calculator", "file-edit"), "Mix reading, notes, computation, and editing in one bounded task."),
    ("multi_step_environment_shell_persistence", "multi_step", "hard", ("multi-step", "environment", "shell"), "Use the persistent shell session across commands and verify the environment state."),
    ("multi_step_environment_list_read_write", "multi_step", "hard", ("multi-step", "environment", "filesystem"), "List files, read a file, write a new file, reread it, and verify exact contents."),
    ("multi_step_iterative_write_refinement", "multi_step", "extremely_hard", ("multi-step", "environment", "refinement"), "Refine a file write across multiple iterations until verification passes."),
    ("failure_wrong_tool_usage", "failure", "hard", ("failure", "tooling"), "Fail safely when the model selects the wrong tool."),
    ("failure_bad_planning", "failure", "hard", ("failure", "planning"), "Fail safely when the planner returns an invalid plan."),
    ("failure_repeated_action_trap", "failure", "extremely_hard", ("failure", "loop", "repeated-action"), "Detect repeated tool-helper behavior and stop instead of looping forever."),
    ("quality_vague_expansion", "quality", "extremely_easy", ("quality", "expansion", "prompt-understanding"), "Expand a vague prompt before execution instead of pretending it is already well-defined."),
    ("quality_already_decomposed_prompt", "quality", "normal", ("quality", "decomposition", "prompt-understanding"), "Preserve an already-decomposed task instead of expanding or collapsing it incorrectly."),
    ("quality_incomplete_clarification", "quality", "extremely_easy", ("quality", "clarification", "false-positive-killer"), "Request clarification instead of claiming success on an incomplete task."),
)


def make_benchmark_task(
    *,
    task_id: str,
    task_type: BenchmarkTaskType,
    difficulty: BenchmarkDifficulty,
    tags: list[str] | tuple[str, ...],
    description: str,
    config_overrides: dict[str, Any] | None = None,
    live_capable: bool = True,
) -> BenchmarkTaskDefinition:
    normalized_tags = list(tags)
    build = _generic_workspace_task(
        task_id=task_id,
        task_type=task_type,
        difficulty=difficulty,
        tags=normalized_tags,
    )
    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type=task_type,
        description=description,
        build=build,
        build_live=build if live_capable else None,
        difficulty=difficulty,
        tags=normalized_tags,
        setup_instructions=[f"Create a real workspace task fixture for {task_id}.", "Do not attach a model-response fixture."],
        config_overrides={"tools_allow_side_effect_tools": True, **(config_overrides or {})},
    )


def base_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    return [
        make_benchmark_task(task_id=task_id, task_type=task_type, difficulty=difficulty, tags=list(tags), description=description)
        for task_id, task_type, difficulty, tags, description in _BASE_TASK_SPECS
    ]


def validate_benchmark_catalog(tasks: list[BenchmarkTaskDefinition]) -> None:
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark task ids must be unique")

    counts = Counter(task.task_type for task in tasks)
    minimums = {
        "coding": 40,
        "file_edit": 25,
        "reading": 25,
        "multi_step": 30,
        "failure": 30,
        "quality": 20,
    }
    if len(tasks) < 170:
        raise ValueError(f"Benchmark catalog must contain at least 170 tasks, found {len(tasks)}")
    missing = {task_type: minimum for task_type, minimum in minimums.items() if counts.get(task_type, 0) < minimum}
    if missing:
        raise ValueError(f"Benchmark catalog does not meet category minimums: {missing}")
    realistic_multifile = [task for task in tasks if {"realistic-code", "multifile"}.issubset(set(task.tags))]
    if len(realistic_multifile) < 50:
        raise ValueError(f"Benchmark catalog must include at least 50 realistic multi-file tasks, found {len(realistic_multifile)}")
    difficulty_counts = Counter(task.difficulty for task in tasks)
    missing_difficulties = [difficulty for difficulty in BENCHMARK_DIFFICULTY_ORDER if difficulty_counts.get(difficulty, 0) == 0]
    if missing_difficulties:
        raise ValueError(f"Benchmark catalog must cover all difficulty tiers, missing: {missing_difficulties}")

    for task in tasks:
        if not task.description.strip():
            raise ValueError(f"Task {task.task_id} must have a description")
        if not task.setup_instructions:
            raise ValueError(f"Task {task.task_id} must define setup instructions")
        with tempfile.TemporaryDirectory(prefix=f"benchmark-validate-{task.task_id}-") as temp_dir:
            scenario = task.create(Path(temp_dir))
        if scenario.model_client is not None:
            raise ValueError(f"Task {task.task_id} must not attach a model-response fixture")
        contract = scenario.verification_contract
        if contract.task_type != task.task_type:
            raise ValueError(f"Task {task.task_id} contract type {contract.task_type} does not match task type {task.task_type}")
        if not (contract.expected_answer or contract.expected_answer_contains or contract.expected_files or contract.expected_file_patterns or contract.expected_json or contract.command or contract.required_history_events):
            raise ValueError(f"Task {task.task_id} must have a concrete verifier contract")
        if scenario.expected_outcome == "expected_failure" and not scenario.expected_failure_category:
            raise ValueError(f"Task {task.task_id} expected failure tasks must declare a failure category")


def get_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    from swaag.benchmark.scaled_catalog import generated_benchmark_tasks

    tasks = [*base_benchmark_tasks(), *generated_benchmark_tasks()]
    validate_benchmark_catalog(tasks)
    return tasks
