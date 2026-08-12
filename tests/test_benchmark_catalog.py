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
    assert sum(1 for task in tasks if task.task_type == "coding" and {"realistic-code", "multifile"}.issubset(set(task.tags))) >= 2
    assert sum(1 for task in tasks if {"multifile", "recovery", "stale-source", "cross-file-sync", "authoritative-source"} & set(task.tags)) >= 9
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
            assert contract.expected_answer_contains
            assert contract.forbid_unexpected_workspace_changes is True


def test_benchmark_catalog_preserves_family_specific_task_shapes(tmp_path) -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}

    coding = tasks["coding_generated_compat_matrix_backfill"].create(tmp_path / "coding")
    coding_files = {path.name for path in coding.workspace.rglob("*") if path.is_file()}
    assert "compatibility_matrix.json" in coding_files
    assert "compatibility_report.md" in coding_files
    assert any(name.endswith("_compatibility.py") for name in coding_files)
    assert any(name.endswith("_report.py") for name in coding_files)
    assert coding.verification_contract.required_tools_used == ["run_tests"]

    reading = tasks["reading_generated_stale_note_null_guard"].create(tmp_path / "reading")
    reading_files = {path.name for path in reading.workspace.rglob("*") if path.is_file()}
    assert {"release_facts.json", "approvals.md", "stale_note.txt"} <= reading_files

    shell_flow = tasks["multi_step_generated_shell_capture_and_verify"].create(tmp_path / "shell")
    shell_files = {path.name for path in shell_flow.workspace.rglob("*") if path.is_file()}
    assert {"release.env", "capture_release.sh", "shell_release_summary.txt"} <= shell_files
    assert "shell_command" in shell_flow.verification_contract.required_tools_used

    failure = tasks["failure_generated_invalid_migration_actions"].create(tmp_path / "failure")
    failure_files = {path.name for path in failure.workspace.rglob("*") if path.is_file()}
    assert "requested_actions.md" in failure_files
    assert {"shell_command", "edit_text", "write_file"} <= set(failure.verification_contract.forbidden_tools_used)

    quality = tasks["quality_generated_conflicting_hints_scope_choice"].create(tmp_path / "quality")
    quality_files = {path.name for path in quality.workspace.rglob("*") if path.is_file()}
    assert {"request.txt", "context.txt"} <= quality_files
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
            assert scenario.verification_contract.expected_answer_contains


def test_capability_benchmark_tasks_cover_new_agent_primitives(tmp_path) -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}
    expected = {
        "capability_history_exact_retrieval",
        "capability_large_output_artifact_recovery",
        "capability_persistent_interactive_terminal",
        "capability_human_duration_wait",
    }
    assert expected <= set(tasks)
    assert len(tasks) >= 63

    history = tasks["capability_history_exact_retrieval"].create(tmp_path / "history")
    assert history.history_messages
    assert {"history_search", "history_window"} <= set(history.verification_contract.required_tools_used)
    assert {"agent_action_selected", "history_retrieved", "history_window_read"} <= set(history.verification_contract.required_history_events)

    large = tasks["capability_large_output_artifact_recovery"].create(tmp_path / "large")
    assert {"shell_command", "read_artifact"} <= set(large.verification_contract.required_tools_used)
    assert {"agent_action_selected", "artifact_created", "artifact_read"} <= set(large.verification_contract.required_history_events)
    assert tasks["capability_large_output_artifact_recovery"].config_overrides["environment_max_capture_chars"] == 512

    terminal = tasks["capability_persistent_interactive_terminal"].create(tmp_path / "terminal")
    assert terminal.verification_contract.required_tools_used == ["terminal"]
    assert {"agent_action_selected", "terminal_create", "terminal_send", "terminal_read", "terminal_close"} <= set(terminal.verification_contract.required_history_events)

    wait = tasks["capability_human_duration_wait"].create(tmp_path / "wait")
    assert wait.verification_contract.required_tools_used == ["wait_seconds"]
    assert wait.verification_contract.required_history_events == ["agent_action_selected", "wait_completed"]


def test_repo_task_budgets_leave_room_for_one_failed_edit_verify_cycle() -> None:
    tasks = {task.task_id: task for task in get_benchmark_tasks()}
    easy = tasks["coding_implement_function"].config_overrides
    normal = tasks["coding_multifile_fix"].config_overrides
    assert easy["runtime_tool_call_budget"] >= 10
    assert easy["runtime_max_total_actions"] >= 12
    assert normal["runtime_tool_call_budget"] >= easy["runtime_tool_call_budget"]
    assert normal["runtime_max_total_actions"] >= easy["runtime_max_total_actions"]


def test_failure_tasks_verify_policy_evidence_and_preserved_state_not_magic_refusal_words(tmp_path) -> None:
    from swaag.benchmark.verifier import verify_benchmark_contract
    from swaag.types import SessionState

    task = next(task for task in get_benchmark_tasks() if task.task_id == "failure_wrong_tool_usage")
    scenario = task.create(tmp_path)
    contract = scenario.verification_contract
    assert contract.expected_answer_any_of == []
    report = verify_benchmark_contract(
        contract,
        assistant_text="The request violates policy. protected.log remains unchanged and preserved.",
        state=SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://model"),
        events=[],
        workspace_before={str(path): path.read_text(encoding="utf-8") for path in scenario.workspace.rglob("*") if path.is_file()},
        workspace_after={str(path): path.read_text(encoding="utf-8") for path in scenario.workspace.rglob("*") if path.is_file()},
        workspace_root=str(scenario.workspace),
    )
    # Other event/tool checks may fail in this synthetic direct call, but refusal wording itself must not.
    assert "expected_answer_any_of" not in report.checks
    assert report.checks["expected_answer_contains"] is True
    assert report.checks["expected_files"] is True


def test_terminal_capability_prompt_matches_trailing_newline_verifier(tmp_path) -> None:
    task = next(task for task in get_benchmark_tasks() if task.task_id == "capability_persistent_interactive_terminal")
    scenario = task.create(tmp_path)
    assert "with a trailing newline after the second line" in scenario.prompt
    expected = next(iter(scenario.verification_contract.expected_files.values()))
    assert expected.endswith("\n")
