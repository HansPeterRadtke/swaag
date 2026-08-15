from pathlib import Path


def test_benchmark_contract_write_allowlist_is_applied_to_runtime_config(tmp_path, monkeypatch) -> None:
    # This verifies the policy transformation independently of model execution by
    # constructing the same paths a benchmark seed provides.
    from swaag.benchmark.task_definitions import get_benchmark_tasks
    from swaag.benchmark.benchmark_runner import _build_config

    task = next(item for item in get_benchmark_tasks() if item.task_id == "coding_multifile_fix")
    scenario = task.create(tmp_path / "workspace", live_mode=True)
    config = _build_config(
        sessions_root=tmp_path / "sessions",
        workspace=scenario.workspace,
        overrides=task.config_overrides,
        base_url="http://127.0.0.1:1",
        seed=11,
    )
    contract = scenario.verification_contract
    assert contract.forbid_unexpected_workspace_changes is True
    allowed = []
    for item in contract.allowed_modified_files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = scenario.workspace / candidate
        allowed.append(str(candidate.resolve()))
    config.editor.allowed_write_paths = allowed
    assert {Path(item).name for item in config.editor.allowed_write_paths} == {"tokenizer.py", "normalizer.py"}
    assert "test_multifile.py" not in {Path(item).name for item in config.editor.allowed_write_paths}
