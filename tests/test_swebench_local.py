from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import builtins
import sys as _sys

import pytest

from swaag.benchmark import swebench_local
from swaag.config import load_config


def test_git_retries_transient_network_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts: list[int] = []

    def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        attempts.append(1)
        if len(attempts) == 1:
            return subprocess.CompletedProcess(
                args=["git", "fetch"],
                returncode=128,
                stdout="",
                stderr="fatal: unable to access 'https://github.com/example/repo.git/': Failure when receiving data from the peer",
            )
        return subprocess.CompletedProcess(
            args=["git", "fetch"],
            returncode=0,
            stdout="ok",
            stderr="",
        )

    sleeps: list[float] = []
    monkeypatch.setattr(swebench_local.subprocess, "run", fake_run)
    monkeypatch.setattr(swebench_local.time, "sleep", sleeps.append)

    swebench_local._git(["fetch"], cwd=tmp_path, timeout_seconds=5)

    assert len(attempts) == 2
    assert sleeps == [1]


def test_git_raises_immediately_for_non_transient_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git", "fetch"],
            returncode=128,
            stdout="",
            stderr="fatal: repository not found",
        )

    monkeypatch.setattr(swebench_local.subprocess, "run", fake_run)

    with pytest.raises(swebench_local.LocalSwebenchFailure, match="repository not found"):
        swebench_local._git(["fetch"], cwd=tmp_path, timeout_seconds=5)


def test_render_prompt_includes_fail_to_pass_test_names() -> None:
    config = load_config()
    assert "\"task_kind\":\"local_repo_code_fix\"" in config.external_benchmarks.agent_generation.prompt_template
    prompt = swebench_local._render_prompt(
        {
            "repo": "demo/repo",
            "instance_id": "demo-1",
            "problem_statement": "Fix the bug.",
            "FAIL_TO_PASS": '["tests/test_demo.py::test_fix"]',
            "hints_text": "",
        },
        config.external_benchmarks.agent_generation.prompt_template,
        config=config,
    )

    assert "tests/test_demo.py::test_fix" in prompt
    assert "\"task_kind\":\"local_repo_code_fix\"" in prompt
    assert "{{" not in prompt


def test_render_prompt_truncates_problem_and_hints_to_policy_budget() -> None:
    config = load_config()
    prompt = swebench_local._render_prompt(
        {
            "repo": "demo/repo",
            "instance_id": "demo-1",
            "problem_statement": "P" * 1000,
            "FAIL_TO_PASS": '["tests/test_demo.py::test_fix"]',
            "hints_text": "H" * 1000,
        },
        "Problem statement:\n{problem_statement}\nKnown failing tests:\n{fail_to_pass_tests}\nHints:\n{hints_text}\n",
        config=config,
    )

    assert "...[truncated]" in prompt
    assert len(prompt) < 1800


def test_empty_patch_retry_prompt_preserves_task_contract_and_failing_tests() -> None:
    config = load_config()
    prompt = swebench_local._render_empty_patch_retry_prompt(
        {
            "repo": "demo/repo",
            "instance_id": "demo-1",
            "problem_statement": "Fix the bug.",
            "FAIL_TO_PASS": '["tests/test_demo.py::test_fix"]',
            "hints_text": "Check parser.py",
        },
        config.external_benchmarks.agent_generation.empty_patch_retry_prompt,
        config=config,
    )

    assert "\"task_kind\":\"local_repo_code_fix\"" in prompt
    assert "tests/test_demo.py::test_fix" in prompt
    assert "Check parser.py" in prompt


def test_prepare_swebench_subset_wraps_dataset_load_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        swebench_local,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Connection reset by peer")),
    )

    class _DummyGeneration:
        default_max_instances = 1

    class _DummyExternal:
        agent_generation = _DummyGeneration()

    class _DummyConfig:
        external_benchmarks = _DummyExternal()

    with pytest.raises(swebench_local.LocalSwebenchFailure, match="Unable to load benchmark dataset"):
        swebench_local.prepare_swebench_subset(
            benchmark_id="swebench_multilingual",
            dataset_name="SWE-bench/SWE-bench_Multilingual",
            instance_ids=["demo__repo-1"],
            output_path=tmp_path / "subset.json",
            config=_DummyConfig(),
        )


def test_prepare_swebench_subset_reports_missing_optional_datasets_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for module_name in list(_sys.modules):
        if module_name == "datasets" or module_name.startswith("datasets."):
            monkeypatch.delitem(_sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets" or name.startswith("datasets."):
            raise ModuleNotFoundError("No module named 'datasets'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    class _DummyGeneration:
        default_max_instances = 1

    class _DummyExternal:
        agent_generation = _DummyGeneration()

    class _DummyConfig:
        external_benchmarks = _DummyExternal()

    with pytest.raises(swebench_local.LocalSwebenchFailure, match="optional 'datasets' package"):
        swebench_local.prepare_swebench_subset(
            benchmark_id="swebench_lite",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            instance_ids=["demo__repo-1"],
            output_path=tmp_path / "subset.json",
            config=_DummyConfig(),
        )


def test_run_agent_turn_uses_real_agent_cli_with_workspace_scoped_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    config = load_config()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_root = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.txt"

    def fake_run(command, **kwargs) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(swebench_local.subprocess, "run", fake_run)

    stdout, stderr, prompt_text = swebench_local._run_agent_turn(
        workspace=workspace,
        session_root=session_root,
        session_name="demo-session",
        prompt="fix the benchmark task",
        prompt_path=prompt_path,
        timeout_seconds=42,
        config=config,
    )

    assert stdout == "ok"
    assert stderr == ""
    assert prompt_text == "fix the benchmark task"
    command = observed["command"]
    kwargs = observed["kwargs"]
    assert command == [sys.executable, "-m", "swaag", "ask", "--session", "demo-session"]
    assert kwargs["cwd"] == workspace
    assert kwargs["input"] == "fix the benchmark task"
    assert kwargs["timeout"] == 42
    env = kwargs["env"]
    assert env["SWAAG__SESSIONS__ROOT"] == str(session_root)
    assert env["SWAAG__MODEL__CONTEXT_LIMIT"] == str(
        min(config.model.context_limit, config.external_benchmarks.agent_generation.agent_context_limit)
    )
    assert env["SWAAG__MODEL__TIMEOUT_SECONDS"] == str(
        config.external_benchmarks.agent_generation.model_timeout_seconds
    )
    assert env["SWAAG__MODEL__STRUCTURED_TIMEOUT_SECONDS"] == str(
        config.external_benchmarks.agent_generation.model_structured_timeout_seconds
    )
    assert env["SWAAG__TOOLS__READ_ROOTS"] == f"[\"{workspace}\"]"
    assert env["SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS"] == "true"
    assert env["SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS"] == "true"
    assert env["SWAAG__PLANNER__MAX_PLAN_STEPS"] == str(config.external_benchmarks.agent_generation.planner_max_plan_steps)
    assert env["SWAAG__RUNTIME__MAX_TOOL_STEPS"] == str(config.external_benchmarks.agent_generation.runtime_max_tool_steps)
    assert str(swebench_local._repo_src_root()) in env["PYTHONPATH"]


def test_run_agent_turn_retries_retryable_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_inputs: list[str] = []
    observed_sessions: list[str] = []
    config = load_config()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run(command, **kwargs) -> subprocess.CompletedProcess[str]:
        observed_sessions.append(command[-1])
        observed_inputs.append(kwargs["input"])
        if len(observed_sessions) == 1:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="ERROR: Model returned invalid JSON for task_plan: '{'",
            )
        return subprocess.CompletedProcess(command, 0, stdout="patched", stderr="")

    monkeypatch.setattr(swebench_local.subprocess, "run", fake_run)

    stdout, stderr, prompt_text = swebench_local._run_agent_turn(
        workspace=workspace,
        session_root=tmp_path / "sessions",
        session_name="demo-session",
        prompt="fix the task",
        prompt_path=tmp_path / "prompt.txt",
        timeout_seconds=15,
        config=config,
    )

    assert stdout == "patched"
    assert "Model returned invalid JSON for task_plan" in stderr
    assert observed_sessions == ["demo-session", "demo-session--retry-2"]
    assert observed_inputs[0] == "fix the task"
    assert "Benchmark recovery note:" in observed_inputs[1]
    assert "Retry from scratch." in prompt_text


def test_real_agent_env_clamps_context_limit_to_server_truth(tmp_path: Path) -> None:
    config = load_config()
    env = swebench_local._real_agent_env(
        workspace=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        config=config,
        discovered_context_limit=1024,
    )

    assert env["SWAAG__MODEL__CONTEXT_LIMIT"] == "1024"
