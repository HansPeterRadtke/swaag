from __future__ import annotations

import json
from pathlib import Path

from swaag.benchmark import external as external_benchmarks
from swaag.benchmark.benchmark_runner import main as benchmark_main
from swaag.benchmark.fixtures import bounded_swebench_fixture_paths, terminal_bench_task_root
from swaag.benchmark.swebench_local import GeneratedSwebenchPredictions, PreparedSwebenchSubset
from swaag.config import load_config


def test_external_benchmark_defaults_cover_all_required_integrations() -> None:
    config = load_config()

    assert set(config.external_benchmarks.targets) == {
        "swebench_lite",
        "swebench_verified",
        "swebench_multilingual",
        "swebench_full",
        "terminal_bench",
    }
    assert config.external_benchmarks.smoke_timeout_seconds == 90
    assert config.external_benchmarks.full_timeout_seconds == 1800
    assert config.external_benchmarks.model_server.preflight_enabled is True
    assert config.external_benchmarks.model_server.healthcheck_timeout_seconds == 5
    assert config.external_benchmarks.model_server.retry_attempts == 3
    assert config.external_benchmarks.model_server.retry_sleep_seconds == 2.0
    assert config.external_benchmarks.terminal_bench.compose_probe_timeout_seconds == 5
    assert config.external_benchmarks.terminal_bench.compose_download_timeout_seconds == 60
    assert config.external_benchmarks.terminal_bench.allow_compose_download is True
    assert config.external_benchmarks.targets["swebench_lite"].allowed_path_literals == ["gold"]
    assert config.external_benchmarks.targets["swebench_lite"].default_variables["namespace_args"] == "--namespace none"
    assert config.external_benchmarks.targets["swebench_lite"].default_variables["dataset_name_args"] == "--dataset_name princeton-nlp/SWE-bench_Lite"
    assert "@args:{cache_args}" in config.external_benchmarks.targets["swebench_lite"].full_command
    assert config.external_benchmarks.targets["swebench_multilingual"].preflight_commands == [["docker", "version"]]
    assert config.external_benchmarks.targets["terminal_bench"].preflight_commands == []
    assert config.external_benchmarks.targets["terminal_bench"].smoke_command == [
        "python3",
        "-m",
        "swaag.benchmark.terminal_bench_local",
        "smoke",
    ]
    assert config.external_benchmarks.targets["terminal_bench"].full_command[:4] == [
        "python3",
        "-m",
        "swaag.benchmark.terminal_bench_local",
        "run",
    ]
    assert "@args:{instance_ids_args}" in config.external_benchmarks.targets["swebench_lite"].full_command
    assert "@args:{dataset_name_args}" in config.external_benchmarks.targets["swebench_lite"].full_command
    assert "@args:{namespace_args}" in config.external_benchmarks.targets["swebench_lite"].full_command
    assert "@args:{task_selection_args}" in config.external_benchmarks.targets["terminal_bench"].full_command
    assert config.external_benchmarks.agent_generation.default_max_instances == 1
    assert config.external_benchmarks.agent_generation.clone_timeout_seconds == 300
    assert config.external_benchmarks.agent_generation.agent_timeout_seconds == 900
    assert config.external_benchmarks.agent_generation.allow_stateful_tools is True
    assert config.external_benchmarks.agent_generation.allow_side_effect_tools is True
    assert config.external_benchmarks.agent_generation.model_timeout_seconds == 180
    assert config.external_benchmarks.agent_generation.model_structured_timeout_seconds == 240
    assert config.external_benchmarks.agent_generation.planner_max_plan_steps == 4
    assert config.external_benchmarks.agent_generation.planner_max_replans == 1
    assert config.external_benchmarks.agent_generation.runtime_max_reasoning_steps == 16
    assert config.external_benchmarks.agent_generation.runtime_max_total_actions == 40
    assert config.external_benchmarks.agent_generation.runtime_max_tool_steps == 24
    assert config.external_benchmarks.agent_generation.runtime_tool_call_budget == 24
    assert config.external_benchmarks.agent_generation.candidate_file_limit == 2
    assert config.external_benchmarks.agent_generation.file_excerpt_char_limit == 900
    assert config.external_benchmarks.agent_generation.issue_prompt_char_limit == 1200
    assert config.external_benchmarks.agent_generation.completion_max_tokens == 192
    assert config.external_benchmarks.agent_generation.solver_max_attempts == 2
    assert config.external_benchmarks.agent_generation.summary_max_chars == 120
    assert config.external_benchmarks.agent_generation.find_max_chars == 800
    assert config.external_benchmarks.agent_generation.replace_max_chars == 1600
    assert config.external_benchmarks.agent_generation.git_remote_base_url == "https://github.com"
    assert "local SWE-bench benchmark instance" in config.external_benchmarks.agent_generation.prompt_template
    assert "read -> edit -> verify -> respond" in config.external_benchmarks.agent_generation.prompt_template
    assert "best-effort concrete code change" in config.external_benchmarks.agent_generation.empty_patch_retry_prompt


def test_repo_local_benchmark_fixtures_exist_for_bounded_proof() -> None:
    fixtures = bounded_swebench_fixture_paths()

    assert set(fixtures) == {
        "swebench_lite",
        "swebench_verified",
        "swebench_multilingual",
        "swebench_full",
    }
    for path in fixtures.values():
        assert path.exists(), path

    task_root = terminal_bench_task_root() / "hello-swaag"
    assert (task_root / "task.yaml").exists()
    assert (task_root / "docker-compose.yaml").exists()
    assert (task_root / "run-tests.sh").exists()
    compose_text = (task_root / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "network_mode: host" in compose_text
    assert "SWAAG_TERMINAL_HOST_NETWORK=1" in compose_text
    run_tests_text = (task_root / "run-tests.sh").read_text(encoding="utf-8")
    assert "/usr/bin/python3 -m pytest" in run_tests_text
    dockerfile_text = (task_root / "Dockerfile").read_text(encoding="utf-8")
    assert "python3-pytest" in dockerfile_text
    assert "python3-requests" in dockerfile_text
    assert "pip install" not in dockerfile_text


def test_external_benchmark_smoke_run_captures_artifacts(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    old_artifact = tmp_path / "artifacts" / "stale.txt"
    old_artifact.parent.mkdir(parents=True, exist_ok=True)
    old_artifact.write_text("old", encoding="utf-8")
    script = tmp_path / "smoke_probe.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "out = Path(sys.argv[1]) / 'artifacts'",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'probe.txt').write_text('ok', encoding='utf-8')",
                "print('probe-ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.workdir = str(tmp_path)
    target.smoke_command = ["python3", str(script), "{output_dir}"]
    target.artifact_globs = ["artifacts/**/*"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="smoke",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["passed"] == 1
    result = report["results"][0]
    assert result["status"] == "passed"
    assert any(path.endswith("probe.txt") for path in result["artifact_paths"])
    assert all(not path.endswith("stale.txt") for path in result["artifact_paths"])
    assert "probe-ok" in Path(result["stdout_path"]).read_text(encoding="utf-8")


def test_external_benchmark_full_run_reports_failed_status_for_missing_input(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_verified"]
    target.workdir = str(tmp_path)

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-verified"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["failed"] == 1
    result = report["results"][0]
    assert result["status"] == "failed"
    assert "predictions_path" in (result["blocker_reason"] or "")


def test_external_benchmark_classifies_missing_python_module_as_external_blocker(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_full"]
    target.workdir = str(tmp_path)
    target.smoke_command = ["python3", "-m", "module_that_does_not_exist_anywhere", "--help"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-full"],
        mode="smoke",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["external_blocked"] == 1
    result = report["results"][0]
    assert result["status"] == "external_blocked"
    assert "No module named" in (result["blocker_reason"] or "")


def test_external_benchmark_accepts_gold_literal_without_existing_path(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_verified"]
    target.workdir = str(tmp_path)
    target.full_command = ["python3", "-c", "print('gold-ok')"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-verified"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
        variables={"predictions_path": "gold"},
    )

    assert report["summary"]["passed"] == 1


def test_external_benchmark_expands_optional_argument_placeholders(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    target.workdir = str(tmp_path)
    script = tmp_path / "args_probe.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "print(json.dumps(sys.argv[1:]))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script), "base", "@args:{instance_ids_args}", "tail"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
        variables={"predictions_path": "gold", "instance_ids_args": "--instance_ids alpha beta"},
    )

    assert report["summary"]["passed"] == 1
    stdout = Path(report["results"][0]["stdout_path"]).read_text(encoding="utf-8")
    assert '["base", "--instance_ids", "alpha", "beta", "tail"]' in stdout


def test_external_benchmark_applies_target_default_variables_before_overrides(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    script = tmp_path / "defaults_probe.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "print(json.dumps(sys.argv[1:]))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script), "@args:{dataset_name_args}", "@args:{namespace_args}"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
        variables={"predictions_path": "gold", "dataset_name_args": "--dataset_name /tmp/row.json"},
    )

    assert report["summary"]["passed"] == 1
    stdout = Path(report["results"][0]["stdout_path"]).read_text(encoding="utf-8")
    assert '["--dataset_name", "/tmp/row.json", "--namespace", "none"]' in stdout


def test_external_benchmark_cache_argument_placeholder_is_optional(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_full"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    script = tmp_path / "cache_probe.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "print(json.dumps(sys.argv[1:]))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script), "head", "@args:{cache_args}", "tail"]

    default_report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-full"],
        mode="full",
        output_dir=tmp_path / "default",
        clean=True,
        config=config,
        variables={"predictions_path": "gold"},
    )
    default_stdout = Path(default_report["results"][0]["stdout_path"]).read_text(encoding="utf-8")
    assert '["head", "tail"]' in default_stdout

    override_report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-full"],
        mode="full",
        output_dir=tmp_path / "override",
        clean=True,
        config=config,
        variables={"predictions_path": "gold", "cache_args": "--cache_level instance"},
    )
    override_stdout = Path(override_report["results"][0]["stdout_path"]).read_text(encoding="utf-8")
    assert '["head", "--cache_level", "instance", "tail"]' in override_stdout


def test_external_benchmark_preflight_failure_is_external_blocker(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["terminal_bench"]
    target.workdir = str(tmp_path)
    target.preflight_commands = [["python3", "-c", "import sys; sys.stderr.write('compose missing\\n'); sys.exit(2)"]]
    target.full_command = ["python3", "-c", "print('should-not-run')"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["terminal-bench"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["external_blocked"] == 1
    assert report["results"][0]["status"] == "external_blocked"
    assert "compose missing" in (report["results"][0]["blocker_reason"] or "")


def test_external_benchmark_classifies_terminal_bench_compose_failure_from_report(
    tmp_path: Path,
) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["terminal_bench"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    script = tmp_path / "terminal_report_probe.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import sys",
                "out = Path(sys.argv[1])",
                "out.mkdir(parents=True, exist_ok=True)",
                "run_dir = out / 'run-1'",
                "run_dir.mkdir(parents=True, exist_ok=True)",
                "(run_dir / 'run.log').write_text('Running docker compose command: docker compose build\\nCommand returned non-zero exit status 2\\n', encoding='utf-8')",
                "(run_dir / 'results.json').write_text(json.dumps({'n_resolved': 0, 'n_unresolved': 1, 'results': [{'failure_mode': 'unknown_agent_error'}]}), encoding='utf-8')",
                "(out / 'terminal_bench_real_agent.json').write_text(json.dumps({'command': ['tb', 'run'], 'compose_mode': 'docker wrapper -> /tmp/docker-compose', 'exit_code': 0, 'task_count': 1, 'result_files': [str((run_dir / 'results.json').resolve())], 'run_log_files': [str((run_dir / 'run.log').resolve())], 'results_summary': {'resolved_count': 0, 'unresolved_count': 1, 'failure_modes': {'unknown_agent_error': 1}}}), encoding='utf-8')",
                "print(f'Report written to {(out / \"terminal_bench_real_agent.json\").resolve()}')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script), "{output_dir}"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["terminal-bench"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["external_blocked"] == 1
    result = report["results"][0]
    assert result["status"] == "external_blocked"
    assert result["environment_status"] == "external_blocked"
    assert result["solved_count_on_bounded_set"] == 0
    assert "docker compose" in (result["blocker_reason"] or "")


def test_benchmark_cli_dispatches_to_external_runner(capsys) -> None:
    exit_code = benchmark_main(["external", "list"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "swebench_lite" in output
    assert "terminal_bench" in output


def test_external_benchmark_timeout_becomes_explicit_external_blocker(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["terminal_bench"]
    target.workdir = str(tmp_path)
    target.smoke_command = ["python3", "-c", "import time; time.sleep(0.2)"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["terminal-bench"],
        mode="smoke",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
        timeout_seconds=1,
    )
    assert report["summary"]["passed"] == 1

    timed_out = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["terminal-bench"],
        mode="smoke",
        output_dir=tmp_path / "timeout",
        clean=True,
        config=config,
        timeout_seconds=1,
    )
    assert timed_out["summary"]["passed"] == 1

    target.smoke_command = ["python3", "-c", "import time; time.sleep(2)"]
    blocked = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["terminal-bench"],
        mode="smoke",
        output_dir=tmp_path / "blocked",
        clean=True,
        config=config,
        timeout_seconds=1,
    )
    assert blocked["summary"]["failed"] == 1
    assert blocked["results"][0]["status"] == "failed"
    assert "timed out" in blocked["results"][0]["blocker_reason"]
    assert blocked["results"][0]["duration_seconds"] >= 1


def test_external_benchmark_agent_run_generates_predictions_and_evaluates(tmp_path: Path, monkeypatch) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    target.default_variables["dataset_name_args"] = "--dataset_name demo/local"
    script = tmp_path / "eval_probe.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import sys",
                "predictions = Path(sys.argv[1])",
                "dataset = Path(sys.argv[sys.argv.index('--dataset_name') + 1])",
                "out = Path(sys.argv[-1]) / 'evaluation_results'",
                "out.mkdir(parents=True, exist_ok=True)",
                "report = Path.cwd() / 'swaag__local.swebench_lite-full.json'",
                "report.write_text(json.dumps({'total_instances': 1, 'submitted_instances': 1, 'completed_instances': 1, 'resolved_instances': 0, 'unresolved_instances': 1, 'empty_patch_instances': 0, 'error_instances': 0, 'schema_version': 2}), encoding='utf-8')",
                "payload = {",
                "  'prediction': predictions.read_text(encoding='utf-8').strip(),",
                "  'dataset': json.loads(dataset.read_text(encoding='utf-8'))[0]['instance_id'],",
                "}",
                "(out / 'result.json').write_text(json.dumps(payload), encoding='utf-8')",
                "print('agent-eval-ok')",
                "print(f'Report written to {report.name}')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script), "{predictions_path}", "@args:{dataset_name_args}", "{output_dir}"]
    target.artifact_globs = ["evaluation_results/**/*", "agent_predictions.jsonl", "dataset_subset.json", "generation_logs/**/*"]

    def fake_prepare(**kwargs):
        dataset_path = Path(kwargs["output_path"])
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "instance_id": "demo__repo-1",
                        "repo": "demo/repo",
                        "base_commit": "abc123",
                        "problem_statement": "Fix the bug.",
                        "hints_text": "",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return PreparedSwebenchSubset(
            dataset_name="demo/local",
            dataset_path=dataset_path,
            instance_ids=["demo__repo-1"],
            instances=[
                {
                    "instance_id": "demo__repo-1",
                    "repo": "demo/repo",
                    "base_commit": "abc123",
                    "problem_statement": "Fix the bug.",
                    "hints_text": "",
                }
            ],
        )

    def fake_generate(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        predictions_path = output_dir / "agent_predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "instance_id": "demo__repo-1",
                    "model_name_or_path": "swaag/real-agent-cli",
                    "model_patch": "diff --git a/demo.txt b/demo.txt",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        logs = output_dir / "generation_logs"
        logs.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / "demo.stdout.txt"
        stderr_path = logs / "demo.stderr.txt"
        stdout_path.write_text("agent-stdout", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return GeneratedSwebenchPredictions(
            predictions_path=predictions_path,
            generation_stdout_paths=[str(stdout_path)],
            generation_stderr_paths=[str(stderr_path)],
            workspace_paths=[str(output_dir / "workspaces" / "demo__repo-1")],
        )

    monkeypatch.setattr(external_benchmarks, "prepare_swebench_subset", fake_prepare)
    monkeypatch.setattr(external_benchmarks, "generate_agent_predictions", fake_generate)

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="agent",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["passed"] == 1
    result = report["results"][0]
    assert result["status"] == "passed"
    assert result["real_agent_run_implemented"] is True
    assert result["real_agent_run_attempted"] is True
    assert result["execution_path"] == "real_agent_cli"
    assert result["prepared_dataset_path"].endswith("dataset_subset.json")
    assert result["predictions_path"].endswith("agent_predictions.jsonl")
    assert result["evaluation_summary"]["completed_instances"] == 1
    assert any(path.endswith("result.json") for path in result["artifact_paths"])
    assert any(path.endswith("swaag__local.swebench_lite-full.json") for path in result["artifact_paths"])
    artifact_payload = json.loads((tmp_path / "out" / "swebench_lite" / "full" / "evaluation_results" / "result.json").read_text(encoding="utf-8"))
    assert artifact_payload["dataset"] == "demo__repo-1"
    assert "model_patch" in artifact_payload["prediction"]


def test_external_benchmark_agent_run_requires_non_empty_patch(tmp_path: Path, monkeypatch) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    target.default_variables["dataset_name_args"] = "--dataset_name demo/local"

    def fake_prepare(**kwargs):
        dataset_path = Path(kwargs["output_path"])
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "instance_id": "demo__repo-1",
                        "repo": "demo/repo",
                        "base_commit": "abc123",
                        "problem_statement": "Fix the bug.",
                        "hints_text": "",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return PreparedSwebenchSubset(
            dataset_name="demo/local",
            dataset_path=dataset_path,
            instance_ids=["demo__repo-1"],
            instances=[
                {
                    "instance_id": "demo__repo-1",
                    "repo": "demo/repo",
                    "base_commit": "abc123",
                    "problem_statement": "Fix the bug.",
                    "hints_text": "",
                }
            ],
        )

    def fake_generate(**kwargs):
        raise external_benchmarks.LocalSwebenchFailure("Agent did not produce any non-empty patch")

    monkeypatch.setattr(external_benchmarks, "prepare_swebench_subset", fake_prepare)
    monkeypatch.setattr(external_benchmarks, "generate_agent_predictions", fake_generate)

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="agent",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["failed"] == 1
    assert "non-empty patch" in (report["results"][0]["blocker_reason"] or "")


def test_external_benchmark_agent_run_classifies_model_server_connection_failure_as_external_blocker(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    target.default_variables["dataset_name_args"] = "--dataset_name demo/local"

    def fake_prepare(**kwargs):
        dataset_path = Path(kwargs["output_path"])
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "instance_id": "demo__repo-1",
                        "repo": "demo/repo",
                        "base_commit": "abc123",
                        "problem_statement": "Fix the bug.",
                        "hints_text": "",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return PreparedSwebenchSubset(
            dataset_name="demo/local",
            dataset_path=dataset_path,
            instance_ids=["demo__repo-1"],
            instances=[
                {
                    "instance_id": "demo__repo-1",
                    "repo": "demo/repo",
                    "base_commit": "abc123",
                    "problem_statement": "Fix the bug.",
                    "hints_text": "",
                }
            ],
        )

    def fake_generate(**kwargs):
        raise external_benchmarks.LocalSwebenchFailure(
            "Agent benchmark run failed: requests.exceptions.ConnectionError: "
            "HTTPConnectionPool(host='127.0.0.1', port=14829): Max retries exceeded with url: /completion"
        )

    monkeypatch.setattr(external_benchmarks, "prepare_swebench_subset", fake_prepare)
    monkeypatch.setattr(external_benchmarks, "generate_agent_predictions", fake_generate)

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="agent",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["external_blocked"] == 1
    result = report["results"][0]
    assert result["status"] == "external_blocked"
    assert "Local model server unavailable" in (result["blocker_reason"] or "")


def test_external_benchmark_agent_run_retries_transient_model_server_failure(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config()
    config.external_benchmarks.model_server.preflight_enabled = True
    config.external_benchmarks.model_server.retry_attempts = 2
    config.external_benchmarks.model_server.retry_sleep_seconds = 0.0

    calls = {"probe": 0, "generate": 0}

    def fake_prepare(**kwargs):
        dataset_path = kwargs["output_path"]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "instance_id": "demo__repo-1",
                        "repo": "demo/repo",
                        "base_commit": "abc123",
                        "problem_statement": "Fix the bug.",
                        "hints_text": "",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return PreparedSwebenchSubset(
            dataset_name="demo/local",
            dataset_path=dataset_path,
            instance_ids=["demo__repo-1"],
            instances=[
                {
                    "instance_id": "demo__repo-1",
                    "repo": "demo/repo",
                    "base_commit": "abc123",
                    "problem_statement": "Fix the bug.",
                    "hints_text": "",
                }
            ],
        )

    def fake_probe(_config):
        calls["probe"] += 1
        if calls["probe"] == 1:
            return external_benchmarks.ModelServerProbeResult(
                False,
                "connection_refused",
                "connection refused",
                "http://127.0.0.1:14829/health",
            )
        return external_benchmarks.ModelServerProbeResult(
            True,
            "ready",
            "ok",
            "http://127.0.0.1:14829/health",
        )

    def fake_generate(**kwargs):
        calls["generate"] += 1
        output_dir = kwargs["output_dir"]
        predictions_path = output_dir / "agent_predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "instance_id": "demo__repo-1",
                    "model_name_or_path": "demo",
                    "model_patch": "diff --git a/file b/file\n",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stdout_path = output_dir / "generation_logs" / "demo.stdout.txt"
        stderr_path = output_dir / "generation_logs" / "demo.stderr.txt"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return GeneratedSwebenchPredictions(
            predictions_path=predictions_path,
            generation_stdout_paths=[str(stdout_path.resolve())],
            generation_stderr_paths=[str(stderr_path.resolve())],
            workspace_paths=[],
        )

    def fake_run_target(**kwargs):
        issue_dir = tmp_path / "out" / "swebench_lite" / "full"
        issue_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = issue_dir / "stdout.txt"
        stderr_path = issue_dir / "stderr.txt"
        stdout_path.write_text("done", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return external_benchmarks.ExternalBenchmarkRunResult(
            benchmark_id="swebench_lite",
            benchmark_label="SWE-bench Lite",
            mode="full",
            harness_implemented=True,
            real_agent_run_implemented=False,
            real_agent_run_attempted=False,
            execution_path="official_harness",
            environment_status="ready",
            result_status="passed",
            solved_count_on_bounded_set=1,
            status="passed",
            command=["python3", "-m", "swebench.harness.run_evaluation"],
            workdir=str(tmp_path),
            output_dir=str(issue_dir),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            artifact_paths=[],
            exit_code=0,
            duration_seconds=0.1,
            evaluation_summary={"resolved_instances": 1},
        )

    monkeypatch.setattr(external_benchmarks, "prepare_swebench_subset", fake_prepare)
    monkeypatch.setattr(external_benchmarks, "_probe_model_server", fake_probe)
    monkeypatch.setattr(external_benchmarks, "generate_agent_predictions", fake_generate)
    monkeypatch.setattr(external_benchmarks, "_run_target", fake_run_target)

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="agent",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["passed"] == 1
    result = report["results"][0]
    assert result["status"] == "passed"
    assert result["environment_status"] == "ready"
    assert result["solved_count_on_bounded_set"] == 1
    assert calls["probe"] == 2
    assert calls["generate"] == 1
    assert any(path.endswith("model_server_probe.json") for path in result["artifact_paths"])


def test_external_benchmark_agent_run_classifies_dataset_fetch_failure_as_external_blocker(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_multilingual"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    target.default_variables["dataset_name_args"] = "--dataset_name SWE-bench/SWE-bench_Multilingual"

    def fake_prepare(**kwargs):
        raise external_benchmarks.LocalSwebenchFailure(
            "Unable to load benchmark dataset SWE-bench/SWE-bench_Multilingual: RuntimeError: Connection reset by peer"
        )

    monkeypatch.setattr(external_benchmarks, "prepare_swebench_subset", fake_prepare)

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-multilingual"],
        mode="agent",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
    )

    assert report["summary"]["external_blocked"] == 1
    result = report["results"][0]
    assert result["status"] == "external_blocked"
    assert "Local benchmark dependencies unavailable" in (result["blocker_reason"] or "")


def test_external_benchmark_classifies_swebench_report_errors_as_external_blocker(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_multilingual"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    target.artifact_globs = ["logs/**/*"]
    script = tmp_path / "report_probe.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import sys",
                "workdir = Path.cwd()",
                "target_output = Path(sys.argv[1])",
                "logs = workdir / 'logs' / 'build_images' / 'instances' / 'demo'",
                "logs.mkdir(parents=True, exist_ok=True)",
                "(logs / 'build_image.log').write_text(\"fatal: unable to access 'https://github.com/demo/repo/': Failure when receiving data from the peer\\n\", encoding='utf-8')",
                "report = {",
                "  'total_instances': 1,",
                "  'submitted_instances': 1,",
                "  'completed_instances': 0,",
                "  'resolved_instances': 0,",
                "  'unresolved_instances': 0,",
                "  'empty_patch_instances': 0,",
                "  'error_instances': 1,",
                "  'schema_version': 2,",
                "}",
                "report_path = workdir / 'swaag__local.swebench_multilingual-full.json'",
                "report_path.write_text(json.dumps(report), encoding='utf-8')",
                "print(f'Report written to {report_path.name}')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script), "{output_dir}"]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-multilingual"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
        variables={"predictions_path": "gold"},
    )

    assert report["summary"]["external_blocked"] == 1
    result = report["results"][0]
    assert result["status"] == "external_blocked"
    assert result["evaluation_summary"]["error_instances"] == 1
    assert any(path.endswith("swaag__local.swebench_multilingual-full.json") for path in result["artifact_paths"])
    assert not (tmp_path / "swaag__local.swebench_multilingual-full.json").exists()


def test_external_benchmark_preserves_passed_swebench_report_summary(tmp_path: Path) -> None:
    config = load_config()
    target = config.external_benchmarks.targets["swebench_lite"]
    target.workdir = str(tmp_path)
    target.preflight_commands = []
    script = tmp_path / "report_pass_probe.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "report = {",
                "  'total_instances': 1,",
                "  'submitted_instances': 1,",
                "  'completed_instances': 1,",
                "  'resolved_instances': 0,",
                "  'unresolved_instances': 1,",
                "  'empty_patch_instances': 0,",
                "  'error_instances': 0,",
                "  'schema_version': 2,",
                "}",
                "report_path = Path.cwd() / 'swaag__local.swebench_lite-full.json'",
                "report_path.write_text(json.dumps(report), encoding='utf-8')",
                "print(f'Report written to {report_path.name}')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target.full_command = ["python3", str(script)]

    report = external_benchmarks.run_external_benchmarks(
        benchmark_ids=["swebench-lite"],
        mode="full",
        output_dir=tmp_path / "out",
        clean=True,
        config=config,
        variables={"predictions_path": "gold"},
    )

    assert report["summary"]["passed"] == 1
    result = report["results"][0]
    assert result["status"] == "passed"
    assert result["evaluation_summary"]["completed_instances"] == 1
    assert result["evaluation_summary"]["unresolved_instances"] == 1
