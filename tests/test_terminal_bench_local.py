from __future__ import annotations

import subprocess
from pathlib import Path

from terminal_bench.agents.failure_mode import FailureMode

from swaag.benchmark import terminal_bench_local
from swaag.benchmark import terminal_bench_agent
from swaag.benchmark.terminal_bench_agent import RealAgentTerminalBenchAgent
from swaag.config import load_config


def test_terminal_bench_local_run_forces_real_agent_import_path(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_runner_env():
        return ({"PYTHONPATH": "demo"}, "docker compose")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(terminal_bench_local, "_runner_env", fake_runner_env)
    monkeypatch.setattr(terminal_bench_local, "_agent_import_ready", lambda: None)
    monkeypatch.setattr(terminal_bench_local.subprocess, "run", fake_run)

    exit_code = terminal_bench_local.main(
        [
            "run",
            "--dataset-path",
            "src/swaag/benchmark/terminal_tasks",
            "--task-id",
            "hello-swaag",
            "--output-path",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert observed["command"][:4] == [
        "tb",
        "run",
        "--agent-import-path",
        "swaag.benchmark.terminal_bench_agent:RealAgentTerminalBenchAgent",
    ]
    assert "--task-id" in observed["command"]
    assert (tmp_path / "out" / "terminal_bench_real_agent.json").exists()


def test_terminal_bench_local_smoke_runs_tb_help(monkeypatch, capsys) -> None:
    def fake_runner_env():
        return ({"PYTHONPATH": "demo"}, "docker compose")

    def fake_run(command, **kwargs):
        assert command == ["tb", "run", "--help"]
        return subprocess.CompletedProcess(command, 0, stdout="tb-help\n", stderr="")

    monkeypatch.setattr(terminal_bench_local, "_runner_env", fake_runner_env)
    monkeypatch.setattr(terminal_bench_local, "_agent_import_ready", lambda: None)
    monkeypatch.setattr(terminal_bench_local.subprocess, "run", fake_run)

    exit_code = terminal_bench_local.main(["smoke"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "tb-help" in out
    assert "docker compose" in out


def test_terminal_bench_local_compose_ready_env_uses_configured_probe_timeout(monkeypatch) -> None:
    config = load_config()
    config.external_benchmarks.terminal_bench.compose_probe_timeout_seconds = 7
    config.external_benchmarks.terminal_bench.compose_download_timeout_seconds = 11
    config.external_benchmarks.terminal_bench.allow_compose_download = True
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="plugin missing")

    monkeypatch.setattr(terminal_bench_local, "load_config", lambda: config)
    monkeypatch.setattr(terminal_bench_local.subprocess, "run", fake_run)
    monkeypatch.setattr(
        terminal_bench_local.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(
        terminal_bench_local,
        "_ensure_compose_binary",
        lambda cache_root, timeout_seconds: Path(f"/tmp/docker-compose-{timeout_seconds}"),
    )
    monkeypatch.setattr(terminal_bench_local, "_write_docker_wrapper", lambda wrapper_dir, compose_binary: wrapper_dir / "docker")

    env, compose_mode = terminal_bench_local._compose_ready_env()

    assert observed["timeout"] == 7
    assert compose_mode == "docker wrapper -> /tmp/docker-compose-11"
    assert str(terminal_bench_local._cache_root() / "bin") in env["PATH"]


def test_real_agent_terminal_bench_agent_builds_bundle(tmp_path: Path) -> None:
    agent = RealAgentTerminalBenchAgent()

    bundle = agent._build_bundle("Fix the bug in app.py.")
    try:
        assert (bundle / "src" / "swaag" / "cli.py").exists()
        assert (bundle / "sitepkgs" / "requests").exists()
        assert "Fix the bug in app.py." in (bundle / "instruction.txt").read_text(encoding="utf-8")
        script = (bundle / "run_swaag.sh").read_text(encoding="utf-8")
        assert "python3 -m swaag ask" in script
        assert "SWAAG__MODEL__BASE_URL" in script
        assert 'export PYTHONPATH="/opt/src:/opt/sitepkgs:${PYTHONPATH:-}"' in script
        assert 'if [[ "${SWAAG_TERMINAL_HOST_NETWORK:-0}" == "1" ]]; then' in script
        assert "< /opt/instruction.txt" in script
    finally:
        bundle.parent.exists() and __import__("shutil").rmtree(bundle.parent, ignore_errors=True)


def test_real_agent_terminal_bench_agent_runs_bundle_script(monkeypatch, tmp_path: Path) -> None:
    class _Session:
        def __init__(self) -> None:
            self.copied = None
            self.keys = None

        def copy_to_container(self, paths, container_dir=None, container_filename=None):
            self.copied = (paths, container_dir, container_filename)

        def send_keys(self, keys, block=False, min_timeout_sec=0.0, max_timeout_sec=0.0):
            self.keys = (keys, block, max_timeout_sec)

    bundle_root = tmp_path / "bundle" / "swaag_bundle"
    bundle_root.mkdir(parents=True)
    (bundle_root / "run_swaag.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    agent = RealAgentTerminalBenchAgent()
    monkeypatch.setattr(agent, "_build_bundle", lambda instruction: bundle_root)
    session = _Session()

    result = agent.perform_task("fix it", session=session)

    assert result.failure_mode == FailureMode.NONE
    assert session.copied == (bundle_root, "/opt", None)
    assert session.keys == (["bash /opt/run_swaag.sh", "Enter"], True, float("inf"))


def test_real_agent_terminal_bench_agent_uses_configured_model_base_url(monkeypatch) -> None:
    config = load_config()
    config.model.base_url = "http://127.0.0.1:19000/v1"
    monkeypatch.setattr(terminal_bench_agent, "load_config", lambda: config)

    agent = RealAgentTerminalBenchAgent()
    bundle = agent._build_bundle("Fix the bug in app.py.")
    try:
        script = (bundle / "run_swaag.sh").read_text(encoding="utf-8")
        assert "MODEL_BASE_URL=http://127.0.0.1:19000/v1" in script
        assert "MODEL_BASE_URL_TEMPLATE=http://__SWAAG_HOST_IP__:19000/v1" in script
        assert 'export SWAAG__MODEL__BASE_URL="$MODEL_BASE_URL"' in script
        assert 'export SWAAG__MODEL__BASE_URL="${MODEL_BASE_URL_TEMPLATE/__SWAAG_HOST_IP__/$HOST_IP}"' in script
    finally:
        bundle.parent.exists() and __import__("shutil").rmtree(bundle.parent, ignore_errors=True)


def test_real_agent_terminal_bench_agent_normalizes_localhostish_model_base_url(monkeypatch) -> None:
    config = load_config()
    config.model.base_url = "http://0.0.0.0:19000/v1"
    monkeypatch.setattr(terminal_bench_agent, "load_config", lambda: config)

    assert RealAgentTerminalBenchAgent._configured_model_base_url() == "http://127.0.0.1:19000/v1"
    assert RealAgentTerminalBenchAgent._container_model_base_url_template() == "http://__SWAAG_HOST_IP__:19000/v1"
