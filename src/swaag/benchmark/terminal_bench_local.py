from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

from swaag.config import load_config
from swaag.utils import stable_json_dumps


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_src_root() -> Path:
    return _repo_root() / "src"


def _cache_root() -> Path:
    return _repo_root() / ".swaag" / "terminal_bench"


def _docker_binary() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("docker is not installed or not on PATH")
    return docker


def _compose_download_url() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux":
        raise RuntimeError(f"Unsupported platform for local compose bootstrap: {platform.system()}")
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    arch = arch_map.get(machine)
    if arch is None:
        raise RuntimeError(f"Unsupported architecture for local compose bootstrap: {platform.machine()}")
    return f"https://github.com/docker/compose/releases/latest/download/docker-compose-linux-{arch}"


def _ensure_compose_binary(cache_root: Path, *, timeout_seconds: int) -> Path:
    compose_binary = cache_root / "docker-compose"
    if compose_binary.exists():
        return compose_binary
    cache_root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_compose_download_url(), timeout=timeout_seconds) as response:
        compose_binary.write_bytes(response.read())
    compose_binary.chmod(compose_binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return compose_binary


def _write_docker_wrapper(wrapper_dir: Path, compose_binary: Path) -> Path:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    docker_wrapper = wrapper_dir / "docker"
    docker_wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"DOCKER_REAL={shlex.quote(_docker_binary())}",
                f"COMPOSE_REAL={shlex.quote(str(compose_binary))}",
                'if [[ "${1:-}" == "compose" ]]; then',
                '  shift',
                '  exec "$COMPOSE_REAL" "$@"',
                "fi",
                'exec "$DOCKER_REAL" "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    docker_wrapper.chmod(docker_wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return docker_wrapper


def _compose_ready_env() -> tuple[dict[str, str], str]:
    config = load_config().external_benchmarks.terminal_bench
    env = dict(os.environ)
    probe_timeout = config.compose_probe_timeout_seconds
    try:
        native = subprocess.run(
            ["docker", "compose", "version"],
            check=False,
            text=True,
            capture_output=True,
            timeout=probe_timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"docker is not installed or not on PATH: {exc.filename}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out after {probe_timeout} seconds while probing docker compose availability"
        ) from exc
    if native.returncode == 0:
        return env, "docker compose"
    compose_binary = shutil.which("docker-compose")
    if compose_binary is None:
        if not config.allow_compose_download:
            raise RuntimeError(
                "docker compose plugin is unavailable, docker-compose is missing, and repo-local compose bootstrap is disabled"
            )
        compose_binary = str(
            _ensure_compose_binary(
                _cache_root(),
                timeout_seconds=config.compose_download_timeout_seconds,
            )
        )
    wrapper_dir = _cache_root() / "bin"
    _write_docker_wrapper(wrapper_dir, Path(compose_binary))
    env["PATH"] = f"{wrapper_dir}{os.pathsep}{env.get('PATH', '')}"
    return env, f"docker wrapper -> {compose_binary}"


def _tb_command(tb_args: Sequence[str]) -> list[str]:
    return [
        "tb",
        "run",
        "--agent-import-path",
        "swaag.benchmark.terminal_bench_agent:RealAgentTerminalBenchAgent",
        *tb_args,
    ]


def _agent_import_ready() -> None:
    module = importlib.import_module("swaag.benchmark.terminal_bench_agent")
    if not hasattr(module, "RealAgentTerminalBenchAgent"):
        raise RuntimeError("RealAgentTerminalBenchAgent is not importable")


def _runner_env() -> tuple[dict[str, str], str]:
    env, compose_mode = _compose_ready_env()
    repo_src = str(_repo_src_root())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_src if not existing_pythonpath else f"{repo_src}{os.pathsep}{existing_pythonpath}"
    return env, compose_mode


def _run_smoke() -> int:
    _agent_import_ready()
    env, compose_mode = _runner_env()
    completed = subprocess.run(
        ["tb", "run", "--help"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode
    print(stable_json_dumps({"compose_mode": compose_mode}, indent=2))
    return 0


def _write_report(output_path: Path, *, command: list[str], compose_mode: str, exit_code: int) -> None:
    result_files = sorted(str(path.resolve()) for path in output_path.rglob("results.json"))
    run_log_files = sorted(str(path.resolve()) for path in output_path.rglob("run.log"))
    resolved_count = 0
    unresolved_count = 0
    failure_modes: dict[str, int] = {}
    for result_file in result_files:
        try:
            payload = json.loads(Path(result_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        resolved_count += int(payload.get("n_resolved", 0) or 0)
        unresolved_count += int(payload.get("n_unresolved", 0) or 0)
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            failure_mode = str(item.get("failure_mode", "") or "")
            if not failure_mode:
                continue
            failure_modes[failure_mode] = failure_modes.get(failure_mode, 0) + 1
    payload = {
        "command": command,
        "compose_mode": compose_mode,
        "exit_code": exit_code,
        "result_files": result_files,
        "run_log_files": run_log_files,
        "task_count": len(result_files),
        "results_summary": {
            "resolved_count": resolved_count,
            "unresolved_count": unresolved_count,
            "failure_modes": failure_modes,
        },
    }
    report_path = output_path / "terminal_bench_real_agent.json"
    report_path.write_text(stable_json_dumps(payload, indent=2), encoding="utf-8")
    print(f"Report written to {report_path.name}")


def _run_tb(tb_args: Sequence[str]) -> int:
    if any(arg in {"--agent", "--agent-import-path"} for arg in tb_args):
        raise SystemExit("terminal_bench_local manages the real-agent adapter itself; do not pass --agent or --agent-import-path")
    _agent_import_ready()
    env, compose_mode = _runner_env()
    command = _tb_command(tb_args)
    completed = subprocess.run(command, check=False, env=env)
    output_path = None
    for index, token in enumerate(tb_args):
        if token == "--output-path" and index + 1 < len(tb_args):
            output_path = Path(tb_args[index + 1]).expanduser().resolve()
            break
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        _write_report(output_path, command=command, compose_mode=compose_mode, exit_code=completed.returncode)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m swaag.benchmark.terminal_bench_local")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Validate the local Terminal-Bench real-agent wrapper.")
    subparsers.add_parser("run", help="Run Terminal-Bench through the real swaag adapter.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "smoke":
        return _run_smoke()
    if args.command == "run":
        forwarded = list(extra)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return _run_tb(forwarded)
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
