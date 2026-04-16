from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from swaag.config import AgentConfig, ExternalBenchmarkTargetConfig, load_config
from swaag.benchmark.swebench_local import (
    LocalSwebenchFailure,
    generate_agent_predictions,
    parse_dataset_name,
    parse_instance_ids,
    prepare_swebench_subset,
)
from swaag.utils import stable_json_dumps


BENCHMARK_LABELS: dict[str, str] = {
    "swebench_lite": "SWE-bench Lite",
    "swebench_verified": "SWE-bench Verified",
    "swebench_multilingual": "SWE-bench Multilingual",
    "swebench_full": "Full SWE-bench",
    "terminal_bench": "Terminal-Bench",
}

BENCHMARK_ALIASES: dict[str, str] = {
    "swebench-lite": "swebench_lite",
    "swebench-verified": "swebench_verified",
    "swebench-multilingual": "swebench_multilingual",
    "swebench-full": "swebench_full",
    "terminal-bench": "terminal_bench",
}


def _coerce_subprocess_text(payload: str | bytes | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


class ExternalBenchmarkBlocked(RuntimeError):
    def __init__(self, reason: str, *, command: Sequence[str] | None = None, workdir: str | None = None):
        super().__init__(reason)
        self.command = list(command or [])
        self.workdir = workdir


class ExternalBenchmarkFailure(RuntimeError):
    def __init__(self, reason: str, *, command: Sequence[str] | None = None, workdir: str | None = None):
        super().__init__(reason)
        self.command = list(command or [])
        self.workdir = workdir


@dataclass(slots=True)
class ExternalBenchmarkRunResult:
    benchmark_id: str
    benchmark_label: str
    mode: str
    harness_implemented: bool
    real_agent_run_implemented: bool
    real_agent_run_attempted: bool
    execution_path: str
    environment_status: str
    result_status: str | None
    solved_count_on_bounded_set: int | None
    status: str
    command: list[str]
    workdir: str
    output_dir: str
    stdout_path: str
    stderr_path: str
    artifact_paths: list[str]
    exit_code: int | None
    duration_seconds: float
    blocker_reason: str | None = None
    prepared_dataset_path: str | None = None
    predictions_path: str | None = None
    generation_stdout_paths: list[str] | None = None
    generation_stderr_paths: list[str] | None = None
    workspace_paths: list[str] | None = None
    evaluation_summary: dict[str, int] | None = None


_REPORT_WRITTEN_RE = re.compile(r"Report written to (?P<path>\S+)")
_SWEBENCH_EXTERNAL_BLOCKER_SNIPPETS = (
    "failure when receiving data from the peer",
    "could not resolve host",
    "temporary failure in name resolution",
    "network is unreachable",
    "connection reset by peer",
    "connection timed out",
    "tls handshake timeout",
    "certificate verify failed",
    "remote end closed connection",
)


@dataclass(slots=True)
class ModelServerProbeResult:
    ready: bool
    classification: str
    detail: str
    url: str


def _canonical_benchmark_id(raw: str) -> str:
    value = raw.strip()
    return BENCHMARK_ALIASES.get(value, value)


def list_external_benchmarks(config: AgentConfig) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for benchmark_id, target in config.external_benchmarks.targets.items():
        items.append(
            {
                "benchmark_id": benchmark_id,
                "label": BENCHMARK_LABELS.get(benchmark_id, benchmark_id),
                "enabled": target.enabled,
                "description": target.description,
                "workdir": target.workdir,
                "smoke_command": list(target.smoke_command),
                "full_command": list(target.full_command),
            }
        )
    items.sort(key=lambda item: item["benchmark_id"])
    return items


def _parse_var_overrides(raw_values: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise SystemExit(f"Invalid --var value {raw!r}; expected key=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --var value {raw!r}; key must not be empty")
        values[key] = value
    return values


def _format_text(template: str, values: dict[str, str]) -> str:
    try:
        return template.format_map(values)
    except KeyError as exc:
        raise ExternalBenchmarkFailure(f"Missing template variable {exc.args[0]!r} for benchmark command") from exc


def _expand_command(command_template: Sequence[str], values: dict[str, str]) -> list[str]:
    command: list[str] = []
    for part in command_template:
        if part.startswith("@args:"):
            rendered = _format_text(part[len("@args:") :], values).strip()
            if rendered:
                command.extend(shlex.split(rendered))
            continue
        rendered = _format_text(part, values)
        if rendered != "":
            command.append(rendered)
    return command


def _resolve_target(
    config: AgentConfig,
    benchmark_id: str,
) -> tuple[str, ExternalBenchmarkTargetConfig]:
    canonical = _canonical_benchmark_id(benchmark_id)
    target = config.external_benchmarks.targets.get(canonical)
    if target is None:
        known = ", ".join(sorted(config.external_benchmarks.targets))
        raise SystemExit(f"Unknown external benchmark id {benchmark_id!r}. Known ids: {known}")
    if not target.enabled:
        raise ExternalBenchmarkFailure(f"External benchmark {canonical!r} is disabled in config")
    return canonical, target


def _collect_artifacts(
    workdir: Path,
    output_dir: Path,
    patterns: Sequence[str],
    values: dict[str, str],
    *,
    snapshot_mtimes: dict[str, int] | None = None,
) -> list[str]:
    found: set[str] = set()
    preexisting = snapshot_mtimes or {}
    for pattern in patterns:
        rendered = _format_text(pattern, values)
        for root in (output_dir, workdir):
            if not root.exists():
                continue
            for path in root.glob(rendered):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                try:
                    stat = path.stat()
                    previous_mtime = preexisting.get(resolved)
                    if previous_mtime is not None and stat.st_mtime_ns <= previous_mtime:
                        continue
                except FileNotFoundError:
                    continue
                found.add(resolved)
    return sorted(found)


def _snapshot_artifacts(
    workdir: Path,
    output_dir: Path,
    patterns: Sequence[str],
    values: dict[str, str],
) -> dict[str, int]:
    snapshot: dict[str, int] = {}
    for pattern in patterns:
        rendered = _format_text(pattern, values)
        for root in (output_dir, workdir):
            if not root.exists():
                continue
            for path in root.glob(rendered):
                if not path.is_file():
                    continue
                try:
                    snapshot[str(path.resolve())] = path.stat().st_mtime_ns
                except FileNotFoundError:
                    continue
    return snapshot


def _run_preflight_commands(
    *,
    target: ExternalBenchmarkTargetConfig,
    values: dict[str, str],
    workdir: Path,
    timeout_seconds: int,
) -> None:
    if not target.preflight_commands:
        return
    # Full benchmark preflights may need to pull official images or validate local
    # container tooling. Reuse the caller's full timeout budget instead of imposing
    # a much tighter hidden cap that can fail a correct local-first setup.
    preflight_timeout = max(10, timeout_seconds)
    for command_template in target.preflight_commands:
        command = _expand_command(command_template, values)
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                check=False,
                text=True,
                capture_output=True,
                timeout=preflight_timeout,
            )
        except FileNotFoundError as exc:
            raise ExternalBenchmarkBlocked(
                f"Benchmark preflight command not available: {exc.filename}",
                command=command,
                workdir=str(workdir),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            stderr_text = _coerce_subprocess_text(exc.stderr).strip()
            message = f"Benchmark preflight timed out after {preflight_timeout} seconds"
            if stderr_text:
                message = f"{message}: {stderr_text}"
            raise ExternalBenchmarkFailure(message, command=command, workdir=str(workdir)) from exc
        if completed.returncode == 0:
            continue
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise ExternalBenchmarkBlocked(
            f"Benchmark preflight failed: {detail}",
            command=command,
            workdir=str(workdir),
        )


def _extract_report_path(stdout_text: str, *, workdir: Path) -> Path | None:
    for line in reversed(stdout_text.splitlines()):
        match = _REPORT_WRITTEN_RE.search(line)
        if match is None:
            continue
        candidate = Path(match.group("path"))
        if not candidate.is_absolute():
            candidate = workdir / candidate
        return candidate
    return None


def _capture_report_artifact(report_path: Path, *, target_output: Path) -> Path | None:
    if not report_path.exists() or not report_path.is_file():
        return None
    destination = target_output / report_path.name
    if report_path.resolve() == destination.resolve():
        return report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(str(report_path), str(destination))
    return destination


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_text_excerpt(path: Path, *, max_chars: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def _swebench_external_blocker_reason(*texts: str) -> str | None:
    combined = "\n".join(texts).lower()
    for marker in _SWEBENCH_EXTERNAL_BLOCKER_SNIPPETS:
        if marker in combined:
            return marker
    return None


def _model_server_health_url(config: AgentConfig) -> str:
    base_url = config.model.base_url.rstrip("/")
    endpoint = config.model.health_endpoint.strip() or "/health"
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base_url}{endpoint}"


def _probe_model_server(config: AgentConfig) -> ModelServerProbeResult:
    url = _model_server_health_url(config)
    timeout_seconds = config.external_benchmarks.model_server.healthcheck_timeout_seconds
    request = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = response.read(512).decode("utf-8", errors="replace").strip()
    except urllib_error.HTTPError as exc:
        body = exc.read(512).decode("utf-8", errors="replace").strip()
        detail = f"health endpoint returned HTTP {exc.code}"
        if body:
            detail = f"{detail}: {body}"
        return ModelServerProbeResult(False, "server_unhealthy", detail, url)
    except urllib_error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError | socket.timeout):
            return ModelServerProbeResult(False, "timeout", "health check timed out", url)
        detail = str(reason).strip() or str(exc).strip()
        lowered = detail.lower()
        if "refused" in lowered:
            classification = "connection_refused"
        elif "timed out" in lowered or "timeout" in lowered:
            classification = "timeout"
        elif "reset" in lowered or "closed" in lowered or "aborted" in lowered:
            classification = "server_crashed"
        else:
            classification = "server_unreachable"
        return ModelServerProbeResult(False, classification, detail, url)
    except TimeoutError:
        return ModelServerProbeResult(False, "timeout", "health check timed out", url)
    if 200 <= status < 300:
        detail = body or f"HTTP {status}"
        return ModelServerProbeResult(True, "ready", detail, url)
    return ModelServerProbeResult(False, "server_unhealthy", f"health endpoint returned HTTP {status}", url)


def _write_model_server_probe_report(
    target_output: Path,
    *,
    attempt: int,
    max_attempts: int,
    probe: ModelServerProbeResult,
) -> Path:
    report_path = target_output / "model_server_probe.json"
    payload = {
        "attempt": attempt,
        "max_attempts": max_attempts,
        "url": probe.url,
        "ready": probe.ready,
        "classification": probe.classification,
        "detail": probe.detail,
    }
    report_path.write_text(stable_json_dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _reset_agent_generation_artifacts(target_output: Path) -> None:
    for path in (
        target_output / "workspaces",
        target_output / "sessions",
        target_output / "generation_logs",
    ):
        if path.exists():
            shutil.rmtree(path)
    predictions_path = target_output / "agent_predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()


def _classify_swebench_report(
    report_payload: dict[str, Any],
    *,
    stdout_text: str,
    stderr_text: str,
    artifact_paths: Sequence[str],
) -> tuple[str, str | None, dict[str, int]]:
    summary = {
        "total_instances": int(report_payload.get("total_instances", 0) or 0),
        "submitted_instances": int(report_payload.get("submitted_instances", 0) or 0),
        "completed_instances": int(report_payload.get("completed_instances", 0) or 0),
        "resolved_instances": int(report_payload.get("resolved_instances", 0) or 0),
        "unresolved_instances": int(report_payload.get("unresolved_instances", 0) or 0),
        "empty_patch_instances": int(report_payload.get("empty_patch_instances", 0) or 0),
        "error_instances": int(report_payload.get("error_instances", 0) or 0),
    }
    if summary["error_instances"] <= 0 and summary["completed_instances"] >= summary["submitted_instances"]:
        return "passed", None, summary
    artifact_texts = [_read_text_excerpt(Path(path)) for path in artifact_paths if path.endswith(".log")]
    blocker_marker = _swebench_external_blocker_reason(stdout_text, stderr_text, *artifact_texts)
    if blocker_marker is not None:
        reason = (
            "Official SWE-bench evaluator reported instance build/runtime errors caused by external infrastructure: "
            f"{blocker_marker}"
        )
        return "external_blocked", reason, summary
    reason = (
        "Official SWE-bench evaluator reported instance errors "
        f"(completed={summary['completed_instances']}, submitted={summary['submitted_instances']}, "
        f"errors={summary['error_instances']})"
    )
    return "failed", reason, summary


def _classify_terminal_bench_report(
    report_payload: dict[str, Any],
    *,
    stdout_text: str,
    stderr_text: str,
) -> tuple[str, str | None, dict[str, int]]:
    summary_payload = report_payload.get("results_summary", {})
    summary = {
        "resolved_count": int(summary_payload.get("resolved_count", 0) or 0),
        "unresolved_count": int(summary_payload.get("unresolved_count", 0) or 0),
        "task_count": int(report_payload.get("task_count", 0) or 0),
    }
    failure_modes = summary_payload.get("failure_modes", {})
    combined = "\n".join(
        [
            stdout_text,
            stderr_text,
            *(
                _read_text_excerpt(Path(path))
                for path in report_payload.get("run_log_files", [])
                if isinstance(path, str)
            ),
        ]
    ).lower()
    if summary["resolved_count"] > 0 and summary["unresolved_count"] == 0:
        return "passed", None, summary
    if "docker compose command" in combined and "returned non-zero exit status" in combined:
        reason = "Terminal-Bench environment failed while running docker compose for the task container"
        return "external_blocked", reason, summary
    if "docker compose" in combined and "not found" in combined:
        reason = "Terminal-Bench environment could not find a working docker compose command"
        return "external_blocked", reason, summary
    if isinstance(failure_modes, dict) and failure_modes.get("unknown_agent_error"):
        return "failed", "Terminal-Bench task reached the agent path but did not resolve the bounded task", summary
    reason = (
        "Terminal-Bench run completed without resolving the bounded task "
        f"(resolved={summary['resolved_count']}, unresolved={summary['unresolved_count']})"
    )
    return "failed", reason, summary


def _run_target(
    *,
    config: AgentConfig,
    benchmark_id: str,
    mode: str,
    output_root: Path,
    variables: dict[str, str],
    timeout_seconds: int,
) -> ExternalBenchmarkRunResult:
    canonical, target = _resolve_target(config, benchmark_id)
    target_output = output_root / canonical / mode
    target_output.mkdir(parents=True, exist_ok=True)
    stdout_path = target_output / "stdout.txt"
    stderr_path = target_output / "stderr.txt"
    values = {
        "output_dir": str(target_output.resolve()),
        "run_id": f"{canonical}-{mode}",
        "instance_ids_args": "",
        "task_selection_args": "--n-tasks 1",
        "dataset_locator_args": "--dataset terminal-bench-core==0.1.1",
        "namespace_args": "",
        "dataset_name_args": "",
        "cache_args": "",
        **target.default_variables,
        **variables,
    }
    command_template = target.smoke_command if mode == "smoke" else target.full_command
    if not command_template:
        raise ExternalBenchmarkFailure(f"No {mode} command configured for {canonical}")
    workdir = Path(_format_text(target.workdir, values)).expanduser()
    if not workdir.is_absolute():
        workdir = Path.cwd() / workdir
    if mode != "smoke":
        _run_preflight_commands(
            target=target,
            values=values,
            workdir=workdir,
            timeout_seconds=timeout_seconds,
        )
    if mode != "smoke":
        missing_env = [name for name in target.required_env if not values.get(name) and not bool(os.environ.get(name))]
        if missing_env:
            raise ExternalBenchmarkBlocked(f"Missing required environment variables: {', '.join(missing_env)}")
        missing_paths: list[str] = []
        for item in target.required_paths:
            rendered = _format_text(item, values)
            if rendered in target.allowed_path_literals:
                continue
            candidate = Path(rendered).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if not candidate.exists():
                missing_paths.append(str(candidate))
        if missing_paths:
            raise ExternalBenchmarkFailure(f"Missing required benchmark inputs: {', '.join(missing_paths)}")
    command = _expand_command(command_template, values)
    started = time.monotonic()
    preexisting_artifacts = _snapshot_artifacts(
        workdir,
        target_output,
        target.artifact_globs,
        values,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ExternalBenchmarkBlocked(
            f"Benchmark command not available: {exc.filename}",
            command=command,
            workdir=str(workdir),
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_coerce_subprocess_text(exc.stdout), encoding="utf-8")
        stderr_text = _coerce_subprocess_text(exc.stderr).strip()
        message = f"Benchmark command timed out after {timeout_seconds} seconds"
        if stderr_text:
            message = f"{message}: {stderr_text}"
        raise ExternalBenchmarkFailure(
            message,
            command=command,
            workdir=str(workdir),
        ) from exc
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    blocker_reason = None
    evaluation_summary: dict[str, int] | None = None
    environment_status = "ready"
    result_status: str | None = None
    solved_count_on_bounded_set: int | None = None
    if completed.returncode == 0:
        status = "passed"
    elif "ModuleNotFoundError" in completed.stderr or "No module named" in completed.stderr:
        status = "external_blocked"
        environment_status = "external_blocked"
        blocker_reason = completed.stderr.strip()
    else:
        status = "failed"
        blocker_reason = completed.stderr.strip() or completed.stdout.strip() or None
    artifacts = _collect_artifacts(
        workdir,
        target_output,
        target.artifact_globs,
        values,
        snapshot_mtimes=preexisting_artifacts,
    )
    report_path = _extract_report_path(completed.stdout, workdir=workdir)
    if report_path is not None:
        captured_report = _capture_report_artifact(report_path, target_output=target_output)
        if captured_report is not None:
            artifacts = sorted({*artifacts, str(captured_report.resolve())})
            report_payload = _read_json_file(captured_report)
            if report_payload is not None and canonical.startswith("swebench_"):
                status, blocker_reason, evaluation_summary = _classify_swebench_report(
                    report_payload,
                    stdout_text=completed.stdout,
                    stderr_text=completed.stderr,
                    artifact_paths=artifacts,
                )
                solved_count_on_bounded_set = evaluation_summary["resolved_instances"]
                result_status = "passed" if status == "passed" else "failed"
                if status == "external_blocked":
                    environment_status = "external_blocked"
            elif report_payload is not None and canonical == "terminal_bench":
                status, blocker_reason, evaluation_summary = _classify_terminal_bench_report(
                    report_payload,
                    stdout_text=completed.stdout,
                    stderr_text=completed.stderr,
                )
                solved_count_on_bounded_set = evaluation_summary["resolved_count"]
                result_status = "passed" if status == "passed" else "failed"
                if status == "external_blocked":
                    environment_status = "external_blocked"
    if result_status is None:
        result_status = "passed" if status == "passed" else "failed"
    if solved_count_on_bounded_set is None and evaluation_summary is not None:
        solved_count_on_bounded_set = next(
            (
                int(evaluation_summary[key])
                for key in ("resolved_instances", "resolved_count")
                if key in evaluation_summary
            ),
            None,
        )
    uses_real_agent_wrapper = canonical == "terminal_bench" and mode == "full"
    return ExternalBenchmarkRunResult(
        benchmark_id=canonical,
        benchmark_label=BENCHMARK_LABELS.get(canonical, canonical),
        mode=mode,
        harness_implemented=True,
        real_agent_run_implemented=uses_real_agent_wrapper,
        real_agent_run_attempted=uses_real_agent_wrapper,
        execution_path="real_agent_terminal_bench" if uses_real_agent_wrapper else "official_harness",
        environment_status=environment_status,
        result_status=result_status,
        solved_count_on_bounded_set=solved_count_on_bounded_set,
        status=status,
        command=command,
        workdir=str(workdir),
        output_dir=str(target_output),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        artifact_paths=artifacts,
        exit_code=completed.returncode,
        duration_seconds=round(time.monotonic() - started, 3),
        blocker_reason=blocker_reason,
        evaluation_summary=evaluation_summary,
    )


def _is_text_swebench_target(benchmark_id: str) -> bool:
    return benchmark_id in {
        "swebench_lite",
        "swebench_verified",
        "swebench_multilingual",
        "swebench_full",
    }


def _classify_model_server_message(message: str) -> str | None:
    lowered = message.lower()
    markers = (
        ("connection_refused", ("max retries exceeded with url: /completion", "failed to establish a new connection", "connection refused")),
        ("server_crashed", ("connection aborted", "remote end closed connection", "remotedisconnected", "connection reset by peer")),
        ("timeout", ("read timed out", "timed out", "timeout")),
        ("server_unreachable", ("requests.exceptions.connectionerror", "requests.exceptions.readtimeout")),
    )
    for classification, snippets in markers:
        if any(marker in lowered for marker in snippets):
            return classification
    return None


def _is_model_server_blocker(message: str) -> bool:
    return _classify_model_server_message(message) is not None


def _is_external_dependency_blocker(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "failure when receiving data from the peer",
            "connection reset by peer",
            "connection aborted",
            "could not resolve host",
            "temporary failure in name resolution",
            "network is unreachable",
            "read timed out",
            "ssl",
            "tls",
            "unable to load benchmark dataset",
            "huggingface.co",
            "github.com",
        )
    )


def _run_agent_target(
    *,
    config: AgentConfig,
    benchmark_id: str,
    output_root: Path,
    variables: dict[str, str],
    timeout_seconds: int,
) -> ExternalBenchmarkRunResult:
    canonical, target = _resolve_target(config, benchmark_id)
    if not _is_text_swebench_target(canonical):
        raise ExternalBenchmarkFailure(
            f"Agent-generated local benchmark runs are only implemented for local text SWE-bench targets; got {canonical}",
        )
    target_output = output_root / canonical / "agent"
    target_output.mkdir(parents=True, exist_ok=True)
    dataset_name_args = variables.get("dataset_name_args", target.default_variables.get("dataset_name_args", ""))
    dataset_name = parse_dataset_name(dataset_name_args)
    instance_ids = parse_instance_ids(variables.get("instance_ids_args", ""))
    probe_report_path: Path | None = None
    try:
        subset = prepare_swebench_subset(
            benchmark_id=canonical,
            dataset_name=dataset_name,
            instance_ids=instance_ids,
            output_path=target_output / "dataset_subset.json",
            config=config,
        )
    except LocalSwebenchFailure as exc:
        if _is_external_dependency_blocker(str(exc)):
            raise ExternalBenchmarkBlocked(
                f"Local benchmark dependencies unavailable during agent benchmark generation: {exc}"
            ) from exc
        raise
    attempts = max(1, config.external_benchmarks.model_server.retry_attempts)
    retry_sleep_seconds = config.external_benchmarks.model_server.retry_sleep_seconds
    generated = None
    last_model_error: str | None = None
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            _reset_agent_generation_artifacts(target_output)
        if config.external_benchmarks.model_server.preflight_enabled:
            probe = _probe_model_server(config)
            probe_report_path = _write_model_server_probe_report(
                target_output,
                attempt=attempt,
                max_attempts=attempts,
                probe=probe,
            )
            if not probe.ready:
                last_model_error = f"{probe.classification}: {probe.detail}"
                if attempt < attempts and retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)
                continue
        try:
            generated = generate_agent_predictions(
                benchmark_id=canonical,
                subset=subset,
                output_dir=target_output,
                config=config,
            )
            break
        except LocalSwebenchFailure as exc:
            if _is_model_server_blocker(str(exc)):
                classification = _classify_model_server_message(str(exc)) or "server_unreachable"
                last_model_error = f"{classification}: {exc}"
                if probe_report_path is not None:
                    probe_report_path.write_text(
                        stable_json_dumps(
                            {
                                "attempt": attempt,
                                "max_attempts": attempts,
                                "url": _model_server_health_url(config),
                                "ready": False,
                                "classification": classification,
                                "detail": str(exc),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if attempt < attempts and retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)
                continue
            if _is_external_dependency_blocker(str(exc)):
                raise ExternalBenchmarkBlocked(
                    f"Local benchmark dependencies unavailable during agent benchmark generation: {exc}"
                ) from exc
            raise
    if generated is None:
        detail = last_model_error or "model server unavailable"
        raise ExternalBenchmarkBlocked(
            f"Local model server unavailable during agent benchmark generation after {attempts} attempt(s): {detail}"
        )
    run_variables = dict(variables)
    run_variables["predictions_path"] = str(generated.predictions_path)
    run_variables["dataset_name_args"] = f"--dataset_name {subset.dataset_path}"
    run_variables["instance_ids_args"] = ""
    result = _run_target(
        config=config,
        benchmark_id=canonical,
        mode="full",
        output_root=output_root,
        variables=run_variables,
        timeout_seconds=timeout_seconds,
    )
    artifact_paths = sorted(
        {
            *result.artifact_paths,
            str(subset.dataset_path.resolve()),
            str(generated.predictions_path.resolve()),
            *generated.generation_stdout_paths,
            *generated.generation_stderr_paths,
            *([str(probe_report_path.resolve())] if probe_report_path is not None else []),
        }
    )
    return ExternalBenchmarkRunResult(
        benchmark_id=result.benchmark_id,
        benchmark_label=result.benchmark_label,
        mode="agent",
        harness_implemented=result.harness_implemented,
        real_agent_run_implemented=True,
        real_agent_run_attempted=True,
        execution_path="real_agent_cli",
        environment_status=result.environment_status,
        result_status=result.result_status,
        solved_count_on_bounded_set=result.solved_count_on_bounded_set,
        status=result.status,
        command=result.command,
        workdir=result.workdir,
        output_dir=result.output_dir,
        stdout_path=result.stdout_path,
        stderr_path=result.stderr_path,
        artifact_paths=artifact_paths,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        blocker_reason=result.blocker_reason,
        prepared_dataset_path=str(subset.dataset_path.resolve()),
        predictions_path=str(generated.predictions_path.resolve()),
        generation_stdout_paths=generated.generation_stdout_paths,
        generation_stderr_paths=generated.generation_stderr_paths,
        workspace_paths=generated.workspace_paths,
        evaluation_summary=result.evaluation_summary,
    )


def run_external_benchmarks(
    *,
    benchmark_ids: Sequence[str],
    mode: str,
    output_dir: Path,
    clean: bool = False,
    variables: dict[str, str] | None = None,
    config: AgentConfig | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if mode not in {"smoke", "full", "agent"}:
        raise ValueError(f"Unsupported external benchmark mode: {mode}")
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config() if config is None else config
    variable_map = dict(variables or {})
    results: list[ExternalBenchmarkRunResult] = []
    started = time.monotonic()
    effective_timeout_seconds = int(
        timeout_seconds
        if timeout_seconds is not None
        else (
            config.external_benchmarks.smoke_timeout_seconds
            if mode == "smoke"
            else config.external_benchmarks.full_timeout_seconds
        )
    )
    for benchmark_id in benchmark_ids:
        canonical = _canonical_benchmark_id(benchmark_id)
        run_started = time.monotonic()
        try:
            if mode == "agent":
                result = _run_agent_target(
                    config=config,
                    benchmark_id=canonical,
                    output_root=output_dir,
                    variables=variable_map,
                    timeout_seconds=effective_timeout_seconds,
                )
            else:
                result = _run_target(
                    config=config,
                    benchmark_id=canonical,
                    mode=mode,
                    output_root=output_dir,
                    variables=variable_map,
                    timeout_seconds=effective_timeout_seconds,
                )
        except (ExternalBenchmarkBlocked, ExternalBenchmarkFailure) as exc:
            issue_dir = output_dir / canonical / mode
            issue_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = issue_dir / "stdout.txt"
            stderr_path = issue_dir / "stderr.txt"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            status = "external_blocked" if isinstance(exc, ExternalBenchmarkBlocked) else "failed"
            result = ExternalBenchmarkRunResult(
                benchmark_id=canonical,
                benchmark_label=BENCHMARK_LABELS.get(canonical, canonical),
                mode=mode,
                harness_implemented=True,
                real_agent_run_implemented=mode == "agent",
                real_agent_run_attempted=mode == "agent",
                execution_path="real_agent_cli" if mode == "agent" else "official_harness",
                environment_status="external_blocked" if isinstance(exc, ExternalBenchmarkBlocked) else "ready",
                result_status="failed",
                solved_count_on_bounded_set=0,
                status=status,
                command=list(exc.command),
                workdir=exc.workdir or str(Path.cwd()),
                output_dir=str(issue_dir),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                artifact_paths=[],
                exit_code=None,
                duration_seconds=round(time.monotonic() - run_started, 3),
                blocker_reason=str(exc),
            )
        except LocalSwebenchFailure as exc:
            issue_dir = output_dir / canonical / mode
            issue_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = issue_dir / "stdout.txt"
            stderr_path = issue_dir / "stderr.txt"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            result = ExternalBenchmarkRunResult(
                benchmark_id=canonical,
                benchmark_label=BENCHMARK_LABELS.get(canonical, canonical),
                mode=mode,
                harness_implemented=True,
                real_agent_run_implemented=mode == "agent",
                real_agent_run_attempted=mode == "agent",
                execution_path="real_agent_cli" if mode == "agent" else "official_harness",
                environment_status="ready",
                result_status="failed",
                solved_count_on_bounded_set=0,
                status="failed",
                command=[],
                workdir=str(Path.cwd()),
                output_dir=str(issue_dir),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                artifact_paths=[],
                exit_code=None,
                duration_seconds=round(time.monotonic() - run_started, 3),
                blocker_reason=str(exc),
            )
        results.append(result)
    payload = {
        "mode": mode,
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item.status == "passed"),
            "failed": sum(1 for item in results if item.status == "failed"),
            "external_blocked": sum(1 for item in results if item.status == "external_blocked"),
        },
        "timeout_seconds": effective_timeout_seconds,
        "results": [asdict(item) for item in results],
        "wall_clock_seconds": round(time.monotonic() - started, 3),
    }
    json_path = output_dir / f"external_benchmark_{mode}_results.json"
    markdown_path = output_dir / f"external_benchmark_{mode}_report.md"
    json_path.write_text(stable_json_dumps(payload, indent=2), encoding="utf-8")
    markdown_lines = [
        f"# External benchmark {mode} report",
        "",
        f"- total: {payload['summary']['total']}",
        f"- passed: {payload['summary']['passed']}",
        f"- failed: {payload['summary']['failed']}",
        f"- external_blocked: {payload['summary']['external_blocked']}",
        "",
        "| Benchmark | Status | Environment | Result | Execution | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        evidence = item.blocker_reason or item.stderr_path
        execution = item.execution_path
        if item.real_agent_run_attempted:
            execution = f"{execution} (real-agent)"
        markdown_lines.append(
            f"| {item.benchmark_label} | {item.status} | {item.environment_status} | "
            f"{item.result_status or '-'} | {execution} | `{evidence}` |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return payload


def _selected_benchmarks(args: argparse.Namespace, config: AgentConfig) -> list[str]:
    if getattr(args, "all", False):
        return sorted(config.external_benchmarks.targets)
    selected = [_canonical_benchmark_id(item) for item in getattr(args, "benchmark", [])]
    if selected:
        return selected
    raise SystemExit("Pass --benchmark <id> (repeatable) or --all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m swaag.benchmark external")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List configured external benchmark integrations.")
    for mode in ("smoke", "run", "agent-run"):
        sub = subparsers.add_parser(mode, help=f"Run the external benchmark {mode} path.")
        sub.add_argument("--benchmark", action="append", default=[], help="Benchmark id to run. Repeat for multiple ids.")
        sub.add_argument("--all", action="store_true", help="Run every configured benchmark.")
        sub.add_argument("--output", default="external_benchmark_output", help="Output directory for reports and captured stdout/stderr.")
        sub.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
        sub.add_argument("--var", action="append", default=[], help="Template variable override in key=value form.")
        sub.add_argument("--timeout-seconds", type=int, help="Override the per-command timeout for this run.")
        sub.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "list":
        for item in list_external_benchmarks(config):
            print(f"{item['benchmark_id']}: {item['description']}")
        return 0
    benchmark_ids = _selected_benchmarks(args, config)
    payload = run_external_benchmarks(
        benchmark_ids=benchmark_ids,
        mode="smoke" if args.command == "smoke" else ("agent" if args.command == "agent-run" else "full"),
        output_dir=Path(args.output),
        clean=bool(args.clean),
        variables=_parse_var_overrides(args.var),
        config=config,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(stable_json_dumps(payload, indent=2))
    else:
        summary = payload["summary"]
        print(f"total={summary['total']}")
        print(f"passed={summary['passed']}")
        print(f"failed={summary['failed']}")
        print(f"external_blocked={summary['external_blocked']}")
    if payload["summary"]["failed"] > 0:
        return 1
    if payload["summary"]["external_blocked"] > 0:
        return 2
    return 0
