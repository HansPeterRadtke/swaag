from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from swaag.benchmark.errors import LocalSwebenchFailure
from swaag.config import AgentConfig
from swaag.fsops import ensure_dir, write_text
from swaag.utils import stable_json_dumps


_TRANSIENT_GIT_ERROR_SNIPPETS = (
    "connection reset by peer",
    "connection was reset",
    "failure when receiving data from the peer",
    "http/2 stream",
    "tls",
    "timed out",
    "unexpected disconnect",
)

_RETRYABLE_AGENT_FAILURE_SNIPPETS = (
    "model returned invalid json for",
    "model returned non-object json for",
    "structured_validation_failure",
    "fatalsemanticengineerror",
)


@dataclass(slots=True)
class PreparedSwebenchSubset:
    dataset_name: str
    dataset_path: Path
    instance_ids: list[str]
    instances: list[dict[str, Any]]


@dataclass(slots=True)
class GeneratedSwebenchPredictions:
    predictions_path: Path
    generation_stdout_paths: list[str]
    generation_stderr_paths: list[str]
    workspace_paths: list[str]


def _load_dataset_loader():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise LocalSwebenchFailure(
            "SWE-bench dataset loading requires the optional 'datasets' package. "
            "Install the benchmark extras with `pip install swaag[official-benchmarks]` "
            "or install `datasets` directly."
        ) from exc
    return load_dataset


def load_dataset(*args, **kwargs):
    return _load_dataset_loader()(*args, **kwargs)


def _parse_value_args(raw: str, option: str) -> list[str]:
    if not raw.strip():
        return []
    parts = shlex.split(raw)
    values: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if token == option and index + 1 < len(parts):
            index += 1
            while index < len(parts) and not parts[index].startswith("--"):
                values.append(parts[index])
                index += 1
            continue
        index += 1
    return values


def parse_dataset_name(dataset_name_args: str) -> str:
    values = _parse_value_args(dataset_name_args, "--dataset_name")
    if not values:
        raise LocalSwebenchFailure("dataset_name_args must include --dataset_name")
    return values[-1]


def parse_instance_ids(instance_ids_args: str) -> list[str]:
    return _parse_value_args(instance_ids_args, "--instance_ids")


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        return [dict(payload)]
    raise LocalSwebenchFailure(f"Unsupported dataset payload in {path}")


def prepare_swebench_subset(
    *,
    benchmark_id: str,
    dataset_name: str,
    instance_ids: Sequence[str],
    output_path: Path,
    config: AgentConfig,
) -> PreparedSwebenchSubset:
    source_path = Path(dataset_name).expanduser()
    rows: list[dict[str, Any]]
    if source_path.exists():
        rows = _load_local_rows(source_path)
    else:
        try:
            dataset = load_dataset(dataset_name, split="test")
        except Exception as exc:
            raise LocalSwebenchFailure(
                f"Unable to load benchmark dataset {dataset_name}: {type(exc).__name__}: {exc}"
            ) from exc
        if instance_ids:
            wanted = set(instance_ids)
            rows = [dict(row) for row in dataset if row.get("instance_id") in wanted]
        else:
            rows = [dict(dataset[index]) for index in range(min(len(dataset), config.external_benchmarks.agent_generation.default_max_instances))]
    if instance_ids:
        wanted = list(instance_ids)
        rows_by_id = {str(row.get("instance_id")): row for row in rows}
        missing = [item for item in wanted if item not in rows_by_id]
        if missing:
            raise LocalSwebenchFailure(
                f"Unable to resolve requested instance ids for {benchmark_id}: {', '.join(missing)}"
            )
        rows = [rows_by_id[item] for item in wanted]
    if not rows:
        raise LocalSwebenchFailure(f"No benchmark rows available for {benchmark_id}")
    ensure_dir(output_path.parent)
    write_text(output_path, stable_json_dumps(rows, indent=2), encoding="utf-8")
    return PreparedSwebenchSubset(
        dataset_name=dataset_name,
        dataset_path=output_path,
        instance_ids=[str(row.get("instance_id", "")) for row in rows],
        instances=rows,
    )


def _git(args: Sequence[str], *, cwd: Path | None = None, timeout_seconds: int) -> None:
    attempts = 3
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode == 0:
            return
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        lowered = detail.lower()
        transient = any(fragment in lowered for fragment in _TRANSIENT_GIT_ERROR_SNIPPETS)
        if transient and attempt < attempts:
            time.sleep(attempt)
            continue
        raise LocalSwebenchFailure(f"git {' '.join(args)} failed: {detail}")


def _prepare_workspace(
    *,
    instance: dict[str, Any],
    workspace_root: Path,
    config: AgentConfig,
) -> Path:
    instance_id = str(instance["instance_id"])
    workspace = workspace_root / instance_id
    if workspace.exists():
        shutil.rmtree(workspace)
    ensure_dir(workspace)
    remote_base = config.external_benchmarks.agent_generation.git_remote_base_url.rstrip("/")
    remote_url = f"{remote_base}/{instance['repo']}.git"
    timeout_seconds = config.external_benchmarks.agent_generation.clone_timeout_seconds
    _git(["init"], cwd=workspace, timeout_seconds=timeout_seconds)
    _git(["remote", "add", "origin", remote_url], cwd=workspace, timeout_seconds=timeout_seconds)
    _git(["fetch", "--depth", "1", "origin", str(instance["base_commit"])], cwd=workspace, timeout_seconds=timeout_seconds)
    _git(["checkout", "--detach", "FETCH_HEAD"], cwd=workspace, timeout_seconds=timeout_seconds)
    _git(["config", "user.email", "benchmark@local.invalid"], cwd=workspace, timeout_seconds=timeout_seconds)
    _git(["config", "user.name", "swaag-benchmark"], cwd=workspace, timeout_seconds=timeout_seconds)
    return workspace


def _truncate_chars(text: str, *, limit: int) -> str:
    value = text.strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 16)].rstrip() + "\n...[truncated]"


def _render_prompt(instance: dict[str, Any], template: str, *, config: AgentConfig) -> str:
    fail_to_pass = instance.get("FAIL_TO_PASS", "")
    if isinstance(fail_to_pass, str):
        with contextlib.suppress(json.JSONDecodeError):
            fail_to_pass = json.loads(fail_to_pass)
    if isinstance(fail_to_pass, list):
        fail_to_pass_text = "\n".join(f"- {item}" for item in fail_to_pass)
    else:
        fail_to_pass_text = str(fail_to_pass or "").strip()
    benchmark_policy = config.external_benchmarks.agent_generation
    combined_limit = max(0, benchmark_policy.issue_prompt_char_limit)
    problem_statement = str(instance.get("problem_statement", "") or "").strip()
    hints_text = str(instance.get("hints_text", "") or "").strip()
    if combined_limit > 0 and len(problem_statement) + len(hints_text) > combined_limit:
        problem_limit = max(200, int(combined_limit * 0.7))
        hint_limit = max(0, combined_limit - problem_limit)
        problem_statement = _truncate_chars(problem_statement, limit=problem_limit)
        hints_text = _truncate_chars(hints_text, limit=hint_limit)
    rendered = template
    replacements = {
        "{repo}": str(instance.get("repo", "")),
        "{instance_id}": str(instance.get("instance_id", "")),
        "{problem_statement}": problem_statement,
        "{fail_to_pass_tests}": fail_to_pass_text,
        "{hints_text}": hints_text,
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    return rendered.strip()


def _render_empty_patch_retry_prompt(instance: dict[str, Any], template: str, *, config: AgentConfig) -> str:
    return _render_prompt(instance, template, config=config)


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _discover_server_context_limit(config: AgentConfig) -> int | None:
    base_url = config.model.base_url.rstrip("/")
    probe_url = f"{base_url}/props"
    timeout_seconds = config.external_benchmarks.model_server.healthcheck_timeout_seconds
    request = urllib.request.Request(probe_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    settings = payload.get("default_generation_settings")
    if not isinstance(settings, dict):
        return None
    params = settings.get("params")
    if not isinstance(params, dict):
        return None
    n_ctx = params.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        return n_ctx
    return None


def _resolved_agent_context_limit(
    config: AgentConfig,
    *,
    discovered_context_limit: int | None = None,
) -> int:
    generation = config.external_benchmarks.agent_generation
    requested_limit = min(config.model.context_limit, generation.agent_context_limit)
    if isinstance(discovered_context_limit, int) and discovered_context_limit > 0:
        return min(requested_limit, discovered_context_limit)
    return requested_limit


def _real_agent_env(
    *,
    workspace: Path,
    session_root: Path,
    config: AgentConfig,
    discovered_context_limit: int | None = None,
) -> dict[str, str]:
    generation = config.external_benchmarks.agent_generation
    env = dict(os.environ)
    repo_src = str(_repo_src_root())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_src if not existing_pythonpath else f"{repo_src}{os.pathsep}{existing_pythonpath}"
    env["SWAAG__SESSIONS__ROOT"] = str(session_root)
    env["SWAAG__MODEL__CONTEXT_LIMIT"] = str(
        _resolved_agent_context_limit(config, discovered_context_limit=discovered_context_limit)
    )
    env["SWAAG__MODEL__TIMEOUT_SECONDS"] = str(generation.model_timeout_seconds)
    env["SWAAG__MODEL__STRUCTURED_TIMEOUT_SECONDS"] = str(generation.model_structured_timeout_seconds)
    env["SWAAG__TOOLS__READ_ROOTS"] = stable_json_dumps([str(workspace)])
    env["SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS"] = (
        "true" if generation.allow_side_effect_tools else "false"
    )
    env["SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS"] = (
        "true" if generation.allow_stateful_tools else "false"
    )
    env["SWAAG__PLANNER__MAX_PLAN_STEPS"] = str(generation.planner_max_plan_steps)
    env["SWAAG__PLANNER__MAX_REPLANS"] = str(generation.planner_max_replans)
    env["SWAAG__RUNTIME__MAX_REASONING_STEPS"] = str(generation.runtime_max_reasoning_steps)
    env["SWAAG__RUNTIME__MAX_TOTAL_ACTIONS"] = str(generation.runtime_max_total_actions)
    env["SWAAG__RUNTIME__MAX_TOOL_STEPS"] = str(generation.runtime_max_tool_steps)
    env["SWAAG__RUNTIME__TOOL_CALL_BUDGET"] = str(generation.runtime_tool_call_budget)
    return env


def _is_retryable_agent_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _RETRYABLE_AGENT_FAILURE_SNIPPETS)


def _retry_prompt(base_prompt: str, attempt: int) -> str:
    return (
        base_prompt.rstrip()
        + "\n\nBenchmark recovery note:\n"
        + f"Previous benchmark attempt {attempt} failed before producing a usable patch because a structured "
        "agent response was malformed.\n"
        "Retry from scratch.\n"
        "Keep the plan extremely short: read -> edit -> verify -> respond.\n"
        "Use repo-relative paths only.\n"
        "Do not echo long absolute paths or restate the full issue text.\n"
        "Make one concrete code change.\n"
    )


def _run_agent_turn(
    *,
    workspace: Path,
    session_root: Path,
    session_name: str,
    prompt: str,
    prompt_path: Path,
    timeout_seconds: int,
    config: AgentConfig,
    discovered_context_limit: int | None = None,
) -> tuple[str, str, str]:
    env = _real_agent_env(
        workspace=workspace,
        session_root=session_root,
        config=config,
        discovered_context_limit=discovered_context_limit,
    )
    max_attempts = max(1, config.external_benchmarks.agent_generation.solver_max_attempts)
    prompt_attempts: list[str] = []
    stdout_attempts: list[str] = []
    stderr_attempts: list[str] = []
    final_detail = ""
    for attempt in range(1, max_attempts + 1):
        current_prompt = prompt if attempt == 1 else _retry_prompt(prompt, attempt - 1)
        prompt_attempts.append(current_prompt)
        write_text(prompt_path, "\n\n--- attempt ---\n\n".join(prompt_attempts), encoding="utf-8")
        attempt_session_name = session_name if attempt == 1 else f"{session_name}--retry-{attempt}"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "swaag",
                "ask",
                "--session",
                attempt_session_name,
            ],
            check=False,
            cwd=workspace,
            text=True,
            capture_output=True,
            input=current_prompt,
            timeout=timeout_seconds,
            env=env,
        )
        stdout_attempts.append(completed.stdout)
        stderr_attempts.append(completed.stderr)
        if completed.returncode == 0:
            return (
                "\n\n--- attempt ---\n\n".join(part for part in stdout_attempts if part),
                "\n\n--- attempt ---\n\n".join(part for part in stderr_attempts if part),
                "\n\n--- attempt ---\n\n".join(prompt_attempts),
            )
        final_detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        if attempt >= max_attempts or not _is_retryable_agent_failure(final_detail):
            break
    raise LocalSwebenchFailure(f"Agent benchmark run failed in {workspace}: {final_detail}")


def generate_agent_predictions(
    *,
    benchmark_id: str,
    subset: PreparedSwebenchSubset,
    output_dir: Path,
    config: AgentConfig,
) -> GeneratedSwebenchPredictions:
    workspaces_root = output_dir / "workspaces"
    sessions_root = output_dir / "sessions"
    logs_root = output_dir / "generation_logs"
    ensure_dir(workspaces_root)
    ensure_dir(sessions_root)
    ensure_dir(logs_root)
    predictions_path = output_dir / "agent_predictions.jsonl"
    model_name = config.external_benchmarks.agent_generation.model_name_or_path
    prompt_template = config.external_benchmarks.agent_generation.prompt_template
    retry_prompt_template = config.external_benchmarks.agent_generation.empty_patch_retry_prompt
    timeout_seconds = config.external_benchmarks.agent_generation.agent_timeout_seconds
    discovered_context_limit = _discover_server_context_limit(config)
    records: list[dict[str, Any]] = []
    stdout_paths: list[str] = []
    stderr_paths: list[str] = []
    workspace_paths: list[str] = []
    for instance in subset.instances:
        instance_id = str(instance["instance_id"])
        workspace = _prepare_workspace(instance=instance, workspace_root=workspaces_root, config=config)
        prompt = _render_prompt(instance, prompt_template, config=config)
        prompt_path = logs_root / f"{instance_id}.prompt.txt"
        stdout_text, stderr_text, prompt_text = _run_agent_turn(
            workspace=workspace,
            session_root=sessions_root,
            session_name=f"{benchmark_id}-{instance_id}",
            prompt=prompt,
            prompt_path=prompt_path,
            timeout_seconds=timeout_seconds,
            config=config,
            discovered_context_limit=discovered_context_limit,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
            timeout=max(30, timeout_seconds // 3),
        )
        if diff.returncode != 0:
            detail = diff.stderr.strip() or diff.stdout.strip() or f"exit code {diff.returncode}"
            raise LocalSwebenchFailure(f"git diff failed for {instance_id}: {detail}")
        if not diff.stdout.strip():
            retry_prompt = _render_empty_patch_retry_prompt(instance, retry_prompt_template, config=config)
            retry_prompt_path = logs_root / f"{instance_id}.retry_prompt.txt"
            retry_stdout, retry_stderr, retry_prompt_text = _run_agent_turn(
                workspace=workspace,
                session_root=sessions_root,
                session_name=f"{benchmark_id}-{instance_id}--empty-patch-retry",
                prompt=retry_prompt,
                prompt_path=retry_prompt_path,
                timeout_seconds=timeout_seconds,
                config=config,
                discovered_context_limit=discovered_context_limit,
            )
            stdout_text = "\n".join(part for part in (stdout_text, retry_stdout) if part)
            stderr_text = "\n".join(part for part in (stderr_text, retry_stderr) if part)
            prompt_text = "\n\n--- retry ---\n\n".join(part for part in (prompt_text, retry_prompt_text) if part)
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=workspace,
                check=False,
                text=True,
                capture_output=True,
                timeout=max(30, timeout_seconds // 3),
            )
            if diff.returncode != 0:
                detail = diff.stderr.strip() or diff.stdout.strip() or f"exit code {diff.returncode}"
                raise LocalSwebenchFailure(f"git diff failed for {instance_id}: {detail}")
        stdout_path = logs_root / f"{instance_id}.stdout.txt"
        stderr_path = logs_root / f"{instance_id}.stderr.txt"
        write_text(stdout_path, stdout_text, encoding="utf-8")
        write_text(stderr_path, stderr_text, encoding="utf-8")
        write_text(prompt_path, prompt_text, encoding="utf-8")
        records.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": diff.stdout,
            }
        )
        stdout_paths.append(str(stdout_path.resolve()))
        stderr_paths.append(str(stderr_path.resolve()))
        workspace_paths.append(str(workspace.resolve()))
    if not any(str(item.get("model_patch", "")).strip() for item in records):
        raise LocalSwebenchFailure(
            f"Agent did not produce any non-empty patch for {benchmark_id}; benchmark proof requires a real agent-generated patch."
        )
    write_text(predictions_path, "\n".join(stable_json_dumps(item) for item in records) + "\n", encoding="utf-8")
    return GeneratedSwebenchPredictions(
        predictions_path=predictions_path,
        generation_stdout_paths=stdout_paths,
        generation_stderr_paths=stderr_paths,
        workspace_paths=workspace_paths,
    )
