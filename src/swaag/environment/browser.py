from __future__ import annotations

import importlib.util
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig
from swaag.environment.process import ProcessManager, ProcessResult


class BrowserAutomationError(RuntimeError):
    pass


@dataclass(slots=True)
class AubroInvocation:
    command_prefix: list[str]
    env_overrides: dict[str, str]
    source_path: str | None


@dataclass(slots=True)
class AubroCommandResult:
    payload: dict[str, Any]
    process_result: ProcessResult
    invocation: AubroInvocation


def _repo_aubro_src_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidate = parent / "aubro" / "src"
        if (candidate / "aubro" / "__init__.py").exists():
            candidates.append(candidate)
    return candidates


def discover_aubro_src(config: AgentConfig) -> Path | None:
    explicit = (config.environment.aubro_src or "").strip()
    env_override = os.environ.get("SWAAG_AUBRO_SRC", "").strip()
    for raw in [env_override, explicit]:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if (candidate / "aubro" / "__init__.py").exists():
            return candidate
    for candidate in _repo_aubro_src_candidates():
        return candidate.resolve()
    return None


def aubro_available(config: AgentConfig) -> bool:
    entrypoint = (os.environ.get("SWAAG_AUBRO_ENTRYPOINT") or config.environment.aubro_entrypoint or "").strip()
    if entrypoint:
        return True
    if importlib.util.find_spec("aubro.cli") is not None:
        return True
    return discover_aubro_src(config) is not None


def resolve_aubro_invocation(config: AgentConfig) -> AubroInvocation:
    entrypoint = (os.environ.get("SWAAG_AUBRO_ENTRYPOINT") or config.environment.aubro_entrypoint or "").strip()
    if entrypoint:
        return AubroInvocation(command_prefix=shlex.split(entrypoint), env_overrides={}, source_path=None)

    command_prefix = [sys.executable, "-m", "aubro.cli"]
    if importlib.util.find_spec("aubro.cli") is not None:
        return AubroInvocation(command_prefix=command_prefix, env_overrides={}, source_path=None)

    source_path = discover_aubro_src(config)
    if source_path is None:
        raise BrowserAutomationError(
            "aubro is unavailable: set environment.aubro_entrypoint, SWAAG_AUBRO_ENTRYPOINT, "
            "or environment.aubro_src/SWAAG_AUBRO_SRC"
        )
    env_overrides: dict[str, str] = {}
    existing = os.environ.get("PYTHONPATH", "")
    env_overrides["PYTHONPATH"] = str(source_path) if not existing else f"{source_path}{os.pathsep}{existing}"
    return AubroInvocation(command_prefix=command_prefix, env_overrides=env_overrides, source_path=str(source_path))


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def run_aubro_command(
    *,
    config: AgentConfig,
    process_manager: ProcessManager,
    command_suffix: list[str],
    cwd: Path,
    env: dict[str, str],
) -> AubroCommandResult:
    invocation = resolve_aubro_invocation(config)
    effective_env = dict(env)
    effective_env.update(invocation.env_overrides)
    process_result = process_manager.run(
        [*invocation.command_prefix, *command_suffix],
        cwd=cwd,
        env=effective_env,
        timeout_seconds=config.environment.aubro_timeout_seconds,
        metadata={"kind": "aubro", "command_suffix": " ".join(command_suffix)},
    )
    stdout = _trim_text(process_result.stdout, config.environment.max_capture_chars)
    stderr = _trim_text(process_result.stderr, config.environment.max_capture_chars)
    process_result.record.stdout = stdout
    process_result.record.stderr = stderr
    if process_result.record.return_code != 0:
        raise BrowserAutomationError(
            f"aubro command failed with exit code {process_result.record.return_code}: {stderr or stdout}"
        )
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise BrowserAutomationError(f"aubro returned invalid JSON: {stdout[:400]!r}") from exc
    if not isinstance(payload, dict):
        raise BrowserAutomationError("aubro returned a non-object JSON payload")
    return AubroCommandResult(payload=payload, process_result=process_result, invocation=invocation)
