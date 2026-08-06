from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swaag.tools.base import ToolValidationError, _validate_schema_value
from swaag.types import HistoryEvent, SessionState


_BENIGN_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_BENIGN_SUFFIXES = (".pyc", ".pyo")


@dataclass(slots=True)
class BenchmarkVerificationReport:
    passed: bool
    task_type: str
    checks: dict[str, bool]
    evidence: dict[str, Any]
    reason: str


def _is_benign_workspace_artifact(path_text: str) -> bool:
    normalized = str(path_text).replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = normalized.split("/")
    if any(part in _BENIGN_DIRS for part in parts):
        return True
    name = parts[-1]
    return name == ".coverage" or name.startswith(".coverage.") or name.endswith(_BENIGN_SUFFIXES)


def _run_command(command: list[str], *, cwd: str | None) -> tuple[bool, dict[str, Any]]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:
        return False, {"command": command, "cwd": cwd, "error": str(exc), "error_type": exc.__class__.__name__}
    evidence = {
        "command": command,
        "cwd": cwd,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    return completed.returncode == 0, evidence


def verify_benchmark_contract(
    contract: Any,
    *,
    assistant_text: str,
    state: SessionState,
    events: list[HistoryEvent],
    workspace_before: dict[str, str] | None = None,
    workspace_after: dict[str, str] | None = None,
    **_ignored: Any,
) -> BenchmarkVerificationReport:
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    event_counts = Counter(event.event_type for event in events)
    seen_events = set(event_counts)

    required_events = set(getattr(contract, "required_history_events", []) or [])
    checks["required_history_events"] = required_events.issubset(seen_events)
    evidence["required_history_events"] = {
        "required": sorted(required_events),
        "seen": sorted(seen_events),
        "missing": sorted(required_events - seen_events),
    }
    forbidden_events = set(getattr(contract, "forbidden_history_events", []) or [])
    if forbidden_events:
        checks["forbidden_history_events"] = forbidden_events.isdisjoint(seen_events)
        evidence["forbidden_history_events"] = {
            "forbidden": sorted(forbidden_events),
            "seen": sorted(seen_events & forbidden_events),
        }
    required_counts = dict(getattr(contract, "required_event_counts", {}) or {})
    if required_counts:
        checks["required_event_counts"] = all(event_counts.get(name, 0) >= count for name, count in required_counts.items())
        evidence["required_event_counts"] = {
            name: {"required": count, "actual": event_counts.get(name, 0)}
            for name, count in sorted(required_counts.items())
        }

    expected_answer = getattr(contract, "expected_answer", None)
    if expected_answer is not None:
        checks["expected_answer"] = assistant_text.strip() == str(expected_answer).strip()
        evidence["expected_answer"] = {"actual": assistant_text.strip(), "expected": str(expected_answer).strip()}
    contains = list(getattr(contract, "expected_answer_contains", []) or [])
    if contains:
        checks["expected_answer_contains"] = all(fragment in assistant_text for fragment in contains)
        evidence["expected_answer_contains"] = {"actual": assistant_text, "required_fragments": contains}
    regex = getattr(contract, "expected_answer_regex", None)
    if regex is not None:
        checks["expected_answer_regex"] = re.search(str(regex), assistant_text) is not None
        evidence["expected_answer_regex"] = {"actual": assistant_text, "pattern": str(regex)}

    expected_json = getattr(contract, "expected_json", None)
    expected_schema = getattr(contract, "expected_json_schema", None)
    if expected_json is not None or expected_schema is not None:
        try:
            parsed = json.loads(assistant_text)
            checks["reading_json_parse"] = True
            evidence["reading_json"] = {"parsed": parsed}
        except json.JSONDecodeError as exc:
            parsed = None
            checks["reading_json_parse"] = False
            evidence["reading_json"] = {"error": str(exc), "text": assistant_text}
        if expected_json is not None:
            checks["reading_json_exact"] = parsed == expected_json
            evidence["reading_json_exact"] = {"actual": parsed, "expected": expected_json}
        if expected_schema is not None:
            try:
                _validate_schema_value(parsed, expected_schema, path="benchmark.reading")
            except ToolValidationError as exc:
                checks["reading_json_schema"] = False
                evidence["reading_json_schema"] = {"error": str(exc), "schema": expected_schema, "actual": parsed}
            else:
                checks["reading_json_schema"] = True
                evidence["reading_json_schema"] = {"schema": expected_schema, "actual": parsed}

    expected_files = dict(getattr(contract, "expected_files", {}) or {})
    if expected_files:
        result: dict[str, Any] = {}
        for path_text, expected_content in expected_files.items():
            path = Path(str(path_text))
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            result[str(path)] = {"exists": path.exists(), "actual": actual, "expected": expected_content}
        checks["expected_files"] = all(item["exists"] and item["actual"] == item["expected"] for item in result.values())
        evidence["expected_files"] = result
    expected_patterns = dict(getattr(contract, "expected_file_patterns", {}) or {})
    if expected_patterns:
        result = {}
        for path_text, patterns in expected_patterns.items():
            path = Path(str(path_text))
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            result[str(path)] = {"exists": path.exists(), "actual": actual, "patterns": list(patterns)}
        checks["expected_file_patterns"] = all(
            item["exists"] and all(pattern in (item["actual"] or "") for pattern in item["patterns"])
            for item in result.values()
        )
        evidence["expected_file_patterns"] = result

    command = list(getattr(contract, "command", []) or [])
    if command:
        checks["command"], evidence["command"] = _run_command(command, cwd=getattr(contract, "command_cwd", None))

    tool_calls = [event for event in events if event.event_type == "tool_called"]
    tool_names = [str(event.payload.get("tool_name", "")) for event in tool_calls]
    required_tools = list(getattr(contract, "required_tools_used", []) or [])
    if required_tools:
        checks["required_tools_used"] = all(tool in tool_names for tool in required_tools)
        evidence["required_tools_used"] = {"required": required_tools, "actual": tool_names}
    forbidden_tools = list(getattr(contract, "forbidden_tools_used", []) or [])
    if forbidden_tools:
        checks["forbidden_tools_used"] = all(tool not in tool_names for tool in forbidden_tools)
        evidence["forbidden_tools_used"] = {"forbidden": forbidden_tools, "actual": tool_names}
    minimum = getattr(contract, "min_tool_calls", None)
    if minimum is not None:
        checks["min_tool_calls"] = len(tool_calls) >= int(minimum)
        evidence["min_tool_calls"] = {"required": int(minimum), "actual": len(tool_calls)}
    maximum = getattr(contract, "max_tool_calls", None)
    if maximum is not None:
        checks["max_tool_calls"] = len(tool_calls) <= int(maximum)
        evidence["max_tool_calls"] = {"required": int(maximum), "actual": len(tool_calls)}

    if workspace_after is not None:
        before = dict(workspace_before or {})
        after = dict(workspace_after)
        raw_changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        changed = [path for path in raw_changed if not _is_benign_workspace_artifact(path)]
        evidence["workspace_changes"] = {"changed_files": changed, "ignored_changed_files": [p for p in raw_changed if p not in changed]}
        allowed = list(getattr(contract, "allowed_modified_files", []) or [])
        if allowed:
            allowed_set = set(allowed) | {f"{path}.bak" for path in allowed}
            checks["allowed_modified_files"] = set(changed).issubset(allowed_set)
            evidence["allowed_modified_files"] = {"allowed": sorted(allowed_set), "changed_files": changed}
        elif getattr(contract, "forbid_unexpected_workspace_changes", False):
            expected = set(expected_files) | set(expected_patterns)
            allowed_set = expected | {f"{path}.bak" for path in expected}
            checks["allowed_modified_files"] = set(changed).issubset(allowed_set)
            evidence["allowed_modified_files"] = {"allowed": sorted(allowed_set), "changed_files": changed}

    task_type = str(getattr(contract, "task_type", ""))
    if task_type == "coding" and command and checks.get("command") and "expected_file_patterns" in checks:
        evidence["expected_file_patterns"]["advisory_for_coding"] = True
        checks["expected_file_patterns"] = True

    passed = all(checks.values()) if checks else False
    failing = [name for name, ok in checks.items() if not ok]
    return BenchmarkVerificationReport(
        passed=passed,
        task_type=task_type or "unknown",
        checks=checks,
        evidence=evidence,
        reason="benchmark_contract_passed" if passed else ",".join(failing) or "benchmark_contract_failed",
    )
