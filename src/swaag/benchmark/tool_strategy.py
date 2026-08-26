from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import shutil
import statistics
import time
from typing import Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.model import LlamaCppClient
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps


@dataclass(slots=True, frozen=True)
class ToolStrategy:
    name: str
    enabled_tools: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ToolStrategyTask:
    task_id: str
    prompt: str
    fixture_files: tuple[tuple[str, str], ...]
    expected_files: tuple[tuple[str, str], ...]
    expected_answer: str


STRATEGIES = {
    "generic_shell": ToolStrategy("generic_shell", ("shell_command",)),
    "structured_tools": ToolStrategy(
        "structured_tools",
        ("list_files", "read_file", "calculator", "edit_text"),
    ),
}

TASKS = (
    ToolStrategyTask(
        task_id="inspect_and_calculate",
        prompt=(
            "Inspect the workspace and report the exact combined unit count from all regional inventory "
            "records. Do not guess. Use the available capabilities to obtain and check the answer."
        ),
        fixture_files=(
            ("README.txt", "Regional inventory records are under inventory/. Ignore archived examples.\n"),
            ("inventory/north.txt", "region=north\nunits=137\n"),
            ("inventory/south.txt", "region=south\nunits=286\n"),
            ("archive/example.txt", "example_only=true\nunits=9999\n"),
        ),
        expected_files=(),
        expected_answer="423",
    ),
    ToolStrategyTask(
        task_id="inspect_edit_and_verify",
        prompt=(
            "Inspect the workspace configuration. Change retry_delay_seconds from its current value to 14 "
            "without changing any other bytes, then verify the resulting file before reporting completion."
        ),
        fixture_files=(
            ("README.txt", "The active service settings are in config/service.conf.\n"),
            ("config/service.conf", "service=collector\nretry_delay_seconds=9\nmode=steady\n"),
        ),
        expected_files=(
            ("README.txt", "The active service settings are in config/service.conf.\n"),
            ("config/service.conf", "service=collector\nretry_delay_seconds=14\nmode=steady\n"),
        ),
        expected_answer="",
    ),
)


def _select_named(items: Iterable, names: Iterable[str], *, kind: str) -> list:
    by_name = {getattr(item, "name", getattr(item, "task_id", "")): item for item in items}
    selected_names = list(names)
    unknown = sorted(set(selected_names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown {kind}: {', '.join(unknown)}")
    return [by_name[name] for name in selected_names] if selected_names else list(items)


def build_execution_matrix(
    *,
    task_ids: Iterable[str] = (),
    strategy_names: Iterable[str] = (),
) -> list[tuple[ToolStrategyTask, ToolStrategy]]:
    tasks = _select_named(TASKS, task_ids, kind="tool-strategy task")
    strategies = _select_named(STRATEGIES.values(), strategy_names, kind="tool strategy")
    matrix: list[tuple[ToolStrategyTask, ToolStrategy]] = []
    for index, task in enumerate(tasks):
        ordered = strategies if index % 2 == 0 else list(reversed(strategies))
        matrix.extend((task, strategy) for strategy in ordered)
    return matrix


def _write_fixture(workspace: Path, files: tuple[tuple[str, str], ...]) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    for relative, content in files:
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _workspace_snapshot(workspace: Path) -> dict[str, str]:
    return {
        str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }


def verify_case(
    task: ToolStrategyTask,
    *,
    workspace: Path,
    initial_snapshot: dict[str, str],
    assistant_text: str,
    tool_names: list[str],
) -> dict:
    initial_expected_snapshot = {
        relative: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for relative, content in task.fixture_files
    }
    expected_files = task.expected_files or task.fixture_files
    expected_snapshot = {
        relative: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for relative, content in expected_files
    }
    final_snapshot = _workspace_snapshot(workspace)
    answer_values = re.findall(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])", assistant_text.replace(",", ""))
    checks = {
        "initial_workspace_exact": initial_snapshot == initial_expected_snapshot,
        "used_domain_tool": any(name != "load_tools" for name in tool_names),
        "workspace_exact": final_snapshot == expected_snapshot,
        "answer_present": not task.expected_answer or task.expected_answer in answer_values,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_snapshot": initial_snapshot,
        "expected_snapshot": expected_snapshot,
        "final_snapshot": final_snapshot,
    }


def _case_config(
    base: AgentConfig,
    *,
    workspace: Path,
    sessions_root: Path,
    strategy: ToolStrategy,
    allowed_write_paths: Iterable[Path] = (),
) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.tools.read_roots = [workspace]
    config.tools.enabled = list(strategy.enabled_tools)
    config.tools.allow_stateful_tools = True
    config.tools.allow_side_effect_tools = True
    config.tools.staged_discovery = True
    allowed = [path.resolve() for path in allowed_write_paths]
    config.editor.allow_writes = bool(allowed)
    config.editor.allowed_write_paths = [str(path) for path in allowed]
    config.model.cache_enabled = False
    config.runtime.completion_evaluation_enabled = False
    return config


def _report_payload(
    *,
    status: str,
    model_identity: dict,
    execution_order: list[dict[str, str]],
    results: list[dict],
) -> dict:
    by_strategy: dict[str, dict] = {}
    selected_strategies = list(dict.fromkeys(item["strategy"] for item in execution_order))
    for strategy_name in selected_strategies:
        rows = [row for row in results if row["strategy"] == strategy_name]
        elapsed = [float(row["elapsed_seconds"]) for row in rows if row.get("elapsed_seconds") is not None]
        by_strategy[strategy_name] = {
            "passed": sum(1 for row in rows if row["passed"]),
            "total": len(rows),
            "mean_elapsed_seconds": statistics.fmean(elapsed) if elapsed else None,
            "median_elapsed_seconds": statistics.median(elapsed) if elapsed else None,
            "model_calls": sum(int(row.get("metrics", {}).get("model_calls", 0)) for row in rows),
            "tool_calls": sum(int(row.get("metrics", {}).get("tool_calls", 0)) for row in rows),
        }
    return {
        "benchmark": "generic_shell_vs_structured_tools",
        "status": status,
        "model_identity": model_identity,
        "execution_order": execution_order,
        "results": results,
        "by_strategy": by_strategy,
        "passed": sum(1 for row in results if row["passed"]),
        "total": len(results),
    }


def run_tool_strategy_benchmark(
    *,
    config: AgentConfig | None = None,
    output_dir: Path,
    task_ids: Iterable[str] = (),
    strategy_names: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identity: dict | None = None,
) -> dict:
    config = config or load_config()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Tool-strategy output already exists; use --clean or a new path: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "tool_strategy_results.json"
    matrix = build_execution_matrix(task_ids=task_ids, strategy_names=strategy_names)
    execution_order = [
        {"task_id": task.task_id, "strategy": strategy.name}
        for task, strategy in matrix
    ]
    if model_identity is None:
        model_identity = LlamaCppClient(config).cache_identity()
    results: list[dict] = []
    report_path.write_text(
        stable_json_dumps(
            _report_payload(
                status="running",
                model_identity=model_identity,
                execution_order=execution_order,
                results=results,
            ),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    for case_index, (task, strategy) in enumerate(matrix, start=1):
        case_root = output_dir / "runs" / f"{case_index:02d}-{task.task_id}-{strategy.name}"
        workspace = case_root / "workspace"
        _write_fixture(workspace, task.fixture_files)
        initial_snapshot = _workspace_snapshot(workspace)
        fixture_by_path = dict(task.fixture_files)
        expected_by_path = dict(task.expected_files or task.fixture_files)
        allowed_write_paths = [
            workspace / relative
            for relative, expected_content in expected_by_path.items()
            if fixture_by_path.get(relative) != expected_content
        ]
        case_config = _case_config(
            config,
            workspace=workspace,
            sessions_root=case_root / "sessions",
            strategy=strategy,
            allowed_write_paths=allowed_write_paths,
        )
        runtime = runtime_factory(case_config)
        state = runtime.create_or_load_session()
        started = time.monotonic()
        assistant_text = ""
        error = ""
        try:
            turn = runtime.run_turn_in_session(state, task.prompt)
            assistant_text = turn.assistant_text
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        events = runtime.history.read_history(state.session_id)
        tool_names = [
            str(event.payload.get("tool_name", ""))
            for event in events
            if event.event_type == "tool_called"
        ]
        rebuilt = runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=False)
        verification = verify_case(
            task,
            workspace=workspace,
            initial_snapshot=initial_snapshot,
            assistant_text=assistant_text,
            tool_names=tool_names,
        )
        row = {
            "case_index": case_index,
            "task_id": task.task_id,
            "strategy": strategy.name,
            "enabled_tools": list(strategy.enabled_tools),
            "session_id": state.session_id,
            "session_root": str(case_config.sessions.root),
            "workspace": str(workspace),
            "assistant_text": assistant_text,
            "tool_names": tool_names,
            "error": error,
            "elapsed_seconds": elapsed,
            "metrics": asdict(rebuilt.metrics),
            "verification": verification,
            "passed": not error and bool(verification["passed"]),
        }
        results.append(row)
        report_path.write_text(
            stable_json_dumps(
                _report_payload(
                    status="running" if case_index < len(matrix) else "complete",
                    model_identity=model_identity,
                    execution_order=execution_order,
                    results=results,
                ),
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return _report_payload(
        status="complete",
        model_identity=model_identity,
        execution_order=execution_order,
        results=results,
    )
