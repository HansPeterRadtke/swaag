from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

# ---------------------------------------------------------------------------
# Public: top-level two-category classification
# ---------------------------------------------------------------------------

TOP_LEVEL_TEST_CATEGORIES = (
    "code_correctness",
    "agent_test",
)

CODE_CORRECTNESS_TEST_FILES = frozenset(
    {
        'tests/test_agent_loop_replay.py',
        'tests/test_agent_support.py',
        'tests/test_benchmark.py',
        'tests/test_benchmark_catalog.py',
        'tests/test_benchmark_import_hygiene.py',
        'tests/test_benchmark_metrics.py',
        'tests/test_benchmark_report.py',
        'tests/test_browser_integration.py',
        'tests/test_budgeting.py',
        'tests/test_clean_install.py',
        'tests/test_cli.py',
        'tests/test_config.py',
        'tests/test_context_builder.py',
        'tests/test_decision.py',
        'tests/test_devcheck.py',
        'tests/test_devcheck_routing.py',
        'tests/test_editing.py',
        'tests/test_end_to_end.py',
        'tests/test_environment.py',
        'tests/test_evaluation_runner.py',
        'tests/test_evaluator.py',
        'tests/test_expander.py',
        'tests/test_external_benchmarks.py',
        'tests/test_failure.py',
        'tests/test_false_positive_killers.py',
        'tests/test_grammar.py',
        'tests/test_guidance.py',
        'tests/test_history.py',
        'tests/test_imports.py',
        'tests/test_live_runtime_profiles.py',
        'tests/test_live_subset_selection.py',
        'tests/test_live_suite_structure.py',
        'tests/test_llm_record_replay.py',
        'tests/test_local_agent_runner.py',
        'tests/test_long_running_tasks.py',
        'tests/test_model_integration.py',
        'tests/test_notes.py',
        'tests/test_orchestrator.py',
        'tests/test_planner.py',
        'tests/test_project_state.py',
        'tests/test_prompt_analyzer.py',
        'tests/test_prompt_understanding_eval.py',
        'tests/test_prompts.py',
        'tests/test_reader.py',
        'tests/test_reasoning.py',
        'tests/test_retrieval.py',
        'tests/test_roles.py',
        'tests/test_runtime.py',
        'tests/test_runtime_verification_flow.py',
        'tests/test_scaled_catalog.py',
        'tests/test_security.py',
        'tests/test_semantic_memory.py',
        'tests/test_session_control.py',
        'tests/test_sessions.py',
        'tests/test_skills.py',
        'tests/test_strategy.py',
        'tests/test_subagents.py',
        'tests/test_subsystems.py',
        'tests/test_swebench_local.py',
        'tests/test_system_benchmark_suite.py',
        'tests/test_terminal_bench_local.py',
        'tests/test_tokens.py',
        'tests/test_tools.py',
        'tests/test_two_categories.py',
        'tests/test_verification.py',
        'tests/test_working_memory.py',
    }
)

AGENT_TEST_FILES = frozenset()

# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------


def _is_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").exists()
        and (path / "src" / "swaag").exists()
        and (path / "tests").exists()
    )


def _discover_project_root() -> Path:
    import os

    if (env_root := os.environ.get("SWAAG_PROJECT_ROOT")):
        configured = Path(env_root).expanduser().resolve()
        if _is_project_root(configured):
            return configured

    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        current = start if start.is_dir() else start.parent
        for candidate in (current, *current.parents):
            if _is_project_root(candidate):
                return candidate

    try:
        dist = importlib.metadata.distribution("swaag")
        direct_url_text: str | None = None
        if hasattr(dist, "read_text"):
            direct_url_text = dist.read_text("direct_url.json")
        if not direct_url_text:
            for file in dist.files or ():
                if str(file).endswith("direct_url.json"):
                    direct_url_path = dist.locate_file(file)
                    if direct_url_path.exists():
                        direct_url_text = direct_url_path.read_text(encoding="utf-8")
                        break
        if direct_url_text:
            payload = json.loads(direct_url_text)
            url = str(payload.get("url", ""))
            if url.startswith("file://"):
                parsed = urlparse(url)
                candidate = Path(unquote(parsed.path)).resolve()
                if _is_project_root(candidate):
                    return candidate
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:
        pass

    fallback = Path(__file__).resolve().parents[2]
    return fallback


def project_root() -> Path:
    return _discover_project_root()


# ---------------------------------------------------------------------------
# Test file discovery
# ---------------------------------------------------------------------------


def all_test_files(root: Path | None = None) -> tuple[str, ...]:
    base = project_root() if root is None else root
    tests_dir = base / "tests"
    if not tests_dir.exists():
        return ()
    return tuple(
        sorted(
            str(path.relative_to(base)).replace("\\", "/")
            for path in tests_dir.glob("test_*.py")
        )
    )


# ---------------------------------------------------------------------------
# Public: top-level two-category API
# ---------------------------------------------------------------------------


def category_files(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    base = project_root() if root is None else root
    known = set(all_test_files(base))
    return {
        "code_correctness": tuple(sorted(CODE_CORRECTNESS_TEST_FILES & known)),
        "agent_test": tuple(sorted(AGENT_TEST_FILES & known)),
    }


def category_for_file(path: str, root: Path | None = None) -> str:
    normalized = path.replace("\\", "/")
    if normalized in AGENT_TEST_FILES:
        return "agent_test"
    if normalized in CODE_CORRECTNESS_TEST_FILES:
        return "code_correctness"
    if normalized in all_test_files(root):
        raise ValueError(f"Test file {path!r} is not classified into a top-level test category")
    raise ValueError(f"Unknown test file: {path}")


def validate_test_category_registry(root: Path | None = None) -> None:
    base = project_root() if root is None else root
    known = set(all_test_files(base))
    referenced = CODE_CORRECTNESS_TEST_FILES | AGENT_TEST_FILES
    missing = sorted(path for path in referenced if path not in known)
    if missing:
        raise RuntimeError(f"Top-level test-category registry references missing test files: {missing}")
    overlap = CODE_CORRECTNESS_TEST_FILES & AGENT_TEST_FILES
    if overlap:
        raise RuntimeError(f"Top-level test-category registry has overlapping test files: {sorted(overlap)}")
    unclassified = sorted(path for path in known if path not in referenced)
    if unclassified:
        raise RuntimeError(f"Top-level test-category registry leaves test files unclassified: {unclassified}")


def build_code_correctness_command(root: Path | None = None) -> list[str]:
    """Build the explicit authoritative code-correctness pytest command.

    Passes the code-correctness file list directly — no marker-based deselection.
    """
    base = project_root() if root is None else root
    files = sorted(str(base / f) for f in CODE_CORRECTNESS_TEST_FILES)
    return [sys.executable, "-m", "pytest", "-q", *files]


def build_agent_tests_command(root: Path | None = None) -> list[str]:
    """Build the authoritative cached benchmark command for agent_test."""
    return [sys.executable, "-m", "swaag.benchmark", "agent-tests"]
