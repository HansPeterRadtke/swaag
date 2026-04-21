from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
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
        "tests/test_benchmark_catalog.py",
        "tests/test_agent_support.py",
        "tests/test_benchmark_import_hygiene.py",
        "tests/test_benchmark_metrics.py",
        "tests/test_benchmark_report.py",
        "tests/test_browser_integration.py",
        "tests/test_budgeting.py",
        "tests/test_clean_install.py",
        "tests/test_cli.py",
        "tests/test_config.py",
        "tests/test_context_builder.py",
        "tests/test_decision.py",
        "tests/test_devcheck.py",
        "tests/test_devcheck_routing.py",
        "tests/test_editing.py",
        "tests/test_evaluation_runner.py",
        "tests/test_evaluator.py",
        "tests/test_expander.py",
        "tests/test_external_benchmarks.py",
        "tests/test_failure.py",
        "tests/test_grammar.py",
        "tests/test_guidance.py",
        "tests/test_imports.py",
        "tests/test_live_runtime_profiles.py",
        "tests/test_live_subset_selection.py",
        "tests/test_live_suite_structure.py",
        "tests/test_llm_record_replay.py",
        "tests/test_local_agent_runner.py",
        "tests/test_model_integration.py",
        "tests/test_notes.py",
        "tests/test_prompt_analyzer.py",
        "tests/test_prompts.py",
        "tests/test_reader.py",
        "tests/test_retrieval.py",
        "tests/test_roles.py",
        "tests/test_security.py",
        "tests/test_skills.py",
        "tests/test_strategy.py",
        "tests/test_swebench_local.py",
        "tests/test_system_benchmark_suite.py",
        "tests/test_terminal_bench_local.py",
        "tests/test_tokens.py",
        "tests/test_tools.py",
        "tests/test_two_categories.py",
        "tests/test_verification.py",
    }
)

AGENT_TEST_FILES = frozenset(
    {
        # Scripted and cached agent behavior tests (LLM calls via FakeModelClient or replay cassettes)
        "tests/test_agent_loop_replay.py",
        "tests/test_benchmark.py",
        "tests/test_clean_install_agent.py",
        "tests/test_end_to_end.py",
        "tests/test_environment.py",
        "tests/test_false_positive_killers.py",
        "tests/test_history.py",
        "tests/test_long_running_tasks.py",
        "tests/test_orchestrator.py",
        "tests/test_planner.py",
        "tests/test_project_state.py",
        "tests/test_prompt_understanding_eval.py",
        "tests/test_reasoning.py",
        "tests/test_runtime.py",
        "tests/test_runtime_verification_flow.py",
        "tests/test_scaled_catalog.py",
        "tests/test_semantic_memory.py",
        "tests/test_session_control.py",
        "tests/test_sessions.py",
        "tests/test_subagents.py",
        "tests/test_subsystems.py",
        "tests/test_working_memory.py",
    }
)

# ---------------------------------------------------------------------------
# Private: internal devcheck routing sets (not part of the public API)
# ---------------------------------------------------------------------------

_DEVCHECK_PROFILE_PRECEDENCE = {
    "fast": 0,
    "system": 1,
    "packaging": 2,
    "manual_validation": 3,
    "heavy_agent": 4,
}

_DEVCHECK_SYSTEM_PROFILE_FILES = frozenset(
    {
        "tests/test_agent_loop_replay.py",
        "tests/test_benchmark_catalog.py",
        "tests/test_benchmark_metrics.py",
        "tests/test_benchmark_report.py",
        "tests/test_browser_integration.py",
        "tests/test_end_to_end.py",
        "tests/test_environment.py",
        "tests/test_false_positive_killers.py",
        "tests/test_history.py",
        "tests/test_live_subset_selection.py",
        "tests/test_long_running_tasks.py",
        "tests/test_orchestrator.py",
        "tests/test_planner.py",
        "tests/test_prompt_understanding_eval.py",
        "tests/test_reasoning.py",
        "tests/test_roles.py",
        "tests/test_runtime.py",
        "tests/test_runtime_verification_flow.py",
        "tests/test_scaled_catalog.py",
        "tests/test_semantic_memory.py",
        "tests/test_sessions.py",
        "tests/test_subagents.py",
        "tests/test_subsystems.py",
        "tests/test_working_memory.py",
    }
)

_DEVCHECK_PACKAGING_PROFILE_FILES = frozenset(
    {
        "tests/test_clean_install.py",
        "tests/test_model_integration.py",
    }
)

MANUAL_VALIDATION_FILES = frozenset({"tests/test_live_llamacpp.py"})

_DEVCHECK_HEAVY_AGENT_PROFILE_FILES = frozenset({"tests/test_benchmark.py"})

_BENCHMARK_STRUCTURE_TESTS = (
    "tests/test_benchmark_catalog.py",
    "tests/test_benchmark_metrics.py",
    "tests/test_benchmark_report.py",
    "tests/test_false_positive_killers.py",
    "tests/test_live_subset_selection.py",
    "tests/test_prompt_understanding_eval.py",
    "tests/test_scaled_catalog.py",
    "tests/test_imports.py",
)

_RUNTIME_FAST_TESTS = (
    "tests/test_budgeting.py",
    "tests/test_decision.py",
    "tests/test_expander.py",
    "tests/test_failure.py",
    "tests/test_prompt_analyzer.py",
    "tests/test_prompts.py",
    "tests/test_strategy.py",
    "tests/test_verification.py",
    "tests/test_imports.py",
)

_RUNTIME_SYSTEM_TESTS = (
    "tests/test_end_to_end.py",
    "tests/test_history.py",
    "tests/test_orchestrator.py",
    "tests/test_planner.py",
    "tests/test_reasoning.py",
    "tests/test_runtime.py",
    "tests/test_runtime_verification_flow.py",
)

_CONTEXT_FAST_TESTS = (
    "tests/test_context_builder.py",
    "tests/test_guidance.py",
    "tests/test_retrieval.py",
    "tests/test_skills.py",
    "tests/test_imports.py",
)

_CONTEXT_SYSTEM_TESTS = (
    "tests/test_environment.py",
    "tests/test_semantic_memory.py",
    "tests/test_subagents.py",
    "tests/test_subsystems.py",
    "tests/test_working_memory.py",
)

_PACKAGING_TESTS = (
    "tests/test_clean_install.py",
    "tests/test_cli.py",
    "tests/test_config.py",
    "tests/test_devcheck.py",
    "tests/test_imports.py",
    "tests/test_model_integration.py",
)

_MANUAL_VALIDATION_PROFILE_TESTS = (
    "tests/test_live_runtime_profiles.py",
    "tests/test_live_suite_structure.py",
    "tests/test_devcheck.py",
)

_FAST_SOURCE_TO_TESTS: dict[str, tuple[str, ...]] = {
    "src/swaag/budgeting.py": ("tests/test_budgeting.py", "tests/test_context_builder.py", "tests/test_imports.py"),
    "src/swaag/cli.py": ("tests/test_cli.py", "tests/test_imports.py"),
    "src/swaag/compression.py": ("tests/test_history.py", "tests/test_imports.py"),
    "src/swaag/editing.py": ("tests/test_editing.py", "tests/test_imports.py"),
    "src/swaag/evaluator.py": ("tests/test_evaluator.py", "tests/test_imports.py"),
    "src/swaag/notes.py": ("tests/test_notes.py", "tests/test_imports.py"),
    "src/swaag/project_state.py": ("tests/test_project_state.py", "tests/test_imports.py"),
    "src/swaag/reader.py": ("tests/test_reader.py", "tests/test_imports.py"),
    "src/swaag/security.py": ("tests/test_security.py", "tests/test_imports.py"),
    "src/swaag/tokens.py": ("tests/test_tokens.py", "tests/test_budgeting.py", "tests/test_imports.py"),
}

_DOC_RUNTIME_TESTS = {
    "doc/context_budgeting.md": ("tests/test_budgeting.py", "tests/test_context_builder.py"),
    "doc/live_runtime_profiles.md": _MANUAL_VALIDATION_PROFILE_TESTS,
    "doc/testing.md": ("tests/test_devcheck.py",),
}

# ---------------------------------------------------------------------------
# Internal devcheck profile specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevcheckProfileSpec:
    name: str
    marker_expression: str
    description: str


@dataclass(frozen=True, slots=True)
class TestmonStatus:
    __test__ = False

    available: bool
    baseline_exists: bool
    mode: str
    reason: str


@dataclass(frozen=True, slots=True)
class DevcheckPlan:
    changed_files: tuple[str, ...]
    profile: str
    candidate_tests: tuple[str, ...]
    reasons: tuple[str, ...]
    marker_expression: str
    testmon: TestmonStatus
    explicit_followup_profiles: tuple[str, ...] = ()


_DEVCHECK_PROFILES = {
    "fast": DevcheckProfileSpec(
        name="fast",
        marker_expression="not agent_test",
        description="Cheap deterministic unit tests only.",
    ),
    "system": DevcheckProfileSpec(
        name="system",
        marker_expression="not agent_test",
        description="Runtime/orchestration/history/system tests.",
    ),
    "packaging": DevcheckProfileSpec(
        name="packaging",
        marker_expression="",
        description="Packaging, install, and model-client code-correctness checks.",
    ),
    "manual_validation": DevcheckProfileSpec(
        name="manual_validation",
        marker_expression="agent_test",
        description="Explicit manual real-model validation tooling, not a test category.",
    ),
    "heavy_agent": DevcheckProfileSpec(
        name="heavy_agent",
        marker_expression="agent_test",
        description="Expensive cached agent behavior checks kept outside routine code-correctness.",
    ),
}

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


def fast_test_files(root: Path | None = None) -> tuple[str, ...]:
    test_files = set(all_test_files(root))
    special = (
        _DEVCHECK_SYSTEM_PROFILE_FILES
        | _DEVCHECK_PACKAGING_PROFILE_FILES
        | MANUAL_VALIDATION_FILES
        | _DEVCHECK_HEAVY_AGENT_PROFILE_FILES
    )
    return tuple(sorted(test_files - special))


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
    referenced = CODE_CORRECTNESS_TEST_FILES | AGENT_TEST_FILES | MANUAL_VALIDATION_FILES
    missing = sorted(path for path in referenced if path not in known)
    if missing:
        raise RuntimeError(f"Top-level test-category registry references missing test files: {missing}")
    overlap = CODE_CORRECTNESS_TEST_FILES & AGENT_TEST_FILES
    if overlap:
        raise RuntimeError(f"Top-level test-category registry has overlapping test files: {sorted(overlap)}")
    unclassified = sorted(path for path in known if path not in referenced)
    if unclassified:
        raise RuntimeError(f"Top-level test-category registry leaves test files unclassified: {unclassified}")


# ---------------------------------------------------------------------------
# Internal devcheck profile API
# ---------------------------------------------------------------------------


def devcheck_profile_test_files(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    return {
        "fast": fast_test_files(root),
        "system": tuple(sorted(_DEVCHECK_SYSTEM_PROFILE_FILES)),
        "packaging": tuple(sorted(_DEVCHECK_PACKAGING_PROFILE_FILES)),
        "manual_validation": tuple(sorted(MANUAL_VALIDATION_FILES)),
        "heavy_agent": tuple(sorted(_DEVCHECK_HEAVY_AGENT_PROFILE_FILES | {"tests/test_clean_install.py"})),
    }


def devcheck_profile_for_test_file(path: str, root: Path | None = None) -> str:
    normalized = path.replace("\\", "/")
    if normalized in _DEVCHECK_HEAVY_AGENT_PROFILE_FILES:
        return "heavy_agent"
    if normalized in MANUAL_VALIDATION_FILES:
        return "manual_validation"
    if normalized in _DEVCHECK_PACKAGING_PROFILE_FILES:
        return "packaging"
    if normalized in _DEVCHECK_SYSTEM_PROFILE_FILES:
        return "system"
    if normalized in all_test_files(root):
        return "fast"
    raise ValueError(f"Unknown test file: {path}")


def validate_devcheck_profile_registry(root: Path | None = None) -> None:
    base = project_root() if root is None else root
    known = set(all_test_files(base))
    referenced = (
        _DEVCHECK_SYSTEM_PROFILE_FILES
        | _DEVCHECK_PACKAGING_PROFILE_FILES
        | MANUAL_VALIDATION_FILES
        | _DEVCHECK_HEAVY_AGENT_PROFILE_FILES
    )
    missing = sorted(path for path in referenced if path not in known)
    if missing:
        raise RuntimeError(f"Test-profile registry references missing test files: {missing}")
    overlaps = set()
    profile_sets = [
        _DEVCHECK_SYSTEM_PROFILE_FILES,
        _DEVCHECK_PACKAGING_PROFILE_FILES,
        MANUAL_VALIDATION_FILES,
        _DEVCHECK_HEAVY_AGENT_PROFILE_FILES,
    ]
    for index, first in enumerate(profile_sets):
        for second in profile_sets[index + 1 :]:
            overlaps.update(first & second)
    if overlaps:
        raise RuntimeError(f"Test-profile registry has overlapping test files: {sorted(overlaps)}")


def devcheck_profile_marker_expression(profile: str) -> str:
    return _DEVCHECK_PROFILES[profile].marker_expression


def detect_testmon(root: Path | None = None) -> TestmonStatus:
    base = project_root() if root is None else root
    available = importlib.util.find_spec("testmon") is not None
    baseline_exists = any((base / name).exists() for name in (".testmondata", ".testmondata-shm", ".testmondata-wal"))
    if not available:
        return TestmonStatus(False, baseline_exists, "disabled", "pytest-testmon is not installed")
    if not baseline_exists:
        return TestmonStatus(True, False, "noselect", "testmon baseline is missing; run the candidate set once to populate it")
    return TestmonStatus(True, True, "forceselect", "testmon baseline is ready")


def _max_devcheck_profile(current: str, candidate: str) -> str:
    if _DEVCHECK_PROFILE_PRECEDENCE[candidate] > _DEVCHECK_PROFILE_PRECEDENCE[current]:
        return candidate
    return current


def _add_targets(targets: list[str], *items: str) -> None:
    for item in items:
        if item and item not in targets:
            targets.append(item)


def _runtime_candidates() -> tuple[str, ...]:
    return (*_RUNTIME_FAST_TESTS, *_RUNTIME_SYSTEM_TESTS)


def _context_candidates() -> tuple[str, ...]:
    return (*_CONTEXT_FAST_TESTS, *_CONTEXT_SYSTEM_TESTS)


def _packaging_candidates() -> tuple[str, ...]:
    return (*_PACKAGING_TESTS, *_MANUAL_VALIDATION_PROFILE_TESTS)


def _normalize_changed_files(changed_files: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(path.replace("\\", "/") for path in changed_files if path.strip())))


def validate_candidate_tests(candidate_tests: Iterable[str], root: Path | None = None) -> tuple[str, ...]:
    base = project_root() if root is None else root
    known = set(all_test_files(base))
    missing = sorted(path for path in candidate_tests if path not in known)
    if missing:
        raise RuntimeError(f"Candidate selector references missing test files: {missing}")
    return tuple(dict.fromkeys(candidate_tests))


def _mark_packaging_only_followup(path: str, reasons: list[str]) -> None:
    reasons.append(f"{path} affects packaging/command wiring; broaden to the packaging profile")


def build_devcheck_plan(
    changed_files: Iterable[str],
    *,
    root: Path | None = None,
    allow_manual_validation: bool = False,
    allow_heavy_agent: bool = False,
) -> DevcheckPlan:
    base = project_root() if root is None else root
    validate_devcheck_profile_registry(base)
    changed = _normalize_changed_files(changed_files)
    if not changed:
        candidate_tests = validate_candidate_tests(("tests/test_imports.py",), base)
        return DevcheckPlan(
            changed_files=changed,
            profile="fast",
            candidate_tests=candidate_tests,
            reasons=("no tracked changes detected; running the smallest smoke import check",),
            marker_expression="not agent_test",
            testmon=detect_testmon(base),
        )

    profile = "fast"
    targets: list[str] = []
    reasons: list[str] = []
    followups: list[str] = []

    for path in changed:
        if path.startswith("tests/"):
            try:
                test_profile = devcheck_profile_for_test_file(path, base)
            except ValueError:
                reasons.append(f"{path} is an unknown test file; no automatic mapping exists")
                continue
            if test_profile == "manual_validation" and not allow_manual_validation:
                followups.append("manual_validation")
                reasons.append(f"{path} is a manual-validation file; explicit real-model validation is required")
                continue
            if test_profile == "heavy_agent" and not allow_heavy_agent:
                followups.append("heavy_agent")
                reasons.append(f"{path} needs expensive agent execution; explicit heavy-agent execution is required")
                continue
            profile = _max_devcheck_profile(profile, test_profile)
            _add_targets(targets, path)
            reasons.append(f"{path} changed; include the test file directly")
            continue

        if path in _DOC_RUNTIME_TESTS:
            _add_targets(targets, *_DOC_RUNTIME_TESTS[path])
            reasons.append(f"{path} has dedicated consistency tests; include only those checks")
            continue
        if path.startswith("doc/"):
            reasons.append(f"{path} is documentation-only and has no dedicated runtime consistency test")
            continue

        if path in {"pyproject.toml", "scripts/archive_proof.py"} or path.startswith("scripts/"):
            profile = _max_devcheck_profile(profile, "packaging")
            _add_targets(targets, *_packaging_candidates())
            _mark_packaging_only_followup(path, reasons)
            continue

        if path in {
            "src/swaag/devcheck.py",
            "src/swaag/finalproof.py",
            "src/swaag/test_categories.py",
            "src/swaag/testlanes.py",
            "src/swaag/testlane.py",
            "tests/conftest.py",
        }:
            profile = _max_devcheck_profile(profile, "packaging")
            _add_targets(targets, *_packaging_candidates())
            _mark_packaging_only_followup(path, reasons)
            continue

        mapped_fast = _FAST_SOURCE_TO_TESTS.get(path)
        if mapped_fast is not None:
            _add_targets(targets, *mapped_fast)
            reasons.append(f"{path} has a narrow deterministic mapping; keep the run in the fast profile")
            continue

        if path.startswith("src/swaag/benchmark/"):
            profile = _max_devcheck_profile(profile, "system")
            _add_targets(targets, *_BENCHMARK_STRUCTURE_TESTS)
            reasons.append(f"{path} changes benchmark structure; include structure/report tests without the heavy proof profile")
            continue

        if path in {"src/swaag/live_runtime_profiles.py"}:
            profile = _max_devcheck_profile(profile, "packaging")
            _add_targets(targets, *_MANUAL_VALIDATION_PROFILE_TESTS)
            reasons.append(f"{path} affects real-model execution policy; include profile consistency checks")
            continue

        if path.startswith("src/swaag/assets/") or path in {
            "src/swaag/config.py",
            "src/swaag/context_builder.py",
            "src/swaag/decision.py",
            "src/swaag/failure.py",
            "src/swaag/grammar.py",
            "src/swaag/model.py",
            "src/swaag/orchestrator.py",
            "src/swaag/planner.py",
            "src/swaag/prompt_analyzer.py",
            "src/swaag/prompts.py",
            "src/swaag/runtime.py",
            "src/swaag/strategy.py",
            "src/swaag/verification.py",
        }:
            profile = _max_devcheck_profile(profile, "system")
            _add_targets(targets, *_runtime_candidates())
            reasons.append(f"{path} affects runtime semantics/orchestration; broaden to the system profile")
            continue

        if (
            path.startswith("src/swaag/environment/")
            or path.startswith("src/swaag/guidance/")
            or path.startswith("src/swaag/retrieval/")
            or path.startswith("src/swaag/skills/")
            or path.startswith("src/swaag/subagents/")
            or path.startswith("src/swaag/tools/")
            or path.startswith("src/swaag/subsystems/")
            or path in {
                "src/swaag/memory_semantic.py",
                "src/swaag/working_memory.py",
                "src/swaag/sessions.py",
                "src/swaag/history.py",
            }
        ):
            profile = _max_devcheck_profile(profile, "system")
            _add_targets(targets, *_context_candidates())
            reasons.append(f"{path} affects context/tool/environment behavior; broaden to the system profile")
            continue

        if path.startswith("src/swaag/"):
            profile = _max_devcheck_profile(profile, "system")
            _add_targets(targets, *_runtime_candidates(), *_context_candidates())
            reasons.append(f"{path} is an unmapped agent core file; degrade safely to the system profile")
            continue

        reasons.append(f"{path} is outside the tracked selector map; no automatic code tests added")

    candidate_tests = validate_candidate_tests(targets, base) if targets else ()
    if profile == "packaging":
        marker_expression = devcheck_profile_marker_expression("packaging")
    elif profile == "manual_validation":
        marker_expression = devcheck_profile_marker_expression("manual_validation")
    elif profile == "heavy_agent":
        marker_expression = devcheck_profile_marker_expression("heavy_agent")
    else:
        marker_expression = "not agent_test"

    return DevcheckPlan(
        changed_files=changed,
        profile=profile,
        candidate_tests=candidate_tests,
        reasons=tuple(reasons),
        marker_expression=marker_expression,
        testmon=detect_testmon(base),
        explicit_followup_profiles=tuple(sorted(dict.fromkeys(followups))),
    )


def build_devcheck_profile_command(
    profile: str,
    *,
    root: Path | None = None,
    use_testmon: bool = False,
    baseline_only: bool = False,
    candidate_tests: Iterable[str] = (),
) -> list[str]:
    if profile == "proof":
        return [sys.executable, "-m", "swaag.finalproof"]
    base = project_root() if root is None else root
    validate_devcheck_profile_registry(base)
    if profile not in _DEVCHECK_PROFILES:
        raise ValueError(f"Unknown test profile: {profile}")
    command = [sys.executable, "-m", "pytest", "-q"]
    profile_files = devcheck_profile_test_files(base).get(profile, ())
    explicit_tests = validate_candidate_tests(candidate_tests, base) if candidate_tests else ()

    marker_expression = ""
    if profile == "manual_validation":
        marker_expression = devcheck_profile_marker_expression("manual_validation")
    elif profile == "heavy_agent":
        marker_expression = devcheck_profile_marker_expression("heavy_agent")

    if marker_expression:
        command.extend(["-m", marker_expression])
    if use_testmon:
        testmon = detect_testmon(base)
        if baseline_only:
            if not testmon.available:
                raise RuntimeError("Cannot build a baseline with pytest-testmon because the plugin is unavailable")
            command.append("--testmon-noselect")
        elif testmon.available:
            command.append("--testmon-forceselect" if testmon.baseline_exists else "--testmon-noselect")
    if explicit_tests:
        command.extend(explicit_tests)
    elif profile_files:
        command.extend(profile_files)
    return command


def build_code_correctness_command(root: Path | None = None) -> list[str]:
    """Build the explicit authoritative code-correctness pytest command.

    Passes the code-correctness file list directly — no marker-based deselection.
    """
    base = project_root() if root is None else root
    files = sorted(str(base / f) for f in CODE_CORRECTNESS_TEST_FILES)
    return [sys.executable, "-m", "pytest", "-q", *files]


def build_agent_tests_command(root: Path | None = None) -> list[str]:
    """Build the explicit authoritative agent-tests pytest command.

    Passes the agent-test file list directly — no marker-based deselection.
    """
    base = project_root() if root is None else root
    files = sorted(str(base / f) for f in AGENT_TEST_FILES)
    return [sys.executable, "-m", "pytest", "-q", *files]
