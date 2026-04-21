from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import swaag.test_categories as test_categories
import tests.conftest as test_conftest
from swaag.test_categories import (
    build_devcheck_plan,
    build_devcheck_profile_command,
    detect_testmon,
    validate_candidate_tests,
    validate_devcheck_profile_registry,
)


def test_validate_devcheck_profile_registry_passes_for_current_tree() -> None:
    validate_devcheck_profile_registry()


def test_validate_candidate_tests_rejects_missing_paths() -> None:
    with pytest.raises(RuntimeError, match="missing test files"):
        validate_candidate_tests(("tests/test_missing_selector_target.py",))


def test_runtime_change_selects_local_nonlive_runtime_candidates() -> None:
    plan = build_devcheck_plan(["src/swaag/runtime.py"])

    assert plan.profile == "system"
    assert "tests/test_runtime.py" in plan.candidate_tests
    assert "tests/test_runtime_verification_flow.py" in plan.candidate_tests
    assert "tests/test_live_llamacpp.py" not in plan.candidate_tests
    assert plan.explicit_followup_profiles == ()
    assert plan.marker_expression == "not agent_test"


def test_tiny_source_change_stays_in_fast_profile() -> None:
    plan = build_devcheck_plan(["src/swaag/tokens.py"])

    assert plan.profile == "fast"
    assert plan.candidate_tests == (
        "tests/test_tokens.py",
        "tests/test_budgeting.py",
        "tests/test_imports.py",
    )


def test_benchmark_structure_change_stays_out_of_heavy_profile() -> None:
    plan = build_devcheck_plan(["src/swaag/benchmark/benchmark_runner.py"])

    assert plan.profile == "system"
    assert "tests/test_benchmark.py" not in plan.candidate_tests
    assert "tests/test_scaled_catalog.py" in plan.candidate_tests


def test_packaging_change_broadens_to_packaging_profile() -> None:
    plan = build_devcheck_plan(["pyproject.toml"])

    assert plan.profile == "packaging"
    assert "tests/test_clean_install.py" in plan.candidate_tests
    assert "tests/test_devcheck.py" in plan.candidate_tests


def test_docs_only_change_can_select_zero_tests() -> None:
    plan = build_devcheck_plan(["README.md"])

    assert plan.candidate_tests == ()
    assert plan.profile == "fast"


def test_docs_with_dedicated_consistency_tests_selects_only_those_tests() -> None:
    plan = build_devcheck_plan(["doc/testing.md"])

    assert plan.candidate_tests == ("tests/test_devcheck.py",)
    assert plan.profile == "fast"


def test_manual_validation_file_requires_explicit_followup() -> None:
    plan = build_devcheck_plan(["tests/test_live_llamacpp.py"])

    assert plan.candidate_tests == ()
    assert plan.explicit_followup_profiles == ("manual_validation",)


def test_heavy_agent_test_change_requires_explicit_followup() -> None:
    plan = build_devcheck_plan(["tests/test_benchmark.py"])

    assert plan.candidate_tests == ()
    assert plan.explicit_followup_profiles == ("heavy_agent",)


def test_detect_testmon_reports_missing_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    status = detect_testmon(tmp_path)

    assert status.available is False
    assert status.mode == "disabled"


def test_detect_testmon_reports_missing_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    status = detect_testmon(tmp_path)

    assert status.available is True
    assert status.baseline_exists is False
    assert status.mode == "noselect"


def test_detect_testmon_reports_ready_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    (tmp_path / ".testmondata").write_text("baseline", encoding="utf-8")

    status = detect_testmon(tmp_path)

    assert status.available is True
    assert status.baseline_exists is True
    assert status.mode == "forceselect"


def test_project_root_uses_direct_url_repo_source_when_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src" / "swaag").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='swaag'\n", encoding="utf-8")

    class _FakeDistribution:
        files = ["swaag-0.1.0.dist-info/direct_url.json"]

        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return None

        def locate_file(self, file) -> Path:  # noqa: ANN001
            return tmp_path / "site-packages" / "swaag-0.1.0.dist-info" / "direct_url.json"

    direct_url_path = tmp_path / "site-packages" / "swaag-0.1.0.dist-info"
    direct_url_path.mkdir(parents=True)
    (direct_url_path / "direct_url.json").write_text(
        json.dumps({"url": repo.as_uri()}),
        encoding="utf-8",
    )

    monkeypatch.delenv("SWAAG_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(test_categories, "__file__", str(tmp_path / "site-packages" / "swaag" / "test_categories.py"))
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path / "workspace"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _FakeDistribution())

    root = test_categories._discover_project_root()

    assert root == repo


def test_build_devcheck_profile_command_uses_explicit_file_lists_for_fast_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(test_categories, "detect_testmon", lambda root=None: test_categories.TestmonStatus(True, True, "forceselect", "ready"))

    command = build_devcheck_profile_command("fast", use_testmon=True)

    assert "--testmon-forceselect" in command
    assert "tests/test_tokens.py" in command
    assert "tests/test_runtime.py" not in command


def test_build_devcheck_profile_command_lists_packaging_files_without_marker_filter() -> None:
    command = build_devcheck_profile_command("packaging")

    assert command[:3] == [command[0], "-m", "pytest"]
    assert "tests/test_clean_install.py" in command
    assert "-m" not in command[3:]


def test_collection_marks_code_correctness_tests_with_correct_category() -> None:
    class _FakeItem:
        def __init__(self, path: Path):
            self.fspath = str(path)
            self.recorded: list[str] = []

        def get_closest_marker(self, name: str):
            return None

        def add_marker(self, mark) -> None:  # noqa: ANN001
            self.recorded.append(mark.name)

    item = _FakeItem(test_categories.project_root() / "tests" / "test_clean_install.py")

    test_conftest.pytest_collection_modifyitems([item])

    assert "code_correctness" in item.recorded
    assert "agent_test" not in item.recorded


def test_unmapped_file_outside_agent_tree_adds_no_tests() -> None:
    plan = build_devcheck_plan(["src/unknown_file.txt"])

    assert plan.profile == "fast"
    assert plan.candidate_tests == ()
    assert any("outside the tracked selector map" in reason for reason in plan.reasons)


def test_unmapped_agent_core_file_degrades_to_system_profile() -> None:
    plan = build_devcheck_plan(["src/swaag/brand_new_module.py"])

    assert plan.profile == "system"
    assert len(plan.candidate_tests) > 0
    assert any("unmapped agent core file" in reason for reason in plan.reasons)


def test_test_only_change_includes_the_test_file_directly() -> None:
    plan = build_devcheck_plan(["tests/test_tokens.py"])

    assert plan.profile == "fast"
    assert "tests/test_tokens.py" in plan.candidate_tests


def test_no_changes_returns_minimal_smoke_check() -> None:
    plan = build_devcheck_plan([])

    assert plan.profile == "fast"
    assert plan.candidate_tests == ("tests/test_imports.py",)


def test_multiple_changes_broaden_profile_correctly() -> None:
    plan = build_devcheck_plan(["src/swaag/tokens.py", "src/swaag/runtime.py"])

    assert plan.profile == "system"
    assert "tests/test_tokens.py" in plan.candidate_tests
    assert "tests/test_runtime.py" in plan.candidate_tests
