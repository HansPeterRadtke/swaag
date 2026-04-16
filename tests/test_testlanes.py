from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import swaag.testlanes as testlanes
import tests.conftest as test_conftest
from swaag.testlanes import build_devcheck_plan, build_lane_command, detect_testmon, validate_candidate_tests, validate_lane_registry


def test_validate_lane_registry_passes_for_current_tree() -> None:
    validate_lane_registry()


def test_validate_candidate_tests_rejects_missing_paths() -> None:
    with pytest.raises(RuntimeError, match="missing test files"):
        validate_candidate_tests(("tests/test_missing_selector_target.py",))


def test_runtime_change_selects_local_nonlive_runtime_candidates() -> None:
    plan = build_devcheck_plan(["src/swaag/runtime.py"])

    assert plan.lane == "system"
    assert "tests/test_runtime.py" in plan.candidate_tests
    assert "tests/test_runtime_verification_flow.py" in plan.candidate_tests
    assert "tests/test_live_llamacpp.py" not in plan.candidate_tests
    assert plan.explicit_followup_lanes == ()
    assert plan.marker_expression == "not integration and not live and not benchmark_heavy"


def test_tiny_source_change_stays_in_fast_lane() -> None:
    plan = build_devcheck_plan(["src/swaag/tokens.py"])

    assert plan.lane == "fast"
    assert plan.candidate_tests == (
        "tests/test_tokens.py",
        "tests/test_budgeting.py",
        "tests/test_imports.py",
    )


def test_benchmark_structure_change_stays_out_of_heavy_lane() -> None:
    plan = build_devcheck_plan(["src/swaag/benchmark/benchmark_runner.py"])

    assert plan.lane == "system"
    assert "tests/test_benchmark.py" not in plan.candidate_tests
    assert "tests/test_scaled_catalog.py" in plan.candidate_tests


def test_packaging_change_broadens_to_integration_lane() -> None:
    plan = build_devcheck_plan(["pyproject.toml"])

    assert plan.lane == "integration"
    assert "tests/test_clean_install.py" in plan.candidate_tests
    assert "tests/test_devcheck.py" in plan.candidate_tests


def test_docs_only_change_can_select_zero_tests() -> None:
    plan = build_devcheck_plan(["README.md"])

    assert plan.candidate_tests == ()
    assert plan.lane == "fast"


def test_docs_with_dedicated_consistency_tests_selects_only_those_tests() -> None:
    plan = build_devcheck_plan(["doc/testing.md"])

    assert plan.candidate_tests == ("tests/test_devcheck.py",)
    assert plan.lane == "fast"


def test_live_test_change_requires_explicit_live_followup() -> None:
    plan = build_devcheck_plan(["tests/test_live_llamacpp.py"])

    assert plan.candidate_tests == ()
    assert plan.explicit_followup_lanes == ("live",)


def test_benchmark_heavy_test_change_requires_explicit_heavy_followup() -> None:
    plan = build_devcheck_plan(["tests/test_benchmark.py"])

    assert plan.candidate_tests == ()
    assert plan.explicit_followup_lanes == ("benchmark_heavy",)


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
    monkeypatch.setattr(testlanes, "__file__", str(tmp_path / "site-packages" / "swaag" / "testlanes.py"))
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path / "workspace"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _FakeDistribution())

    root = testlanes._discover_project_root()

    assert root == repo


def test_build_lane_command_uses_explicit_file_lists_for_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(testlanes, "detect_testmon", lambda root=None: testlanes.TestmonStatus(True, True, "forceselect", "ready"))

    command = build_lane_command("fast", use_testmon=True)

    assert "--testmon-forceselect" in command
    assert "tests/test_tokens.py" in command
    assert "tests/test_runtime.py" not in command


def test_build_lane_command_filters_integration_lane() -> None:
    command = build_lane_command("integration")

    assert command[:4] == [command[0], "-m", "pytest", "-q"]
    assert "integration" in command
    assert "tests/test_clean_install.py" in command


def test_collection_marker_precedence_keeps_benchmark_heavy_test_out_of_integration_lane() -> None:
    class _FakeItem:
        def __init__(self, path: Path, explicit_markers: set[str]):
            self.fspath = str(path)
            self._explicit_markers = explicit_markers
            self.recorded: list[str] = []

        def get_closest_marker(self, name: str):
            return object() if name in self._explicit_markers else None

        def add_marker(self, mark) -> None:  # noqa: ANN001
            self.recorded.append(mark.name)

    item = _FakeItem(testlanes.project_root() / "tests" / "test_clean_install.py", {"benchmark_heavy"})

    test_conftest.pytest_collection_modifyitems([item])

    assert item.recorded == ["benchmark_heavy"]


def test_unmapped_file_outside_agent_tree_adds_no_tests() -> None:
    plan = build_devcheck_plan(["src/unknown_file.txt"])

    assert plan.lane == "fast"
    assert plan.candidate_tests == ()
    assert any("outside the tracked selector map" in reason for reason in plan.reasons)


def test_unmapped_agent_core_file_degrades_to_system_lane() -> None:
    plan = build_devcheck_plan(["src/swaag/brand_new_module.py"])

    assert plan.lane == "system"
    assert len(plan.candidate_tests) > 0
    assert any("unmapped agent core file" in reason for reason in plan.reasons)


def test_test_only_change_includes_the_test_file_directly() -> None:
    plan = build_devcheck_plan(["tests/test_tokens.py"])

    assert plan.lane == "fast"
    assert "tests/test_tokens.py" in plan.candidate_tests


def test_no_changes_returns_minimal_smoke_check() -> None:
    plan = build_devcheck_plan([])

    assert plan.lane == "fast"
    assert plan.candidate_tests == ("tests/test_imports.py",)


def test_multiple_changes_broaden_lane_correctly() -> None:
    plan = build_devcheck_plan(["src/swaag/tokens.py", "src/swaag/runtime.py"])

    assert plan.lane == "system"
    assert "tests/test_tokens.py" in plan.candidate_tests
    assert "tests/test_runtime.py" in plan.candidate_tests
