from __future__ import annotations

from pathlib import Path

import pytest

from swaag.devcheck import build_pytest_command, main
from swaag.finalproof import build_finalproof_commands, build_finalproof_environment
from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation
from swaag.testlanes import DevcheckPlan, TestmonStatus


def _plan(*, lane: str = "fast", marker_expression: str = "not integration and not live and not benchmark_heavy", candidate_tests: tuple[str, ...] = ("tests/test_imports.py",), testmon: TestmonStatus | None = None) -> DevcheckPlan:
    return DevcheckPlan(
        changed_files=("src/swaag/runtime.py",),
        lane=lane,
        candidate_tests=candidate_tests,
        reasons=("reason",),
        marker_expression=marker_expression,
        testmon=testmon or TestmonStatus(True, True, "forceselect", "baseline ready"),
    )


def test_build_pytest_command_uses_testmon_forceselect_when_available() -> None:
    command = build_pytest_command(_plan())
    assert command[:4] == [command[0], "-m", "pytest", "-q"]
    assert "--testmon-forceselect" in command
    assert command[-1] == "tests/test_imports.py"


def test_build_pytest_command_uses_testmon_noselect_without_baseline() -> None:
    command = build_pytest_command(_plan(testmon=TestmonStatus(True, False, "noselect", "baseline missing")))
    assert "--testmon-noselect" in command
    assert "--testmon-forceselect" not in command


def test_build_pytest_command_falls_back_explicitly_when_testmon_is_missing() -> None:
    command = build_pytest_command(_plan(testmon=TestmonStatus(False, False, "disabled", "missing plugin")))
    assert "--testmon-forceselect" not in command
    assert "--testmon-noselect" not in command


def test_build_pytest_command_can_require_testmon() -> None:
    with pytest.raises(RuntimeError, match="pytest-testmon is required"):
        build_pytest_command(
            _plan(testmon=TestmonStatus(False, False, "disabled", "missing plugin")),
            require_testmon=True,
        )


def test_devcheck_requires_explicit_live_lane(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--dry-run", "--changed-file", "tests/test_live_llamacpp.py"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "followup_lanes=['live']" in captured.out
    assert "python3 -m swaag.testlane live" in captured.out


def test_finalproof_builds_required_commands() -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()
    commands = build_finalproof_commands(
        benchmark_output=Path("/tmp/bench"),
        live_benchmark_output=Path("/tmp/live"),
    )
    env = build_finalproof_environment()

    flattened = [" ".join(command) for command in commands]
    assert any("tests/test_imports.py" in command for command in flattened)
    assert any("--live-subset" in command for command in flattened)
    assert any(f"--structured-output-mode {recommendation.structured_output_mode}" in command for command in flattened)
    assert any(f"--model-profile {recommendation.model_profile}" in command for command in flattened)
    assert any(f"--seeds {','.join(str(seed) for seed in recommendation.seeds)}" in command for command in flattened)
    assert env["SWAAG_LIVE_MODEL_PROFILE"] == recommendation.model_profile
    assert env["SWAAG_LIVE_STRUCTURED_OUTPUT_MODE"] == recommendation.structured_output_mode
    assert env["SWAAG_LIVE_SEEDS"] == ",".join(str(seed) for seed in recommendation.seeds)
    assert any("scripts/archive_proof.py" in command for command in flattened)


def test_testing_doc_describes_incremental_lanes() -> None:
    text = (Path(__file__).resolve().parents[1] / "doc" / "testing.md").read_text(encoding="utf-8")

    assert "python3 -m swaag.devcheck" in text
    assert "python3 -m swaag.testlane fast" in text
    assert "python3 -m swaag.testlane system" in text
    assert "pytest-testmon baseline" in text
    assert "candidate tests" in text
