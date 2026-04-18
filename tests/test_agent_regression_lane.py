from __future__ import annotations

from pathlib import Path

from swaag.benchmark import benchmark_runner
from swaag.benchmark.agent_regression import get_agent_regression_families


def test_agent_regression_families_cover_replay_and_runtime_regressions() -> None:
    families = {item.family_id: item for item in get_agent_regression_families()}

    assert "continuation_background" in families
    assert "runtime_recovery_contracts" in families
    assert "direct_response_guardrails" in families
    assert "tool_routing_context_focus" in families
    assert "subagent_traceability" in families
    assert "record_replay_runtime" in families
    assert "scripted_benchmark_runtime" in families
    assert families["record_replay_runtime"].nodeids == (
        "tests/test_agent_loop_replay.py::test_record_replay_client_replays_runtime_tool_flow",
    )


def test_benchmark_runner_regression_command_uses_agent_regression_lane(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_agent_loop_regression_lane(*, output_dir: Path, family_ids, clean: bool):
        observed["output_dir"] = output_dir
        observed["family_ids"] = family_ids
        observed["clean"] = clean
        return {
            "summary": {"total_families": 1, "passed_families": 1, "failed_families": 0, "percent": 100.0},
            "results": [],
        }

    monkeypatch.setattr("swaag.benchmark.agent_regression.run_agent_loop_regression_lane", fake_run_agent_loop_regression_lane)

    exit_code = benchmark_runner.main(
        ["regression", "--family", "record_replay_runtime", "--output", str(tmp_path / "out"), "--clean"]
    )

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "out"
    assert observed["family_ids"] == ["record_replay_runtime"]
    assert observed["clean"] is True


def test_benchmark_runner_three_lane_command_uses_three_lane_evaluation(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_three_lane_evaluation(**kwargs):
        observed.update(kwargs)
        return {
            "lane_scores": {
                "deterministic_correctness": 100.0,
                "agent_loop_regression": 100.0,
                "live_agent_evaluation": 100.0,
            },
            "overall_percent": 100.0,
        }

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_three_lane_evaluation", fake_run_three_lane_evaluation)

    exit_code = benchmark_runner.main(
        ["three-lane-evaluate", "--output", str(tmp_path / "eval"), "--clean", "--task", "live-demo"]
    )

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "eval"
    assert observed["clean"] is True
    assert observed["benchmark_task_ids"] == ["live-demo"]
