from pathlib import Path

from swaag.benchmark.orchestration_live_matrix import CASE_IDS


def test_live_orchestration_matrix_covers_high_risk_real_worker_transitions():
    assert set(CASE_IDS) == {
        "bounded_parallel_live",
        "semantic_branch_live",
        "dynamic_replan_replace_live",
        "completion_only_notifications_live",
    }


def test_live_orchestration_matrix_is_cli_registered():
    text = Path("src/swaag/benchmark/benchmark_runner.py").read_text()
    assert '"orchestration-live-matrix"' in text
    assert "run_live_orchestration_matrix" in text


def test_live_orchestration_matrix_isolates_each_case_workspace():
    text = Path("src/swaag/benchmark/orchestration_live_matrix.py").read_text()
    assert 'workspace = root / "workspace"' in text
    assert "config.tools.read_roots = [workspace]" in text
    assert 'config.communication.model_routes = {}' in text
