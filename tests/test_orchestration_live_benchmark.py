from pathlib import Path

from swaag.benchmark.orchestration_live import DEFAULT_PROMPT


def test_live_orchestration_prompt_covers_dependency_and_start_contract():
    assert "exactly two workers" in DEFAULT_PROMPT
    assert "sequential scheduling" in DEFAULT_PROMPT
    assert "depend on the first worker completing" in DEFAULT_PROMPT
    assert "dependency input mapping" in DEFAULT_PROMPT
    assert "Validate the plan before starting it" in DEFAULT_PROMPT
    assert "then start it" in DEFAULT_PROMPT


def test_live_orchestration_benchmark_is_cli_registered():
    text = Path("src/swaag/benchmark/benchmark_runner.py").read_text()
    assert '"orchestration-live"' in text
    assert "run_live_orchestration_benchmark" in text


def test_live_orchestration_benchmark_isolates_workspace_manifest():
    text = Path("src/swaag/benchmark/orchestration_live.py").read_text()
    assert 'workspace_root = output_dir / "workspace"' in text
    assert "config.tools.read_roots = [workspace_root]" in text


def test_live_orchestration_prompt_requires_overall_plan_objective():
    from swaag.benchmark.orchestration_live import DEFAULT_PROMPT

    assert "The plan objective must be exactly" in DEFAULT_PROMPT
    assert "Complete ALPHA and then BETA using the durable dependency flow." in DEFAULT_PROMPT


def test_live_orchestration_benchmark_bounds_action_output_reserve():
    text = Path("src/swaag/benchmark/orchestration_live.py").read_text()
    assert 'output_ratio_by_kind["action"] = 0.08' in text
    assert 'output_floor_ratio_by_kind["action"] = 0.05' in text


def test_live_orchestration_prompt_prefers_bulk_plan_apply():
    from swaag.benchmark.orchestration_live import DEFAULT_PROMPT

    assert "plan.apply" in DEFAULT_PROMPT
    assert "complete known graph in one tool call" in DEFAULT_PROMPT


def test_live_objective_scoring_ignores_terminal_punctuation_only():
    from swaag.benchmark.orchestration_live import _normalize_objective

    expected = "Complete ALPHA and then BETA using the durable dependency flow."
    assert _normalize_objective(expected) == _normalize_objective(expected.rstrip("."))
    assert _normalize_objective(expected) != _normalize_objective("Return exactly ALPHA COMPLETE")
