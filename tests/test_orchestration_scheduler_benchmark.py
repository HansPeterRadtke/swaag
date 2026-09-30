from swaag.benchmark.orchestration_scheduler import (
    CASES,
    run_orchestration_scheduler_benchmark,
)


def test_orchestration_scheduler_benchmark_covers_required_use_cases(tmp_path):
    report = run_orchestration_scheduler_benchmark(
        output_dir=tmp_path / "benchmark", clean=True
    )
    assert report["total"] == len(CASES) >= 27
    assert report["all_passed"], [
        (item["case_id"], item["error"])
        for item in report["results"]
        if not item["passed"]
    ]
    required = {
        "bulk_plan_apply",
        "dependency_output_flow",
        "sequential_exclusive",
        "bounded_parallel",
        "semantic_branch_resolution",
        "failure_branch",
        "dynamic_replan_replace",
        "named_model_routing",
        "model_route_replacement",
        "missing_model_fails_closed",
        "empty_plan_validation",
        "cycle_guard",
        "plan_cancel",
        "plan_auto_completion",
        "resource_budget_admission",
        "impossible_resource_fails_closed",
        "finish_abort_criteria",
        "reporting_completion_only",
        "critical_reporting_bypass",
        "semantic_interesting_reporting",
        "equal_round_robin_share",
        "weighted_fair_share",
        "multi_worker_weighted_share",
        "dynamic_weight_change",
        "multi_slot_admission",
        "control_priority_preemption_replay",
        "aging_prevents_starvation",
    }
    assert required <= {item["case_id"] for item in report["results"]}
