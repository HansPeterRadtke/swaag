from __future__ import annotations

from swaag.budgeting import compute_call_budget, compute_section_budgets, structured_output_token_floor
from swaag.grammar import plain_text_contract, plan_contract, prompt_analysis_contract
from swaag.runtime import AgentRuntime
from swaag.types import PromptAssembly, PromptComponent, SessionState


def test_call_budget_scales_with_context_and_call_kind(make_config) -> None:
    small = make_config(model__context_limit=1024)
    large = make_config(model__context_limit=8192)

    small_analysis = compute_call_budget(small, call_kind="analysis")
    small_plan = compute_call_budget(small, call_kind="plan")
    small_answer = compute_call_budget(small, call_kind="answer")
    large_answer = compute_call_budget(large, call_kind="answer")

    assert small_plan.output_tokens > small_analysis.output_tokens
    assert small_answer.output_tokens > small_analysis.output_tokens
    assert small_plan.output_tokens >= small_answer.output_tokens
    assert large_answer.output_tokens > small_answer.output_tokens
    assert small_answer.safe_input_budget < small_analysis.safe_input_budget
    assert large_answer.safe_input_budget > small_answer.safe_input_budget


def test_section_budgets_scale_without_fixed_tiny_caps(make_config) -> None:
    config = make_config()

    compact = compute_section_budgets(config, call_kind="analysis", safe_input_budget=240)
    roomy = compute_section_budgets(config, call_kind="plan", safe_input_budget=2400)

    assert compact.available_input_tokens == 240
    assert roomy.available_input_tokens == 2400
    assert roomy.history_tokens > compact.history_tokens
    assert roomy.environment_files_tokens > compact.environment_files_tokens
    assert roomy.guidance_tokens > compact.guidance_tokens
    assert roomy.skills_tokens > compact.skills_tokens


def test_runtime_budget_report_uses_dynamic_call_budget_not_legacy_reserved_tokens(make_config) -> None:
    config = make_config(
        model__context_limit=2048,
        context__reserved_response_tokens=999,
        context__safety_margin_tokens=17,
    )
    runtime = AgentRuntime(config)
    assembly = PromptAssembly(
        kind="analysis",
        prompt_text="hello",
        components=[PromptComponent(name="body", text="hello world", category="current_user")],
        prompt_mode="standard",
    )
    report = runtime._budget_report(
        SessionState(
            session_id="s1",
            created_at="t0",
            updated_at="t0",
            config_fingerprint="cfg",
            model_base_url=config.model.base_url,
        ),
        assembly,
        plain_text_contract(),
    )
    dynamic = compute_call_budget(config, call_kind="analysis")

    assert report.reserved_response_tokens == dynamic.output_tokens
    assert report.safety_margin_tokens == dynamic.safety_margin_tokens
    assert report.reserved_response_tokens != 999


def test_runtime_budget_report_raises_structured_reserve_for_bounded_contracts(make_config) -> None:
    config = make_config(model__context_limit=2048)
    runtime = AgentRuntime(config)
    state = SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url=config.model.base_url,
    )

    analysis_assembly = PromptAssembly(
        kind="analysis",
        prompt_text="classify",
        components=[PromptComponent(name="body", text="Read /tmp/example.txt and return line 3.", category="current_user")],
        prompt_mode="standard",
    )
    plan_assembly = PromptAssembly(
        kind="plan",
        prompt_text="plan",
        components=[PromptComponent(name="body", text="Edit /tmp/example.txt and then answer.", category="current_user")],
        prompt_mode="standard",
    )

    analysis_report = runtime._budget_report(state, analysis_assembly, prompt_analysis_contract())
    plan_report = runtime._budget_report(state, plan_assembly, plan_contract(["read_file", "write_file", "edit_text"]))

    assert analysis_report.reserved_response_tokens > compute_call_budget(config, call_kind="analysis").output_tokens
    assert plan_report.reserved_response_tokens >= compute_call_budget(config, call_kind="plan").output_tokens
    assert plan_report.required_tokens <= config.model.context_limit


def test_structured_output_floor_uses_bounded_schema_instance_not_schema_text(make_config) -> None:
    config = make_config(model__context_limit=2048)
    runtime = AgentRuntime(config)
    counter = runtime._get_budget_counter(None)
    contract = plan_contract(["read_file", "write_file", "edit_text"])

    floor = structured_output_token_floor(contract, config=config, counter=counter, call_kind="plan")
    schema_tokens = counter.count_text(str(contract.json_schema)).tokens

    assert floor > 0
    assert floor <= schema_tokens


def test_budget_policy_ratios_are_read_from_config_object(make_config) -> None:
    config = make_config(model__context_limit=2048)
    config.budget_policy.output_ratio_by_kind["plan"] = 0.5
    budget = compute_call_budget(config, call_kind="plan")

    assert budget.output_tokens == 1024


def test_tool_input_budget_can_override_default_small_ratio(make_config) -> None:
    config = make_config(model__context_limit=2048)

    tool_input_budget = compute_call_budget(config, call_kind="tool_input")
    decision_budget = compute_call_budget(config, call_kind="decision")

    assert tool_input_budget.output_tokens > decision_budget.output_tokens


def test_structured_output_floor_uses_budget_policy_config(make_config) -> None:
    config = make_config(model__context_limit=2048)
    config.budget_policy.structured_output_json_factor_by_contract["task_plan"] = 2.0
    runtime = AgentRuntime(config)
    counter = runtime._get_budget_counter(None)
    contract = plan_contract(["read_file", "write_file", "edit_text"])

    floor = structured_output_token_floor(contract, config=config, counter=counter, call_kind="plan")
    schema_tokens = counter.count_text(str(contract.json_schema)).tokens

    assert floor > 0
    assert floor <= schema_tokens


def test_safe_input_floor_is_honoured_from_config(make_config) -> None:
    """safe_input_floor_tokens from config must be the floor for safe_input_budget."""
    # Degenerate tiny context: overhead likely exceeds available space, floor must apply.
    config = make_config(model__context_limit=300)
    config.budget_policy.safe_input_floor_tokens = 64

    budget = compute_call_budget(config, call_kind="plan")

    assert budget.safe_input_budget >= 64


def test_safe_input_floor_default_is_128(make_config) -> None:
    """Default safe_input_floor_tokens must be 128."""
    config = make_config()
    assert config.budget_policy.safe_input_floor_tokens == 128
