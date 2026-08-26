from __future__ import annotations

from swaag.context_compiler import ContextCompiler
from swaag.grammar import agent_action_contract
from swaag.tokens import ConservativeEstimator
from swaag.types import ContractSpec, PromptAssembly, PromptComponent


def test_context_compiler_accounts_named_components_and_output_reserve(make_config):
    config = make_config(model__context_limit=4096)
    compiler = ContextCompiler(config)
    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="system user",
        components=[
            PromptComponent(name="system", category="system_prompt", text="system "),
            PromptComponent(name="history", category="history", text="user"),
        ],
    )
    result = compiler.compile(
        assembly,
        agent_action_contract([]),
        ConservativeEstimator(chars_per_token=1.0),
        minimum_output_tokens=200,
    )
    accounting = result.accounting()
    assert accounting["reserved_response_tokens"] >= 200
    assert accounting["available_input_tokens"] == (
        accounting["context_limit"]
        - accounting["reserved_response_tokens"]
        - accounting["safety_margin_tokens"]
    )
    assert {item["name"] for item in accounting["components"]} >= {"system", "history", "constraint_schema"}


def test_context_compiler_reports_exact_overflow_tokens(make_config):
    config = make_config(model__context_limit=256)
    compiler = ContextCompiler(config)
    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="x" * 1000,
        components=[PromptComponent(name="huge", text="x" * 1000)],
    )
    result = compiler.compile(
        assembly,
        agent_action_contract([]),
        ConservativeEstimator(chars_per_token=1.0),
        minimum_output_tokens=64,
    )
    assert not result.report.fits
    assert result.overflow_tokens == result.report.required_tokens - result.report.context_limit


def test_estimated_counting_uses_percentage_safety(make_config):
    config = make_config(model__context_limit=10000, context__safety_margin_tokens=64)
    compiler = ContextCompiler(config)
    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="x" * 100,
        components=[PromptComponent(name="input", text="x" * 100)],
    )
    result = compiler.compile(
        assembly,
        agent_action_contract([]),
        ConservativeEstimator(chars_per_token=1.0),
        minimum_output_tokens=200,
    )
    assert result.report.safety_margin_tokens > 64


def test_full_fidelity_input_not_rejected_only_for_desired_output_ratio(make_config):
    config = make_config(model__context_limit=1000, context__safety_margin_tokens=10)
    compiler = ContextCompiler(config)
    class Exact:
        def count_text(self, text):
            from swaag.tokens import CountResult

            return CountResult(tokens=len(text), exact=True, strategy="test_exact")

    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="x" * 600,
        components=[PromptComponent(name="input", text="x" * 600)],
    )
    result = compiler.compile(
        assembly,
        agent_action_contract([]),
        Exact(),
        minimum_output_tokens=100,
    )
    assert result.report.fits
    assert result.report.input_tokens == 600
    assert result.report.safety_margin_tokens == 10
    assert result.report.reserved_response_tokens <= 390
    assert result.context_limit_source == "configured"


def test_per_call_desired_output_is_soft_and_accounted(make_config):
    config = make_config(
        model__context_limit=1000,
        context__safety_margin_tokens=10,
        budget_policy__structured_output_json_floor_tokens=20,
    )
    compiler = ContextCompiler(config)
    contract = ContractSpec(
        name="test_output",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    class Exact:
        def count_text(self, text):
            from swaag.tokens import CountResult

            return CountResult(tokens=len(text), exact=True, strategy="test_exact")

    roomy = compiler.compile(
        PromptAssembly(
            kind="tool_result_projection",
            prompt_mode="lean",
            prompt_text="x" * 100,
            components=[PromptComponent(name="input", text="x" * 100)],
        ),
        contract,
        Exact(),
        minimum_output_tokens=50,
        desired_output_tokens=400,
    )
    assert roomy.report.reserved_response_tokens == 400
    assert roomy.accounting()["desired_output_tokens"] == 400

    pressured = compiler.compile(
        PromptAssembly(
            kind="tool_result_projection",
            prompt_mode="lean",
            prompt_text="x" * 800,
            components=[PromptComponent(name="input", text="x" * 800)],
        ),
        contract,
        Exact(),
        minimum_output_tokens=100,
        desired_output_tokens=400,
    )
    assert pressured.report.fits
    assert pressured.report.input_tokens == 800
    assert pressured.report.reserved_response_tokens == 190
    assert pressured.desired_output_tokens == 400


def test_live_context_limit_override_is_authoritative(make_config):
    config = make_config(model__context_limit=2048, context__safety_margin_tokens=10)
    compiler = ContextCompiler(config)
    class Exact:
        def count_text(self, text):
            from swaag.tokens import CountResult

            return CountResult(tokens=len(text), exact=True, strategy="test_exact")

    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="x" * 3000,
        components=[PromptComponent(name="input", text="x" * 3000)],
    )
    result = compiler.compile(
        assembly,
        agent_action_contract([]),
        Exact(),
        minimum_output_tokens=100,
        context_limit=4096,
        context_limit_source="server_props:n_ctx",
    )
    assert result.report.context_limit == 4096
    assert result.report.fits
    assert result.context_limit_source == "server_props:n_ctx"


def test_unbounded_schema_content_does_not_inflate_required_output(make_config):
    config = make_config(
        model__context_limit=1000,
        budget_policy__structured_output_json_floor_tokens=20,
    )
    compiler = ContextCompiler(config)
    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="x" * 100,
        components=[PromptComponent(name="input", text="x" * 100)],
    )

    no_tools = compiler.compile(
        assembly,
        agent_action_contract([]),
        ConservativeEstimator(chars_per_token=1.0),
        minimum_output_tokens=30,
    )
    many_tools = compiler.compile(
        assembly,
        agent_action_contract(
            [
                (
                    f"tool_{index}",
                    "description",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                )
                for index in range(20)
            ]
        ),
        ConservativeEstimator(chars_per_token=1.0),
        minimum_output_tokens=30,
    )

    # A valid action may contain an empty tool-call list, so schema breadth is
    # grammar overhead, not a deterministic generated-output requirement.
    assert many_tools.structured_output_floor_tokens == no_tools.structured_output_floor_tokens


def test_context_limit_is_not_silently_inflated_to_policy_floor(make_config):
    config = make_config(
        model__context_limit=128,
        budget_policy__structured_output_json_floor_tokens=20,
        context__safety_margin_tokens=5,
    )
    compiler = ContextCompiler(config)
    assembly = PromptAssembly(
        kind="action",
        prompt_mode="standard",
        prompt_text="x" * 80,
        components=[PromptComponent(name="input", text="x" * 80)],
    )
    result = compiler.compile(
        assembly,
        agent_action_contract([]),
        ConservativeEstimator(chars_per_token=1.0),
        minimum_output_tokens=40,
    )

    assert result.report.context_limit == 128
    assert not result.report.fits
