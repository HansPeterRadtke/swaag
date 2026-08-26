from __future__ import annotations

from swaag.context_compiler import ContextCompiler
from swaag.grammar import agent_action_contract
from swaag.tokens import ConservativeEstimator
from swaag.types import PromptAssembly, PromptComponent


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
