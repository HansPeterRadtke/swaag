from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from swaag.benchmark.local_agent_runner import _build_edit_contract, _build_file_selection_contract
from swaag.benchmark.local_agent_runner import LocalRunnerPolicy
from swaag.context_builder import build_context
from swaag.grammar import (
    action_selection_contract,
    active_session_control_contract,
    failure_classification_contract,
    plan_contract,
    prompt_analysis_contract,
    relevance_scoring_contract,
    strategy_selection_contract,
    subagent_selection_contract,
    summary_contract,
    task_decision_contract,
    task_expansion_contract,
    text_response_contract,
    tool_decision_contract,
    tool_input_contract,
    verification_contract,
    yes_no_contract,
)
from swaag.model import LlamaCppClient
from swaag.prompts import PromptBuilder
from swaag.retrieval.embeddings import EmbeddingBackend
from swaag.schema_portability import assert_portable_json_schema
from swaag.subagents import default_subagent_specs
from swaag.tokens import ConservativeEstimator
from swaag.tools.registry import ToolRegistry
from swaag.types import ContractMode, Message, SessionState


class _PositiveBackend(EmbeddingBackend):
    mode = "llm_scoring"
    degraded = False

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        del query
        return [1.0 for _text in texts]


def _state() -> SessionState:
    return SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        messages=[Message(role="user", content="Complete the current task.", created_at="t1")],
    )


def _tool_specs() -> list[tuple[str, str, dict[str, Any], str]]:
    return [
        (
            "alpha_tool",
            "Alpha tool registered description.",
            {
                "type": "object",
                "properties": {"alpha": {"type": "string"}},
                "required": ["alpha"],
                "additionalProperties": False,
            },
            "Alpha registered usage guidance.",
        ),
        (
            "beta_tool",
            "Beta tool registered description.",
            {
                "type": "object",
                "properties": {"beta": {"type": "boolean"}},
                "required": ["beta"],
                "additionalProperties": False,
            },
            "Beta registered usage guidance.",
        ),
    ]


def _production_contracts(config) -> list:
    tool_registry = ToolRegistry()
    tool_names = tool_registry.tool_names(config)
    contracts = [
        yes_no_contract(),
        text_response_contract("answer_response"),
        text_response_contract("clarification_response"),
        prompt_analysis_contract(),
        task_decision_contract(tool_names),
        task_expansion_contract(),
        active_session_control_contract(),
        summary_contract(),
        plan_contract(tool_names),
        strategy_selection_contract(),
        failure_classification_contract(),
        action_selection_contract(),
        subagent_selection_contract(default_subagent_specs().keys()),
        relevance_scoring_contract(3),
        verification_contract(["criterion"]),
        tool_decision_contract(tool_names),
        _build_file_selection_contract(["pkg/a.py", "pkg/b.py"], policy=LocalRunnerPolicy()),
        _build_edit_contract(["pkg/a.py", "pkg/b.py"], policy=LocalRunnerPolicy()),
    ]
    for tool in tool_registry.enabled_tools(config):
        contracts.append(tool_input_contract(tool.name, tool.input_schema))
    return contracts


def test_selected_skills_never_hide_enabled_tools(make_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("swaag.skills.selector.build_backend", lambda *args, **kwargs: _PositiveBackend())
    monkeypatch.setattr("swaag.retrieval.retriever.build_backend", lambda *args, **kwargs: _PositiveBackend())
    monkeypatch.setattr("swaag.guidance.resolver.build_backend", lambda *args, **kwargs: _PositiveBackend())
    config = make_config(retrieval__backend="llm_scoring")
    tools = _tool_specs()

    bundle = build_context(config, _state(), ConservativeEstimator(), goal="Complete the current task.", available_tools=tools)

    assert bundle.selected_skill_ids
    assert bundle.exposed_tool_names == [tool[0] for tool in tools]
    assert bundle.tool_prompt_tuples == tools


def test_planning_and_tool_choice_prompts_include_full_enabled_registry(make_config) -> None:
    builder = PromptBuilder(make_config())
    messages = [Message(role="user", content="Use the right tool.", created_at="t1")]
    tools = _tool_specs()

    task_decision_prompt = builder.build_task_decision_prompt(
        "Use the right tool.",
        '{"task_type":"structured"}',
        prompt_mode="lean",
        tools=tools,
    )
    decision_prompt = builder.build_decision_prompt(messages, tools, prompt_mode="lean")
    plan_prompt = builder.build_plan_prompt(
        "Use the right tool.",
        prompt_mode="lean",
        context_components=[],
        tools=tools,
    )

    for prompt_text in [task_decision_prompt.prompt_text, decision_prompt.prompt_text, plan_prompt.prompt_text]:
        for name, description, schema, guidance in tools:
            assert f"- {name}" in prompt_text
            assert description in prompt_text
            assert json.dumps(schema, sort_keys=True, separators=(",", ":")) in prompt_text
            assert guidance in prompt_text
    assert 'Use verification_type="composite" for every step' in plan_prompt.prompt_text
    assert "Allowed check_type values are:" in plan_prompt.prompt_text
    assert "Do not invent check_type values" in plan_prompt.prompt_text
    assert "tool_effect_verified" in plan_prompt.prompt_text
    assert "objective_verification_check" not in plan_prompt.prompt_text
    assert "runtime installs any tool-registered automatic mechanical objective check" in plan_prompt.prompt_text
    assert "set expected exactly equal to that step's expected_tool" in plan_prompt.prompt_text
    assert "artifact and expected_json are not substitutes for expected" in plan_prompt.prompt_text
    assert "For json_schema_valid, set actual_source and schema_json" in plan_prompt.prompt_text
    assert "For function_exists, set path and function_name" in plan_prompt.prompt_text
    assert "For symbol_exists, set path and symbol" in plan_prompt.prompt_text
    assert "A non-dry-run edit_text call atomically persists its edit" in plan_prompt.prompt_text
    assert "do not add a follow-up write_file step for the same file" in plan_prompt.prompt_text
    assert "never artifact/input/output labels" in plan_prompt.prompt_text
    assert "file_exists, tool_files_changed, artifact_present, and tool_output_nonempty are not substitutes" in plan_prompt.prompt_text
    assert "in an initial plan depends_on names only earlier step_id values" in plan_prompt.prompt_text
    assert "completed prior step id supplied in replan evidence" in plan_prompt.prompt_text
    assert "runtime derives the structural done condition" in plan_prompt.prompt_text
    assert "expected_outputs is a non-empty list of output labels for the step, including respond steps" in plan_prompt.prompt_text
    assert "For respond/reasoning, presence and value checks are compiled to assistant_text" in plan_prompt.prompt_text
    assert "success_criteria is the authoritative semantic criterion" in plan_prompt.prompt_text
    assert "weaken the requested final state" in plan_prompt.prompt_text
    assert "require tool_result_success when tests must pass" in plan_prompt.prompt_text
    assert "Never emit tool_effect_verified or file_contains" in plan_prompt.prompt_text
    assert "read_file has one file and one output_ref per step" in plan_prompt.prompt_text
    assert "split multiple files into ordered steps" in plan_prompt.prompt_text
    assert "allow the registered persisted-effect check and later whole-goal review" in plan_prompt.prompt_text
    assert "that dependency is already satisfied and will be removed mechanically" in plan_prompt.prompt_text
    assert "Require dependencies_completed when dependencies exist" in plan_prompt.prompt_text


def test_tool_input_prompt_uses_registered_docs_and_has_no_tool_name_branches(make_config) -> None:
    builder = PromptBuilder(make_config())
    tool_spec = _tool_specs()[0]

    prompt = builder.build_tool_input_prompt(
        [Message(role="user", content="Call alpha.", created_at="t1")],
        tool_spec=tool_spec,
        prompt_mode="lean",
    )
    source = inspect.getsource(PromptBuilder.build_tool_input_prompt)

    assert tool_spec[1] in prompt.prompt_text
    assert tool_spec[3] in prompt.prompt_text
    assert json.dumps(tool_spec[2], sort_keys=True, separators=(",", ":")) in prompt.prompt_text
    assert "Generated arguments must be executable against the latest observed state" in prompt.prompt_text
    assert "do not repeat arguments that depend only on that stale target" in prompt.prompt_text
    assert "if tool_name" not in source
    assert "elif tool_name" not in source
    assert "match tool_name" not in source


def test_live_contracts_are_json_schema_only(make_config) -> None:
    assert ContractMode.__args__ == ("json_schema",)
    for contract in _production_contracts(make_config(tools__allow_stateful_tools=True, tools__allow_side_effect_tools=True)):
        assert contract.mode == "json_schema"
        assert contract.json_schema is not None


def test_provider_request_shapes_are_constrained_and_nonportable_forms_absent(make_config) -> None:
    openai_config = make_config(model__base_url="https://openrouter.ai/api/v1", model__profile_name="openai/gpt-4o-mini")
    llama_config = make_config(model__base_url="http://127.0.0.1:8080")
    contract = tool_decision_contract(["alpha_tool"])

    openai_payload = LlamaCppClient(openai_config).build_completion_request("prompt", max_tokens=64, contract=contract)
    llama_payload = LlamaCppClient(llama_config).build_completion_request("prompt", max_tokens=64, contract=contract)

    assert openai_payload["response_format"]["type"] == "json_schema"
    assert openai_payload["response_format"]["json_schema"]["strict"] is True
    assert openai_payload["response_format"]["json_schema"]["schema"] == contract.json_schema
    assert openai_payload["provider"]["require_parameters"] is True
    assert "json_object" not in json.dumps(openai_payload)
    assert "grammar" not in openai_payload
    assert "json_schema" not in openai_payload

    assert llama_payload["json_schema"] == contract.json_schema
    assert "response_format" not in llama_payload
    assert "grammar" not in llama_payload


def test_all_production_generation_schemas_are_portable(make_config) -> None:
    config = make_config(tools__allow_stateful_tools=True, tools__allow_side_effect_tools=True)
    for contract in _production_contracts(config):
        assert contract.json_schema is not None
        assert_portable_json_schema(contract.json_schema, schema_name=contract.name)


def test_file_mutation_tools_require_model_owned_result_review(make_config) -> None:
    registry = ToolRegistry()
    config = make_config(tools__allow_stateful_tools=True, tools__allow_side_effect_tools=True)
    tools = {tool.name: tool for tool in registry.enabled_tools(config)}

    assert tools["edit_text"].semantic_result_review_required is True
    assert tools["edit_text"].automatic_objective_verification_check_type == "tool_effect_verified"
    assert tools["write_file"].semantic_result_review_required is True
    assert tools["write_file"].automatic_objective_verification_check_type == "tool_effect_verified"
    assert tools["read_file"].max_plan_output_refs == 1
    assert verification_contract(["result_satisfies_step"]).json_schema is not None


def test_plan_contract_excludes_runtime_rejected_verification_types() -> None:
    schema = plan_contract(["read_file", "edit_text"]).json_schema
    assert schema is not None
    verification_enum = schema["properties"]["steps"]["items"]["properties"]["verification_type"]["enum"]
    assert verification_enum == ["composite"]
    assert "execution" not in verification_enum
    assert "structural" not in verification_enum
    assert "value" not in verification_enum
    assert "llm_fallback" not in verification_enum
