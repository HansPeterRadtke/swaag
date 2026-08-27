from __future__ import annotations

import json
from typing import Any

from swaag.grammar import yes_no_contract
from swaag.model import CompletionRequestPolicy
from swaag.prompt_instructions import make_prompt_instruction
from swaag.runtime import AgentRuntime, PreparedCall
from swaag.tokens import ExactTokenCounter
from swaag.tools.base import SemanticCallRequest
from swaag.types import (
    CompletionResult,
    ContractSpec,
    PromptComponent,
)


class _InstructionProjectionClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy(
            "test", "server_schema", contract.mode, 10, 0.01
        )

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
        }

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        if payload["contract"] == "yes_no":
            text = json.dumps({"answer": "yes"})
            return CompletionResult(
                text=text,
                raw_request=payload,
                raw_response={"content": text},
                prompt_tokens=None,
                completion_tokens=None,
                finish_reason="stop",
            )
        assert payload["contract"] == "prompt_instruction_projection"
        prompt = str(payload["prompt"])
        self.prompts.append(prompt)
        assert (
            "[DURABLE MODEL-AUTHORED INSTRUCTIONS FOR THIS CALL KIND]"
            not in prompt
        )
        retained: list[str] = []
        if "NEVER_DROP_ALPHA" in prompt:
            retained.append("Never violate NEVER_DROP_ALPHA.")
        if "ALWAYS_KEEP_BETA" in prompt:
            retained.append("Always preserve ALWAYS_KEEP_BETA.")
        if not retained:
            retained.append("Preserve every exact fragment requirement.")
        text = json.dumps(
            {
                "projection": " ".join(retained)
            }
        )
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


class _OutputStarvingInstructionClient(_InstructionProjectionClient):
    def __init__(self) -> None:
        super().__init__()
        self.yes_no_calls = 0

    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        if payload["contract"] != "yes_no":
            return super().send_completion(payload, **kwargs)
        self.yes_no_calls += 1
        if self.yes_no_calls == 1:
            return CompletionResult(
                text='{"answer":',
                raw_request=payload,
                raw_response={"content": '{"answer":'},
                prompt_tokens=None,
                completion_tokens=None,
                finish_reason="length",
            )
        text = json.dumps({"answer": "yes"})
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_prompt_instructions_project_only_after_measured_overflow(
    make_config,
) -> None:
    config = make_config(
        model__context_limit=700,
        context__max_compaction_rounds=2,
        prompt_instructions__max_instruction_chars=30_000,
        prompt_instructions__max_total_chars=32_000,
    )
    client = _InstructionProjectionClient()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    state.prompt_instructions.append(
        make_prompt_instruction(
            config,
            title="Large exact correction",
            content=(
                "NEVER_DROP_ALPHA "
                + "repeated contextual wording " * 350
                + " ALWAYS_KEEP_BETA"
            ),
            scopes=["action"],
        )
    )
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="action",
        system_instruction="Choose the next valid action.",
        components=[
            PromptComponent(
                name="request",
                category="current_user",
                text="Continue.",
            )
        ],
    )
    failed = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(700, "test"),
    )
    assert failed.report.fits is False
    assert failed.overflow_tokens > 0

    recovered = runtime._recover_prompt_instruction_overflow(
        state,
        assembly,
        yes_no_contract(),
        failed,
        minimum_output_tokens=64,
        context_limit_resolution=(700, "test"),
    )

    assert recovered is not None
    assert recovered.report.fits is True
    projection = next(
        component
        for component in assembly.components
        if component.name == "durable_prompt_instruction_projection"
    )
    assert "NEVER_DROP_ALPHA" in projection.text
    assert "ALWAYS_KEEP_BETA" in projection.text
    assert not any(
        component.name == "durable_prompt_instructions"
        for component in assembly.components
    )
    events = runtime.history.read_history(state.session_id)
    created = next(
        event
        for event in events
        if event.event_type == "prompt_instruction_projection_created"
    )
    assert created.payload["overflow_tokens"] == failed.overflow_tokens
    assert created.payload["source_instruction_references"][0][
        "instruction_id"
    ] == state.prompt_instructions[0].instruction_id
    assert created.payload["projected_tokens"] < created.payload["source_tokens"]


def test_prompt_instructions_remain_exact_when_full_context_fits(make_config) -> None:
    config = make_config(model__context_limit=4_000)
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    instruction = make_prompt_instruction(
        config,
        title="Exact rule",
        content="Preserve this exact rule.",
        scopes=["action"],
    )
    state.prompt_instructions.append(instruction)
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="action",
        system_instruction="Choose an action.",
        components=[
            PromptComponent(
                name="request",
                category="current_user",
                text="Continue.",
            )
        ],
    )
    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(4_000, "test"),
    )

    assert compilation.report.fits is True
    exact = next(
        component
        for component in assembly.components
        if component.name == "durable_prompt_instructions"
    )
    assert instruction.content in exact.text
    assert not any(
        event.event_type == "prompt_instruction_projection_created"
        for event in runtime.history.read_history(state.session_id)
    )


def test_action_preparation_uses_instruction_projection_only_as_final_fallback(
    make_config,
) -> None:
    config = make_config(
        model__context_limit=1_800,
        context__max_compaction_rounds=2,
        prompt_instructions__max_instruction_chars=60_000,
        prompt_instructions__max_total_chars=64_000,
    )
    client = _InstructionProjectionClient()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    state.prompt_instructions.append(
        make_prompt_instruction(
            config,
            title="Oversized operating rules",
            content=(
                "NEVER_DROP_ALPHA "
                + "repeated contextual wording " * 1_000
                + " ALWAYS_KEEP_BETA"
            ),
            scopes=["action"],
        )
    )

    prepared = runtime._prepare_action_call(
        state,
        original_request="Continue the task.",
        pending_messages=[],
        tool_specs=[],
        contract=yes_no_contract(),
        validation_feedback="",
        minimum_output_tokens=64,
    )

    assert prepared.report.fits is True
    assert any(
        component.name == "durable_prompt_instruction_projection"
        for component in prepared.assembly.components
    )
    context_events = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "context_compiled"
        and event.payload.get("kind") == "action"
    ]
    assert context_events[0].payload["cap_error"] == "context_limit_exceeded"
    assert context_events[-1].payload["accounting"]["fits"] is True
    assert context_events[-1].payload["prompt_instruction_projection"] is True


def test_semantic_capability_can_explicitly_allow_final_instruction_projection(
    make_config,
) -> None:
    config = make_config(
        model__context_limit=700,
        context__max_compaction_rounds=2,
        prompt_instructions__max_instruction_chars=30_000,
        prompt_instructions__max_total_chars=32_000,
    )
    runtime = AgentRuntime(
        config,
        model_client=_InstructionProjectionClient(),
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()
    state.prompt_instructions.append(
        make_prompt_instruction(
            config,
            title="Large history rules",
            content=(
                "NEVER_DROP_ALPHA "
                + "repeated contextual wording " * 350
                + " ALWAYS_KEEP_BETA"
            ),
            scopes=["history_analysis"],
        )
    )
    payload = runtime._execute_tool_semantic_call(
        state,
        SemanticCallRequest(
            kind="history_analysis",
            system_instruction="Answer the exact yes/no question.",
            components=[
                PromptComponent(
                    name="question",
                    category="current_user",
                    text="Is the supplied evidence present?",
                )
            ],
            contract=yes_no_contract(),
            minimum_output_tokens=64,
            allow_prompt_instruction_projection=True,
        ),
    )

    assert payload == {"answer": "yes"}
    assert any(
        event.event_type == "prompt_instruction_projection_created"
        and event.payload["kind"] == "history_analysis"
        for event in runtime.history.read_history(state.session_id)
    )


def test_output_starvation_rebuild_projects_instructions_only_after_overflow(
    make_config,
) -> None:
    config = make_config(
        model__context_limit=700,
        model__max_retries=1,
        context__max_compaction_rounds=2,
        prompt_instructions__max_instruction_chars=30_000,
        prompt_instructions__max_total_chars=32_000,
    )
    client = _OutputStarvingInstructionClient()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(
            lambda text: len(text.split()) if text.strip() else 0
        ),
    )
    state = runtime.create_or_load_session()

    assembly = None
    compilation = None
    for repetitions in range(50, 700, 10):
        state.prompt_instructions = [
            make_prompt_instruction(
                config,
                title="Large exact correction",
                content=(
                    "NEVER_DROP_ALPHA "
                    + "repeated contextual wording " * repetitions
                    + " ALWAYS_KEEP_BETA"
                ),
                scopes=["action"],
            )
        ]
        candidate = runtime.prompts.build_semantic_operation_prompt(
            kind="action",
            system_instruction="Answer the exact yes/no question.",
            components=[
                PromptComponent(
                    name="question",
                    category="current_user",
                    text="Is the requirement present?",
                )
            ],
        )
        candidate_compilation = runtime._compile_context(
            state,
            candidate,
            yes_no_contract(),
            minimum_output_tokens=64,
            desired_output_tokens=64,
            context_limit_resolution=(700, "test"),
        )
        if (
            candidate_compilation.report.fits
            and 0
            <= 700 - candidate_compilation.report.required_tokens
            < 64
        ):
            assembly = candidate
            compilation = candidate_compilation
            break

    assert assembly is not None
    assert compilation is not None
    payload, final_prepared = runtime._execute_with_output_recovery(
        state,
        PreparedCall(
            assembly,
            compilation.report,
            "lean",
            yes_no_contract(),
        ),
        minimum_output_tokens=64,
        desired_output_tokens=64,
        context_limit_resolution=(700, "test"),
        allow_prompt_instruction_projection=True,
    )

    assert payload == {"answer": "yes"}
    assert final_prepared.report.reserved_response_tokens > 64
    assert client.yes_no_calls == 2
    assert any(
        component.name == "durable_prompt_instruction_projection"
        for component in assembly.components
    )
    assert any(
        event.event_type == "context_compiled"
        and event.payload.get("output_retry") == 1
        and event.payload.get("prompt_instruction_projection") is True
        for event in runtime.history.read_history(state.session_id)
    )
