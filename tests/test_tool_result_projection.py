from __future__ import annotations

import json

from swaag.grammar import tool_result_projection_contract
from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.prompts import PromptBuilder
from swaag.types import CompletionResult, ContractSpec, Message


class _CharacterCountProjectionClient:
    is_deterministic_test_client = True

    def __init__(self, markers: list[str]):
        self.markers = markers
        self.requests = []

    def tokenize(self, text: str) -> int:
        return len(text)

    def tokenize_selection(self, text: str) -> int:
        return len(text)

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy(
            "test", "server_schema", contract.mode, 30, 0.01
        )

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature=None,
    ):
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload, **_kwargs):
        self.requests.append(payload)
        projection = " ".join(
            marker for marker in self.markers if marker in payload["prompt"]
        ) or "fragment retained"
        response = json.dumps({"projection": projection})
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_projection_contract_is_closed_json_schema():
    contract = tool_result_projection_contract()
    assert contract.name == "tool_result_projection"
    assert contract.json_schema["additionalProperties"] is False
    assert contract.json_schema["required"] == ["projection"]


def test_prompt_builder_substitutes_projection_but_keeps_source_reference(make_config):
    builder = PromptBuilder(make_config())
    messages = [
        Message(role="user", content="find the answer", created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="RAW BULK " * 500,
            created_at="t1",
            metadata={"source_event_sequence": 77, "source_event_hash": "deadbeef"},
        ),
    ]
    assembly = builder.build_agent_action_prompt(
        messages,
        [],
        original_request="find the answer",
        pending_user_messages=[],
        prompt_mode="standard",
        tool_result_projections={77: "only the semantically relevant fact"},
    )
    assert "SOURCE EVENT sequence=77 hash=deadbeef" in assembly.prompt_text
    assert "SEMANTIC PROJECTION" in assembly.prompt_text
    assert "only the semantically relevant fact" in assembly.prompt_text
    assert "RAW BULK RAW BULK" not in assembly.prompt_text


def test_projection_prompt_contains_goal_source_and_target(make_config):
    builder = PromptBuilder(make_config())
    assembly = builder.build_tool_result_projection_prompt(
        original_request="locate the failing test",
        tool_name="shell_command",
        raw_tool_result="lots of output",
        source_event_sequence=12,
        source_event_hash="abc",
        target_tokens=222,
    )
    assert assembly.kind == "tool_result_projection"
    assert "locate the failing test" in assembly.prompt_text
    assert "sequence=12 hash=abc" in assembly.prompt_text
    assert "222 tokens" in assembly.prompt_text


def test_runtime_reuses_only_matching_projection_that_meets_new_target(
    make_config, tmp_path
):
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    event = runtime.history.record_event(
        state,
        "tool_result_projected",
        {
            "source_event_sequence": 12,
            "source_event_hash": "abc",
            "tool_name": "shell_command",
            "target_tokens": 100,
            "original_tokens": 800,
            "projected_tokens": 80,
            "overflow_tokens": 400,
            "projection": "durable semantic projection",
        },
    )

    assert runtime._stored_tool_result_projection(
        state,
        source_event_sequence=12,
        source_event_hash="abc",
        target_tokens=90,
    ) == (event.sequence, "durable semantic projection", 80)
    assert runtime._stored_tool_result_projection(
        state,
        source_event_sequence=12,
        source_event_hash="abc",
        target_tokens=70,
    ) is None
    assert runtime._stored_tool_result_projection(
        state,
        source_event_sequence=12,
        source_event_hash="different",
        target_tokens=90,
    ) is None


def test_oversized_tool_result_projection_preserves_every_fragment(make_config) -> None:
    markers = [f"critical-tool-marker-{index}" for index in range(4)]
    source_text = "".join("A" * 2_000 + marker for marker in markers)
    config = make_config(
        model__context_limit=2_000,
        context__max_compaction_rounds=4,
        context__safety_margin_tokens=32,
    )
    client = _CharacterCountProjectionClient(markers)
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    source = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "generic_reader",
            "raw_input": {},
            "validated_input": {},
            "output": {"text": source_text},
        },
    )
    message = Message(
        role="tool",
        name="generic_reader",
        content=source_text,
        created_at="t",
        metadata={
            "source_event_sequence": source.sequence,
            "source_event_hash": source.hash,
            "source_event_references": [],
        },
    )

    projection = runtime._create_tool_result_projection(
        state,
        original_request="Find the critical marker without losing exact source data.",
        message=message,
        target_tokens=256,
        original_tokens=8_000,
        overflow_tokens=4_000,
    )

    assert projection == " ".join(markers)
    assert len(client.requests) > 1
    projected = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "tool_result_projected"
    ][-1]
    assert projected.payload["source_event_sequence"] == source.sequence
    assert projected.payload["source_event_hash"] == source.hash
