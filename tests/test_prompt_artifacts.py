from __future__ import annotations

from swaag.grammar import yes_no_contract
from swaag.prompts import PromptBuilder
from swaag.runtime import AgentRuntime
from swaag.tokens import ConservativeEstimator
from swaag.types import Message, PromptComponent
from swaag.utils import sha256_text


def _artifact_map(assembly) -> dict[str, str]:
    return {artifact.source: artifact.sha256 for artifact in assembly.prompt_artifacts}


def test_template_backed_prompt_records_canonical_artifact_versions(make_config) -> None:
    config = make_config()
    builder = PromptBuilder(config)
    assembly = builder.build_agent_action_prompt(
        [Message(role="user", content="inspect the current state", created_at="now")],
        [],
        original_request="inspect the current state",
        pending_user_messages=[],
        prompt_mode="standard",
    )
    artifacts = _artifact_map(assembly)

    assert "prompt_protocol:explicit_text_fallback_v1" in artifacts
    assert "rendered_system:action" in artifacts
    assert f"assets/prompts/{config.prompts.standard_system_template}" in artifacts
    action_source = f"assets/prompts/{config.prompts.action_template}"
    assert artifacts[action_source] == sha256_text(
        builder._load_template(config.prompts.action_template)
    )
    assert all(len(value) == 64 for value in artifacts.values())


def test_inline_semantic_prompt_versions_rendered_system(make_config) -> None:
    assembly = PromptBuilder(make_config()).build_semantic_operation_prompt(
        kind="history_analysis",
        system_instruction="Semantically inspect exact evidence.",
        components=[PromptComponent(name="evidence", text="exact evidence")],
    )
    artifacts = _artifact_map(assembly)

    assert artifacts["rendered_system:history_analysis"] == sha256_text(
        "Semantically inspect exact evidence."
    )
    assert not any(source.startswith("assets/prompts/") for source in artifacts)


def test_prompt_built_event_persists_prompt_and_artifact_hashes(make_config) -> None:
    config = make_config(model__context_limit=8_000)
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ConservativeEstimator(),
    )
    state = runtime.create_or_load_session()
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="history_analysis",
        system_instruction="Inspect evidence without changing it.",
        components=[PromptComponent(name="evidence", text="source event 7")],
    )
    contract = yes_no_contract()
    compilation = runtime._compile_context(
        state,
        assembly,
        contract,
        minimum_output_tokens=64,
    )

    runtime._record_prompt_built(state, assembly, contract, compilation.report)

    event = runtime.history.read_history(state.session_id)[-1]
    assert event.event_type == "prompt_built"
    assert event.payload["prompt_sha256"] == sha256_text(assembly.prompt_text)
    assert event.payload["prompt_artifacts"] == [
        {"source": artifact.source, "sha256": artifact.sha256}
        for artifact in assembly.prompt_artifacts
    ]
    assert event.payload["message_ranges"] == [
        {
            "role": message_range.role,
            "component_start": message_range.component_start,
            "component_end": message_range.component_end,
        }
        for message_range in assembly.message_ranges
    ]


def test_live_prompt_materialization_accounts_exact_server_chat_template(
    make_config,
) -> None:
    template_hash = sha256_text("qwen-template")
    protocol_hash = sha256_text("qwen-model-and-template")

    class TemplateClient:
        def render_chat_prompt(self, messages):
            assert [item["role"] for item in messages] == ["system", "user"]
            return {
                "prompt": (
                    "<|im_start|>system\n"
                    + messages[0]["content"]
                    + "<|im_end|>\n<|im_start|>user\n"
                    + messages[1]["content"]
                    + "<|im_end|>\n<|im_start|>assistant\n"
                ),
                "chat_template_sha256": template_hash,
                "prompt_protocol_sha256": protocol_hash,
            }

    runtime = AgentRuntime(
        make_config(model__context_limit=8_000),
        model_client=TemplateClient(),
        token_counter=ConservativeEstimator(),
    )
    state = runtime.create_or_load_session()
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="history_analysis",
        system_instruction="Exact system instruction.",
        components=[PromptComponent(name="evidence", text="Exact evidence.")],
    )

    runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
    )

    assert assembly.prompt_text.startswith("<|im_start|>system\n")
    assert "".join(component.text for component in assembly.components) == assembly.prompt_text
    assert not any(component.name.startswith("fallback_") for component in assembly.components)
    assert _artifact_map(assembly)["prompt_protocol:server_chat_template"] == protocol_hash
    assert [message["content"] for message in runtime._assembly_chat_messages(assembly)] == [
        "Exact system instruction.",
        "Exact evidence.",
    ]
