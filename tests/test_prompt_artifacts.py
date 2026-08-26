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

    assert "prompt_protocol:llama3" in artifacts
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
