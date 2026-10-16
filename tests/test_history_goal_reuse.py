import pytest

from swaag.compression import message_source_event_references, summary_message_payload
from swaag.grammar import yes_no_contract
from swaag.prompt_instructions import make_prompt_instruction
from swaag.runtime import AgentRuntime, BudgetExceededError
from swaag.tokens import ExactTokenCounter
from swaag.types import Message, PromptComponent


@pytest.fixture
def reduction_case(make_config, monkeypatch):
    runtime = AgentRuntime(
        make_config(model__context_limit=400, runtime__lean_on_overflow=False),
        model_client=object(),
        token_counter=ExactTokenCounter(lambda text: len(text.split())),
    )
    state = runtime.create_or_load_session()
    for role, text in [
        ("user", "Do the local task."),
        ("assistant", "earlier detail " * 1000),
        ("user", "Report progress."),
    ]:
        runtime._record_message(state, Message(role=role, content=text, created_at="now"))
    compacted = []

    def compact(state, **kwargs):
        compacted.append(kwargs)
        sources = kwargs.get("authoritative_messages") or list(state.messages)
        references = message_source_event_references([sources[1]])
        summary = summary_message_payload(
            "Earlier task evidence remains available.",
            source_message_start=1, source_message_count=1, created_at="now",
            source_event_references=references,
        )
        summary["metadata"]["projection_objective_sha256"] = kwargs.get("projection_objective_sha256")
        runtime._record_history_reduction(
            state,
            event_payload={
                "source_message_start": 1, "source_message_count": 1,
                "source_event_references": references,
                "source_event_ranges": summary["metadata"]["source_event_ranges"],
                "summary_message": summary, "summary_budget_report": {},
            },
            complete_source_messages=sources,
            reproject_authoritative=kwargs.get("authoritative_messages") is not None,
            hierarchical=False,
        )
        return True

    monkeypatch.setattr(runtime, "_compact_once", compact)
    current = {"system": "Report the current task state.", "snapshot": "WORKER_RUNNING"}

    def build(messages, components, projections, mode):
        return runtime.prompts.build_semantic_operation_prompt(
            kind="action", system_instruction=(current.get("summary_system", current["system"])
                if any(m.role == "summary" for m in messages) else current["system"]), prompt_mode=mode,
            components=[
                PromptComponent(name="history", category="history", text="\n".join(m.content for m in messages)),
                PromptComponent(name="runtime_system_snapshot", category="runtime_context", text=current["snapshot"]),
            ],
        )

    def prepare(**changes):
        options = dict(
            original_request="Report progress.", pending_messages=[], tool_specs=[],
            contract=yes_no_contract(), validation_feedback="", minimum_output_tokens=64,
            assembly_builder=build,
        )
        options.update(changes)
        return runtime._prepare_action_call(state, **options)

    return runtime, state, compacted, current, prepare


def test_same_task_reuses_derived_history_after_raw_overflow(reduction_case):
    runtime, state, compacted, current, prepare = reduction_case
    assert prepare().report.fits
    assert len(compacted) == 1
    assert compacted[0]["projection_objective_sha256"]
    runtime._record_message(state, Message(role="assistant", content="NEW_OBSERVATION_713", created_at="now"))
    current["snapshot"] = "WORKER_COMPLETED"
    prepared = prepare()
    assert prepared.report.fits
    assert "NEW_OBSERVATION_713" in prepared.assembly.prompt_text
    assert "WORKER_COMPLETED" in prepared.assembly.prompt_text
    assert "WORKER_RUNNING" not in prepared.assembly.prompt_text
    assert len(compacted) == 1, "The same task must not repeat successful history reduction"
    assert any("earlier detail" in m.content for m in runtime.history.read_authoritative_messages(state.session_id))
    rebuilt = runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=False)
    assert rebuilt.messages == state.messages
    events = runtime.history.read_history(state.session_id)
    compiled = [e for e in events if e.event_type == "context_compiled"]
    assert compiled[-2].payload["history_source"] == "authoritative_message_events"
    assert compiled[-2].payload["candidate_only"] is True
    assert compiled[-1].payload["history_source"] == "reused_goal_history_projection"


@pytest.mark.parametrize("change", ["request", "pending", "feedback", "user_event", "system", "policy", "tools", "unkeyed", "missing_provenance"])
def test_changed_authority_or_unknown_summary_requires_fresh_source_reduction(reduction_case, change):
    runtime, state, compacted, current, prepare = reduction_case
    assert prepare().report.fits
    previous_objective = compacted[0]["projection_objective_sha256"]
    changes = {}
    if change == "request":
        changes["original_request"] = "Explain the earlier details."
    elif change == "pending":
        changes["pending_messages"] = ["Do not publish anything."]
    elif change == "feedback":
        changes["validation_feedback"] = "Cite the original user evidence."
    elif change == "user_event":
        runtime._record_message(state, Message(role="user", content="Report progress.", created_at="later"))
    elif change == "system":
        current["system"] += " Preserve each source date."
    elif change == "policy":
        state.prompt_instructions.append(make_prompt_instruction(runtime.config,
            title="Output", content="Preserve each source date.", scopes=["action"]))
    elif change == "tools":
        changes["tool_specs"] = [("new_read_tool", "Read evidence", {}, "Evidence")]
    else:
        summary = next(m for m in state.messages if m.role == "summary")
        summary.metadata.pop("projection_objective_sha256" if change == "unkeyed" else "source_event_references")
    assert prepare(**changes).report.fits
    assert len(compacted) == 2
    assert compacted[-1]["authoritative_messages"]
    if change not in {"unkeyed", "missing_provenance"}:
        assert compacted[-1]["projection_objective_sha256"] != previous_objective
    summary = next(m for m in state.messages if m.role == "summary")
    assert summary.metadata["projection_event_type"] == "history_reprojected"
    assert summary.metadata["projection_event_hash"]
    rebuilt = runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=False)
    assert rebuilt.messages == state.messages


def test_exact_history_still_wins_when_capacity_grows(reduction_case):
    runtime, state, compacted, current, prepare = reduction_case
    assert prepare().report.fits
    runtime.config.model.context_limit = 4000
    prepared = prepare()
    assert "earlier detail" in prepared.assembly.prompt_text
    assert "Earlier task evidence remains available." not in prepared.assembly.prompt_text
    assert len(compacted) == 1


def test_disabled_compaction_does_not_reuse_a_lossy_view(reduction_case):
    runtime, state, compacted, current, prepare = reduction_case
    assert prepare().report.fits
    runtime.config.context.compact_on_overflow = False
    with pytest.raises(BudgetExceededError):
        prepare()
    assert len(compacted) == 1


def test_changed_instruction_selection_during_reuse_rejects_candidate(reduction_case):
    runtime, state, compacted, current, prepare = reduction_case
    assert prepare().report.fits
    current["summary_system"] = "Include the source date in each progress report."
    assert prepare().report.fits
    assert len(compacted) == 2
    assert compacted[-1]["authoritative_messages"]
