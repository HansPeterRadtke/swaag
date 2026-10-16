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


@pytest.mark.parametrize("protected_repetitions, should_fit", [(1500, True), (2500, False)])
def test_disjoint_safe_history_reductions_combine_across_protected_direction(make_config, monkeypatch, protected_repetitions, should_fit):
    runtime = AgentRuntime(make_config(model__context_limit=6000, runtime__lean_on_overflow=False),
        model_client=object(), token_counter=ExactTokenCounter(lambda text: len(text.split())))
    state = runtime.create_or_load_session()
    protected = "Do not publish. " * protected_repetitions
    for role, text in [("user", "Report the task facts."),
                       ("assistant", "FACT_A " + "detail " * 2000),
                       ("user", protected),
                       ("assistant", "FACT_B " + "detail " * 2000),
                       ("user", "Report both facts and preserve my prohibition.")]:
        runtime._record_message(state, Message(role=role, content=text, created_at="now"))
    authoritative_before = runtime.history.read_authoritative_messages(state.session_id)

    def select(_state, messages):
        # Controlled semantic judgments: two safe regions separated by protected authority.
        return [{"source_message_start": i, "source_message_count": 1,
                 "criticality": "compressible", "reason": "retain exact fact, omit routine detail",
                 "projection": "", "hierarchical": False}
                for i, message in enumerate(messages)
                if message.role == "assistant" and "detail " in message.content]

    def summarize(_state, prepared, **kwargs):
        marker = "FACT_A" if "FACT_A" in prepared.assembly.prompt_text else "FACT_B"
        return {"summary": "Earlier task evidence.", "verbatim_spans": [marker], "preserve_recent_messages": 0}, prepared

    def build(messages, components, projections, mode):
        return runtime.prompts.build_semantic_operation_prompt(kind="action", prompt_mode=mode,
            system_instruction="Report both exact facts without changing files.",
            components=[PromptComponent(name="history", category="history", text="\n".join(m.content for m in messages))])

    monkeypatch.setattr(runtime, "_select_history_compaction_spans", select)
    monkeypatch.setattr(runtime, "_execute_with_output_recovery", summarize)
    options = dict(original_request="Report the task facts.", pending_messages=[],
        tool_specs=[], contract=yes_no_contract(), validation_feedback="",
        minimum_output_tokens=64, assembly_builder=build)
    if should_fit:
        prepared = runtime._prepare_action_call(state, **options)
        assert prepared.report.fits
        final_view = prepared.assembly.prompt_text
    else:
        with pytest.raises(BudgetExceededError):
            runtime._prepare_action_call(state, **options)
        final_view = "\n".join(message.content for message in state.messages)
    assert "FACT_A" in final_view and "FACT_B" in final_view
    assert protected in final_view
    assert runtime.history.read_authoritative_messages(state.session_id) == authoritative_before
    reductions = [e for e in runtime.history.read_history(state.session_id) if e.event_type == "history_compressed"]
    assert len(reductions) == 2
    assert 0 < reductions[0].payload["actual_recovered_tokens"] < reductions[0].payload["required_recovery_tokens"]
    assert runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=False).messages == state.messages
