from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from swaag.history import HistoryStore
from swaag.questions import RESOLUTIONS, validate_question_capacity
from swaag.runtime import AgentRuntime
from swaag.system_context import runtime_system_context_sources
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry
from swaag.types import Message


def ask(runtime, state):
    return runtime.history.record_event(state, "agent_question", {
        "action_index":1, "question":"Which destination?", "criticality":"optional",
        "reason":"The destination is unspecified", "assumption_if_unanswered":"Use the local draft"})


def resolve_input(question_id, sequence, resolution="answered_by_user"):
    return dict(operation="resolve", question_id=question_id, resolution=resolution,
        answer="The staging destination" if resolution.startswith("answered_") else "",
        reason="Recorded evidence resolves this question", evidence_sequences=[sequence])


def setup(make_config):
    config=make_config()
    runtime=AgentRuntime(config, model_client=object())
    state=runtime.create_or_load_session()
    question=ask(runtime,state)
    return config,runtime,state,question


def test_question_survives_new_messages_checkpoint_and_exact_replay(make_config):
    config,runtime,state,question=setup(make_config)
    runtime.history.record_event(state,"message_added",{"message":asdict(Message(
        role="user",content="Continue inspecting the repository",created_at="now"))})
    for checkpoint in (True,False):
        rebuilt=runtime.history.rebuild_from_history(state.session_id,prefer_checkpoint=checkpoint)
        assert rebuilt.open_questions[0]["question_id"]==question.id
        assert rebuilt.open_questions[0]["assumption_if_unanswered"]=="Use the local draft"
    sources=runtime_system_context_sources(config,state)
    source=next(item for item in sources if item.name=="open_questions")
    assert "Which destination?" in source.text
    assert not source.optional


def test_legacy_checkpoint_rebuilds_previously_unprojected_questions(make_config):
    _,runtime,state,question=setup(make_config)
    path=runtime.history.checkpoint_path(state.session_id)
    payload=json.loads(path.read_text()); payload.pop("open_questions")
    path.write_text(json.dumps(payload))
    rebuilt=runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt.open_questions[0]["question_id"]==question.id


@pytest.mark.parametrize("resolution",RESOLUTIONS)
def test_all_resolutions_replay_without_erasing_original_evidence(make_config,resolution):
    config,runtime,state,question=setup(make_config)
    if resolution=="answered_by_research":
        evidence=runtime.history.record_event(state,"tool_result",{
            "tool_name":"read_file","success":True,"output":{"destination":"staging"},
            "content":"The configured destination is staging","error":None,"duration_ms":1,"raw_input":{},"validated_input":{}})
    else:
        evidence=runtime.history.record_event(state,"message_added",{"message":asdict(Message(
            role="user",content="Use staging; the previous destination question is settled.",created_at="now"))})
    _,result=ToolRegistry().dispatch("questions",resolve_input(question.id,evidence.sequence,resolution),config,state)
    assert len(state.open_questions)==1  # tool emits events; it cannot mutate canonical state
    event=result.generated_events[0]
    runtime.history.record_event(state,event.event_type,event.payload)
    rebuilt=runtime.history.rebuild_from_history(state.session_id,prefer_checkpoint=False)
    assert rebuilt.open_questions==[]
    events=runtime.history.read_history(state.session_id)
    assert any(item.id==question.id for item in events)
    assert events[-1].payload["resolution"]==resolution
    assert events[-1].payload["evidence_sequences"]==[evidence.sequence]
    with pytest.raises(ToolValidationError,match="already resolved"):
        ToolRegistry().dispatch("questions",resolve_input(question.id,evidence.sequence,resolution),config,state)


def test_answer_cannot_cite_invented_evidence_or_an_assumption_as_user_reply(make_config):
    config,runtime,state,question=setup(make_config)
    for sequence,match in ((99999,"missing"),(question.sequence,"recorded user")):
        with pytest.raises(ToolValidationError,match=match):
            ToolRegistry().dispatch("questions",resolve_input(question.id,sequence),config,state)
    assert len(runtime.history.rebuild_from_history(state.session_id).open_questions)==1


def test_open_question_pressure_rejects_new_questions_without_deleting_old_ones(make_config):
    config,runtime,state,question=setup(make_config)
    config.runtime.max_open_questions=1
    candidate=SimpleNamespace(question="Another?",reason="Need a choice",assumption_if_unanswered="Keep existing")
    with pytest.raises(ValueError,match="capacity"):
        validate_question_capacity(state,[candidate],config)
    validate_question_capacity(state,[],config)  # useful work can still continue
    assert state.open_questions[0]["question_id"]==question.id
    config.runtime.max_open_questions=2
    config.runtime.max_open_question_chars=1
    with pytest.raises(ValueError,match="character budget"):
        validate_question_capacity(state,[candidate],config)


def test_question_schema_is_closed_and_default_capability_enabled(make_config):
    tool=ToolRegistry().get("questions")
    schema=tool.input_schema
    assert set(schema["required"])==set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert "questions" in make_config().tools.enabled
    with pytest.raises(ToolValidationError):
        tool.validate({"operation":"list","unexpected":True})
