"""A routing paraphrase cannot replace exact user authority at a worker boundary."""
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from swaag.types import Message
from tests.test_preemption import _ImmediateClient


def test_worker_id_control_preserves_original_authority_and_order(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['agent_control', 'agent_status_lookup'],
        tools__allow_side_effect_tools=True), model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    exact = '\n  Finish tests, but do not deploy until I approve the report.  \n'
    correction = 'Keep the report private as well.'
    try:
        worker = service.workers.create('Run tests and deploy after success.')
        source = service._orchestrator_state()
        runtime._record_message(source, Message(role='user', content=exact, created_at='2026-10-14T00:00:00Z'))
        runtime.history.record_event(source, 'turn_started', {'user_text':exact, 'turn_index':1})
        runtime._record_message(source, Message(role='user', content=correction, created_at='2026-10-14T00:00:01Z'))
        run = runtime.execute_tool_once('agent_control', {'session_ref':worker.worker_id, 'message':'Hold deployment.'}, session_id=source.session_id)
        assert run.error is None
        receipt = run.tool_result.output
        assert receipt['queued'] is True and receipt['session_id'] == worker.session_id
        pending = runtime.history.list_pending_control_messages(worker.session_id)
        assert pending[0]['message'] == 'Hold deployment.'
        assert [item['message'] for item in pending[0]['authority']] == [exact, correction]
        runtime.history.mark_control_message_processed(worker.session_id, receipt['control_id'])
        restored = runtime.history.list_authoritative_control_messages(worker.session_id)
        assert restored[0]['authority'] == pending[0]['authority']
        state = runtime.history.rebuild_from_history(worker.session_id)
        prompt = runtime._authoritative_intervention_texts(state)[0]
        assert exact in prompt and prompt.index(exact) < prompt.index(correction)
        assert 'Hold deployment.' not in prompt
        status = runtime.execute_tool_once('agent_status_lookup', {'session_ref':worker.worker_id}, session_id=source.session_id)
        assert status.error is None and status.tool_result.output['session_id'] == worker.session_id
    finally:
        service.close()


def test_control_authority_idempotency_preserves_first_accepted_payload(make_config):
    runtime = AgentRuntime(make_config())
    target = runtime.create_or_load_session()
    original = [{'source_session_id':'source', 'message':'Do not deploy.'}]
    first = runtime.history.enqueue_control_message(target.session_id, 'hold', control_id='stable_control', authority=original)
    second = runtime.history.enqueue_control_message(target.session_id, 'deploy', control_id='stable_control', authority=[{'source_session_id':'source','message':'Changed retry'}])
    assert second == first
    assert runtime.history.list_pending_control_messages(target.session_id)[0]['authority'] == original


def test_routing_note_is_not_promoted_to_worker_action_or_completion_authority(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=_ImmediateClient('unused'))
    worker = runtime.create_or_load_session()
    original = 'Translate the quoted prohibition; leave the worker task unchanged.'
    wrong_note = 'ROUTING_NOTE_783: stop the worker and demand approval.'
    runtime.history.enqueue_control_message(worker.session_id, wrong_note, source='agent_control',
        authority=[{'source_session_id':'foreground','source_event_sequence':7,'message':original}])
    assert runtime._authoritative_intervention_texts(worker) == [original]
    components = runtime._current_authority_components(worker, 'Complete the authorized work.')
    assert original in '\n'.join(c.text for c in components)
    assert wrong_note not in '\n'.join(c.text for c in components)
    from swaag.grammar import agent_action_contract
    prepared = runtime._prepare_action_call(worker,
        original_request='Complete the authorized work.',
        pending_messages=runtime._authoritative_intervention_texts(worker),
        tool_specs=[], contract=agent_action_contract([]), validation_feedback='')
    assert original in prepared.assembly.prompt_text
    assert wrong_note not in prepared.assembly.prompt_text
    # Retaining exact audit evidence is separate from promoting it to authority.
    assert runtime.history.list_authoritative_control_messages(worker.session_id)[0]['message'] == wrong_note
