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
