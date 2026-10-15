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
    interventions = runtime._authoritative_intervention_texts(worker)
    assert len(interventions) == 1 and original in interventions[0]
    assert 'foreground orchestrator' in interventions[0]
    assert 'does not replace the worker objective' in interventions[0]
    assert wrong_note not in interventions[0]
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


def test_foreground_control_retries_reuse_receipt_but_new_turns_do_not(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['agent_control'], tools__allow_side_effect_tools=True),
        model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    exact = 'Hold publication until I approve; continue the checks.'
    try:
        worker = service.workers.create('Check and publish Alpha.')
        other = service.workers.create('Check and publish Beta.')
        source = service._orchestrator_state()
        def turn():
            runtime._record_message(source, Message(role='user', content=exact,
                created_at='2026-10-15T17:00:00Z'))
            runtime.history.record_event(source, 'turn_started',
                {'user_text': exact, 'turn_index': source.turn_count + 1})
        def send(target, note):
            result = runtime.execute_tool_once('agent_control',
                {'session_ref': target, 'message': note}, session_id=source.session_id)
            assert result.error is None
            return result.tool_result.output
        turn()
        first = send(worker.worker_id, 'First routing note.')
        retry = send(worker.session_id, 'A different routing paraphrase of the same instruction.')
        assert retry['control_id'] == first['control_id']
        assert retry['message'] == first['message']
        assert first['status'] == retry['status'] == 'pending'
        assert len(runtime.history.list_pending_control_messages(worker.session_id)) == 1
        runtime.history.mark_control_message_processed(worker.session_id, first['control_id'])
        processed = send(worker.worker_id, 'Retry after acknowledgement.')
        assert processed['control_id'] == first['control_id']
        assert processed['status'] == 'processed'
        assert runtime.history.list_pending_control_messages(worker.session_id) == []
        # Repeating the same words in a NEW user turn is a new instruction.
        source = runtime.history.rebuild_from_history(source.session_id)
        turn()
        later = send(worker.worker_id, 'Same wording, new user event.')
        assert later['control_id'] != first['control_id']
        second_target = send(other.worker_id, 'Another target in this turn.')
        assert second_target['control_id'] != later['control_id']
        assert len(runtime.history.list_authoritative_control_messages(worker.session_id)) == 2
        assert len(runtime.history.list_pending_control_messages(other.session_id)) == 1
    finally:
        service.close()



def test_concurrent_foreground_retries_and_reopened_store_share_one_control(make_config):
    from concurrent.futures import ThreadPoolExecutor
    from swaag.history import HistoryStore
    from swaag.tools.base import ToolContext
    from swaag.tools.control import AgentControlTool
    from swaag.environment.environment import AgentEnvironment
    runtime = AgentRuntime(make_config(model__context_limit=32000),
        model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create('Check and publish.')
        source = service._orchestrator_state()
        original = 'Continue checking, and hold publication.'
        runtime._record_message(source, Message(role='user', content=original,
            created_at='2026-10-15T17:00:00Z'))
        runtime.history.record_event(source, 'turn_started',
            {'user_text': original, 'turn_index': 1})
        tool = AgentControlTool()
        context = ToolContext(runtime.config, source, AgentEnvironment(runtime.config, source),
            runtime_capabilities={'orchestration':service.orchestration_api,
                                  'system_snapshot_provider':service._orchestrator_runtime_snapshot})
        def send(note):
            return tool.execute({'session_ref':worker.session_id,'message':note},context).output
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(send, ['first','second','third','fourth']))
        assert len({r['control_id'] for r in replies}) == 1
        reopened = HistoryStore(runtime.config.sessions.root)
        pending = reopened.list_pending_control_messages(worker.session_id)
        assert len(pending) == 1
        assert all(r['message']==pending[0]['message'] for r in replies)
        reopened.mark_control_message_processed(worker.session_id, pending[0]['control_id'])
        context.session_state = reopened.rebuild_from_history(source.session_id)
        reply = send('Replay with a freshly rebuilt source state.')
        assert reply['status'] == 'processed'
        assert reopened.list_pending_control_messages(worker.session_id) == []
        # A correction received within this same turn changes the authority set.
        runtime._record_message(context.session_state, Message(role='user',
            content='Also keep the report private.',
            created_at='2026-10-15T17:00:01Z',
            metadata={'orchestrator_control_id':'correction_private'}))
        corrected = send('Apply the later correction.')
        assert corrected['control_id'] != reply['control_id']
        assert [a['message'] for a in corrected['authority']] == [
            original, 'Also keep the report private.']
    finally:
        service.close()


def test_direct_controls_without_foreground_provenance_are_not_collapsed(make_config):
    runtime = AgentRuntime(make_config(tools__enabled=['agent_control'],
        tools__allow_side_effect_tools=True), model_client=_ImmediateClient('unused'))
    source = runtime.create_or_load_session()
    target = runtime.create_or_load_session()
    receipts = []
    for _ in range(2):
        result = runtime.execute_tool_once('agent_control',
            {'session_ref':target.session_id,'message':'Continue the next step.'},
            session_id=source.session_id)
        assert result.error is None
        receipts.append(result.tool_result.output)
    assert receipts[0]['control_id'] != receipts[1]['control_id']
    assert all(not r['authority'] for r in receipts)
    assert len(runtime.history.list_pending_control_messages(target.session_id)) == 2


def test_existing_foreground_policy_is_upgraded_once_without_losing_user_rules(make_config):
    from dataclasses import asdict
    from swaag.prompt_instructions import make_prompt_instruction
    runtime = AgentRuntime(make_config(model__context_limit=32000),
        model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        state = service._load_orchestrator_state()
        for instruction_id in ['instruction_swaag_user_controls_v2', 'user_keep_private']:
            instruction = make_prompt_instruction(runtime.config, title=instruction_id,
                content='Keep the report private.', scopes=['all'], categories=['orchestration'],
                instruction_id=instruction_id, authority='project_policy',
                source_kind='swaag_core', source_ref='test', specificity=100)
            runtime.history.record_event(state, 'prompt_instruction_added',
                {'instruction':asdict(instruction),'instruction_store':'session'})
        state = service._orchestrator_state()
        active = {x.instruction_id:x for x in state.prompt_instructions}
        assert 'instruction_swaag_user_controls_v2' not in active
        assert active['user_keep_private'].content == 'Keep the report private.'
        current = active['instruction_swaag_user_controls_v3']
        assert 'assistant_message' in current.content
        assert 'Describe unobserved actions as intentions' in current.content
        before = state.event_count
        assert service._orchestrator_state().event_count == before
    finally:
        service.close()
