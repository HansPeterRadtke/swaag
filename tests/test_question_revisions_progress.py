from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from swaag.history import HistoryStore
from swaag.orchestration import OrchestrationManager, OrchestrationStore
from swaag.orchestration_api import OrchestrationApi
from swaag.progress import plan_progress, worker_progress
from swaag.questions import apply_revision_controls, revision_event
from swaag.runtime import AgentRuntime
from swaag.session_lock import session_execution_lock
from swaag.system_context import runtime_system_context_sources
from swaag.tools.registry import ToolRegistry
from swaag.workers import WorkerManager


def setup(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    workers = WorkerManager(runtime)
    worker = workers.create('Inspect available evidence')
    state = runtime.history.rebuild_from_history(worker.session_id)
    question = runtime.history.record_event(state, 'agent_question', {
        'action_index': 1, 'question': 'Which target?', 'criticality': 'optional',
        'reason': 'Target is unknown', 'assumption_if_unanswered': 'Use the local draft'})
    manager = OrchestrationManager(workers)
    return runtime, workers, worker, state, question, OrchestrationApi(manager)


def revision(question, expected=0, blocking=False):
    return {'question_id': question.id, 'expected_revision': expected, 'question': 'Which verified target?',
        'criticality': 'blocking' if blocking else 'optional', 'importance': 'critical' if blocking else 'major',
        'reason': 'New evidence changes the target', 'assumption_if_unanswered': '' if blocking else 'Keep a local draft',
        'revision_reason': 'Reassessed the applicable guidelines'}


def test_owner_revision_preserves_original_and_replays(make_config):
    runtime, workers, worker, state, question, api = setup(make_config)
    try:
        _, result = ToolRegistry().dispatch('revise_question', revision(question), runtime.config, state)
        assert state.open_questions[0]['revision'] == 0
        event = result.generated_events[0]
        runtime.history.record_event(state, event.event_type, event.payload)
        for checkpoint in (True, False):
            rebuilt = runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=checkpoint)
            assert rebuilt.open_questions[0]['revision'] == 1
            assert rebuilt.open_questions[0]['question'] == 'Which verified target?'
            assert rebuilt.open_questions[0]['source_event_sequence'] == question.sequence
        assert runtime.history.read_history(state.session_id)[question.sequence-1].payload['question'] == 'Which target?'
        with pytest.raises(ValueError, match='Stale'):
            revision_event(state, revision(question), runtime.config, actor='owner')
    finally:
        workers.shutdown()


def test_orchestrator_revision_is_queued_during_work_and_cas_rejects_stale_edit(make_config):
    runtime, workers, worker, state, question, api = setup(make_config)
    entered, release = threading.Event(), threading.Event()
    def run():
        with session_execution_lock(runtime.history, state.session_id):
            entered.set()
            assert release.wait(5)
            apply_revision_controls(runtime, state)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(2)
        response = api.execute('questions.revise', {'worker_id': worker.worker_id, 'revision': revision(question), 'control_id': 'first'})
        assert response['pending'] and response['outcome'] is None
        assert response['inventory']['questions'][0]['revision'] == 0
        assert len(response['inventory']['pending_revisions']) == 1
        release.set(); thread.join(3); assert not thread.is_alive()
        response = api.execute('questions.revise', {'worker_id': worker.worker_id, 'revision': revision(question), 'control_id': 'first'})
        assert not response['pending']
        assert response['outcome']['event_type'] == 'agent_question_revised'
        response = api.execute('questions.revise', {'worker_id': worker.worker_id, 'revision': revision(question), 'control_id': 'stale'})
        assert response['outcome']['event_type'] == 'agent_question_revision_rejected'
        assert response['inventory']['questions'][0]['revision'] == 1
        events = runtime.history.read_history(state.session_id)
        assert sum(event.event_type == 'agent_question_revised' for event in events) == 1
        assert not any(event.event_type == 'agent_question_resolved' for event in events)
    finally:
        release.set(); thread.join(3); workers.shutdown()


def test_complete_inventory_prioritizes_questions_and_preserves_exact_source(make_config):
    runtime, workers, worker, state, question, api = setup(make_config)
    try:
        other = workers.create('Review another component')
        other_state = runtime.history.rebuild_from_history(other.session_id)
        for index in range(18):
            runtime.history.record_event(other_state, 'agent_question', {
                'action_index': 1, 'question': f'Exact outstanding question {index}?',
                'criticality': 'blocking' if index == 17 else 'optional', 'importance': 'critical' if index == 17 else 'minor',
                'reason': 'Evidence is absent', 'assumption_if_unanswered': '' if index == 17 else 'Retain draft'})
        inventory = api.execute('questions.list')['inventory']
        assert inventory['complete'] and len(inventory['questions']) == 19
        assert inventory['questions'][0]['question'] == 'Exact outstanding question 17?'
        runtime.history.record_event(state, 'worker_question_inventory', inventory)
        source = next(source for source in runtime_system_context_sources(runtime.config, state) if source.name == 'worker_question_inventory')
        assert not source.optional
        assert all(item['question'] in source.text for item in inventory['questions'])
    finally:
        workers.shutdown()


def test_reentrant_session_ownership_and_idle_application_never_calls_model(make_config):
    runtime, workers, worker, state, question, api = setup(make_config)
    try:
        with session_execution_lock(runtime.history, state.session_id):
            with session_execution_lock(runtime.history, state.session_id, blocking=False) as acquired:
                assert acquired
        response = api.execute('questions.revise', {'worker_id': worker.worker_id, 'revision': revision(question, blocking=True)})
        assert not response['pending']
        assert response['inventory']['questions'][0]['criticality'] == 'blocking'
        assert runtime.run_pending_controls_in_session(state) is None
    finally:
        workers.shutdown()


def test_progress_history_finite_endless_and_qualified_estimates(make_config):
    runtime, workers, worker, state, _, api = setup(make_config)
    try:
        report = {'steps': [{'id': 'a', 'label': 'Inspect evidence', 'state': 'completed', 'weight': 2},
                            {'id': 'b', 'label': 'Verify result', 'state': 'in_progress', 'weight': 3}],
                  'reason': 'Inspection finished, verification is running', 'estimated_remaining_seconds': None, 'estimate_conditions': ''}
        _, result = ToolRegistry().dispatch('report_progress', report, runtime.config, state)
        event = result.generated_events[0]
        runtime.history.record_event(state, event.event_type, event.payload)
        rebuilt = runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=False)
        assert worker_progress(worker, rebuilt)['overall_percent'] == 40
        endless = worker_progress(replace(worker, completion_mode='continuous'), rebuilt)
        assert endless['overall_percent'] is None and endless['current_cycle_percent'] == 40
        assert endless['estimated_remaining_seconds'] is None
        assert workers.inspect(worker.worker_id)['progress']['overall_percent'] == 40
        with pytest.raises(ValueError, match='conditions'):
            ToolRegistry().get('report_progress').validate({**report, 'estimated_remaining_seconds': 10})
        plan = api.execute('create', {'objective': 'Continuous research with finite preparation'})['plan']['plan_id']
        api.execute('node.add', {'plan_id': plan, 'objective': 'Research continually', 'completion_mode': 'continuous'})
        api.execute('node.add', {'plan_id': plan, 'objective': 'Prepare sources'})
        progress = api.execute('get', {'plan_id': plan})['progress']
        assert progress['overall_percent'] is None and progress['continuous_nodes'] == 1
        assert progress['total_finite_nodes'] == 1
    finally:
        workers.shutdown()


def test_explicit_foreground_orchestrator_has_guideline_reading_and_control_tools(make_config):
    from swaag.communication import CommunicationService
    runtime = AgentRuntime(make_config(), model_client=object())
    service = CommunicationService.from_runtime(runtime)
    try:
        assert runtime.config.communication.enabled is False
        tools = set(service.orchestrator_runtime.config.tools.enabled)
        assert {'read_file', 'search_repo', 'worker_questions', 'orchestration_control', 'report_progress'} <= tools
        state = service._orchestrator_state()
        assert 'orchestration' in service.orchestrator_runtime.tool_runtime_capabilities(state.session_id)
        assert any('Review the complete' in item.content for item in state.prompt_instructions)
    finally:
        service.close()


def test_continuous_mode_reaches_worker_creation(make_config):
    runtime, workers, worker, state, question, api = setup(make_config)
    started = []
    workers.start = lambda worker_id: started.append(worker_id)
    try:
        response = api.execute('plan.apply', {'plan_spec': {
            'objective': 'Ongoing improvement', 'nodes': [{'key': 'improve', 'objective': 'Improve continually',
                'completion_mode': 'continuous'}], 'start': True}})
        assert len(started) == 1
        assert workers.store.get(started[0]).completion_mode == 'continuous'
        assert response['progress']['overall_percent'] is None
    finally:
        workers.shutdown()


def test_supervision_transport_bypasses_occupied_semantic_request_slots(make_config):
    import asyncio
    from swaag.communication import CommunicationService
    runtime = AgentRuntime(make_config(), model_client=object())
    service = CommunicationService(runtime, max_concurrency=1)
    async def verify():
        await service._semaphore.acquire()
        try:
            result = await asyncio.wait_for(service._dispatch_json_line_request(
                {'op': 'orchestration.supervision'}), timeout=.5)
            assert result['supervision']['automatic_semantic_intervention'] is False
        finally:
            service._semaphore.release()
    try:
        asyncio.run(verify())
    finally:
        service.close()
