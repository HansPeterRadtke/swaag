#!/usr/bin/env python3
"""Uncached live orchestrator/inventory/backend-supervision acceptance on one backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.utils import utc_now_iso


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:14829')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = output.parent / (output.stem + '-state')
    root.mkdir(parents=True, exist_ok=False)
    config = load_config(env={
        'SWAAG__SESSIONS__ROOT': str(root / 'sessions'),
        'SWAAG__MODEL__BASE_URL': args.base_url,
        'SWAAG__MODEL__CACHE_ENABLED': 'false',
        'SWAAG__TOOLS__READ_ROOTS': json.dumps([str(root)]),
        'SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED': 'true',
    })
    runtime = AgentRuntime(config)
    service = CommunicationService.from_runtime(runtime)
    questions = [
        ('Which staging target is authorized?', 'blocking', 'critical', ''),
        ('Which accent color do you prefer?', 'optional', 'normal', 'Keep the default color'),
        ('Which footer wording do you prefer?', 'optional', 'minor', 'Keep the existing footer'),
    ]
    first = service.workers.create('Prepare a staging release')
    second = service.workers.create('Prepare appearance options', completion_mode='continuous')
    for index, (question, criticality, importance, assumption) in enumerate(questions):
        worker = first if index == 0 else second
        state = runtime.history.rebuild_from_history(worker.session_id)
        runtime.history.record_event(state, 'agent_question', {
            'action_index': 0, 'question': question, 'criticality': criticality,
            'importance': importance, 'reason': 'This requires the user\'s decision',
            'assumption_if_unanswered': assumption})
    before = service.orchestration_api.execute('questions.list')['inventory']
    result, errors = {}, []
    started = time.monotonic()
    def invoke():
        try:
            result.update(service.orchestrator_message(
                'Report the complete outstanding question list from the worker inventory already supplied in your context. '
                'State the critical blocking question first, followed by both optional questions and their provisional assumptions. '
                'Quote every question exactly. Explain that the appearance worker is intentionally continuous and has no overall completion percentage. '
                'This is a status-only request: do not answer or change any question, start workers, or call tools.'))
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
    thread = threading.Thread(target=invoke)
    thread.start()
    samples = []
    try:
        with (root / 'supervision.jsonl').open('w') as stream:
            while thread.is_alive():
                sample = service.supervisor.snapshot()
                samples.append(sample)
                stream.write(json.dumps(sample) + '\n'); stream.flush()
                thread.join(1)
        state = service._orchestrator_state()
        events = runtime.history.read_history(state.session_id)
        requests = [event for event in events if event.event_type == 'model_request_sent' and event.payload['kind'] == 'action']
        prompts = [event.payload['request'].get('prompt', '') for event in requests]
        answer = result.get('answer', '')
        backend_samples = [backend for sample in samples for backend in sample['backends'].values()]
        active_samples = [item for sample in samples for item in sample['active_sessions']]
        checks = {
            'no_runtime_error': not errors,
            'all_exact_questions_in_action_input': any(all(item[0] in prompt for item in questions) for prompt in prompts),
            'all_exact_questions_reported': all(item[0] in answer for item in questions),
            'critical_question_first': all(answer.find(questions[0][0]) < answer.find(item[0]) for item in questions[1:]),
            'provisional_assumptions_preserved': all(item[3] in answer for item in questions[1:]),
            'continuous_state_in_action_input': any('intentionally_endless' in prompt for prompt in prompts),
            'questions_unchanged': before == service.orchestration_api.execute('questions.list')['inventory'],
            'no_tool_calls': not any(event.event_type == 'tool_called' for event in events),
            'orchestrator_independently_supervised': any('orchestrator' in item['roles'] for item in active_samples),
            'actual_backend_processing_observed': any(item.get('state') == 'processing' for item in backend_samples),
            'backend_progress_observed': any(item.get('progress_observed_at') for item in backend_samples),
            'no_speculative_inference_deadline': any(event.event_type == 'model_backend_activity' and
                event.payload.get('timeout_policy') == 'observed_local_activity_with_fail_safe_backstop' for event in events),
            'independent_completion_accepted': any(event.event_type == 'completion_evaluated' and event.payload.get('complete') is True for event in events),
            'no_repeated_action': sum(event.event_type == 'agent_action_selected' for event in events) == 1,
        }
        calls = [dict(kind=event.payload['kind'], elapsed_seconds=event.payload.get('elapsed_seconds'),
            timings=event.payload['completion']['raw_response'].get('timings'),
            timeout_policy=event.payload['completion']['raw_response'].get('timeout_policy'))
            for event in events if event.event_type == 'model_response_received']
        report = {'generated_at': utc_now_iso(), 'passed': all(checks.values()), 'checks': checks,
                  'elapsed_seconds': time.monotonic() - started, 'answer': answer, 'errors': errors,
                  'session_id': state.session_id, 'calls': calls, 'samples': len(samples),
                  'conditions': 'Jetson production 27B model, native llama.cpp, uncached; software regression suite running concurrently under existing CPU quota'}
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return 0 if report['passed'] else 1
    finally:
        service.close()


if __name__ == '__main__':
    raise SystemExit(main())
