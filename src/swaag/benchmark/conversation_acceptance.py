"""Real, isolated canonical SWAAG conversational acceptance.

Unlike benchmark_runner's developer-session action loop, this invokes exactly
CommunicationService.orchestrator_message, the actual human text/voice API.
No production session/worker state, tool access, or external writes are admitted.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps, utc_now_iso


def _workspace_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file()
    }


def _answer_checks(answer: str, contract: Any) -> dict[str, bool]:
    return {
        'nonempty_answer': bool(answer.strip()),
        'expected_answer_contains': all(x.casefold() in answer.casefold() for x in contract.expected_answer_contains),
        'expected_answer_regex': contract.expected_answer_regex is None
        or re.search(contract.expected_answer_regex, answer) is not None,
    }


def run_conversation_acceptance(
    *, output_dir: Path, config_paths: Sequence[Path] = (), case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Run independent, durable, model-backed acceptance on selected simple turns.

    Never clears an existing output, replays uncertain effects, or mutates a
    deployed session. A previously checked operation is recovered by its exact
    matching id through the canonical SWAAG durable operation API.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = conversational_smoke_benchmark_tasks()
    unknown = set(case_ids) - {x.task_id for x in tasks}
    if unknown:
        raise ValueError('unknown conversational cases: ' + ', '.join(sorted(unknown)))
    if case_ids:
        tasks = [task for task in tasks if task.task_id in case_ids]
    report: dict[str, Any] = {
        'benchmark': 'canonical-conversation', 'started_at': utc_now_iso(),
        'user_surface': 'CommunicationService.orchestrator_message',
        'isolated_from_production': True, 'cases': [],
    }
    for task in tasks:
        case_root = output_dir / "results" / task.task_id
        case_root.mkdir(parents=True, exist_ok=True)
        scenario = task.create(output_dir / "fixtures", live_mode=True)
        workspace = scenario.workspace
        before = _workspace_hashes(workspace)
        config = load_config(config_paths=list(config_paths), env={})
        config.sessions.root = output_dir / "isolated_sessions" / task.task_id
        config.tools.read_roots = [workspace]
        config.tools.enabled = []
        config.tools.allow_stateful_tools = False
        config.tools.allow_side_effect_tools = False
        config.communication.enabled = True
        config.communication.enabled_tools = []
        config.communication.model_routes = {}
        config.communication.fast_initial_reply = True
        config.model.cache_enabled = False
        config.budget_policy.output_ratio_by_kind['action'] = 0.08
        config.budget_policy.output_floor_ratio_by_kind['action'] = 0.05
        service = CommunicationService.from_runtime(AgentRuntime(config))
        started = time.monotonic()
        error = ''
        response: dict[str, Any] = {}
        recorded: dict[str, Any] = {}
        operation_id = 'conversation-acceptance-' + task.task_id
        workers_count = -1
        history_events: list[Any] = []
        try:
            operation = service.orchestrator_operations.get(operation_id)
            if operation is None:
                state = service._load_orchestrator_state()
                existing = service.orchestrator_runtime.history.read_history(state.session_id)
                # Session creation itself is an expected durable initialization
                # event. Reject only pre-existing user activity or side effects.
                if any(e.event_type in {'message_added', 'turn_started', 'tool_called'}
                       for e in existing):
                    raise RuntimeError('benchmark session already has user activity without matching durable operation')
                for msg in scenario.history_messages:
                    service.orchestrator_runtime._record_message(state, msg)
            response = service.orchestrator_message(scenario.prompt, operation_id=operation_id)
            recorded = service.orchestrator_operations.get(operation_id) or {}
            state = service._load_orchestrator_state()
            history_events = service.orchestrator_runtime.history.read_history(state.session_id)
            workers_count = len(service.workers.store.list(include_archived=False))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            service.workers.shutdown(wait=False)
            for manager in service.worker_model_managers.values():
                manager.shutdown(wait=False)
            service.close()
        answer = str(response.get('answer') or response.get('assistant_text') or '')
        checks = _answer_checks(answer, scenario.verification_contract)
        observed_user = [e for e in history_events if e.event_type == 'message_added'
                         and isinstance(e.payload.get('message'), dict)
                         and e.payload['message'].get('role') == 'user']
        checks.update({
            'durable_operation_completed': recorded.get('status') == 'completed',
            'zero_tools_used': not any(e.event_type == 'tool_called' for e in history_events),
            'zero_workers_created': workers_count == 0,
            'workspace_unchanged': _workspace_hashes(workspace) == before,
            'actual_final_user_turn_recorded': any(
                str(e.payload['message'].get('content') or '') == scenario.prompt
                for e in observed_user),
        })
        result = {
            'case_id': task.task_id, 'prompt': scenario.prompt,
            'seed_history_count': len(scenario.history_messages),
            'answer': answer, 'checks': checks,
            'passed': not error and all(checks.values()),
            'error': error, 'model_call_count': sum(e.event_type == 'model_request_sent' for e in history_events),
            'tool_count': sum(e.event_type == 'tool_called' for e in history_events),
            'worker_count': workers_count, 'seconds': round(time.monotonic()-started,3),
            'duplicate_receipt': bool(response.get('duplicate')),
        }
        report['cases'].append(result)
        (case_root / 'result.json').write_text(stable_json_dumps(result, indent=2)+'\n', encoding='utf-8')
        print(f"[canonical-conversation {len(report['cases'])}/{len(tasks)}] {task.task_id} passed={result['passed']} seconds={result['seconds']} answer={answer[:180]!r} error={error[:90]!r}", flush=True)
    report['passed'] = sum(bool(case['passed']) for case in report['cases'])
    report['failed'] = len(report['cases']) - report['passed']
    report['finished_at'] = utc_now_iso()
    (output_dir / 'summary.json').write_text(stable_json_dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
