"""Two-process acceptance of the canonical conversational durable-history route.

Execute first/second in SEPARATE OS Python processes. Never repeat a previously
claimed uncertain operation. The caller chooses a fresh private output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import stable_json_dumps

FIRST_PROMPT = 'What is the project codename? Answer with only the codename.'
CORRECTION = 'Correction: replace the project codename Rook with Cedar.'
LATEST_PROMPT = 'What is the latest project codename? Answer only with the codename.'


def run_phase(*, output_dir: Path, config_paths: Sequence[Path], phase: str) -> dict:
    if phase not in {'first', 'second'}:
        raise ValueError('phase must be first or second')
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / 'workspace'
    workspace.mkdir(exist_ok=True)
    config = load_config(config_paths=list(config_paths), env={})
    config.sessions.root = root / 'sessions'
    config.tools.read_roots = [workspace]
    config.tools.enabled = []
    config.tools.allow_stateful_tools = False
    config.tools.allow_side_effect_tools = False
    config.communication.enabled = True
    config.communication.enabled_tools = []
    config.communication.model_routes = {}
    config.communication.fast_initial_reply = True
    config.model.cache_enabled = False
    service = CommunicationService.from_runtime(AgentRuntime(config))

    def history():
        state = service._load_orchestrator_state()
        return service.orchestrator_runtime.history.read_history(state.session_id)

    def operation(text: str, name: str):
        return service.orchestrator_message(text, operation_id=f'canonical-conversation-restart-{name}')

    def answer(reply):
        return str(reply.get('answer') or '')

    try:
        if phase == 'first':
            if (root / 'first.json').exists():
                raise RuntimeError('first phase already recorded; never replay that operation')
            if service.orchestrator_operations.get('canonical-conversation-restart-initial') is not None:
                raise RuntimeError('first operation already exists; inspect its receipt, do not repeat')
            state = service._load_orchestrator_state()
            if any(e.event_type == 'message_added' for e in history()):
                raise RuntimeError('unexpected earlier user conversation in this supposedly fresh session')
            for idx, (role, content) in enumerate((
                ('user', 'In this isolated project, the current codename is Rook.'),
                ('assistant', 'Recorded: Rook.'),
            ), 1):
                service.orchestrator_runtime._record_message(
                    state, Message(role=role, content=content, created_at=f'2026-01-01T00:0{idx}:00+00:00'))
            reply = operation(FIRST_PROMPT, 'initial')
            current = history()
            if answer(reply).strip().casefold().rstrip('.') != 'rook':
                raise AssertionError('initial answer did not preserve the exact seeded fact')
            if any(e.event_type == 'tool_called' for e in current):
                raise AssertionError('unexpected tool in chat-only first phase')
            result = {
                'phase': phase, 'answer': answer(reply), 'number_events': len(current),
                'event_last_hash': current[-1].hash,
                'model_calls': sum(e.event_type == 'model_request_sent' for e in current),
            }
        else:
            first_path = root / 'first.json'
            if not first_path.exists():
                raise RuntimeError('first phase receipt missing; run first in a separate process')
            if (root / 'second.json').exists():
                raise RuntimeError('second phase already recorded; never replay side-effect identity')
            first = json.loads(first_path.read_text(encoding='utf-8'))
            previous = history()
            if len(previous) != first['number_events'] or previous[-1].hash != first['event_last_hash']:
                raise AssertionError('durable history changed across restart before follow-up')
            duplicate = operation(FIRST_PROMPT, 'initial')
            if duplicate.get('duplicate') is not True or answer(duplicate) != first['answer']:
                raise AssertionError('idempotent previous-operation reply could not be recovered')
            current = history()
            if [(e.sequence, e.hash) for e in current] != [(e.sequence, e.hash) for e in previous]:
                raise AssertionError('duplicate operation unexpectedly altered durable history')
            for key in ('correction', 'latest'):
                existing = service.orchestrator_operations.get('canonical-conversation-restart-' + key)
                if existing is not None:
                    raise RuntimeError(key + ' may have been attempted; do not replay an uncertain operation')
            corrected = operation(CORRECTION, 'correction')
            # This step may be intentionally refused/uncertain; never turn it
            # into automatic retries, including after a disconnect/timeout.
            latest = operation(LATEST_PROMPT, 'latest')
            final = history()
            if answer(latest).strip().casefold().rstrip('.') != 'cedar':
                raise AssertionError('latest answer did not use the new correction')
            if any(e.event_type == 'tool_called' for e in final):
                raise AssertionError('chat-only correction executed external tools')
            if service.workers.store.list(include_archived=False):
                raise AssertionError('chat-only revision created unexpected workers')
            result = {
                'phase': phase, 'duplicate_recovered_without_new_events': True,
                'correction_answer': answer(corrected), 'latest_answer': answer(latest),
                'events_before': len(previous), 'events_after': len(final),
                'model_calls_increment': sum(e.event_type == 'model_request_sent' for e in final)
                    - first['model_calls'],
            }
        (root / f'{phase}.json').write_text(stable_json_dumps(result, indent=2) + '\n', encoding='utf-8')
        return result
    finally:
        service.workers.shutdown(wait=False)
        for manager in service.worker_model_managers.values():
            manager.shutdown(wait=False)
        service.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, action='append', required=True)
    parser.add_argument('--phase', choices=['first', 'second'], required=True)
    args = parser.parse_args(argv)
    result = run_phase(output_dir=args.output, config_paths=args.config, phase=args.phase)
    print(stable_json_dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
