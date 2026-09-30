#!/usr/bin/env python3
"""Real-model question resolution with a second deliberately unanswered question."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import utc_now_iso


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:14829')
    args=parser.parse_args()
    output=Path(args.output).resolve()
    root=output.parent/(output.stem+'-state')
    if root.exists():
        raise ValueError('Choose a fresh output path; live evidence is never overwritten')
    root.mkdir(parents=True)
    config=load_config(env={
        'SWAAG__SESSIONS__ROOT':str(root/'sessions'),
        'SWAAG__MODEL__BASE_URL':args.base_url,
        'SWAAG__MODEL__CACHE_ENABLED':'false',
        'SWAAG__TOOLS__ENABLED':'["questions"]',
        'SWAAG__TOOLS__STAGED_DISCOVERY':'false',
        'SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS':'true',
        'SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS':'false',
        'SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED':'true',
    })
    for key in ('timeout_seconds','simple_timeout_seconds','structured_timeout_seconds',
                'verification_timeout_seconds','benchmark_timeout_seconds'):
        setattr(config.model,key,1800)
    config.runtime.max_total_actions=8
    config.runtime.tool_call_budget=4
    runtime=AgentRuntime(config)
    state=runtime.create_or_load_session()
    questions=[]
    for text,assumption in (('Which destination should the draft use?','Keep the draft local'),
                            ('Which accent color should the draft use?','Keep the default color')):
        event=runtime.history.record_event(state,'agent_question',{
            'action_index':0,'question':text,'criticality':'optional',
            'reason':'This preference was not supplied','assumption_if_unanswered':assumption})
        questions.append(event)
    answer=runtime.history.record_event(state,'message_added',{'message':asdict(Message(
        role='user',content='Use Oslo as the destination. I have not decided the accent color.',created_at=utc_now_iso()))})
    before=list(state.open_questions)
    started=time.monotonic()
    error=None;result=None
    try:
        result=runtime.run_turn_in_session(state,
            f'The recorded user answer at history event {answer.sequence} says to use Oslo as the destination. '
            'Use the questions capability to resolve only the destination question as answered by the user, '
            'citing that exact event. The accent-color question remains unresolved with its existing assumption. '
            'Do not add questions or change any files. After observing the successful resolution, '
            'return exactly QUESTION-RESOLVED-OK.')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    rebuilt=runtime.history.rebuild_from_history(state.session_id,prefer_checkpoint=False)
    events=runtime.history.read_history(state.session_id)
    resolutions=[event for event in events if event.event_type=='agent_question_resolved']
    checks={
        'no_runtime_error':error is None,
        'exact_final_response':result is not None and result.assistant_text.strip()=='QUESTION-RESOLVED-OK',
        'only_target_resolved':len(resolutions)==1 and resolutions[0].payload['question_id']==questions[0].id,
        'user_answer_provenance':len(resolutions)==1 and resolutions[0].payload['resolution']=='answered_by_user'
            and answer.sequence in resolutions[0].payload['evidence_sequences']
            and 'oslo' in resolutions[0].payload['answer'].casefold(),
        'unanswered_question_preserved':len(rebuilt.open_questions)==1
            and rebuilt.open_questions[0]['question_id']==questions[1].id
            and rebuilt.open_questions[0]['assumption_if_unanswered']=='Keep the default color',
        'independent_completion_observed':any(event.event_type=='completion_evaluated' for event in events),
        'exact_replay_matches_active_state':rebuilt.open_questions==state.open_questions,
    }
    report={'benchmark':'question_lifecycle_live_v1','generated_at':utc_now_iso(),
            'session_id':state.session_id,'elapsed_seconds':time.monotonic()-started,
            'checks':checks,'passed':all(checks.values()),'error':error,
            'before':before,'after':rebuilt.open_questions,
            'resolutions':[event.payload for event in resolutions],
            'assistant_text':result.assistant_text if result else None}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':checks,'elapsed_seconds':report['elapsed_seconds']}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
