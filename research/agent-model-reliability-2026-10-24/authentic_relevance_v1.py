#!/usr/bin/env python3
"""Source-exact role-specific relevance contrasts based on genuine SWAAG investigation reports.

Researcher-defined relevance anchors, not independent multi-rater human labels.
Original documents are not modified; paragraphs are copied byte-for-byte.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
DOCS={
 'delegation':'docs/dual-role-foreground-latency-2026-10-18.md',
 'identity':'docs/foreground-identity-routing-2026-10-19.md',
 'first_reply':'docs/fast-initial-user-facing-2026-10-20.md',
}

# The goals derive from actual completed operations and owner requests recorded in the reports.
# Gold=8 means essential to this exact task; gold=0 means extraneous to it.
# This is a contract-derived developer annotation, NOT a certified human relevance label.
GOALS={
 'spoken_status':{
  'role':'user_answer',
  'owner_goal':'Explain plainly whether the Qwen-Coder foreground and Qwen3.8 worker handoff actually works and whether it is responsive enough for spoken use. State the verified result, current-status risk and important incompleteness; omit programmer-only internals.',
  'operational_context':'The owner asked for a working Android voice agent, correct background workers, truthful progress and fast spoken answers. The recorded study is a real isolated SWAAG test, not a physical Android end-to-end acceptance.',
 },
 'fix_planning':{
  'role':'worker_quality',
  'owner_goal':'Investigate and correct the actual Coder plan.apply invalid argument and repeated wrong worker routing attempts without bypassing schema authority or claiming an action took effect. Require exact source contracts and negative evidence.',
  'operational_context':'The SWAAG foreground uses Qwen3-Coder; Qwen3.8 is the default worker. Model route identifiers are not necessarily permitted worker route keys. Keep evidence and avoid mutating live conversations or model servers.',
 },
 'coordination':{
  'role':'orchestration',
  'owner_goal':'Determine what orchestration and GPU scheduling must preserve when a foreground request overlaps Qwen3.8 workers, especially current worker lifecycle, durable receipts, and whether it can actually make a truthful first reply.',
  'operational_context':'One initial capable communication call must answer promptly and hand off unchanged user intent; SWAAG worker and foreground backends share Thor GPU, and existing work cannot be blindly preempted.',
 },
 'identity_answer':{
  'role':'user_answer',
  'owner_goal':'Answer the user’s exact question: who are you, and which configured model answers the user-facing message? Do not confuse the background worker model with the voice conversation model or invent phone playback success.',
  'operational_context':'Known SWAAG identity/route errors were reproduced in real-model tests. The task is a concise factual identity reply, not a software patch.',
 }
}

# Stable paragraph-unique substrings, each present verbatim in a recorded engineering report.
CANDIDATES={
 'd_scope':('delegation','After the user\'s model-role reassignment,'),
 'd_validator':('delegation','The orchestration-control tool explicitly instructs'),
 'd_checker':('delegation','The existing independent foreground completion checker also receives'),
 'd_results':('delegation','The source change was tested using the same isolated real-model'),
 'd_calls':('delegation','The isolated test recorded that the orchestrator produced multiple'),
 'd_tests':('delegation','Six focused deterministic tests in'),
 'd_remaining':('delegation','The recorded final answer can become outdated'),
 'd_status':('delegation','An actual fresh isolated **Coder foreground'),
 'd_audit':('delegation','The previously failing global-snapshot unit test was investigated'),
 'd_priority':('delegation','The existing active voice source'),
 'i_fail':('identity','The user-facing SWAAG/Coder model is expected'),
 'i_benchmark':('identity','**Important benchmark correction:**'),
 'i_snapshot':('identity','`CommunicationService._orchestrator_runtime_snapshot()` now includes'),
 'i_unknown':('identity','The independent checker treats installed role metadata'),
 'i_result_identity':('identity','- With independently grounded role evidence'),
 'i_playback':('identity','- A question asking whether Android audio playback'),
 'i_exact_words':('identity','- A five-word identity question was answered'),
 'i_model_route':('identity','- After adding live configured-model routing disclosure'),
 'i_route_failure':('identity','The first staged-discovery *cross-role* live trial'),
 'i_route_repair':('identity','The repaired snapshot now independently exposes'),
 'f_slow':('first_reply','The existing production foreground was still slow'),
 'f_fast':('first_reply','`communication.fast_initial_reply` is an'),
 'f_handoff':('first_reply','An eligible self-contained conversation answer'),
 'f_prompts':('first_reply','Prompt-instruction stores, explicit session instructions'),
 'f_live_identity':('first_reply','- A live canonical isolated orchestration identity request'),
 'f_tool_task':('first_reply','- A live actual calculator task emitted'),
 'f_calibration':('first_reply','- An earlier four-case two-field fast classifier'),
 'f_android':('first_reply','- The existing dedicated voice-transport'),
 'f_limits':('first_reply','A quick semantic `answer` is not independently LLM-reviewed.'),
}

# Positive vs negative flags strictly limited to extreme relevance where a clear
# user request or tool contract supplies an anchor. Ambiguous cases excluded.
CONTRASTS={
 'spoken_status':{
  'critical':['d_scope','d_results','d_remaining','d_status','f_slow','f_live_identity','f_tool_task','f_android'],
  'omit':['d_validator','d_tests','d_audit','i_exact_words','i_route_repair','f_prompts','i_benchmark','d_checker'],
 },
 'fix_planning':{
  'critical':['d_scope','d_validator','d_checker','d_calls','d_audit','i_route_failure','i_route_repair','d_priority'],
  'omit':['i_exact_words','i_playback','f_android','f_live_identity','f_tool_task','i_result_identity','f_slow','f_calibration'],
 },
 'coordination':{
  'critical':['d_scope','d_calls','d_remaining','d_status','d_priority','f_handoff','f_fast','f_limits'],
  'omit':['i_exact_words','d_tests','d_audit','i_benchmark','f_live_identity','f_android','i_playback','f_calibration'],
 },
 'identity_answer':{
  'critical':['i_fail','i_snapshot','i_result_identity','i_model_route','i_unknown','f_live_identity','i_playback','d_scope'],
  'omit':['d_validator','d_audit','d_tests','d_priority','d_checker','f_tool_task','i_route_repair','f_prompts'],
 }
}

def build():
    docs={}
    for k,filename in DOCS.items():
        content=(ROOT/filename).read_text(encoding='utf8')
        docs[k]={'file':filename,'sha256':hashlib.sha256(content.encode()).hexdigest(),
                 'paragraphs':[]}
        for match in re.finditer(r'[^\n]+(?:\n[^\n]+)*',content):
            # Preserve original offsets while separating individual Markdown list items.
            # Otherwise one line about a five-word reply leaks entire neighboring
            # passages about model routing and the benchmark gold becomes meaningless.
            block=match.group()
            parts=list(re.finditer(r'[^\n]+(?:\n(?!- )[^\n]+)*',block))
            for part in parts:
                text=part.group();start=match.start()+part.start();end=match.start()+part.end()
                if text.strip().startswith('#'):continue
                assert content[start:end]==text
                docs[k]['paragraphs'].append({'text':text,'char_start':start,'char_end':end})
    excerpts={}
    for item,(source,marker) in CANDIDATES.items():
        found=[p for p in docs[source]['paragraphs'] if marker in p['text']]
        if len(found)!=1:raise ValueError('missing_or_ambiguous_source:'+item+':'+str(len(found)))
        assert len(found[0]['text'])>=50
        excerpts[item]={'source':docs[source]['file'],'source_sha256':docs[source]['sha256'],
                        'source_char_start':found[0]['char_start'],'source_char_end':found[0]['char_end'],
                        'source_text':found[0]['text']}
    cases=[]
    for goal in GOALS:
        labels=CONTRASTS[goal]
        if set(labels['critical']) & set(labels['omit']):raise ValueError('conflicting label')
        for kind,score in [('critical',8),('omit',0)]:
            for candidate in labels[kind]:
                cases.append(dict(case_id=goal+':'+candidate,goal_id=goal,
                                  **GOALS[goal],gold_score=score,
                                  researcher_label='critical_required' if score==8 else 'omit_from_this_purpose',
                                  annotation_provenance='source/owner contract interpreted by one researcher; no independent adjudicator',
                                  **excerpts[candidate]))
    return cases

if __name__=='__main__':
    cases=build()
    dest=Path(__file__).parent/'authentic_relevance_v1.json'
    if dest.exists():raise SystemExit('existing frozen fixture; refusing to overwrite')
    dest.write_text(json.dumps({'version':'2026-10-24-v1','source_description':'Authentic unchanged SWAAG real-model investigation report paragraphs; developer labels not human-validated',
                                'cases':cases},ensure_ascii=False,indent=2)+'\n')
    print('cases',len(cases),'source_documents',len(DOCS),'bytes',dest.stat().st_size,'max_source_characters',max(len(x['source_text']) for x in cases))
