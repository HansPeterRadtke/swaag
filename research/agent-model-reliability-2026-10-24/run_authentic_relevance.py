#!/usr/bin/env python3
"""Preregistered, bounded pairwise Qwen rating of source-exact SWAAG work messages.

Source is 3 documented genuine SWAAG real-model engineering investigations.
Owner-goal binary annotations were made by this investigator and are NOT
independent human judgements. Never replay an attempted model call.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request
from run_live import PORTS, LABELS, fetch, preflight

ROOT=Path(__file__).resolve().parent
VERSION='2026-10-24-source-authentic-v1'

def utc_now():return dt.datetime.now(dt.timezone.utc).isoformat()

def plan(rows):
    pool=[]
    for goal in ('spoken_status','fix_planning','coordination','identity_answer'):
        for score in (8,0):
            labeled=[c for c in rows if c['goal_id']==goal and c['gold_score']==score]
            assert len(labeled)==8
            # Each role is represented with 4 required, 4 omitted original messages.
            pool.extend(labeled[:4])
    assert len(pool)==32 and len({x['case_id'] for x in pool})==32
    rng=random.Random(20261024)
    rng.shuffle(pool)
    for case in pool:
        configs=[(p,encoding) for p in PORTS for encoding in ('number','category')]
        # Rotate exposure order; no one model or encoding always benefits from warm cache.
        rng.shuffle(configs)
        for port,encoding in configs:
            yield {'trial_id':f"authentic-v1:{case['case_id']}:{port}:{encoding}:seed17",'case':case,
                'port':port,'encoding':encoding,'seed':17,'temperature':0.0}

def append(path,row):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(row,at_utc=utc_now()),ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())

def prompt(case,encoding):
    rubric=' '.join(f'{i}={name}' for i,name in enumerate(LABELS))
    instruction=('Return the 0-to-8 integer as score.' if encoding=='number' else 'Return the exact named category as score.')
    content=(f"TASK-SPECIFIC PURPOSE: {case['role']}\nORIGINAL OWNER GOAL: {case['owner_goal']}\n"
             f"REAL PROJECT AND ACTION CONTEXT: {case['operational_context']}\n"
             f"CANDIDATE EXACT REPORTED MESSAGE (source from a genuine SWAAG engineering study):\n{case['source_text']}\n"
             'Rate importance to satisfying this SPECIFIC goal, not general topic similarity. '
             'Use 0 for unnecessary material and 8 for an indispensable fact or constraint; intermediate levels for useful but nonessential material.\n'
             f"Ordered levels: {rubric}\n{instruction}")
    return [{'role':'system','content':'You are evaluating the relevance of a real engineering report message to the stated current role and owner task. Preserve the distinction between information needed to perform worker work and information useful to tell the user. Do not treat historical status as current. Answer only the strict JSON output.'},
            {'role':'user','content':content}]

def run_trial(t,path,models):
    case=t['case'];port=t['port'];encoding=t['encoding'];model=models[port]
    observed=preflight(port)
    msgs=prompt(case,encoding)
    field={'type':'integer','minimum':0,'maximum':8} if encoding=='number' else {'type':'string','enum':list(LABELS)}
    schema={'type':'object','properties':{'score':field},'required':['score'],'additionalProperties':False}
    body={'model':model,'messages':msgs,'temperature':t['temperature'],'top_p':1.0,'seed':t['seed'],
          'max_tokens':50,'stream':False,'response_format':{'type':'json_schema',
          'json_schema':{'name':'relevance_v1_'+encoding,'strict':True,'schema':schema}},
          'chat_template_kwargs':{'enable_thinking':False}}
    metadata={'trial_id':t['trial_id'],'event':'request_submitted','case_id':case['case_id'],
        'source_file':case['source'],'source_sha256':case['source_sha256'],'source_char_start':case['source_char_start'],
        'source_char_end':case['source_char_end'],'gold':case['gold_score'],'gold_origin':case['annotation_provenance'],
        'role':case['role'],'goal_id':case['goal_id'],'encoding':encoding,'port':port,'model':model,
        'admission':observed,'schema':schema,'request':body,'study_version':VERSION}
    append(path,metadata)  # durable before external request: never retry on uncertainty.
    http=urllib.request.Request(f'http://10.8.0.7:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(http,timeout=30) as response:
        result=json.load(response)
    elapsed=time.monotonic()-start
    raw=result.get('choices',[{}])[0]
    try:val=json.loads(raw.get('message',{}).get('content') or '{}')['score']
    except (ValueError,KeyError,TypeError):val=None
    if encoding=='category':score=LABELS.index(val) if val in LABELS else None
    else:score=val if type(val) is int and 0<=val<=8 else None
    append(path,{'trial_id':t['trial_id'],'event':'response','score':score,'raw_value':val,
                 'valid':score is not None,'elapsed_seconds':elapsed,'usage':result.get('usage'),
                 'raw_response':raw,'finish_reason':raw.get('finish_reason')})
    return score

def main():
    p=argparse.ArgumentParser();p.add_argument('--journal',type=Path,required=True);p.add_argument('--max-new',type=int,default=12);a=p.parse_args()
    if not 1<=a.max_new<=20:raise SystemExit('max 20 model calls per invocation')
    fixture=json.loads((ROOT/'authentic_relevance_v1.json').read_text())
    expected=hashlib.sha256((ROOT/'authentic_relevance_v1.json').read_bytes()).hexdigest()
    planned=list(plan(fixture['cases']))
    assert len(planned)==128 and len(set(t['trial_id'] for t in planned))==128
    seen=set()
    if a.journal.exists():
        for raw in a.journal.read_text().splitlines():
            entry=json.loads(raw)
            if entry.get('fixture_sha256') and entry['fixture_sha256']!=expected:raise SystemExit('fixture/version mismatch')
            seen.add(entry['trial_id'])
    models={port:fetch(port,'/v1/models')['data'][0]['id'] for port in PORTS}
    a.journal.parent.mkdir(parents=True,exist_ok=True)
    newly=0
    for t in planned:
        if t['trial_id'] in seen:continue
        if newly>=a.max_new:break
        try:
            # This journal's metadata identifies the frozen input and version.
            score=run_trial(t,a.journal,models)
            newly+=1
            print(json.dumps({'trial':t['trial_id'],'score':score,'gold':t['case']['gold_score'],'model':models[t['port']]}),flush=True)
        except Exception as exc:
            # Any previous request_submitted is permanently an uncertain effect.
            append(a.journal,{'trial_id':t['trial_id'],'event':'halt','reason':type(exc).__name__+': '+str(exc)[:120],
                              'fixture_sha256':expected,'not_retried':True})
            print(json.dumps({'trial':t['trial_id'],'halt':str(exc)[:120]}),flush=True)
            break
    print('new_completed_calls',newly,flush=True)
if __name__=='__main__': main()
