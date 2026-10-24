#!/usr/bin/env python3
"""Finite, single-request-at-a-time, fail-closed Qwen causal-entailment study.

Trial identity includes condition, seed and prompt form. A durable journal records
EVERY stage before a request is submitted and immediately after its response.
No retry/replay of any previously journaled trial, including interrupted ones.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time
import urllib.request

from logic_oracle import load
from run_live import PORTS, fetch, preflight

ROOT=Path(__file__).resolve().parent
VERSION='2026-10-24-formal-holdout-v2-predeclared'
CHOICES=('yes','no','unknown')
SCHEMA={'type':'object','properties':{'answer':{'type':'string','enum':list(CHOICES)}},'required':['answer'],'additionalProperties':False}
FACT_SCHEMA={'type':'object','properties':{'facts':{'type':'string'}},'required':['facts'],'additionalProperties':False}


def plan(cases):
    """Frozen ordering: paired, cross-model first; seed/prompt stability afterward."""
    for case in cases:
        for port in PORTS:
            for method in ('fused','split'):
                yield {'trial':f"r3:{case['id']}:{port}:17:base:{method}",'case':case,'port':port,'seed':17,'temperature':0.0,'style':'base','method':method}
    for case in cases:
        for port in PORTS:
            yield {'trial':f"r3:{case['id']}:{port}:42:base:fused",'case':case,'port':port,'seed':42,'temperature':0.25,'style':'base','method':'fused'}
    for case in cases:
        for port in PORTS:
            yield {'trial':f"r3:{case['id']}:{port}:17:paraphrase:fused",'case':case,'port':port,'seed':17,'temperature':0.25,'style':'paraphrase','method':'fused'}


def prompts(case,style,method,extracted=None):
    question=case['question']
    if method=='fused':
        if style=='paraphrase':
            return [{'role':'system','content':'Choose yes if the documented facts require yes, no if they require no, and unknown if either can be true. A quoted claim is not a verified observation. Respond only in the JSON schema.'},
                {'role':'user','content':f"What can be established without unsupported assumptions?\nDocument: {case['source']}\nQuery: {question}"}]
        return [{'role':'system','content':'Reason only from reliable source statements. Answer yes if entailed, no if disproved, otherwise unknown. Do not promote quotes to facts. Respond using the JSON schema.'},
            {'role':'user','content':f"Source: {case['source']}\nQuestion: {question}"}]
    if method=='facts':
        return [{'role':'system','content':'Extract all verifiable rules and facts useful for the question. Retain whether a ledger is complete, source authority, quotation scope, negations and unknowns. Do not infer new observations. JSON only.'},
            {'role':'user','content':f"Source: {case['source']}\nQuestion to resolve later: {question}\nWrite complete relevant facts for another model without looking at the source."}]
    if method=='from_facts':
        return [{'role':'system','content':'Determine yes/no/unknown only from the provided fact extract. If essential information was omitted or ambiguous, choose unknown. JSON only.'},
            {'role':'user','content':f"Extracted facts: {extracted}\nQuestion: {question}"}]
    raise ValueError('bad prompt mode')


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def append(path,row):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(row,utc_time=now()),ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())


def generate(p,stage,messages,schema):
    port=p['port']
    model=p['model']
    req={'model':model,'messages':messages,'temperature':p['temperature'],'top_p':1.0,
         'seed':p['seed'],'max_tokens':128,'stream':False,
         'response_format':{'type':'json_schema','json_schema':{'name':'round3_'+stage,'strict':True,'schema':schema}},
         'chat_template_kwargs':{'enable_thinking':False}}
    request=urllib.request.Request(f'http://10.8.0.7:{port}/v1/chat/completions',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(request,timeout=65) as response:
        body=json.load(response)
    delta=time.monotonic()-start
    raw=body.get('choices',[{}])[0]
    try:parsed=json.loads(raw.get('message',{}).get('content',''))
    except (TypeError,ValueError):parsed={}
    value=parsed.get('answer' if stage!='facts' else 'facts')
    return {'request':req,'raw_response':raw,'usage':body.get('usage'),
            'elapsed_seconds':delta,'value':value,'valid':value in CHOICES if stage!='facts' else isinstance(value,str)}


def run_one(p,path):
    trial=p['trial'];case=p['case'];port=p['port'];pid=trial
    # Both model slots and the durable worker roster must be idle before ANY new inference.
    initial=preflight(port)
    append(path,{'trial':trial,'event':'begin','case':case,'method':p['method'],'seed':p['seed'],
                 'temperature':p['temperature'],'style':p['style'],'port':port,'model':p['model'],
                 'ground_truth':case['gold'],'source':'formal_holdout_v2.json','admission':initial,'version':VERSION})
    stages=[('verdict',prompts(case,p['style'],'fused'),SCHEMA)] if p['method']=='fused' else [('facts',prompts(case,p['style'],'facts'),FACT_SCHEMA)]
    observations=[]
    for stage,messages,output_schema in stages:
        append(path,{'trial':trial,'event':'request_start','stage':stage,'messages':messages,'schema':output_schema,
                     'status':'UNKNOWN_IF_INTERRUPTED_DO_NOT_RETRY'})
        result=generate(p,stage,messages,output_schema)
        append(path,{'trial':trial,'event':'response','stage':stage,'result':result})
        observations.append(result)
    if p['method']=='split':
        extracted=observations[0]['value']
        if not isinstance(extracted,str):
            append(path,{'trial':trial,'event':'completed','status':'invalid_extract','correct':False,'valid':False})
            return 'invalid_extract'
        # A new check is required because a real production user may arrive during stage one.
        preflight_next=preflight(port)
        append(path,{'trial':trial,'event':'admission_second_stage','data':preflight_next})
        messages=prompts(case,p['style'],'from_facts',extracted=extracted)
        append(path,{'trial':trial,'event':'request_start','stage':'verdict','messages':messages,'schema':SCHEMA,
                     'status':'UNKNOWN_IF_INTERRUPTED_DO_NOT_RETRY'})
        result=generate(p,'verdict',messages,SCHEMA)
        append(path,{'trial':trial,'event':'response','stage':'verdict','result':result})
        observations.append(result)
    verdict=observations[-1]['value']
    complete={'trial':trial,'event':'completed','status':'scored','gold':case['gold'],'answer':verdict,
              'valid':verdict in CHOICES, 'correct':verdict==case['gold'],
              'total_elapsed_seconds':sum(x['elapsed_seconds'] for x in observations),
              'total_tokens':sum((x['usage'] or {}).get('total_tokens',0) for x in observations),
              'stage_count':len(observations)}
    append(path,complete)
    return 'correct' if complete['correct'] else 'wrong'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--max-trials',type=int,default=6)
    a=parser.parse_args()
    if not 1<=a.max_trials<=12:raise SystemExit('max 12 trials per invocation')
    cases=load(ROOT/'formal_holdout_v2.json')
    planned=list(plan(cases))
    ids=[p['trial'] for p in planned]
    assert len(ids)==len(set(ids)) and len(planned)==48
    # The journal is append-only, and *any* event means trial must never replay.
    seen=set()
    if a.journal.exists():
        for raw in a.journal.read_text().splitlines():
            row=json.loads(raw); seen.add(row['trial'])
    a.journal.parent.mkdir(parents=True,exist_ok=True)
    models={port:fetch(port,'/v1/models')['data'][0]['id'] for port in PORTS}
    completed=0
    for t in planned:
        if t['trial'] in seen:continue
        if completed>=a.max_trials:break
        t['model']=models[t['port']]
        try:
            result=run_one(t,a.journal)
            completed+=1
            print(json.dumps({'trial':t['trial'],'status':result}),flush=True)
        except Exception as exc:
            # Any request_start means the effect is uncertain; never repeat.
            append(a.journal,{'trial':t['trial'],'event':'stopped','exception_class':type(exc).__name__,
                              'reason':str(exc)[:180], 'no_retry':True})
            print(json.dumps({'trial':t['trial'],'stopped':type(exc).__name__+': '+str(exc)[:90]}),flush=True)
            break
    print('completed_new',completed,flush=True)

if __name__=='__main__':main()
