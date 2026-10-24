#!/usr/bin/env python3
"""Bounded production-friendly Qwen probe. Synthetic cases only; preserves raw judgments.

Do not run without a contemporaneous idle SWAAG supervisor and idle native slots.
No model restarts, model loading, new training, remote writes or retries on failure.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import socket
import time
import urllib.request

PORTS=(14833,14835)
SCHEMA_VERSION='2026-10-24-r1'
LABELS=('irrelevant','very_low','low','somewhat_low','medium','somewhat_high','high','very_high','critical')
ROOT=pathlib.Path(__file__).resolve().parent


def json_rpc(op,params=None):
    with socket.create_connection(('127.0.0.1',13401),timeout=2) as s:
        s.settimeout(4)
        s.sendall((json.dumps({'op':op,'params':params or {}})+'\n').encode())
        raw=s.makefile('rb').readline()
    if not raw or len(raw)>2000000: raise RuntimeError('unavailable_or_oversize_supervisor_reply')
    r=json.loads(raw)
    if not r.get('ok'): raise RuntimeError('supervisor_rpc_error')
    return r['result']


def fetch(port, path):
    return json.load(urllib.request.urlopen(f'http://10.8.0.7:{port}{path}',timeout=4))


def preflight(port):
    s=json_rpc('orchestration.supervision')['supervision']
    b=s.get('backends',{})
    if s.get('active_sessions'): raise RuntimeError('supervisor_sessions_active')
    if len(b)<2: raise RuntimeError('supervisor_missing_backend')
    now=dt.datetime.now(dt.timezone.utc)
    for val in b.values():
        if val.get('state')!='idle' or val.get('stale') or not val.get('monitor_running'):
            raise RuntimeError('supervisor_backend_not_verified_idle')
        observed=dt.datetime.fromisoformat(val['observed_at'].replace('Z','+00:00'))
        if not 0<= (now-observed).total_seconds()<5:raise RuntimeError('supervisor_observation_stale')
    workers=json_rpc('task.list').get('workers',[])
    if any(w.get('status') in {'queued','running','suspended','input_required'} for w in workers):
        raise RuntimeError('production_worker_pending')
    for p in PORTS:
        slots=fetch(p,'/slots')
        if not isinstance(slots,list) or not slots or any(x.get('is_processing',True) for x in slots):
            raise RuntimeError('llama_slot_busy_or_unknown')
    with open('/proc/meminfo') as f:
        mem=next(int(s.split()[1])*1024 for s in f if s.startswith('MemAvailable:'))
    disk=os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize
    if mem < 12*1024**3 or disk < 10*1024**3:raise RuntimeError('memory_or_disk_reserve')
    return {'supervisor_observed':{k:v.get('observed_at') for k,v in b.items()},'idle_slots':True,'workers_total':len(workers),'mem_available_bytes':mem,'disk_available_bytes':disk}


def schema(field,choices):
    property={'type':'string','enum':list(choices)} if choices else {'type':'integer','minimum':0,'maximum':8}
    return {'type':'object','properties':{field:property},'required':[field],'additionalProperties':False}


def tasks(rows):
    # Frozen four relevance corner cases; no posterior case replacement.
    for item in rows:
        if item['kind']!='relevance':continue
        for encoding in ('number','category'):
            for port in PORTS:
                for seed in (17,42):
                    yield {'id':f"relevance:{item['id']}:{encoding}:{port}:{seed}", 'kind':'relevance',
                           'case':item,'encoding':encoding,'port':port,'seed':seed,'stage':'single'}
    # First factual extraction / reasoning cases only; split vs fused kept distinct.
    for item in rows:
        if item['kind'] not in ('extraction','reasoning'): continue
        for variant in ('fused','decomposed'):
            for port in PORTS:
                for seed in (17,42):
                    yield {'id':f"{item['kind']}:{item['id']}:{variant}:{port}:{seed}", 'kind':item['kind'],
                           'case':item,'encoding':variant,'port':port,'seed':seed,'stage':'single'}


def send(port,model,messages,format_schema,field,seed):
    request={'model':model,'messages':messages,'temperature':0,'top_p':1,'seed':seed,
        'max_tokens':100,'stream':False, 'response_format':{'type':'json_schema',
        'json_schema':{'name':'agent_reliability','strict':True,'schema':format_schema}},
        'chat_template_kwargs':{'enable_thinking':False}}
    http=urllib.request.Request(f'http://10.8.0.7:{port}/v1/chat/completions', data=json.dumps(request).encode(),
                                headers={'Content-Type':'application/json'})
    start=time.monotonic(); result=json.load(urllib.request.urlopen(http,timeout=75)); elapsed=time.monotonic()-start
    output=result.get('choices',[{}])[0].get('message') or {}
    try: value=json.loads(output.get('content') or '{}')[field]
    except (ValueError,KeyError,TypeError): value=None
    return {'request':request,'elapsed_seconds':elapsed,'raw_message':output,
            'value':value,'usage':result.get('usage'),'finish_reason':result.get('choices',[{}])[0].get('finish_reason')}


def run_trial(t,models):
    c=t['case'];enc=t['encoding'];port=t['port'];seed=t['seed'];model=models[port]
    gate=preflight(port)
    if t['kind']=='relevance':
        choices=LABELS if enc=='category' else None
        labels=', '.join(f'{i}={name}' for i,name in enumerate(LABELS))
        instruction='Output the ordered textual category; '+labels if choices else 'Output an integer from 0 to 8, with 0 completely irrelevant and 8 critical.'
        prompt=(f"Judge relevance specifically for '{c['dimension']}' using only user request and candidate. "
                f"User request: {c['user_request']}\nCandidate: {c['candidate']}\n"+instruction)
        first=send(port,model,[{'role':'system','content':'Return only a valid JSON relevance judgement in the requested schema. Do not confuse user-facing, worker-quality and orchestration relevance.'},{'role':'user','content':prompt}], schema('score',choices),'score',seed)
        value=LABELS.index(first['value']) if enc=='category' and first['value'] in LABELS else first['value']
        valid=(type(value) is int and 0<=value<=8)
        correct=valid and abs(value-c['gold'])<=c.get('tolerance',0)
        return {'trial':t['id'],'kind':t['kind'],'input':c,'model':model,'port':port,'seed':seed,'encoding':enc,'admission':gate,'steps':[first], 'canonical_answer':value, 'valid':valid,'correct':correct,'total_elapsed_seconds':first['elapsed_seconds']}
    choices=('yes','no','unknown','10')
    if enc=='fused':
        question=f"Source: {c['source']}\nQuestion: {c['question']}\nAnswer using only the source. If not entailed, choose unknown."
        first=send(port,model,[{'role':'system','content':'Answer an evidence-based factual question. Quote no unsupported facts.'},{'role':'user','content':question}], schema('answer',choices),'answer',seed)
        steps=[first]
    else:
        # The second stage cannot inspect the full original source; this is the real decomposition risk.
        extract=send(port,model,[{'role':'system','content':'Extract only statements that are asserted as facts by the source. Preserve negation and uncertainty; distinguish quotations from established facts.'},{'role':'user','content':f"Source: {c['source']}\nReturn a concise facts string."}], {'type':'object','properties':{'facts':{'type':'string'}},'required':['facts'],'additionalProperties':False},'facts',seed)
        if not isinstance(extract['value'],str):
            return {'trial':t['id'],'kind':t['kind'],'input':c,'model':model,'port':port,'seed':seed,'encoding':enc,'admission':gate,'steps':[extract], 'canonical_answer':None,'valid':False,'correct':False,'total_elapsed_seconds':extract['elapsed_seconds']}
        # Cannot start a second call if a production worker arrives between stages.
        try:
            mid_gate=preflight(port)
        except Exception as exc:
            return {'trial':t['id'],'kind':t['kind'],'input':c,'model':model,'port':port,'seed':seed,'encoding':enc,'admission':gate,'steps':[extract], 'canonical_answer':None,'valid':False,'correct':False,'incomplete':True,'blocked_after_first_stage':type(exc).__name__ + ': ' + str(exc), 'total_elapsed_seconds':extract['elapsed_seconds']}
        second=send(port,model,[{'role':'system','content':'Answer only from the given extracted facts; absent information is unknown.'},{'role':'user','content':f"Extracted facts: {extract['value']}\nQuestion: {c['question']}"}],schema('answer',choices),'answer',seed)
        steps=[extract,second]
    answer=steps[-1]['value'];valid=answer in choices
    return {'trial':t['id'],'kind':t['kind'],'input':c,'model':model,'port':port,'seed':seed,'encoding':enc,'admission':gate,'steps':steps,'canonical_answer':answer,'valid':valid,'correct':valid and answer==c['gold'],'total_elapsed_seconds':sum(s['elapsed_seconds'] for s in steps)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=8);parser.add_argument('--out',type=pathlib.Path,required=True);args=parser.parse_args()
    if not 1<=args.limit<=128:raise SystemExit('bad bounded call count')
    rows=json.loads((ROOT/'fixtures.json').read_text());models={p:fetch(p,'/v1/models')['data'][0]['id'] for p in PORTS}
    seen=set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                a=json.loads(line)
                seen.add(a['trial'])
            except Exception: raise SystemExit('invalid_previous_trial_journal')
    submitted=0
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for t in tasks(rows):
        if t['id'] in seen:continue
        if submitted>=args.limit:break
        try:record=run_trial(t,models)
        except Exception as e:
            # No retry on uncertain effects. Preserve exact trial identity and stop.
            record={'trial':t['id'],'failed_or_interrupted':type(e).__name__, 'reason':str(e)[:120], 'timestamp':dt.datetime.now(dt.timezone.utc).isoformat()}
            with args.out.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({'trial':t['id'],'stopped':record['reason']}),flush=True)
            break
        record['timestamp']=dt.datetime.now(dt.timezone.utc).isoformat();record['study_revision']=SCHEMA_VERSION
        with args.out.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        submitted+=1;print(json.dumps({'trial':t['id'],'valid':record['valid'],'correct':record['correct'],'seconds':round(record['total_elapsed_seconds'],2),'model':record['model']}),flush=True)
    print('new_trial_count',submitted,flush=True)

if __name__=='__main__':main()
