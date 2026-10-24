#!/usr/bin/env python3
"""Bounded ARES example-data benchmark on resident Qwen; NOT human-labelled gold.

Every request is independently admitted and durably logged before submission.
Never retry a trial when any journal event exists, even on ambiguous network loss.
"""
import argparse,datetime as dt,hashlib,json,os,pathlib,random,time,urllib.request
from run_live import PORTS, LABELS,fetch,preflight

ROOT=pathlib.Path(__file__).resolve().parent
VERSION='external-ares-v1-20261025'

def jsonl(path,row):
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps({'at':dt.datetime.now(dt.timezone.utc).isoformat(),**row},sort_keys=True,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def plan(cases):
    items=[]
    rng=random.Random(20261025)
    for case in cases:
        combos=[(port,e) for port in PORTS for e in ('numeric','ordinal')]
        rng.shuffle(combos)
        for port,encoding in combos:
            items.append({'id':f"ares-v1:{case['item_id']}:{port}:{encoding}:s17",'item':case,'port':port,'encoding':encoding,'seed':17})
    assert len(items)==192 and len({x['id'] for x in items})==192
    return items

def trial(t,models,path,source_digest):
    c=t['item'];encoding=t['encoding'];model=models[t['port']]
    initial=preflight(t['port'])
    levels=', '.join(f'{i}={name}' for i,name in enumerate(LABELS))
    instruction=('Return an integer score between 0 and 8.' if encoding=='numeric'
                 else 'Return the exact ordered text level as score.')
    msgs=[{'role':'system','content':'Assess how relevant the quoted document text is as evidence for answering the exact current user question. A document may mention the topic but still not answer it. Output only the schema value.'},
          {'role':'user','content':f"ORIGINAL QUESTION: {c['query']}\nEXACT CANDIDATE DOCUMENT:\n{c['document']}\n"
           f"Rank by relevance to answering this question, not superficial word overlap. Ordered levels: {levels}. "
           f"0 means it does not help answer this question, 8 means essential direct answer evidence. {instruction}"}]
    prop={'type':'integer','minimum':0,'maximum':8} if encoding=='numeric' else {'type':'string','enum':list(LABELS)}
    schema={'type':'object','properties':{'score':prop},'required':['score'],'additionalProperties':False}
    body={'model':model,'messages':msgs,'temperature':0,'seed':17,'top_p':1,'max_tokens':45,'stream':False,
          'response_format':{'type':'json_schema','json_schema':{'name':'ares_relevance','strict':True,'schema':schema}},
          'chat_template_kwargs':{'enable_thinking':False}}
    meta={'trial_id':t['id'],'event':'request_submitted','dataset_sha256':source_digest,'case_id':c['item_id'],
          'source_row_index':c['source_row_index'],'source_label':c['label'],'source_label_provenance':'ARES public example, not verified human rated',
          'document_sha256':c['passage_sha256'],'model':model,'port':t['port'],'encoding':encoding,
          'seed':17,'admission':initial,'request':body,'version':VERSION}
    jsonl(path,meta)
    http=urllib.request.Request(f"http://10.8.0.7:{t['port']}/v1/chat/completions",data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(http,timeout=45) as response:r=json.load(response)
    duration=time.monotonic()-start
    raw=(r.get('choices') or [{}])[0]
    try:v=json.loads(raw.get('message',{}).get('content') or '')['score']
    except (KeyError,ValueError,TypeError):v=None
    sc=(v if type(v) is int and 0<=v<=8 else None) if encoding=='numeric' else (LABELS.index(v) if v in LABELS else None)
    binary=(sc>=5) if sc is not None else None
    jsonl(path,{'trial_id':t['id'],'event':'response','value':v,'ordinal_score':sc,'source_label':c['label'],
                'predicted_binary':binary,'source_label_concordant':binary==bool(c['label']) if binary is not None else None,
                'full_wall_seconds':duration,'usage':r.get('usage'),'finish_reason':raw.get('finish_reason'),'raw_message':raw})
    return sc

def main():
    p=argparse.ArgumentParser();p.add_argument('--journal',required=True,type=pathlib.Path);p.add_argument('--max-new',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.max_new<=12:raise SystemExit('max 12 bounded model calls per invocation')
    fp=ROOT/'ares_nq_reference_v1.json';cases=json.loads(fp.read_text())['items'];digest=hashlib.sha256(fp.read_bytes()).hexdigest()
    if len(cases)!=48:raise RuntimeError('unexpected dataset size')
    seen=set()
    if a.journal.exists():
        for row in a.journal.read_text().splitlines():
            x=json.loads(row)
            if x.get('dataset_sha256') and x['dataset_sha256']!=digest:raise RuntimeError('prior_journal_different_dataset')
            seen.add(x['trial_id'])
    models={port:fetch(port,'/v1/models')['data'][0]['id'] for port in PORTS}
    a.journal.parent.mkdir(parents=True,exist_ok=True)
    finished=0
    for t in plan(cases):
        if t['id'] in seen:continue
        if finished>=a.max_new:break
        try:
            score=trial(t,models,a.journal,digest)
            finished+=1
            print(json.dumps({'trial':t['id'],'score':score,'source_label':t['item']['label']}),flush=True)
        except Exception as exc:
            jsonl(a.journal,{'trial_id':t['id'],'event':'stopped','reason':type(exc).__name__+': '+str(exc)[:120],
                            'dataset_sha256':digest,'never_replay':True})
            print(json.dumps({'trial':t['id'],'stopped':type(exc).__name__+': '+str(exc)[:120]}),flush=True)
            break
    print('new_completed',finished,flush=True)
if __name__=='__main__':main()
