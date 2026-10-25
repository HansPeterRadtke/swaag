#!/usr/bin/env python3
"""Run bounded resident-Qwen scientific-claim judgments on expert SciFact dev.

Source claims/abstracts stay in private scratch, not in the public Git branch.
Exact request/raw output is retained in an append-only private JSONL journal.
Every stage requires fresh verified idleness; uncertain trials never repeat.
"""
import argparse,datetime as dt,hashlib,json,os,pathlib,random,time,urllib.request
from run_live import PORTS,fetch,preflight

HERE=pathlib.Path(__file__).resolve().parent
VERSION='2026-10-25-expert-scifact-v1'
LABELS=('SUPPORT','CONTRADICT','NOT_ENOUGH_INFO')

def utc_now():return dt.datetime.now(dt.timezone.utc).isoformat()

def append(path,row):
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps({'time_utc':utc_now(),**row},ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def validate(manifest,private):
    assert manifest['study']==private['study']== 'scifact-expert-claims-v1'
    assert len(manifest['cases'])==len(private['cases'])==48
    assert manifest['upstream_sha256']==private['original_archive_sha256']
    for a,b in zip(manifest['cases'],private['cases']):
        for src,dest in (('case_id','case_id'),('gold','gold'),('source_claim_id','source_claim_id'),
                         ('source_doc_id','source_doc_id'),('abstract_sha256','abstract_sha256'),('claim_sha256','claim_sha256')):
            if a[src]!=b[dest]:raise ValueError('manifest_source_mismatch')
        if a['rationale_indices']!=b['rationale_indices']:raise ValueError('expert_indices_mismatch')
        if hashlib.sha256(b['claim'].encode()).hexdigest()!=a['claim_sha256']:raise ValueError('claim_sha_mismatch')
        actual=b['title'].strip()+'\n'+'\n'.join(b['sentences'])
        if hashlib.sha256(actual.encode()).hexdigest()!=a['abstract_sha256']:raise ValueError('abstract_sha_mismatch')
    return private['cases']

def plan(cases):
    rng=random.Random(20261025)
    out=[]
    for c in cases:
        configs=[(p,condition) for p in PORTS for condition in ('direct','expert_rationale','select_then_answer','alternate_direct')]
        # Prioritize at least one direct and a curated-rationale condition per case.
        configs.sort(key=lambda x:({'direct':0,'expert_rationale':1,'select_then_answer':2,'alternate_direct':3}[x[1]],x[0]))
        # Alternate two available models on each condition, with reproducible order.
        for cond in ('direct','expert_rationale','select_then_answer','alternate_direct'):
            same=[x for x in configs if x[1]==cond]
            rng.shuffle(same)
            for port,_ in same:
                out.append({'trial_id':f"scifact-v1:{c['case_id']}:{port}:{cond}",
                            'case':c,'port':port,'condition':cond,'seed':42 if cond=='alternate_direct' else 17,
                            'temperature':0.25 if cond=='alternate_direct' else 0.0})
    if len(out)!=48*8 or len({x['trial_id'] for x in out})!=48*8:raise AssertionError('bad_selection_matrix')
    return out

def source_text(case,indices):
    return '\n'.join(f"[{i}] {case['sentences'][i]}" for i in indices)

def input_messages(case,condition,selected=None):
    if condition=='select':
        return [{'role':'system','content':'Choose up to four most probative *existing abstract sentence indices* for whether the scientific claim is supported or contradicted. Preserve counterevidence and explicit limitations. Answer JSON only.'},
                {'role':'user','content':f"CLAIM: {case['claim']}\nSCIENTIFIC ABSTRACT TITLE: {case['title']}\n"
                 f"SOURCE ABSTRACT SENTENCES:\n{source_text(case,range(len(case['sentences'])))}\nChoose evidence sentence identifiers for another independent inference stage."}]
    if condition=='expert_rationale':ids=case['rationale_indices']
    elif condition=='select_then_answer':ids=selected
    else:ids=range(len(case['sentences']))
    if ids is None:raise ValueError('missing_selection')
    if condition=='alternate_direct':
        system='Independently evaluate a scientific statement against the cited experimental results. Look for direct contradiction, negation, comparison direction, magnitude, and study populations. If not established, say NOT_ENOUGH_INFO. JSON only.'
        question=f"What is the evidence-grounded relation between claim and the original paper?\nCLAIM: {case['claim']}\nARTICLE: {case['title']}\nORIGINAL SENTENCES:\n{source_text(case,ids)}"
    else:
        system='Interpret the claim using only the supplied scientific abstract evidence. Output SUPPORT if the text entails the claim, CONTRADICT if it entails an incompatible proposition, or NOT_ENOUGH_INFO if neither is established. Do not answer from prior background knowledge. JSON only.'
        question=f"SCIENTIFIC CLAIM: {case['claim']}\nPAPER TITLE: {case['title']}\nORIGINAL ABSTRACT SENTENCES:\n{source_text(case,ids)}\nChoose the correct evidence relation."
    return [{'role':'system','content':system},{'role':'user','content':question}]

def schema_for(stage,case):
    if stage=='select':
        return {'type':'object','properties':{'sentence_ids':{'type':'array','items':{'type':'integer','enum':list(range(len(case['sentences'])))},'minItems':1,'maxItems':4}},'required':['sentence_ids'],'additionalProperties':False}
    return {'type':'object','properties':{'relation':{'type':'string','enum':list(LABELS)}},'required':['relation'],'additionalProperties':False}

def call(t,stage,messages,schema,model,journal):
    gate=preflight(t['port'])
    body={'model':model,'messages':messages,'temperature':t['temperature'],'seed':t['seed'],'top_p':1,
        'max_tokens':65,'stream':False,'response_format':{'type':'json_schema',
        'json_schema':{'name':'science_'+stage,'strict':True,'schema':schema}},
        'chat_template_kwargs':{'enable_thinking':False}}
    append(journal,{'trial':t['trial_id'],'event':'request_start','stage':stage,
                    'model':model,'admission':gate,'request':body,'uncertain_if_interrupted':True})
    start=time.monotonic()
    http=urllib.request.Request(f'http://10.8.0.7:{t["port"]}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(http,timeout=65) as conn:response=json.load(conn)
    delta=time.monotonic()-start
    choices=response.get('choices') or [{}]
    output=choices[0].get('message') or {}
    try:value=json.loads(output.get('content') or '{}')
    except (TypeError,ValueError):value={}
    result={'stage':stage,'content':value,'raw_message':output,'usage':response.get('usage'),
            'finish_reason':choices[0].get('finish_reason'),'wall_seconds':delta}
    append(journal,{'trial':t['trial_id'],'event':'response','result':result})
    return result

def run_one(t,model,journal):
    c=t['case']
    # Only after preflight of BOTH model backends and all active worker queues.
    gate=preflight(t['port'])
    append(journal,{'trial':t['trial_id'],'event':'begin','version':VERSION,'case_id':c['case_id'],
                    'claim_sha256':c['claim_sha256'],'abstract_sha256':c['abstract_sha256'],
                    'source_doc_id':c['source_doc_id'],'source_claim_id':c['source_claim_id'],
                    'expert_label':c['gold'],'model':model,'condition':t['condition'],
                    'seed':t['seed'],'temperature':t['temperature'],'admission':gate})
    steps=[];picked=None
    if t['condition']=='select_then_answer':
        r=call(t,'select',input_messages(c,'select'),schema_for('select',c),model,journal);steps.append(r)
        ids=r['content'].get('sentence_ids')
        if not isinstance(ids,list) or not 1<=len(ids)<=4 or len(set(ids))!=len(ids) or any(type(x) is not int or x<0 or x>=len(c['sentences']) for x in ids):
            append(journal,{'trial':t['trial_id'],'event':'completed','status':'invalid_selector','stages':1,
                            'total_wall_seconds':sum(x['wall_seconds'] for x in steps),
                            'total_tokens':sum((x['usage'] or {}).get('total_tokens',0) for x in steps)})
            return 'invalid_selector'
        picked=ids
        # Recheck active ownership and native slots between semantically separate calls.
        preflight(t['port'])
    r=call(t,'verdict',input_messages(c,t['condition'],picked),schema_for('verdict',c),model,journal)
    steps.append(r)
    answer=r['content'].get('relation')
    valid=answer in LABELS
    result={'trial':t['trial_id'],'event':'completed','status':'scored','expert_gold':c['gold'],
           'answer':answer,'valid':valid,'correct':valid and answer==c['gold'],
           'condition':t['condition'],'selected_ids':picked,
           'expert_rationale_covered':len(set(picked or [])&set(c['rationale_indices']))/len(c['rationale_indices']) if picked is not None else None,
           'stages':len(steps),'total_wall_seconds':sum(x['wall_seconds'] for x in steps),
           'total_tokens':sum((x['usage'] or {}).get('total_tokens',0) for x in steps)}
    append(journal,result)
    return 'correct' if result['correct'] else ('wrong' if valid else 'invalid')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-private',type=pathlib.Path,required=True)
    p.add_argument('--journal',type=pathlib.Path,required=True);p.add_argument('--max-new',type=int,default=6)
    a=p.parse_args()
    if not 1<=a.max_new<=12:raise SystemExit('max twelve trials per admission batch')
    manifest=json.loads((HERE/'scifact_expert_manifest_v1.json').read_text())
    private=json.loads(a.source_private.read_text())
    cases=validate(manifest,private)
    seen=set()
    if a.journal.exists():
        for row in a.journal.read_text().splitlines():seen.add(json.loads(row)['trial'])
    a.journal.parent.mkdir(parents=True,exist_ok=True)
    models={port:fetch(port,'/v1/models')['data'][0]['id'] for port in PORTS}
    complete=0
    for t in plan(cases):
        if t['trial_id'] in seen:continue
        if complete>=a.max_new:break
        try:
            state=run_one(t,models[t['port']],a.journal)
            complete+=1
            print(json.dumps({'trial':t['trial_id'],'outcome':state}),flush=True)
        except Exception as exc:
            # No retry after any external request might have been accepted.
            append(a.journal,{'trial':t['trial_id'],'event':'stopped','exception':type(exc).__name__,
                              'reason':str(exc)[:130],'never_replay':True})
            print(json.dumps({'trial':t['trial_id'],'stopped':type(exc).__name__+': '+str(exc)[:100]}),flush=True)
            break
    print('new_completed',complete,flush=True)
if __name__=='__main__':main()
