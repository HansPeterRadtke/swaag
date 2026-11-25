#!/usr/bin/env python3
"""Independent read-only adjudication of the completed W104 pilot.

No inference, model loading, network, service or agent actions. Verify
original source identities, disjoint sampling, causal boundaries and
all 480 recorded scores. The stopped full attempt remains incomplete.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

PINS={
 'tiny':['86dddafbb85ee3ca19c6efc74ed6f69b27fd1d9c',10118687,'7312bd478f3a32fa272c056718cb454048fc82df6dce77f02bead40849bbae9d'],
 'base':['b16033db5f46856684572f668012e721dd199e16',23776923,'d0ef84bbcb490b154295fd494819af82c562c3bf1c45036604f9e3e7cf0232b0'],
 'small':['bed671b2981e69ba7f019a32f3ba1ab212f3be7a',94937548,'460d9367967a079ccbb9ae19683ea1b7b5998735fa7d02d6f58bcee35aef9d45'],
}
FILES={
 'inputs.json':'6f6decf1532c30ef1b04e38526de743afcecdee8b98b589e5568ae0729456cdd',
 'balanced-selection.json':'e6e191b02d1e035b78a28ef942b8acff786ece1250779cb132765493d453f8b7',
 'full-attempt-stop-receipt.json':'7ec17f5e12dd255d5898a122b31b3cd63b0e21a4a3f1e35bc2755846b7f8e0f9',
 'scicom-three-graphs-once/attempt.json':'a61c028ddf24f43b679a2749792ea4fcd8552e82ba83e38fcfb8d7db140d0ed8',
 'scicom-three-graphs-once/decisions.jsonl':'d6ede16fdfc8ea4952a277c184e3100b2eab1cb453a1e686ff5e7f83644e793f',
 'scicom-balanced-distinct-once/attempt.json':'e4ca285ef2a42b041afabeb6234a36d2c1e05cb8d7a14fc69b4583c995672b4c',
 'scicom-balanced-distinct-once/decisions.jsonl':'a474737586c2cce6aa7ecfa6be0b6ddbf4786f905760236a21e461976965cdba',
 'scicom-balanced-distinct-once/result.json':'d27f5270b4dfbe7d588d5659b4f2721c7bf9c8203f6a99b38fa97adf1429b9af',
}
SCRIPTS={
 'prepare_scicom_causal_pause_inputs.py':'ed601a775265fd38890b32b62e6b650a232e96f7bfefb53e31cb80e7f241dc5a',
 'run_scicom_balanced_causal_pilot_once.py':'6cbebdbaea4f93e7ba1054bc522cb4a5c3203f186af6819f01e2bd9729fdc2a2',
 'run_scicom_causal_audio_once.py':'e8601d4fbe17696abcf587ff04d814df7079aa6879a5576c764f399e5ec41b0f',
}
OLD='f4e217a4c14287c3b1d4eaa8c872a2273d5a37ff7da38c9d114d639ed804892e'
SEED='swaag-20261125-w104-balanced-distinct-v1'
THRESHOLDS=(.5,.55,.8,.95)

def check(ok,why):
    if not ok:raise ValueError(why)

def sha(b):return hashlib.sha256(b).hexdigest()

def file(root,name,size_limit=110_000_000):
    p=root/name
    check(p.is_file() and not p.is_symlink() and p.stat().st_size<=size_limit,'invalid evidence file')
    try:p.resolve().relative_to(root.resolve())
    except ValueError as e:raise ValueError('external evidence path') from e
    return p

def pinned(root,name,h):
    p=file(root,name)
    with p.open('rb') as f:got=hashlib.file_digest(f,'sha256').hexdigest()
    check(got==h,'changed evidence: '+name)
    return p

def build_cases(source,excluded):
    picked=[];used=set(excluded)
    for lang in ('en','de'):
        for label in ('hold','eot'):
            candidates=[]
            for row in source:
                if row['language']!=lang:continue
                for s in row['spans']:
                    if s['label']!=label or s['decision_sample'] is None:continue
                    ident=row['id']
                    rank=sha(f"{SEED}|{ident}|{s['index']}".encode())
                    candidates.append((rank,dict(id=ident,language=lang,wav_sha256=row['wav_sha256'],**s)))
            group=[]
            for _,case in sorted(candidates,key=lambda z:z[0]):
                if case['id'] in used:continue
                used.add(case['id']);group.append(case)
                if len(group)==40:break
            check(len(group)==40,'incomplete sample')
            picked.extend(group)
    return sorted(picked,key=lambda c:sha((SEED+'|order|'+c['id']).encode()))

def validate_decision(case,row):
    check(set(row)==set(case)|{'feature_seconds','models'},'wrong decision fields')
    check(all(row[k]==v for k,v in case.items()),'changed case identity or annotation')
    check(type(row['feature_seconds']) in (int,float) and math.isfinite(row['feature_seconds'])
          and row['feature_seconds']>=0,'invalid feature time')
    check(set(row['models'])==set(PINS),'missing model score')
    for x in row['models'].values():
        check(set(x)=={'score','inference_seconds'},'wrong score fields')
        p,t=x['score'],x['inference_seconds']
        check(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1
              and type(t) in (int,float) and math.isfinite(t) and t>=0,'invalid score or compute time')

def metrics(rows):
    out={}
    for name in PINS:
        out[name]={}
        for lang in ('en','de'):
            g=[r for r in rows if r['language']==lang]
            n=Counter(r['label'] for r in g)
            check(n=={'hold':40,'eot':40},'wrong balanced strata')
            durations=sorted(r['feature_seconds']+r['models'][name]['inference_seconds'] for r in g)
            table={}
            for th in THRESHOLDS:
                positive=Counter(r['label'] for r in g if r['models'][name]['score']>=th)
                table[str(th)]={'false_holds':positive['hold'],'hold_count':40,
                    'detected_ends':positive['eot'],'end_count':40,'recall':positive['eot']/40}
            p95=durations[math.ceil(.95*len(durations))-1]
            out[name][lang]={'thresholds':table,'compute_p50_s':statistics.median(durations),
                'compute_p95_s':p95,'compute_max_s':durations[-1],
                'passes_primary_research_screen':table['0.5']['false_holds']==0 and table['0.5']['recall']>=.95 and p95<=.25}
    return out

def verify(root,old,scripts):
    data={n:pinned(root,n,h).read_bytes() for n,h in FILES.items()}
    for n,h in SCRIPTS.items():pinned(scripts,n,h)
    for name,(_,size,h) in PINS.items():
        check(pinned(root,name+'-onnx-model.int8.onnx',h).stat().st_size==size,'wrong model size')
    old_rows=json.loads(pinned(old,'corpus/manifest.json',OLD).read_bytes())['rows']
    source=json.loads(data['inputs.json'])
    check(source['old_manifest_sha256']==OLD,'wrong public corpus')
    rows=source['rows']
    check(len(rows)==len(old_rows)==800,'missing public input')
    for a,b in zip(old_rows,rows):
        check({k:v for k,v in b.items() if k!='spans'}=={k:v for k,v in a.items() if k!='spans'},
              'wrong public row')
        check(len(a['spans'])==len(b['spans']),'missing labeled pause')
        for s,t in zip(a['spans'],b['spans']):
            # Enumerate legal native decision boundaries independently.
            first=next((i for i in range(2560,a['frames']+1,2560)
                        if s['start_sample']+3200<=i<=s['end_sample']),None)
            check(t==dict(s,decision_sample=first),'noncausal input decision')
    all_cases=[dict(id=r['id'],language=r['language'],wav_sha256=r['wav_sha256'],**s)
               for r in rows for s in r['spans'] if s['decision_sample'] is not None]
    check(len(all_cases)==1730,'wrong eligible total')
    stop=json.loads(data['full-attempt-stop-receipt.json'])
    prefix=[json.loads(l) for l in data['scicom-three-graphs-once/decisions.jsonl'].splitlines()]
    check(len(prefix)==22 and stop['complete_rows']==22 and stop['process_stopped'] is True
          and stop['completed_original_result'] is False and stop['never_resume_or_replay_original_attempt'] is True
          and not (root/'scicom-three-graphs-once/result.json').exists(),'stopped run incorrectly completed')
    for a,b in zip(all_cases,prefix):validate_decision(a,b)
    nxt=all_cases[22]
    excluded=sorted({x['id'] for x in prefix}|{nxt['id']})
    check(stop['uncertain_next_pause']==[nxt['id'],nxt['index']] and stop['excluded_row_ids']==excluded,'wrong interrupted inputs')
    plan=json.loads(data['balanced-selection.json'])
    cases=build_cases(rows,excluded)
    check(plan['seed']==SEED and plan['per_stratum']==40 and plan['no_prior_scores_used'] is True
          and plan['excluded_recordings']==excluded and plan['cases']==cases,'sample selection mismatch')
    a=json.loads(data['scicom-balanced-distinct-once/attempt.json'])
    r=json.loads(data['scicom-balanced-distinct-once/result.json'])
    check(a['source_script_sha256']==SCRIPTS['run_scicom_balanced_causal_pilot_once.py']
          and a['base_runner_sha256']==SCRIPTS['run_scicom_causal_audio_once.py']
          and a['model_pins']==r['model_pins']==PINS and a['selection_sha256']==r['selection_sha256']==FILES['balanced-selection.json']
          and a['thresholds']==list(THRESHOLDS) and a['primary']==.5
          and a['gate']=={'zero_false_holds':True,'minimum_end_recall_all':.95,'maximum_p95_compute_s':.25}
          and a['threads']==r['threads']==2 and a['one_use_no_replay'] is True
          and a['no_agent_model_service_phone_or_tool_calls'] is True,'run admission mismatch')
    observed=[json.loads(l) for l in data['scicom-balanced-distinct-once/decisions.jsonl'].splitlines()]
    check(len(observed)==len(cases)==r['rows']==160 and len({c['id'] for c in cases})==160
          and not set(excluded)&{c['id'] for c in cases} and r['actual_predictions']==480
          and r['decisions_sha256']==FILES['scicom-balanced-distinct-once/decisions.jsonl'],'incomplete or repeated run')
    byid={x['id']:x for x in rows}
    for case,row in zip(cases,observed):
        validate_decision(case,row)
        n=sha(case['id'].encode())+'.wav'
        check(byid[case['id']]['file']==n,'wrong input file')
        pinned(root,'input-wavs/'+n,case['wav_sha256'])
    calculated=metrics(observed)
    check(calculated==r['metrics'] and r['status']=='completed_balanced_distinct_cpu_pilot'
          and r['provider']==['CPUExecutionProvider'] and r['remote_model_calls']==0,'metric or provider mismatch')
    for k in ('release_authorized','automatic_command_authorized','german_officially_supported',
              'official_benchmark_score','speaker_or_interruption_tested'):
        check(r[k] is False,'unsupported claim')
    return {'status':'independently_verified_scicom_balanced_pilot','distinct_recordings':160,
            'predictions':480,'original_interrupted_complete_rows':22,'excluded_recordings':len(excluded),
            'metrics':calculated,'any_model_passed_both_languages':any(all(
                v[l]['passes_primary_research_screen'] for l in ('en','de')) for v in calculated.values()),
            'production_deployed':False,'no_new_inference':True,'result_sha256':FILES['scicom-balanced-distinct-once/result.json']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','old','scripts'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.root,a.old,a.scripts),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
