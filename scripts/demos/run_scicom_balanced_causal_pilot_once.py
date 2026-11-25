#!/usr/bin/env python3
"""Bounded distinct bilingual acoustic pilot after the stopped full attempt.

The original 22 complete decisions and uncertain next decision are never
replayed. Exclude every affected recording, select by frozen hash order with
no model scores, and use one eligible pause per row. This is a new diagnostic
sample, not a resumption or representative holdout claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from run_scicom_causal_audio_once import (
    ROOT,PINS,INPUT_SHA,THRESHOLDS,PRIMARY,GATE,WINDOW,SR,model_paths,inputs,
    pcm_for,causal_window,disable_network,lease_valid,require,dump,sha_file,q95,
)

STOP_SHA='d6ede16fdfc8ea4952a277c184e3100b2eab1cb453a1e686ff5e7f83644e793f'
FIRST_ATTEMPT_SHA='a61c028ddf24f43b679a2749792ea4fcd8552e82ba83e38fcfb8d7db140d0ed8'
SEED='swaag-20261125-w104-balanced-distinct-v1'
PER_STRATUM=40
THREADS=2


def select_cases(manifest,excluded):
    used=set(excluded);selected=[]
    for lang in ('en','de'):
        for label in ('hold','eot'):
            candidates=[]
            for r in manifest:
                if r['language']!=lang:continue
                for s in r['spans']:
                    if s['label']==label and s['decision_sample'] is not None:
                        key=hashlib.sha256((SEED+'|'+r['id']+'|'+str(s['index'])).encode()).hexdigest()
                        candidates.append((key,r,s))
            count=0
            for _,r,s in sorted(candidates,key=lambda x:x[0]):
                if r['id'] in used:continue
                used.add(r['id'])
                selected.append({'id':r['id'],'language':lang,'wav_sha256':r['wav_sha256'],**s})
                count+=1
                if count==PER_STRATUM:break
            require(count==PER_STRATUM,'insufficient distinct eligible rows')
    # Deterministic interleaving of strata by second independent hash.
    return sorted(selected,key=lambda c:hashlib.sha256((SEED+'|order|'+c['id']).encode()).hexdigest())


def original_exclusions(root,manifest):
    run=root/'scicom-three-graphs-once'
    require(sha_file(run/'attempt.json')==FIRST_ATTEMPT_SHA and sha_file(run/'decisions.jsonl')==STOP_SHA,
            'original interrupted evidence changed')
    stop=json.loads((root/'full-attempt-stop-receipt.json').read_text())
    require(stop['process_stopped'] is True and stop['completed_original_result'] is False
            and stop['raw_decisions_sha256']==STOP_SHA and not (run/'result.json').exists(),
            'original experiment not safely stopped')
    old=[json.loads(x) for x in (run/'decisions.jsonl').read_text().splitlines()]
    keys=[(r['id'],s['index']) for r in manifest for s in r['spans'] if s['decision_sample'] is not None]
    require(len(old)==22 and [(r['id'],r['index']) for r in old]==keys[:len(old)]
            and tuple(stop['uncertain_next_pause'])==keys[len(old)],'stopped prefix mismatch')
    excluded=sorted({r['id'] for r in old}|{keys[len(old)][0]})
    require(stop['excluded_row_ids']==excluded,'unsafe original exclusions')
    return excluded


def selection(root):
    manifest=inputs(root);exclude=original_exclusions(root,manifest)
    cases=select_cases(manifest,exclude)
    return {'input_manifest_sha256':INPUT_SHA,'stopped_original_rows_sha256':STOP_SHA,
            'excluded_recordings':exclude,'seed':SEED,'per_stratum':PER_STRATUM,
            'no_prior_scores_used':True,'cases':cases}


def metrics(rows):
    result={}
    for name in PINS:
        result[name]={}
        for lang in ('en','de'):
            group=[r for r in rows if r['language']==lang]
            holds=[r for r in group if r['label']=='hold'];ends=[r for r in group if r['label']=='eot']
            require(len(holds)==len(ends)==PER_STRATUM,'incomplete pilot stratum')
            times=[r['feature_seconds']+r['models'][name]['inference_seconds'] for r in group]
            thresholds={}
            for threshold in THRESHOLDS:
                fp=sum(r['models'][name]['score']>=threshold for r in holds)
                tp=sum(r['models'][name]['score']>=threshold for r in ends)
                thresholds[str(threshold)]={'false_holds':fp,'hold_count':len(holds),'detected_ends':tp,
                     'end_count':len(ends),'recall':tp/len(ends)}
            primary=thresholds[str(PRIMARY)]
            result[name][lang]={'thresholds':thresholds,'compute_p50_s':statistics.median(times),
                    'compute_p95_s':q95(times),'compute_max_s':max(times),
                    'passes_primary_research_screen':primary['false_holds']==0 and primary['recall']>=.95 and q95(times)<=.25}
    return result


def run(root,out,lease):
    paths=model_paths(root);manifest=inputs(root);plan=selection(root)
    plan_path=root/'balanced-selection.json'
    require(plan_path.is_file() and not plan_path.is_symlink() and json.loads(plan_path.read_text())==plan,
            'frozen selection changed')
    r=lease_valid(lease)
    require(r['expires_at']-time.time()>=600,'not enough admitted time for pilot')
    require(out.parent.resolve()==root.resolve()==ROOT and not root.is_symlink() and not out.exists() and not out.is_symlink(),
            'fresh private output required')
    out.mkdir(mode=0o700)
    dump(out/'attempt.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_script_sha256':sha_file(Path(__file__).resolve()),
             'base_runner_sha256':sha_file(Path(__file__).with_name('run_scicom_causal_audio_once.py')),
             'selection_sha256':sha_file(plan_path),'model_pins':PINS,'threads':THREADS,
             'thresholds':THRESHOLDS,'primary':PRIMARY,'gate':GATE,'lease':lease,
             'one_use_no_replay':True,'no_agent_model_service_phone_or_tool_calls':True})
    started=time.monotonic();rows=[];result={'status':'failed_or_incomplete','release_authorized':False}
    try:
        disable_network()
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        fe=WhisperFeatureExtractor(feature_size=80,sampling_rate=SR,chunk_length=8)
        opts=ort.SessionOptions();opts.intra_op_num_threads=THREADS;opts.inter_op_num_threads=1
        opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sessions={}
        for name,path in paths.items():
            s=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
            require(s.get_providers()==['CPUExecutionProvider'] and [i.name for i in s.get_inputs()]==['input_features']
                    and [o.name for o in s.get_outputs()]==['probability'],'wrong model contract')
            sessions[name]=s
        byid={r['id']:r for r in manifest}
        # No repeated warm-up from the interrupted attempt. Use a new fixed
        # non-oracle signal, distinct from any input recording.
        def feature(x):
            f=fe([x],sampling_rate=SR,return_tensors='np',padding='max_length',max_length=WINDOW,
                  truncation=True,do_normalize=False)['input_features'].astype(np.float32)
            require(f.shape==(1,80,800) and np.isfinite(f).all(),'bad feature')
            return f
        def predict(s,f):
            start=time.perf_counter();p=s.run(['probability'],{'input_features':f})[0];seconds=time.perf_counter()-start
            require(p.shape==(1,1) and np.isfinite(p).all() and 0<=float(p[0,0])<=1,'bad probability')
            return {'score':float(p[0,0]),'inference_seconds':seconds}
        warm=feature(np.zeros(WINDOW,np.float32)+np.float32(.0001))
        for s in sessions.values():predict(s,warm)
        with (out/'decisions.jsonl').open('xb') as f:
            for c in plan['cases']:
                require(time.time()+15<r['expires_at'],'lease approaching expiry')
                pcm=pcm_for(root,byid[c['id']]);t=time.perf_counter()
                feat=feature(causal_window(pcm,c['decision_sample'],np));elapsed=time.perf_counter()-t
                item={**c,'feature_seconds':elapsed,'models':{}}
                for name,s in sessions.items():item['models'][name]=predict(s,feat)
                f.write(json.dumps(item,allow_nan=False).encode()+b'\n');f.flush();rows.append(item)
                if len(rows)%20==0:print('balanced pilot',len(rows),'of',len(plan['cases']),flush=True)
        result={'status':'completed_balanced_distinct_cpu_pilot','rows':len(rows),'actual_predictions':len(rows)*3,
                'selection_sha256':sha_file(plan_path),'model_pins':PINS,'provider':['CPUExecutionProvider'],
                'onnxruntime_version':ort.__version__,'threads':THREADS,'metrics':metrics(rows),
                'decisions_sha256':sha_file(out/'decisions.jsonl'),'release_authorized':False,
                'automatic_command_authorized':False,'german_officially_supported':False,
                'official_benchmark_score':False,'speaker_or_interruption_tested':False,'remote_model_calls':0}
    except Exception as e:
        result.update(error=type(e).__name__+': '+str(e)[:300],rows_completed=len(rows))
    result['wall_seconds']=time.monotonic()-started
    dump(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
    return 0 if result['status']=='completed_balanced_distinct_cpu_pilot' else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--lease')
    a=p.parse_args();os.umask(0o077)
    if a.prepare:
        require(not a.run and a.root.resolve()==ROOT,'preparation is separate from inference')
        plan=selection(a.root);dump(a.root/'balanced-selection.json',plan)
        print(json.dumps({'prepared':len(plan['cases']),'selection_sha256':sha_file(a.root/'balanced-selection.json'),
                          'excluded_recordings':len(plan['excluded_recordings'])}));return 0
    require(a.run and a.lease and os.getenv('SWAAG_W104_SCICOM_BALANCED_ONCE')=='1','fresh explicit opt-in required')
    return run(a.root,a.output,a.lease)

if __name__=='__main__':raise SystemExit(main())
