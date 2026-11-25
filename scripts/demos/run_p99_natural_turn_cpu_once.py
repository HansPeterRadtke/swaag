#!/usr/bin/env python3
"""One-use offline p99lab single-channel EOT screen, real annotated public audio.

No microphone, model service, remote inference, agent, action or GPU.
Uses only full native 160 ms steps, their last score and actual annotations.
The v1 ONNX graph is not the v2 multichannel interruption model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import wave
from prepare_public_natural_turn_corpus import (
    PRIVATE_PREFIX, DATASET, DATA_REVISION, FILES, sha_file, sha_bytes,
)

MODEL_SHA='e747b182d685c7db489a3ea2e12bdcd65bac506fcddec00ed3cd0bf9f209fd11'
MODEL_REV='d548240397d5449901559eb09f87cdfcad70420e'
MANIFEST_SHA='f4e217a4c14287c3b1d4eaa8c872a2273d5a37ff7da38c9d114d639ed804892e'
MODEL_BYTES=21825125
OWNER='owner-20261103-swaag-voice-agent-newchat/w103-p99-natural-audio'
STEP=2560
SR=16000
MIN_PAUSE=3200
THRESHOLDS=(0.06,0.16,0.36,0.50,0.80,0.95)
PRIMARY=0.36
STATES=('audio','conv1','conv2','k','v','k_valid','h6')
SHAPES={'state_audio':[1,304],'state_conv1':[1,80,2],
        'state_conv2':[1,256,2],'state_k':[6,4,60,64],
        'state_v':[6,4,60,64],'state_k_valid':[60],'state_h6':[60,256]}
OUT_NAMES=['probs']+['new_state_'+k for k in STATES]


def require(ok,msg):
    if not ok:raise ValueError(msg)


def unique(p,data):
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def dump(p,obj):
    unique(p,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False).encode())


def lease_check(lease):
    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
         'raspi','/data/bin/resource-broker','status']
    d=json.loads(subprocess.run(cmd,capture_output=True,text=True,timeout=12,check=True).stdout)
    m=d['machines']['nitro']
    require(m['snapshot_status']=='ok' and 0<=m['snapshot_age_seconds']<=90,
            'stale broker snapshot')
    found=[x for x in m['requests'] if x.get('request_id')==lease]
    require(len(found)==1,'no exact CPU lease')
    r=found[0]
    require(r['owner']==OWNER and r['state']=='active' and r['ram_bytes']>=3*1024**3
            and r['cpu_millicores']>=2000,'insufficient exact CPU lease')
    return r


def sources(root):
    model=root/'p99-model.onnx';manifest=root/'corpus/manifest.json'
    require(model.stat().st_size==MODEL_BYTES and sha_file(model)==MODEL_SHA,'model drift')
    require(sha_file(manifest)==MANIFEST_SHA,'pinned corpus manifest drift')
    require('Apache License' in (root/'p99-LICENSE').read_text()
            and 'p99lab' in (root/'p99-NOTICE').read_text(),'model license notice missing')
    rows=json.loads(manifest.read_text())['rows']
    require(len(rows)==800 and len({r['id'] for r in rows})==800,'wrong corpus count')
    for lang,(_,_,_,nhold) in FILES.items():
        g=[r for r in rows if r['language']==lang]
        require(len(g)==400 and sum(s['label']=='hold' for r in g for s in r['spans'])==nhold,
                'wrong stratum or pause labels')
    return model,rows


def step_endpoints(span):
    # Never use padded future samples or earlier per-frame outputs from a
    # chunk with audio beyond the decision time. No score outside a pause.
    start,end=span['start_sample'],span['end_sample']
    first=((start+MIN_PAUSE+STEP-1)//STEP)*STEP
    return list(range(first,end+1,STEP))


def latency_at(points,span,threshold):
    for end,score in points:
        if score>=threshold:
            return (end-span['start_sample'])/SR
    return None


def summarize(rows):
    out={}
    for lang in ('en','de'):
        group=[r for r in rows if r['language']==lang]
        tab={}
        for threshold in THRESHOLDS:
            eot,hold,delays=[],[],[]
            false_turns=0
            for row in group:
                false_turn=False
                for s in row['spans']:
                    pts=s['points']
                    delay=latency_at(pts,s,threshold)
                    if s['label']=='hold':
                        hold.append((bool(pts),delay is not None))
                        false_turn|=delay is not None
                    else:
                        eot.append((bool(pts),delay is not None))
                        if delay is not None:delays.append(delay)
                false_turns+=false_turn
            tab[str(threshold)]={'hold_spans_total':len(hold),'hold_spans_eligible':sum(a for a,b in hold),
                'false_hold_spans':sum(b for a,b in hold),'turns_with_false_hold':false_turns,
                'eot_spans_total':len(eot),'eot_spans_eligible':sum(a for a,b in eot),
                'eot_detected':sum(b for a,b in eot),
                'eot_recall_all':sum(b for a,b in eot)/len(eot),
                'hold_false_positive_rate_eligible':sum(b for a,b in hold)/max(1,sum(a for a,b in hold)),
                'detected_eot_latency_median_s':statistics.median(delays) if delays else None}
        out[lang]=tab
    return out


def checked_audio(root,row):
    name=row['file']
    require(name==sha_bytes(row['id'].encode())+'.wav','unexpected corpus filename')
    p=root/'corpus'/name
    require(p.is_file() and not p.is_symlink(),'missing pinned audio')
    raw=p.read_bytes()
    require(sha_bytes(raw)==row['wav_sha256'],'changed audio')
    with wave.open(io.BytesIO(raw),'rb') as w:
        require((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==
                (SR,1,2,row['frames']),'wrong PCM format')
        return w.readframes(w.getnframes())


def run(root,out,lease):
    model,rows=sources(root)
    lease_check(lease)
    require(not out.exists() and not out.is_symlink() and
            str(out.resolve()).startswith(PRIVATE_PREFIX),'unique private run required')
    out.mkdir(mode=0o700)
    src=Path(__file__).resolve()
    attempt={'started_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha':MODEL_SHA,'model_revision':MODEL_REV,'corpus_manifest_sha':MANIFEST_SHA,
        'script_sha256':sha_file(src),'lease':lease,'thresholds':THRESHOLDS,
        'primary':PRIMARY,'minimum_pause_samples':MIN_PAUSE,'step_samples':STEP,
        'research_gate':{'zero_false_hold':True,'min_eot_recall_all':0.95,'max_step_p95_s':0.025},
        'no_model_service_agent_phone_or_tools':True}
    dump(out/'attempt.json',attempt)
    started=time.monotonic()
    all_rows=[];times=[]
    status='failed_or_incomplete'
    error=None
    try:
        import numpy as np
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sess=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        require(sess.get_providers()==['CPUExecutionProvider'],'unexpected model provider')
        require({i.name:i.shape for i in sess.get_inputs()}=={'audio':[1,STEP],**SHAPES},
                'unexpected input shapes')
        require([o.name for o in sess.get_outputs()]==OUT_NAMES,'unexpected output names')
        def reset():
            return {n:np.zeros(shape,dtype=np.float32) for n,shape in SHAPES.items()}
        # One non-corpus silence warm-up; not used for latency or labels.
        sess.run(OUT_NAMES,{'audio':np.zeros((1,STEP),np.float32),**reset()})
        with (out/'rows.jsonl').open('xb') as log:
            for index,row in enumerate(rows):
                pcm=checked_audio(root,row)
                x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.0
                state=reset(); endpoints={}
                spans=[{**s,'points':[]} for s in row['spans']]
                for span in spans:
                    for end in step_endpoints(span):
                        require(end not in endpoints,'overlapping scored pause')
                        endpoints[end]=span
                rt=[]
                for start in range(0,len(x)-STEP+1,STEP):
                    before=time.perf_counter()
                    res=sess.run(OUT_NAMES,{'audio':x[start:start+STEP].reshape(1,STEP),**state})
                    elapsed=time.perf_counter()-before
                    rt.append(elapsed)
                    require(res[0].shape==(8,) and np.isfinite(res[0]).all()
                            and np.all((res[0]>=0)&(res[0]<=1)),'invalid probability')
                    state={'state_'+k:v for k,v in zip(STATES,res[1:])}
                    if start+STEP in endpoints:
                        endpoints[start+STEP]['points'].append([start+STEP,float(res[0][-1])])
                item={'id':row['id'],'language':row['language'],'wav_sha256':row['wav_sha256'],
                      'frames':row['frames'],'spans':spans,'step_count':len(rt),'step_seconds':rt}
                log.write(json.dumps(item,allow_nan=False).encode()+b'\n')
                log.flush()
                all_rows.append(item);times.extend(rt)
                if (index+1)%50==0:
                    print('rows',index+1,'of',len(rows),'steps',len(times),flush=True)
        report=summarize(all_rows)
        q=sorted(times)
        p95=q[math.ceil(len(q)*.95)-1]
        passes=all(report[l][str(PRIMARY)]['false_hold_spans']==0
                   and report[l][str(PRIMARY)]['eot_recall_all']>=.95 for l in ('en','de')) and p95<=.025
        status='completed_narrow_research_screen'
        result={'status':status,'model_sha':MODEL_SHA,'model_revision':MODEL_REV,
                'manifest_sha':MANIFEST_SHA,'rows':len(all_rows),'steps':len(times),
                'actual_ort_version':ort.__version__,'numpy_version':np.__version__,
                'provider':sess.get_providers(),'step_p50_s':statistics.median(times),
                'step_p95_s':p95,'step_max_s':max(times),'metrics':report,
                'passes_predeclared_primary_screen':passes,
                'release_authorized':False,'automatic_command_authorized':False,
                'speaker_or_interrupt_tested':False,'german_officially_supported':False,
                'official_eot_benchmark_score':False,'model_or_agent_remote_calls':0,
                'rows_sha256':sha_file(out/'rows.jsonl')}
    except Exception as e:
        error=type(e).__name__+': '+str(e)[:300]
        result={'status':status,'error':error,'rows_completed':len(all_rows),
                'release_authorized':False}
    result['wall_seconds']=time.monotonic()-started
    dump(out/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','rows','steps','step_p50_s','step_p95_s','passes_predeclared_primary_screen','wall_seconds','error')}),flush=True)
    return 0 if status=='completed_narrow_research_screen' else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease');p.add_argument('--run',action='store_true')
    a=p.parse_args();os.umask(0o077)
    if not a.run:
        _,rows=sources(a.root);print(json.dumps({'preflight':True,'rows':len(rows),'no_model_loaded':True}));return 0
    require(os.getenv('SWAAG_W103_P99_CPU_ONCE')=='1' and a.lease,'explicit opt-in and lease required')
    return run(a.root,a.output,a.lease)

if __name__=='__main__':raise SystemExit(main())
