#!/usr/bin/env python3
"""New one-use CPU experiment on aligned prefixes of real public speech.

Reuses already verified Namo model bytes, not the previous synthetic oracle
experiment. No speech recognition, remote inference, actions or production
services. Gold-prefix information is optimistic and is not live ASR proof.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from prepare_public_natural_turn_corpus import PRIVATE_PREFIX, sha_bytes, sha_file
from run_p99_natural_turn_cpu_once import MANIFEST_SHA, dump, require
from run_isolated_namo_multilingual_cpu_once import MODEL_FILES, REVISION, predict

TEXT_SHA='c25e8871ab61930edbad6a385b2cef21096077f994e56088604d7d4e38ca5169'
OWNER='owner-20261103-swaag-voice-agent-newchat/w103-namo-natural-aligned-text'
THRESHOLDS=(0.5,0.8)
# Freeze before any aligned-prefix inference; no threshold fitting.
FUSION=((0.36,0.5),(0.36,0.8),(0.8,0.8))


def check_lease(lease):
    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
         'raspi','/data/bin/resource-broker','status']
    d=json.loads(subprocess.run(cmd,capture_output=True,text=True,timeout=12,check=True).stdout)
    m=d['machines']['nitro']
    require(m['snapshot_status']=='ok' and 0<=m['snapshot_age_seconds']<=90,'stale lease snapshot')
    found=[x for x in m['requests'] if x.get('request_id')==lease]
    require(len(found)==1,'no exact lease')
    r=found[0]
    require(r['owner']==OWNER and r['state']=='active' and r['ram_bytes']>=2*1024**3
        and r['cpu_millicores']>=1000 and r['gpu_bytes']==0,'wrong CPU admission')


def inputs(model,manifest):
    require(model.is_dir() and not model.is_symlink(),'invalid model directory')
    for n,(h,size) in MODEL_FILES.items():
        f=model/n
        require(f.stat().st_size==size and sha_file(f)==h,'model source drift '+n)
    require(sha_file(manifest)==TEXT_SHA,'aligned transcript source changed')
    d=json.loads(manifest.read_text())
    require(d['source_manifest_sha256']==MANIFEST_SHA and d['no_current_model_or_phone'] is True,'wrong corpus')
    rows=d['cases']
    require(len(rows)==1730 and sum(not x['has_prefix'] for x in rows)==27,'wrong case count')
    require(len({(c['id'],c['span_index']) for c in rows})==len(rows),'duplicate span')
    for c in rows:
        require(c['language'] in ('en','de') and c['label'] in ('hold','eot')
            and c['text_sha256']==sha_bytes(c['text'].encode()) and c['has_prefix']==bool(c['text']),
            'bad source case')
    return rows


def run(model,manifest,out,lease):
    cases=inputs(model,manifest)
    check_lease(lease)
    require(not out.exists() and not out.is_symlink() and str(out.resolve()).startswith(PRIVATE_PREFIX),
            'unused private result required')
    out.mkdir(mode=0o700)
    dump(out/'attempt.json',{'utc':datetime.now(timezone.utc).isoformat(),'model_revision':REVISION,
        'model_sha':MODEL_FILES['model_quant.onnx'][0],'manifest_sha':TEXT_SHA,
        'script_sha':sha_file(Path(__file__).resolve()),'thresholds':THRESHOLDS,
        'fusion':FUSION,'lease':lease,'one_use_no_remote_model_or_agent':True,
        'ideal_word_alignments_not_asr':True,'missing_prefix_never_authorizes':True})
    t=time.monotonic();rows=[];times=[]
    result={'status':'failed_or_incomplete','release_authorized':False}
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sess=ort.InferenceSession(str(model/'model_quant.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        require(sess.get_providers()==['CPUExecutionProvider'],'unexpected provider')
        tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
        predict(sess,tokenizer,np,'A local non-oracle warm-up.')
        with (out/'cases.jsonl').open('xb') as f:
            for index,c in enumerate(cases):
                record={k:v for k,v in c.items() if k!='text'}
                if not c['has_prefix']:
                    record.update(status='no_aligned_prefix')
                elif len(tokenizer.encode(c['text']).ids)>512:
                    record.update(status='oversized_prefix')
                else:
                    # A failed/uncertain inference is not retried or counted
                    # as a true negative. The source experiment ends on error.
                    record.update(status='valid',**predict(sess,tokenizer,np,c['text']))
                    times.append(record['inference_seconds'])
                f.write(json.dumps(record,allow_nan=False).encode()+b'\n');f.flush()
                rows.append(record)
                if (index+1)%100==0:
                    print('aligned pauses',index+1,'of',len(cases),flush=True)
        valid=[x for x in rows if x['status']=='valid']
        q=sorted(times)
        result={'status':'completed_actual_cpu_aligned_prefix_screen','model_revision':REVISION,
                'model_sha':MODEL_FILES['model_quant.onnx'][0],'manifest_sha':TEXT_SHA,
                'source_script_sha':sha_file(Path(__file__).resolve()),'provider':sess.get_providers(),
                'rows':len(rows),'valid':len(valid),'missing_or_oversized':len(rows)-len(valid),
                'inference_p50_s':statistics.median(q),'inference_p95_s':q[math.ceil(len(q)*0.95)-1],
                'inference_max_s':q[-1],'cases_sha256':sha_file(out/'cases.jsonl'),
                'remote_model_calls':0,'agent_actions':0,'live_asr_tested':False,
                'release_authorized':False,'automatic_command_authorized':False}
    except Exception as e:
        result.update(error=type(e).__name__+': '+str(e)[:300],rows_completed=len(rows))
    result['wall_seconds']=time.monotonic()-t
    dump(out/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['status'].startswith('completed_actual') else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','manifest','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--lease');p.add_argument('--run',action='store_true')
    a=p.parse_args();os.umask(0o077)
    if not a.run:
        print(json.dumps({'preflight_only':True,'cases':len(inputs(a.model,a.manifest))}));return 0
    require(os.getenv('SWAAG_W103_NAMO_ALIGNED_ONCE')=='1' and a.lease,'explicit fresh opt-in and lease required')
    return run(a.model,a.manifest,a.output,a.lease)

if __name__=='__main__':raise SystemExit(main())
