#!/usr/bin/env python3
"""One-use local CPU comparison of three pinned Scicom acoustic EOT models.

Scores the same first causal decision sample for all models, on the exact
public labeled W103 corpus. This is a diagnostic, not official benchmark
scoring, live speech detection, speaker authentication, or tool authority.
The publisher already evaluated on this corpus; German is exploratory.
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
import socket
import statistics
import subprocess
import time
import wave

from prepare_scicom_causal_pause_inputs import ROOT
from prepare_public_natural_turn_corpus import sha_file, sha_bytes
from run_p99_natural_turn_cpu_once import MANIFEST_SHA, dump, require

PINS={
 'tiny':('86dddafbb85ee3ca19c6efc74ed6f69b27fd1d9c',10118687,'7312bd478f3a32fa272c056718cb454048fc82df6dce77f02bead40849bbae9d'),
 'base':('b16033db5f46856684572f668012e721dd199e16',23776923,'d0ef84bbcb490b154295fd494819af82c562c3bf1c45036604f9e3e7cf0232b0'),
 'small':('bed671b2981e69ba7f019a32f3ba1ab212f3be7a',94937548,'460d9367967a079ccbb9ae19683ea1b7b5998735fa7d02d6f58bcee35aef9d45'),
}
INPUT_SHA='6f6decf1532c30ef1b04e38526de743afcecdee8b98b589e5568ae0729456cdd'
OWNER='owner-20261103-swaag-voice-agent-newchat/w104-scicom-audio'
THRESHOLDS=(.5,.55,.8,.95)
PRIMARY=.5
GATE={'zero_false_holds':True,'minimum_end_recall_all':.95,'maximum_p95_compute_s':.25}
WINDOW=128000
SR=16000


def lease_valid(lease):
    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
         'raspi','/data/bin/resource-broker','status']
    status=json.loads(subprocess.run(cmd,capture_output=True,text=True,timeout=12,check=True).stdout)
    m=status['machines']['nitro']
    require(m['snapshot_status']=='ok' and 0<=m['snapshot_age_seconds']<=90,'stale broker snapshot')
    found=[r for r in m['requests'] if r.get('request_id')==lease]
    require(len(found)==1,'no matching CPU lease')
    r=found[0]
    require(r['owner']==OWNER and r['state']=='active' and r['ram_bytes']>=3*1024**3
            and r['cpu_millicores']>=2000 and r.get('gpu_bytes',0)==0,'wrong resource allowance')
    return r


def model_paths(root):
    paths={}
    receipt=json.loads((root/'download-attempt.json').read_text())
    files={x['local']:x for x in receipt['items']}
    for name,(rev,size,h) in PINS.items():
        p=root/(name+'-onnx-model.int8.onnx')
        require(p.stat().st_size==size and sha_file(p)==h,'changed model '+name)
        for artifact in ('README.md','onnx-eot_window.json','onnx-export_report.json','preprocessor_config.json'):
            n=name+'-'+artifact
            path=root/n;f=files[n]
            require(path.is_file() and not path.is_symlink() and f['revision']==rev,'wrong source metadata')
            raw=path.read_bytes()
            blob=hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
            require(len(raw)==f['size'] and f['kind']=='gitblob' and blob==f['pin'],'metadata source mismatch')
        card=(root/(name+'-README.md')).read_text()
        require('license: apache-2.0' in card,'model license declaration missing')
        c=json.loads((root/(name+'-onnx-eot_window.json')).read_text())
        require(c['window_seconds']==8 and c['num_mel_bins']==80 and c['sample_rate']==16000
                and c['do_normalize'] is False and c['family']=='whisper','input contract drift')
        paths[name]=p
    return paths


def inputs(root):
    path=root/'inputs.json'
    require(sha_file(path)==INPUT_SHA,'fixed corpus changed')
    data=json.loads(path.read_text());rows=data['rows']
    require(data['old_manifest_sha256']==MANIFEST_SHA and len(rows)==800
            and len({r['id'] for r in rows})==800,'wrong public row corpus')
    for row in rows:
        require(row['file']==sha_bytes(row['id'].encode())+'.wav' and row['language'] in ('en','de'),'unsafe row identity')
        for s in row['spans']:
            first=((s['start_sample']+3200+2559)//2560)*2560
            want=first if first<=s['end_sample'] else None
            require(s['decision_sample']==want,'wrong causal endpoint')
    require(sum(len(r['spans']) for r in rows)==2357
            and sum(s['decision_sample'] is not None for r in rows for s in r['spans'])==1730,
            'wrong frozen pause coverage')
    return rows


def pcm_for(root,row):
    p=root/'input-wavs'/row['file']
    require(p.is_file() and not p.is_symlink(),'missing original waveform')
    raw=p.read_bytes()
    require(sha_bytes(raw)==row['wav_sha256'],'input waveform changed')
    with wave.open(io.BytesIO(raw),'rb') as f:
        require((f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes())==
                (SR,1,2,row['frames']),'wrong native waveform')
        pcm=f.readframes(f.getnframes())
    require(len(pcm)==row['frames']*2,'truncated waveform')
    return pcm


def causal_window(pcm,decision,np):
    require(type(decision) is int and 0<decision<=len(pcm)//2 and len(pcm)%2==0,'bad decision sample')
    # No bytes beyond the decision point, even during feature extraction.
    x=np.frombuffer(pcm[max(0,decision-WINDOW)*2:decision*2],dtype='<i2').astype(np.float32)/32768
    if len(x)<WINDOW:
        x=np.pad(x,(WINDOW-len(x),0))
    require(x.shape==(WINDOW,),'incorrect model window')
    return x


def q95(seq):
    return sorted(seq)[math.ceil(.95*len(seq))-1]


def summarize(rows,manifest):
    result={}
    observed={(x['id'],x['index']):x for x in rows}
    require(len(observed)==len(rows)==1730,'missing or duplicate decisions')
    for name in PINS:
        result[name]={}
        for lang in ('en','de'):
            selected=[(r,s,observed.get((r['id'],s['index']))) for r in manifest if r['language']==lang for s in r['spans']]
            cases=[(s,x) for _,s,x in selected if x is not None]
            ends=sum(s['label']=='eot' for _,s,_ in selected)
            hold=sum(s['label']=='hold' for _,s,_ in selected)
            eh=sum(s['label']=='hold' for s,x in cases)
            ee=sum(s['label']=='eot' for s,x in cases)
            times=[x['feature_seconds']+x['models'][name]['inference_seconds'] for s,x in cases]
            metrics={}
            for threshold in THRESHOLDS:
                wrong=[s for s,x in cases if s['label']=='hold' and x['models'][name]['score']>=threshold]
                hits=[s for s,x in cases if s['label']=='eot' and x['models'][name]['score']>=threshold]
                metrics[str(threshold)]={'ends_all':ends,'holds_all':hold,'ends_eligible':ee,'holds_eligible':eh,
                    'end_detected':len(hits),'end_recall_all':len(hits)/ends,'false_holds':len(wrong),
                    'false_hold_rate_eligible':len(wrong)/eh}
            primary=metrics[str(PRIMARY)]
            result[name][lang]={'thresholds':metrics,'compute_p50_s':statistics.median(times),'compute_p95_s':q95(times),
               'passes_primary_research_screen':primary['false_holds']==0 and primary['end_recall_all']>=.95 and q95(times)<=.25}
    return result


def disable_network():
    # New isolated test process only. The broker read-only admission already
    # finished; inference may not open any socket or run another program.
    def denied(*args,**kwargs):
        raise RuntimeError('network forbidden in offline inference')
    socket.socket.connect=denied
    socket.socket.connect_ex=denied
    socket.create_connection=denied


def run(root,out,lease):
    paths=model_paths(root);manifest=inputs(root)
    lease_valid(lease)
    require(root.resolve()==ROOT and not root.is_symlink(),'wrong private root')
    require(out.parent.resolve()==root and not out.exists() and not out.is_symlink(),'fresh private output required')
    out.mkdir(mode=0o700)
    dump(out/'attempt.json',{'utc':datetime.now(timezone.utc).isoformat(),'model_pins':PINS,'input_manifest_sha256':INPUT_SHA,
      'source_script_sha256':sha_file(Path(__file__).resolve()),'thresholds':THRESHOLDS,'primary':PRIMARY,'gate':GATE,
      'lease':lease,'one_use_no_replay':True,'no_agent_model_service_phone_or_tool_calls':True})
    started=time.monotonic();rows=[];result={'status':'failed_or_incomplete','release_authorized':False}
    try:
        disable_network()
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        fe=WhisperFeatureExtractor(feature_size=80,sampling_rate=SR,chunk_length=8)
        def features(x):
            f=fe([x],sampling_rate=SR,return_tensors='np',padding='max_length',max_length=WINDOW,
                 truncation=True,do_normalize=False)['input_features'].astype(np.float32)
            require(f.shape==(1,80,800) and np.isfinite(f).all(),'invalid feature shape/value')
            return f
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sessions={}
        def predict(sess,f):
            t=time.perf_counter();raw=sess.run(['probability'],{'input_features':f})[0];elapsed=time.perf_counter()-t
            require(raw.shape==(1,1) and np.isfinite(raw).all() and 0<=float(raw[0,0])<=1,'invalid probability')
            return {'score':float(raw[0,0]),'inference_seconds':elapsed}
        # One unrelated silence warm-up per graph, outside measured inputs.
        warm=features(np.zeros(WINDOW,np.float32))
        for name,path in paths.items():
            sess=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
            require(sess.get_providers()==['CPUExecutionProvider'] and [i.name for i in sess.get_inputs()]==['input_features']
                    and [o.name for o in sess.get_outputs()]==['probability'],'unexpected model contract')
            predict(sess,warm)
            sessions[name]=sess
        with (out/'decisions.jsonl').open('xb') as f:
            for row in manifest:
                pcm=pcm_for(root,row)
                for s in row['spans']:
                    at=s['decision_sample']
                    if at is None:continue
                    t=time.perf_counter();x=features(causal_window(pcm,at,np));feature_time=time.perf_counter()-t
                    observed={'id':row['id'],'language':row['language'],'wav_sha256':row['wav_sha256'],**s,
                              'feature_seconds':feature_time,'models':{}}
                    for name,sess in sessions.items():observed['models'][name]=predict(sess,x)
                    f.write(json.dumps(observed,allow_nan=False).encode()+b'\n');f.flush();rows.append(observed)
                    if len(rows)%100==0:print('scored',len(rows),'of 1730 pauses with all three graphs',flush=True)
        result={'status':'completed_actual_cpu_causal_pause_screen','model_pins':PINS,'input_manifest_sha256':INPUT_SHA,
                'rows':len(rows),'actual_predictions':len(rows)*3,'onnxruntime_version':ort.__version__,
                'numpy_version':np.__version__,'transformers_version':__import__('transformers').__version__,
                'provider':['CPUExecutionProvider'],'decisions_sha256':sha_file(out/'decisions.jsonl'),
                'metrics':summarize(rows,manifest),'publisher_already_tested_public_corpus':True,
                'german_officially_supported':False,'official_benchmark_score':False,'release_authorized':False,
                'automatic_command_authorized':False,'speaker_or_interruption_tested':False,'remote_model_calls':0}
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc)[:300],rows_completed=len(rows))
    result['wall_seconds']=time.monotonic()-started
    dump(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},ensure_ascii=False),flush=True)
    return 0 if result['status']=='completed_actual_cpu_causal_pause_screen' else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease');p.add_argument('--run',action='store_true')
    a=p.parse_args();os.umask(0o077)
    if not a.run:
        paths=model_paths(a.root);rows=inputs(a.root)
        print(json.dumps({'preflight_only':True,'models':list(paths),'rows':len(rows),'no_inference':True}));return 0
    require(os.getenv('SWAAG_W104_SCICOM_ONCE')=='1' and a.lease,'explicit fresh CPU opt-in required')
    return run(a.root,a.output,a.lease)


if __name__=='__main__':raise SystemExit(main())
