#!/usr/bin/env python3
"""One-use CPU-only local ONNX FireRed multilingual turn detector benchmark for SWAAG.

Non-destructive: fixed public 17-case SWAAG turn oracle + independent 34
assistant-authored EN/DE text holdout, all created before any model inference.
Never sends to SWAAG, microphone, provider, messaging, tools or phone. No
production model config. No GPU. Does NOT qualify the model for semantic
interruption or speaker-authorized tool actions. Every completed/failed
one-use result gets a distinct private no-replay marker and raw scores.
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
import subprocess
import time
from typing import Any

MODEL_FILES={
 'multilingual_best_model_q8.onnx':('285cafedb0263b065d83964ecc728b795bb281688b0d6525f6c7ca6cb1f756df',178152959),
 'tokenizer/tokenizer.json':('bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088',2919362),
 'tokenizer/tokenizer_config.json':('214f78f6c5d034a8b405632d301ee3710eb6f075b3f5faf8241a273a13908746',1191),
 'tokenizer/special_tokens_map.json':('b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3',125),
 'tokenizer/vocab.txt':('fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c',995526),
 'tokenizer/config.json':('a8ecea4db687cce851674916271ce65d54982763afedbbef347b7a95913882db',844),
 'README.md':('41404835c4b3dbb1fb22ee276ae69801149ff3c5207e53ae0a02deedc3c36686',2009),
}

ORACLE_SHA='929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e'
HOLDOUT_SHA='f84271eb1cb110ce1fcdd6fbf129ec9bd09281c0396248a4dd9a8c888b7b9ef8'
REPOSITORY='FireRedTeam/FireRedChat-turn-detector'
REVISION='1a53b2145cdac39f247a8d55922630635aadfaef'
EXPECTED_OWNER='owner-20261103-swaag-voice-agent-newchat/w72-firered-cpu-standalone'
LEASE='rb_b10b2325420d47758394520c0dd95441'
OPTIN='SWAAG_W72_NEW_FIRERED_CPU_CLASSIFIER_ONCE'


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(2*1024*1024):h.update(chunk)
    return h.hexdigest()


def regular(path:Path)->None:
    if not path.is_file() or path.is_symlink():
        raise ValueError('Missing or unsafe pinned source file '+path.name)


def checked_sources(model:Path,oracle:Path,holdout:Path)->list[dict]:
    if model.is_symlink() or not model.is_dir():raise ValueError('Model directory missing/symlinked')
    for filename,(digest,size) in MODEL_FILES.items():
        p=model/filename;regular(p)
        if p.stat().st_size!=size or sha(p)!=digest:
            raise ValueError('Published pinned source hash/size mismatch: '+filename)
    card=(model/'README.md').read_text()
    if 'license: apache-2.0' not in card:
        raise ValueError('Public repository model card is not Apache-2.0 licensed')
    for path,digest in ((oracle,ORACLE_SHA),(holdout,HOLDOUT_SHA)):
        regular(path)
        if sha(path)!=digest:raise ValueError('Oracle/holdout source changed after labels fixed: '+path.name)
    orig=json.loads(oracle.read_text())
    new=json.loads(holdout.read_text())
    original=[{'case_id':q['case_id'],'expected':q['expected'],'text':q['text'],
        'language':'de' if 'german' in q['case_id'] else 'en','source':'original_SWAAG_oracle'}
      for q in orig['cases'] if q['family']=='turn_completion']
    hold=[{**q,'source':'assistant_authored_bilingual_holdout'} for q in new['cases']]
    cases=original+hold
    if len(original)!=17 or len(hold)!=34 or len({c['case_id'] for c in cases})!=51:
        raise ValueError('Expected original 17 plus predeclared 34 independent synthetic holdouts')
    for c in cases:
        if (type(c['expected']) is not bool or not c['text'].strip()
            or c['language'] not in ('en','de') or len(c['text'])>800):
            raise ValueError('Unsafe or invalid fixed source case')
    return cases


def lease_valid():
    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
         'raspi','/data/bin/resource-broker','status']
    x=subprocess.run(cmd,capture_output=True,text=True,timeout=12,check=True)
    obj=json.loads(x.stdout)
    data=[r for d in obj['machines'].values() for r in d.get('requests',[])
        if r.get('request_id')==LEASE]
    if len(data)!=1 or data[0].get('state')!='active' or data[0].get('owner')!=EXPECTED_OWNER:
        raise RuntimeError('CPU model inference lease unavailable, do not start')
    if data[0].get('cpu_millicores',0)<2000 or data[0].get('ram_bytes',0)<3*1024**3:
        raise RuntimeError('Insufficient authorized CPU/RAM reservation')


def unique_file(path:Path,data:bytes):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def p95(seq:list[float])->float:
    if not seq:return float('nan')
    a=sorted(seq);idx=math.ceil(.95*len(a))-1
    return round(a[idx],4)


def stats(cases:list[dict])->dict[str,Any]:
    complete=[c for c in cases if not c.get('invalid')]
    lat=[c['inference_seconds'] for c in complete]
    return {'selected':len(cases),'completed_valid':len(complete),
      'invalid_or_errors':len(cases)-len(complete),
      'passed_vendor_threshold_0_5':sum(c.get('pass_0_5') is True for c in cases),
      'failed_vendor_threshold_0_5':sum(c.get('pass_0_5') is False for c in cases),
      'false_finished_incomplete_0_5':sum(c.get('expected') is False and c.get('predicted_complete_0_5') is True for c in cases),
      'false_finished_incomplete_0_8':sum(c.get('expected') is False and c.get('predicted_complete_0_8') is True for c in cases),
      'missed_complete_0_5':sum(c.get('expected') is True and c.get('predicted_complete_0_5') is False for c in cases),
      'inference_p50_seconds':round(statistics.median(lat),4) if lat else None,
      'inference_p95_seconds':p95(lat) if lat else None,
      'inference_max_seconds':round(max(lat),4) if lat else None}


def predict(onnx,tokenizer,np,text):
    # The model's own inference_onxx.py uses BERT tokenization with NO added
    # specials and 128 padded tokens (including exact attention mask). Left
    # truncation is declared explicitly and no remote model is accessed.
    tokens=tokenizer(text,truncation=True,padding='max_length',max_length=128,
                    add_special_tokens=False,return_tensors='np')
    ids=tokens['input_ids'].astype(np.int64)
    mask=tokens['attention_mask'].astype(np.int64)
    if ids.shape!=(1,128) or mask.shape!=(1,128) or not 1<=int(mask.sum())<=128:
        raise ValueError('Unexpected FireRed original tokenizer layout')
    start=time.perf_counter()
    raw=onnx.run(None,{'input_ids':ids,'attention_mask':mask})[0]
    elapsed=time.perf_counter()-start
    if raw.shape!=(1,2) or not np.all(np.isfinite(raw)):
        raise ValueError('Unexpected FireRed ONNX output/logits/NaN')
    z=raw[0].astype(np.float64)
    probs=np.exp(z-z.max());probs/=probs.sum()
    return {'p_finished':float(probs[1]),'p_incomplete':float(probs[0]),
      'tokens':int(mask.sum()),'inference_seconds':round(elapsed,6)}


def run(model:Path,oracle:Path,holdout:Path,output:Path)->dict:
    cases=checked_sources(model,oracle,holdout)
    lease_valid()
    if output.exists() or output.is_symlink() or not output.is_absolute() or not str(output).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('One-use private model experiment must use unused private root')
    if os.getenv(OPTIN)!='1':raise ValueError('Explicit one-time CPU-only model opt-in required')
    output.mkdir(mode=0o700,parents=True,exist_ok=False);os.chmod(output,0o700)
    unique_file(output/'attempt.json',json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
      'source_revision':REVISION,'model_files':MODEL_FILES,
      'original_oracle_sha':ORACLE_SHA,'bilingual_holdout_sha':HOLDOUT_SHA,
      'selected_cases':len(cases),'cpu_only':True,'no_models_providers_phone_or_tools_outside_local_ONNX':True,
      'kind':'one_use_started_do_not_replay_uncertain_or_finished_inference'},indent=2).encode())
    started=time.perf_counter()
    record={'status':'started','source_revision':REVISION,'repository':REPOSITORY,
      'source_oracle_sha':ORACLE_SHA,'holdout_sha':HOLDOUT_SHA,
      'onnx_model_sha':MODEL_FILES['multilingual_best_model_q8.onnx'][0],
      'model_local_immutable_path':str(model),'cpu_only':True,
      'model_provider_requests':0,'swaag_tool_or_phone_calls':0,
      'one_use_selected_cases':len(cases),'original_interruption_cases_not_supported_or_inferred':18,
      'German_holdout_not_in_vendor_documented_training_languages':True,
      'single_primary_threshold':0.5,'secondary_predeclared_conservative_threshold':0.8,
      'predeclared_qualify_text_only':{'original_correct':17,'bilingual_holdout_correct':34,
         'incomplete_false_finished_total':0,'inference_p95_max_seconds':0.3,'model_invalid':0},
      'cases':[]}
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
        t0=time.perf_counter()
        session_opts=ort.SessionOptions()
        session_opts.intra_op_num_threads=2
        session_opts.inter_op_num_threads=1
        session_opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sess=ort.InferenceSession(str(model/'multilingual_best_model_q8.onnx'),sess_options=session_opts,providers=['CPUExecutionProvider'])
        if sess.get_providers()!=['CPUExecutionProvider']:
            raise RuntimeError('Only CPU provider permitted')
        tokenizer=AutoTokenizer.from_pretrained(str(model/'tokenizer'),local_files_only=True,trust_remote_code=False, truncation_side='left')
        record['load_seconds']=round(time.perf_counter()-t0,3)
        record['onnxruntime_version']=ort.__version__
        record['tokenizers_version']=__import__('transformers').__version__
        record['onnx_inputs']=[x.name for x in sess.get_inputs()]
        # This is not any labeled test case and is not repeated in the case list.
        record['fresh_nonoracle_warmup']=predict(sess,tokenizer,np,'This local model test contains no user audio or external actions.')
        lease_valid()
        for q in cases:
            entry={k:q[k] for k in ('case_id','source','language','expected','text')}
            try:
                entry.update(predict(sess,tokenizer,np,q['text']))
                v=entry['p_finished']
                entry['predicted_complete_0_5']=v>=.5
                entry['predicted_complete_0_8']=v>=.8
                entry['pass_0_5']=entry['predicted_complete_0_5'] is entry['expected']
                entry['invalid']=False
            except BaseException as e:
                entry.update({'invalid':True,'error_type':type(e).__name__,'pass_0_5':False})
            record['cases'].append(entry)
            print('CASE',entry['case_id'],'p_finished',round(entry.get('p_finished',-1),4),'pass',entry['pass_0_5'],'elapsed_ms',round(1000*entry.get('inference_seconds',0),2),flush=True)
        record['original_turn_oracle']=stats([c for c in record['cases'] if c['source']=='original_SWAAG_oracle'])
        record['holdout_all']=stats([c for c in record['cases'] if c['source']=='assistant_authored_bilingual_holdout'])
        record['holdout_english']=stats([c for c in record['cases'] if c['source']=='assistant_authored_bilingual_holdout' and c['language']=='en'])
        record['holdout_german']=stats([c for c in record['cases'] if c['source']=='assistant_authored_bilingual_holdout' and c['language']=='de'])
        record['all_51']=stats(record['cases'])
        record['status']='completed_actual_CPU_ONNX_inference'
        strong=(record['original_turn_oracle']['passed_vendor_threshold_0_5']==17
            and record['holdout_all']['passed_vendor_threshold_0_5']==34
            and record['all_51']['false_finished_incomplete_0_5']==0
            and record['all_51']['invalid_or_errors']==0
            and record['all_51']['inference_p95_seconds']<=.3)
        record['quality']='meets_fixed_TEXT_ONLY_oracle_not_live_speech_authority' if strong else 'rejected_for_automatic_voice_completion_on_fixed_oracle'
    except BaseException as e:
        record['status']='failed_or_unknown'
        record['error_type']=type(e).__name__
        raise
    finally:
        record['total_run_wall_seconds']=round(time.perf_counter()-started,3)
        record['finished_utc']=datetime.now(timezone.utc).isoformat()
        unique_file(output/'result.json',json.dumps(record,indent=2,ensure_ascii=False).encode())
    return record


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--private-output',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    cases=checked_sources(args.model_dir,args.oracle,args.holdout)
    if not args.run:
        print(json.dumps({'status':'CPU_source_and_holdout_preflight_only','cases':len(cases),
         'source_revision':REVISION,'model_file_SHA256':MODEL_FILES['multilingual_best_model_q8.onnx'][0],
         'no_model_loaded_or_source_written':True}))
        return 0
    result=run(args.model_dir,args.oracle,args.holdout,args.private_output)
    print('FINAL_FIRERED_CPU',json.dumps({'status':result['status'],'quality':result.get('quality'),
      'original':result.get('original_turn_oracle'),'holdout':result.get('holdout_all'),
      'english':result.get('holdout_english'),'german':result.get('holdout_german'),
      'wall_seconds':result['total_run_wall_seconds']}))
    return 0 if result.get('quality')=='meets_fixed_TEXT_ONLY_oracle_not_live_speech_authority' else 3

if __name__=='__main__':raise SystemExit(main())
