#!/usr/bin/env python3
"""One-use, source-pinned no-tool voice classification using ALREADY running Coder.

This is NOT a new classifier service, installed runtime change, autonomous
SWAAG turn, voice sample or production-ready alternative. Executes the
currently installed Thor gateway's exact turn/interruption classification
functions in an isolated Python interpreter with its classifier pointed to
the EXISTING, healthy Coder endpoint. Original thirty-five-case oracle is
split: six prior pilot cases are explicitly excluded and NEVER repeated;
remaining twenty-nine cases can be run once. A new, previously withheld
thirty-case holdout is a separately predeclared file before any inference.
No tool schema, SWAAG operation, TTS or client submission is sent to models.
Failed and uncertain cases are recorded durably and not retried.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path('/data/var/swaag-benchmarks/owner-serial-000062-20261123')
GATEWAY_DIR=Path('/data/src/worktrees/voice-guideline-repair-20261012/voice_agent_live_runtime')
CONFIG=Path('/etc/swaag/voice-gateway.toml')
ORIGINAL=Path('/data/src/github/swaag/src/swaag/benchmark/fixtures/voice_classifier_oracle_20261104.json')
CODER_URL='http://10.8.0.7:14833/v1/chat/completions'
CODER_MODEL='qwen3-coder-30b-a3b-instruct-q4-k-m'
OPT_IN='SWAAG_CODER_CLASSIFIER_EXTENDED_NOACTION'
EXPECTED_HASHES={
 'gateway.py':'23d5206072297c5910dc3d2fb98aa7136308a5fc96471af14843df7f3130de78',
 'gateway_config.py':'38439178736d363f73ef9d63e1162495b8f26d7c684c9b4766a9e4bcc3ed10ee',
 'gateway_runtime_toml':'8e9a1fa4593ec26b6b21b87004eda81a64a757430961024df52ab9a5964e1226',
 'original_oracle':'929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e',
 'nov23_holdout':'dd8ef8f94f0bee6624013462c6f314da52f8e33cdef7b0e09d66addd71c107fc',
}
PILOT_IDS={'base_turn_completion_unfinished_object','base_turn_completion_completed_correction',
 'base_turn_completion_hold_command','base_turn_completion_german_partial',
 'initial_interruption_ack','initial_interruption_correction'}


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def source_gate(holdout:Path)->None:
    for label,path in [('gateway.py',GATEWAY_DIR/'voice_agent_gateway.py'),
                       ('gateway_config.py',GATEWAY_DIR/'voice_agent_gateway_config.py'),
                       ('gateway_runtime_toml',CONFIG),('original_oracle',ORIGINAL),
                       ('nov23_holdout',holdout)]:
        if (not path.is_file() or path.is_symlink() or
                path.stat().st_size>150000 or sha(path)!=EXPECTED_HASHES[label]):
            raise ValueError('Source or fixed test oracle drift: '+label)


def cases_for(phase:str,holdout:Path)->list[dict]:
    if phase=='remaining29':
        cases=json.loads(ORIGINAL.read_text())['cases']
        selected=[c for c in cases if c['case_id'] not in PILOT_IDS]
        if len(cases)!=35 or len(selected)!=29:
            raise ValueError('Original oracle changed or repeated pilot input')
    elif phase=='fresh_holdout30':
        selected=json.loads(holdout.read_text())['cases']
        if len(selected)!=30 or any(x['case_id'] in PILOT_IDS for x in selected):
            raise ValueError('Holdout not the predeclared thirty new cases')
    else:raise ValueError('Unrecognized one-use classification phase')
    for c in selected:
        if (not isinstance(c,dict) or type(c.get('expected')) is not bool
            or c.get('family') not in ('turn_completion','interruption')
            or not isinstance(c.get('text'),str) or not c['text'].strip()
            or len(c['text'])>500):
            raise ValueError('Malformed or unbounded fixed decision oracle')
    if len({c['case_id'] for c in selected})!=len(selected):
        raise ValueError('Repeated case identities prohibited')
    return selected


def write_once(path:Path,data:dict)->None:
    b=json.dumps(data,ensure_ascii=False,indent=2).encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())


def update_owned_receipt(path:Path,data:dict)->None:
    # This updates ONLY an existing private receipt that THIS isolated process
    # created; it never sends model instructions again on resume.
    temp=path.parent/(path.name+'.atomic-pending')
    if temp.exists():raise FileExistsError('Interrupted receipt writer; do not resume ambiguous state')
    write_once(temp,data)
    os.replace(temp,path)


def model_config():
    sys.path.insert(0,str(GATEWAY_DIR))
    sys.path.insert(0,'/data/src/github/swaag/src')
    gateway=importlib.import_module('voice_agent_gateway')
    config=importlib.import_module('voice_agent_gateway_config')
    from swaag.benchmark.voice_classifier_acceptance import check_oracle_result
    if getattr(gateway,'CONFIG') is not None:
        raise ValueError('Classifier test interpreter is not isolated')
    installed=config.load_gateway_config(str(CONFIG))
    if installed.foreground.chat_url!=CODER_URL or installed.foreground.model!=CODER_MODEL:
        raise ValueError('Coder endpoint or identity differs from verified shared foreground model')
    gateway.CONFIG=replace(installed,classifier=config.ClassifierConfig(
        chat_url=CODER_URL,health_url='http://10.8.0.7:14833/health',
        model=CODER_MODEL,request_timeout_seconds=8.0,enable_thinking=False))
    original_json_http=gateway.json_http
    count={'calls':0}
    def strict_noaction_model_http(url,payload=None,**kwargs):
        if (url!=CODER_URL or not isinstance(payload,dict) or
                payload.get('model')!=CODER_MODEL or payload.get('stream') is not False
                or 'tools' in payload or len(payload.get('messages',[]))!=2):
            raise ValueError('Only fixed Coder no-tool semantic inference is permitted')
        count['calls']+=1
        return original_json_http(url,payload,**kwargs)
    gateway.json_http=strict_noaction_model_http
    return gateway,check_oracle_result,count


def execute(cases:list[dict],out:Path,phase:str)->dict:
    receipt={'status':'started','started_utc':datetime.now(timezone.utc).isoformat(),
       'phase':phase,'expected_cases':len(cases),'case_ids':[c['case_id'] for c in cases],
       'model':CODER_MODEL,'model_endpoint':CODER_URL,
       'fixed_source_sha256':EXPECTED_HASHES,
       'no_tools_or_SWAAG_actions':True,'no_running_gateway_mutation':True,
       'no_new_models_loaded':True,'remaining_cases_only':phase=='remaining29',
       'prior_six_pilot_not_repeated':True,'results':[]}
    write_once(out,receipt)
    gateway,check,count=model_config()
    consecutive_invalid=0
    for case in cases:
        event={'case_id':case['case_id'],'family':case['family'],
               'expected':case['expected'],'state':'request_started',
               'started_utc':datetime.now(timezone.utc).isoformat()}
        receipt['results'].append(event)
        update_owned_receipt(out,receipt)
        before=count['calls'];started=time.monotonic()
        try:
            if case['family']=='turn_completion':
                actual=gateway.classify_turn_completion(case['text'])
            else:
                actual=gateway.classify_interruption(case['text'],case.get('assistant_text') or '')
            outcome=check(case['family'],case['expected'],actual)
        except Exception as exc:
            outcome={'passed':False,'classifier_valid':False,'exception_type':type(exc).__name__}
        event.update({'state':'finished','passed':bool(outcome.get('passed')),
           'classifier_valid':bool(outcome.get('classifier_valid')),
           'observed':outcome.get('observed'),'kind':outcome.get('kind'),
           'confidence':outcome.get('confidence'),
           'elapsed_seconds':round(time.monotonic()-started,3),
           'model_requests':count['calls']-before})
        if 'exception_type' in outcome:event['exception_type']=outcome['exception_type']
        update_owned_receipt(out,receipt)
        print('CASE_OBSERVED',case['case_id'],event['passed'],event['elapsed_seconds'],flush=True)
        # A shared production model can become heavily contended. Keep our
        # own classifier probes bounded: two invalid/time-out outputs in a
        # row are sufficient to reject the candidate without consuming
        # additional inference on unfinished cases. Never retry either case.
        consecutive_invalid=(consecutive_invalid+1) if not event['classifier_valid'] else 0
        if consecutive_invalid>=2:
            receipt['status']='aborted_due_to_consecutive_invalid_classifier_responses'
            receipt['unattempted_case_ids']=[x['case_id'] for x in cases[len(receipt['results']):]]
            receipt['abort_threshold']=2
            update_owned_receipt(out,receipt)
            break
    else:
        receipt['status']='completed'
    receipt['completed_utc']=datetime.now(timezone.utc).isoformat()
    receipt['passed']=sum(x['passed'] for x in receipt['results'])
    receipt['failed']=len(receipt['results'])-receipt['passed']
    receipt['model_requests']=count['calls']
    receipt['response_p95_seconds']=sorted(x['elapsed_seconds'] for x in receipt['results'])[int((len(receipt['results'])-1)*.95)]
    receipt['response_max_seconds']=max(x['elapsed_seconds'] for x in receipt['results'])
    receipt['production_release_qualified']=False
    update_owned_receipt(out,receipt)
    return receipt


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['remaining29','fresh_holdout30'],required=True)
    ap.add_argument('--holdout',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args(argv)
    source_gate(args.holdout)
    cases=cases_for(args.phase,args.holdout)
    if not args.run:
        print(json.dumps({'status':'read_only_no_model_call','phase':args.phase,
           'expected_cases':len(cases),'case_ids':[c['case_id'] for c in cases],
           'model_actions':0}))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('Explicit existing model classification opt-in required')
    if not args.evidence.is_absolute() or not str(args.evidence).startswith(str(BASE)+'/'):
        raise ValueError('One-use receipt outside the project-owned private directory')
    if args.evidence.is_symlink() or args.evidence.exists():
        raise FileExistsError('Prior/uncertain model calls must never be repeated')
    receipt=execute(cases,args.evidence,args.phase)
    print('PHASE_DONE',args.phase,receipt['status'],receipt['passed'],'of',len(cases),
          'attempted',len(receipt['results']),'requests',receipt['model_requests'],
          'p95_observed',receipt['response_p95_seconds'],flush=True)
    return 0 if receipt['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
