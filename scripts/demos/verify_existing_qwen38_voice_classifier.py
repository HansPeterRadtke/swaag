#!/usr/bin/env python3
"""One-use source-pinned no-tool voice classification using ALREADY running Qwen3.8 Q8.

This is NOT a new classifier service, installed runtime change, autonomous
SWAAG turn, voice sample or production-ready alternative. Executes the
currently installed Thor gateway's exact turn/interruption classification
functions in an isolated Python interpreter with its classifier pointed to
the EXISTING, healthy Qwen3.8 Q8 endpoint. The original thirty-five-case oracle is split into eight predeclared pilot
cases and twenty-seven distinct remaining cases. No case or uncertain outcome
is repeated for this model. A separate thirty-case holdout was fixed before
this pilot and is withheld until the model qualifies under the earlier tests.
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
import urllib.request

BASE=Path('/data/var/swaag-benchmarks/owner-serial-000063-20261123')
GATEWAY_DIR=Path('/data/src/worktrees/voice-guideline-repair-20261012/voice_agent_live_runtime')
CONFIG=Path('/etc/swaag/voice-gateway.toml')
ORIGINAL=Path('/data/src/github/swaag/src/swaag/benchmark/fixtures/voice_classifier_oracle_20261104.json')
QWEN_URL='http://10.8.0.7:14835/v1/chat/completions'
QWEN_MODEL='qwen38-27b-q8-full-262k'
OPT_IN='SWAAG_QWEN38_CLASSIFIER_NOACTION'
EXPECTED_HASHES={
 'gateway.py':'23d5206072297c5910dc3d2fb98aa7136308a5fc96471af14843df7f3130de78',
 'gateway_config.py':'38439178736d363f73ef9d63e1162495b8f26d7c684c9b4766a9e4bcc3ed10ee',
 'gateway_runtime_toml':'8e9a1fa4593ec26b6b21b87004eda81a64a757430961024df52ab9a5964e1226',
 'original_oracle':'929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e',
 'nov23_holdout':'dd8ef8f94f0bee6624013462c6f314da52f8e33cdef7b0e09d66addd71c107fc',
}
QWEN_PILOT_ORDER=[
 'base_turn_completion_unfinished_object',
 'base_turn_completion_hold_command',
 'initial_interruption_ack',
 'base_turn_completion_german_partial',
 'base_turn_completion_completed_correction',
 'acknowledgment_interruption_correct',
 'base_turn_completion_german_finished',
 'initial_interruption_correction',
]
QWEN_PILOT_IDS=set(QWEN_PILOT_ORDER)


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
    historical=json.loads(ORIGINAL.read_text())['cases']
    if len(historical)!=35 or len({x['case_id'] for x in historical})!=35:
        raise ValueError('Historical voice classifier oracle unexpectedly changed')
    by_id={x['case_id']:x for x in historical}
    if not QWEN_PILOT_IDS.issubset(by_id):
        raise ValueError('Missing fixed predeclared pilot inputs')
    if phase=='pilot8':
        selected=[by_id[k] for k in QWEN_PILOT_ORDER]
    elif phase=='remaining27':
        selected=[c for c in historical if c['case_id'] not in QWEN_PILOT_IDS]
        if len(selected)!=27:
            raise ValueError('Expected exactly 27 fresh original cases')
    elif phase=='fresh_holdout30':
        selected=json.loads(holdout.read_text())['cases']
        if len(selected)!=30 or any(x['case_id'] in QWEN_PILOT_IDS for x in selected):
            raise ValueError('Predeclared holdout changed')
    else:raise ValueError('Unknown one-use classifier evaluation phase')
    for case in selected:
        if (not isinstance(case,dict) or type(case.get('expected')) is not bool
            or case.get('family') not in ('turn_completion','interruption')
            or not isinstance(case.get('text'),str) or not case['text'].strip()
            or len(case['text'])>500):
            raise ValueError('Malformed or unbounded fixed decision oracle')
    if len({x['case_id'] for x in selected})!=len(selected):
        raise ValueError('Duplicate case identities are forbidden')
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
    if installed.strong.health_url!='http://10.8.0.7:14835/v1/models':
        raise ValueError('Expected already-running strong model configuration changed')
    if installed.foreground.chat_url!='http://10.8.0.7:14833/v1/chat/completions':
        raise ValueError('SWAAG foreground routing unexpectedly changed')
    # Confirm the exact already-running strong model instead of loading or
    # silently redirecting a different resident model instance.
    with urllib.request.urlopen(installed.strong.health_url,timeout=3) as response:
        model_list=json.loads(response.read(64000))
    names=[item.get('id') for item in model_list.get('data',[])]
    if names!=[QWEN_MODEL]:
        raise ValueError('The running strong-model identity differs from the pinned alias')
    gateway.CONFIG=replace(installed,classifier=config.ClassifierConfig(
        chat_url=QWEN_URL,health_url='http://10.8.0.7:14835/health',
        model=QWEN_MODEL,request_timeout_seconds=8.0,enable_thinking=False))
    original_json_http=gateway.json_http
    count={'calls':0}
    def strict_noaction_model_http(url,payload=None,**kwargs):
        if (url!=QWEN_URL or not isinstance(payload,dict) or
                payload.get('model')!=QWEN_MODEL or payload.get('stream') is not False
                or 'tools' in payload or len(payload.get('messages',[]))!=2):
            raise ValueError('Only fixed Qwen3.8 Q8 no-tool semantic inference is permitted')
        count['calls']+=1
        return original_json_http(url,payload,**kwargs)
    gateway.json_http=strict_noaction_model_http
    return gateway,check_oracle_result,count


def execute(cases:list[dict],out:Path,phase:str)->dict:
    receipt={'status':'started','started_utc':datetime.now(timezone.utc).isoformat(),
       'phase':phase,'expected_cases':len(cases),'case_ids':[c['case_id'] for c in cases],
       'model':QWEN_MODEL,'model_endpoint':QWEN_URL,
       'fixed_source_sha256':EXPECTED_HASHES,
       'no_tools_or_SWAAG_actions':True,'no_running_gateway_mutation':True,
       'no_new_models_loaded':True,'remaining_cases_only':phase=='remaining27',
       'pilot_cannot_be_replayed_on_later_phases':True,'results':[]}
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
        if consecutive_invalid>=2 or (event['classifier_valid'] and not event['passed']):
            receipt['status']=('aborted_due_to_consecutive_invalid_classifier_responses'
                               if consecutive_invalid>=2
                               else 'disqualified_on_incorrect_real_model_decision')
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



def pilot_permits_extended_inference() -> dict:
    """Never spend more shared-model inference after an unqualified pilot.

    The provisional two-second ninety-fifth-percentile decision budget is
    an evaluation gate, not an owner-specified production service guarantee.
    A new source/model/config revision would need a distinct new experiment;
    earlier evidence must never be overwritten or the same inputs replayed.
    """
    file=BASE/'qwen38-pilot8-original-noaction-result.json'
    if file.is_symlink() or not file.is_file() or file.stat().st_size>200000:
        raise ValueError('Missing unique completed source-pinned Qwen3.8 pilot receipt')
    receipt=json.loads(file.read_text())
    entries=receipt.get('results') or []
    if (receipt.get('status')!='completed' or receipt.get('phase')!='pilot8'
        or receipt.get('model')!=QWEN_MODEL or receipt.get('model_endpoint')!=QWEN_URL
        or receipt.get('fixed_source_sha256')!=EXPECTED_HASHES
        or receipt.get('passed')!=8 or receipt.get('failed')!=0
        or receipt.get('model_requests')!=8 or len(entries)!=8
        or [x.get('case_id') for x in entries]!=QWEN_PILOT_ORDER
        or any(x.get('state')!='finished' or x.get('passed') is not True
               or x.get('classifier_valid') is not True for x in entries)):
        raise ValueError('Pilot result is not the exact successful no-action validation')
    p95=receipt.get('response_p95_seconds')
    if (type(p95) not in (float,int) or p95>2.0 or p95<0):
        raise ValueError('Pilot latency exceeds provisional two-second voice-turn budget; do not consume more model calls')
    return {'source_pinned_pilot_passed':8,'p95_seconds':p95,
            'provisional_max_p95_seconds':2.0}


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['pilot8','remaining27','fresh_holdout30'],required=True)
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
    if args.phase!='pilot8':
        pilot_permits_extended_inference()
    if not args.evidence.is_absolute() or not str(args.evidence).startswith(str(BASE)+'/'):
        raise ValueError('One-use receipt outside the project-owned private directory')
    if args.evidence.is_symlink() or args.evidence.exists():
        raise FileExistsError('Prior/uncertain Qwen3.8 model inference must never be repeated')
    receipt=execute(cases,args.evidence,args.phase)
    print('PHASE_DONE',args.phase,receipt['status'],receipt['passed'],'of',len(cases),
          'attempted',len(receipt['results']),'requests',receipt['model_requests'],
          'p95_observed',receipt['response_p95_seconds'],flush=True)
    return 0 if receipt['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
