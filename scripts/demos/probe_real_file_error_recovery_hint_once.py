#!/usr/bin/env python3
"""One-use *model-only* test of recovery guidance on an actual failed tool context.

Read an already completed/aborted hash-pinned developer journal. Take its
second genuine action-model request, which has a real missing-file tool_error
in its context, and APPEND new explicit no-action recovery advice.
Send exactly ONE NEW AND DISTINCT local inference to an existing resident
private model. Never execute even a valid proposed tool, restart any model,
replay old model requests or touch the original durable session. A proposed
listing is not itself a real source read or verified autonomous recovery.
The raw prompt, model output and token timing are all PRIVATE evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

JOURNAL_SHA='c0f70b01637b77467fe988ed00e55215b7573f6ed7635dea41f2f3f166738f01'
MODEL_ENDPOINT='http://10.8.0.7:14833/completion'
MODEL_CATALOG='http://10.8.0.7:14833/v1/models'
EXPECTED_MODEL_ALIAS='qwen3-coder-30b-a3b-instruct-q4-k-m'
ERROR_PATH='tests/test_voice_classification_legacy.py'
OPTIN='SWAAG_PRIVATE_FRESH_RECOVERY_HINT_ONCE'

RECOVERY_HINT=(
    "\n\n[NEW_EXPLICIT_MODEL_RECOVERY_GUIDANCE_NOT_PRESENT_IN_ORIGINAL_MODEL_REQUEST]\n"
    "The concrete previous read_file invocation FAILED mechanically: FilesystemError, "
    "requested path tests/test_voice_classification_legacy.py does NOT exist. "
    "The exact failure is already visible in your context. A subsequent identical read "
    "cannot succeed unless independent evidence shows the file was created, which has not happened. "
    "Choose the available list_files tool for the tests directory to find the real source path, "
    "or report the missing file from evidence. Do not invent a successful result. "
    "Never repeat an unverified Send, provider, or other external side effect. "
    "This is a new diagnostic action choice; no selected tool will actually execute.\n"
)


def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def read_original_request(journal:Path)->tuple[dict,dict]:
    if journal.is_symlink() or not journal.is_file() or sha(journal.read_bytes())!=JOURNAL_SHA:
        raise ValueError('Actual previous one-use journal source differs from pinned immutable result')
    rows=[json.loads(s) for s in journal.read_text().splitlines() if s.strip()]
    errors=[r for r in rows if r['type']=='tool_error'
        and (r.get('payload') or {}).get('tool_name')=='read_file'
        and (r.get('payload') or {}).get('error_type')=='FilesystemError'
        and ((r.get('payload') or {}).get('tool_input') or {}).get('path')==ERROR_PATH]
    if len(errors)!=4:raise ValueError('Missing exact original four real failures')
    reqs=[r for r in rows if r['type']=='model_request_sent']
    if len(reqs)!=5:raise ValueError('Original model request count drifted')
    original=reqs[1]
    if not errors[0]['sequence']<original['sequence']<errors[1]['sequence']:
        raise ValueError('Model request was not the immediately measured post-error context')
    request=(original['payload'] or {}).get('request')
    if not isinstance(request,dict):raise ValueError('Original model request not recorded')
    prompt=request.get('prompt')
    if (not isinstance(prompt,str) or 'FilesystemError' not in prompt
        or ERROR_PATH not in prompt or not ('list_files' in prompt)):
        raise ValueError('Actual model input lacks essential missing-file evidence and tool catalog')
    if not isinstance(request.get('json_schema'),dict):
        raise ValueError('Missing real structured SWAAG action schema')
    if 'n_predict' not in request:
        raise ValueError('Original action completion budget missing')
    return request,{'original_sequence':original['sequence'],
      'original_request_sha256':sha(json.dumps(request,sort_keys=True).encode()),
      'original_context_prompt_sha256':sha(prompt.encode()),
      'genuine_tool_error_sequence':errors[0]['sequence'],
      'real_file_error_count':len(errors)}


def modified_request(original:dict)->dict:
    new=dict(original)
    new['prompt']=original['prompt']+RECOVERY_HINT
    new['n_predict']=min(768,int(original['n_predict']))
    new['seed']=20261123065
    new['temperature']=0.0
    new['top_p']=1.0
    new.pop('stream',None)
    if sha(new['prompt'].encode())==sha(original['prompt'].encode()):
        raise ValueError('Refusing replay of an identical model prompt')
    if not (0<new['n_predict']<=768):raise ValueError('Unbounded model output')
    if set(new)-set(original):
        raise ValueError('Unexpected new model request parameters')
    return new


def create_unique(path:Path,data:dict)->None:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)
        f.flush();os.fsync(f.fileno())


def evaluate_proposal(text:str)->dict:
    try:parsed=json.loads(text)
    except (TypeError,ValueError):
        return {'valid_json':False,'has_proposed_listing':False,'error':'invalid_model_action_json'}
    if not isinstance(parsed,dict):
        return {'valid_json':False,'has_proposed_listing':False,'error':'not_action_object'}
    calls=parsed.get('tool_calls')
    if not isinstance(calls,list):
        return {'valid_json':True,'has_proposed_listing':False,'error':'tool_calls_missing'}
    options=[]
    for tool in calls:
        if not isinstance(tool,dict):continue
        name=tool.get('tool_name')
        args=tool.get('arguments')
        if not isinstance(name,str) or not isinstance(args,dict):continue
        options.append({'tool':name,'arguments':args})
    return {'valid_json':True,'has_proposed_listing':any(x['tool']=='list_files'
          and x['arguments']=={'path':'tests'} for x in options),
          'proposed_tool_names':[x['tool'] for x in options],
          'proposed_tools_executed':0}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    original,info=read_original_request(args.journal)
    unique=modified_request(original)
    if not args.run:
        print(json.dumps({'status':'offline_preflight_only','original_journal_sha':JOURNAL_SHA,
          'true_previous_missing_file_errors':info['real_file_error_count'],
          'new_prompt_sha':sha(unique['prompt'].encode()),'original_prompt_sha':info['original_context_prompt_sha256'],
          'model_calls':0,'tools_called':0}))
        return 0
    if os.environ.get(OPTIN)!='1':
        raise ValueError('Explicit opt-in required for distinct private model-only diagnostic')
    if (not args.evidence.is_absolute() or args.evidence.is_symlink() or
        args.evidence.exists() or '/data/var/swaag-benchmarks/' not in str(args.evidence)):
        raise ValueError('Destination must be a new private absolute SWAAG evidence directory')
    args.evidence.mkdir(mode=0o700,parents=True,exist_ok=False)
    os.chmod(args.evidence,0o700)
    started=datetime.now(timezone.utc).isoformat()
    info.update({'status':'one_new_model_call_about_to_start','started_utc':started,
        'source_existing_model_url':MODEL_ENDPOINT,
        'model_context_prompt_sha256':sha(unique['prompt'].encode()),
        'recovery_hint_sha256':sha(RECOVERY_HINT.encode()),
        'new_distinct_seed':unique['seed'],'model_output_token_cap':unique['n_predict'],
        'model_calls_admitted':1,'tools_allowed_to_execute':[],
        'original_request_not_replayed':True,'previous_model_session_untouched':True})
    create_unique(args.evidence/'attempt.json',info)
    # This may timeout or be interrupted; a single preexisting attempt record
    # permanently forbids an automatic retry of uncertain model output.
    result={'status':'failed_or_unknown','started_utc':started,
        'model_request_sha256':sha(json.dumps(unique,sort_keys=True).encode()),
        'request_distinct_from_original':True,'tools_executed':0,
        'production_service_or_user_session_changed':False}
    wall=time.perf_counter()
    try:
        # Read-only model identity check: never silently send the private
        # source action prompt to a provider or wrong self-hosted model.
        with urllib.request.urlopen(MODEL_CATALOG,timeout=4) as info_http:
            if info_http.status!=200:
                raise ValueError('Resident model identity HTTP request failed')
            catalogue=json.loads(info_http.read(32000))
        names=[x.get('id') for x in (catalogue.get('data') or [])]
        if names!=[EXPECTED_MODEL_ALIAS]:
            raise ValueError('The resident endpoint is not the pinned Coder model')
        result['model_catalogue_verified']=True
        data=json.dumps(unique,ensure_ascii=False).encode()
        request=urllib.request.Request(MODEL_ENDPOINT,data=data,
            headers={'Content-Type':'application/json','Accept':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=115) as http:
            if http.status!=200:
                raise ValueError('Unexpected Coder private local model status')
            raw=http.read(1500000)
        response=json.loads(raw)
        content=response.get('content')
        if not isinstance(content,str) or len(content)>200000:
            raise ValueError('Unparseable or oversized model response')
        result.update({'status':'completed_model_only','http_status':200,
             'wall_seconds':round(time.perf_counter()-wall,3),
             'raw_private_model_content':content,
             'response_metadata':{k:response.get(k) for k in ('tokens_predicted','tokens_evaluated',
               'generation_settings','stop','stopped_eos','stopped_limit','timings') if k in response},
             'proposal':evaluate_proposal(content),
             'output_characters':len(content),
             'model_inference_only':True,'simulated_tool_receipts':False,
             'model_not_authorized_to_execute_any_tools':True,
             'actual_multistep_agent_success_claimed':False})
    except Exception as exc:
        result['error_type']=type(exc).__name__
        result['wall_seconds']=round(time.perf_counter()-wall,3)
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    create_unique(args.evidence/'result.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['wall_seconds'],
        'proposed_listing':(result.get('proposal') or {}).get('has_proposed_listing'),
        'proposed_tools_executed':0,'result_sha256':sha((args.evidence/'result.json').read_bytes())}))
    return 0 if result['status']=='completed_model_only' else 3

if __name__=='__main__':raise SystemExit(main())
