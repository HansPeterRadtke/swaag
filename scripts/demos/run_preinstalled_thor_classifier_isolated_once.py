#!/usr/bin/env python3
"""Self-contained no-action local turn classifier candidate benchmark on Thor.

Only an already installed SHA-pinned model is considered. Runs a short-lived
llama-server bound to 127.0.0.1:14921; uses the exact published SWAAG voice
classifier oracle against the genuine Thor gateway function; never restarts,
changes, or talks to a live SWAAG/Jetson voice service or user/phone.

Requires prior broker reservation and a unique evidence directory. The script
stops and reaps ONLY its own child model and benchmark processes, even after
errors. Unknown outcomes are one-use, never automatically retried.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/data/var/swaag-benchmarks/owner-serial-000067-20261124')
MODEL = Path('/data/models/vlm/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf')
MODEL_SHA = 'd02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12'
MODEL_BYTES = 1929901056
SERVER = Path('/data/src/external/llama.cpp-upstream/build-sm110/bin/llama-server')
BENCH_SHA = 'ed10fb53850003a463d9a01f99c06ea4daf17ddbfb55910cdb3e59522534e947'
ORACLE_SHA = '929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e'
GATEWAY_DIR = Path('/data/src/worktrees/voice-guideline-repair-20261012/voice_agent_live_runtime')
GATEWAY_SOURCE_SHA = '23d5206072297c5910dc3d2fb98aa7136308a5fc96471af14843df7f3130de78'
GATEWAY_CONFIG_SOURCE_SHA = '38439178736d363f73ef9d63e1162495b8f26d7c684c9b4766a9e4bcc3ed10ee'
CONFIG = Path('/etc/swaag/voice-gateway.toml')
MODEL_ALIAS = 'swaag-qwen25vl-3b-isolated-20261124'
HOST = '127.0.0.1'
PORT = 14921
URL = f'http://{HOST}:{PORT}'
OPTIN = 'SWAAG_ISOLATED_THOR_CLASSIFIER_NOACTION_W67'
BENCH = ROOT/'voice_classifier_acceptance.py'
ORACLE = ROOT/'voice_classifier_oracle_20261104.json'


def sha(path:Path)->str:
    value=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024*1024):value.update(data)
    return value.hexdigest()


def exclusive_json(path:Path,obj:dict)->None:
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())


def preflight()->dict:
    if not ROOT.is_dir() or ROOT.is_symlink():raise RuntimeError('Private SWAAG benchmark root missing')
    pins={MODEL:(MODEL_SHA,MODEL_BYTES),BENCH:(BENCH_SHA,None),ORACLE:(ORACLE_SHA,None),
          GATEWAY_DIR/'voice_agent_gateway.py':(GATEWAY_SOURCE_SHA,None),
          GATEWAY_DIR/'voice_agent_gateway_config.py':(GATEWAY_CONFIG_SOURCE_SHA,None)}
    for p,(want,size) in pins.items():
        if not p.is_file() or p.is_symlink() or (size is not None and p.stat().st_size!=size):
            raise RuntimeError('Missing/mismatched pinned model/source: '+p.name)
        if sha(p)!=want:raise RuntimeError('Pinned model/benchmark/gateway bytes changed: '+p.name)
    if not SERVER.is_file() or not os.access(SERVER,os.X_OK) or SERVER.is_symlink():
        raise RuntimeError('Preinstalled GPU llama.cpp server missing')
    if not CONFIG.is_file():raise RuntimeError('Original gateway config missing')
    model_info=json.loads(ORACLE.read_text())
    if len(model_info.get('cases',[]))!=35:
        raise RuntimeError('Original real 35-case turn classifier oracle changed')
    if (sum(v.get('family')=='turn_completion' for v in model_info['cases'])!=17 or
        sum(v.get('family')=='interruption' for v in model_info['cases'])!=18):
        raise RuntimeError('Invalid source-grounded family counts')
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((HOST,PORT))==0:
            raise RuntimeError('Dedicated private model port was already occupied')
    for name in ('attempt.json','result.json','isolated-model.log','isolated-benchmark-console.log','model-health-receipt.json','voice-classifier-35-case.json'):
        if (ROOT/name).exists():raise RuntimeError('Existing/uncertain classifier attempt: never replay')
    mem_available_kib=0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            mem_available_kib=int(line.split()[1]);break
    if mem_available_kib<8*1024*1024:raise RuntimeError('Insufficient free unified RAM before new model')
    return {'model_path':str(MODEL),'model_sha256':MODEL_SHA,
      'model_bytes':MODEL_BYTES,'source_sha256':BENCH_SHA,'oracle_sha256':ORACLE_SHA,
      'gateway_sha256':GATEWAY_SOURCE_SHA,'gateway_parser_sha256':GATEWAY_CONFIG_SOURCE_SHA,
      'llama_binary_sha256':sha(SERVER),'gateway_config_sha256':sha(CONFIG),
      'model_alias':MODEL_ALIAS,'localhost_only':URL,
      'oracle_cases':35,'turn_completion_cases':17,'interruption_cases':18,
      'unified_mem_available_gib':round(mem_available_kib/1024**2,2),
      'shared_services_untouched':True,'external_provider_calls':0,'phone_actions':0}


def request_json(url:str,body:dict|None,timeout:float)->tuple[int,dict]:
    data=json.dumps(body).encode('utf8') if body is not None else None
    headers={'Content-Type':'application/json'} if data is not None else {}
    req=urllib.request.Request(url,data=data,method='POST' if data is not None else 'GET',headers=headers)
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        return int(resp.status),json.loads(resp.read(64000))


def run(args:argparse.Namespace)->int:
    pins=preflight()
    if not args.run:
        print('ISOLATED_MODEL_PRECHECK_NO_ACTION',json.dumps(pins));return 0
    if os.getenv(OPTIN)!='1':raise RuntimeError('Explicit one-use model startup opt-in required')
    if not args.broker_lease_id or not args.broker_lease_id.startswith('rb_'):
        raise RuntimeError('A real previously granted broker lease ID is required')
    info={**pins,'broker_lease_id':args.broker_lease_id,
      'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'model_spawn_attempted':True,'one_use_no_action':True,
      'remote_model_provider_requests':0,'no_phone_actions':True}
    exclusive_json(ROOT/'attempt.json',info)
    started=time.monotonic();server=None;benchproc=None
    output={'status':'unknown_or_failed','attempt_sha256':sha(ROOT/'attempt.json'),
      'model_alias':MODEL_ALIAS,'separate_loopback_port':PORT,
      'no_production_configuration_or_service_change':True,
      'broker_lease_id':args.broker_lease_id,'model_started':False,
      'model_stopped':False,'benchmark_started':False,'benchmark_exit_code':None}
    model_log=ROOT/'isolated-model.log'
    console=ROOT/'isolated-benchmark-console.log'
    try:
        command=[str(SERVER),'--model',str(MODEL),'--alias',MODEL_ALIAS,
         '--host',HOST,'--port',str(PORT),'--ctx-size','2048',
         '--n-gpu-layers','99','--parallel','1','--threads','2',
         '--threads-batch','2','--no-warmup']
        with model_log.open('xb') as log:
            server=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
              stderr=subprocess.STDOUT,start_new_session=True,
              env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'})
        output['model_started']=True
        output['own_model_child_pid']=server.pid
        output['model_command']=command
        status=None
        for _ in range(75):
            if server.poll() is not None:raise RuntimeError('Isolated model child exited before health')
            try:
                code,data=request_json(URL+'/health',None,timeout=2)
                if code==200 and data.get('status')=='ok':status=data;break
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,OSError):pass
            if time.monotonic()-started>105:raise TimeoutError('Model startup health window elapsed')
            time.sleep(1.5)
        if status is None:raise RuntimeError('Model health not 200 ready')
        code,catalog=request_json(URL+'/v1/models',None,timeout=4)
        identities=[x.get('id') for x in catalog.get('data',[])]
        if code!=200 or identities!=[MODEL_ALIAS]:raise RuntimeError('Isolated model alias mismatch')
        output['model_health_ok']=True
        output['model_ready_elapsed_seconds']=round(time.monotonic()-started,3)
        exclusive_json(ROOT/'model-health-receipt.json',{'status':'ready',
          'model_id':MODEL_ALIAS,'elapsed_seconds':output['model_ready_elapsed_seconds']})
        # One unique warmup request: does NOT use any oracle case and does
        # not touch SWAAG's live gateway. Allows a cold model to initialize.
        unique='For a unit test only: Is the statement "This is a one-use diagnostic." a finished utterance?'
        preflight_request={
          'model':MODEL_ALIAS,'stream':False,'max_tokens':48,'temperature':0.0,
          'messages':[{'role':'system','content':'Return a short JSON object only with finished as a boolean.'},
                      {'role':'user','content':unique}],
          'response_format':{'type':'json_object'}}
        t=time.perf_counter()
        code,warm=request_json(URL+'/v1/chat/completions',preflight_request,timeout=35)
        if code!=200 or not warm.get('choices'):
            raise RuntimeError('No genuine isolated warmup model answer')
        content=warm['choices'][0]['message']['content']
        output['warmup_model_response_seconds']=round(time.perf_counter()-t,3)
        output['warmup_response_sha256']=hashlib.sha256(str(content).encode()).hexdigest()
        output['warmup_model_response_observed']=True
        output['warmup_not_an_oracle_case']=True
        # Real SWAAG classification from published source, 35 cases, one
        # preexisting third-party GGUF. Strict schema and 8 second timeout.
        cmd=[sys.executable,str(BENCH),'--gateway-source-dir',str(GATEWAY_DIR),
         '--gateway-config',str(CONFIG),'--model-origin',URL,
         '--model-identity',MODEL_ALIAS,'--oracle',str(ORACLE),
         '--output',str(ROOT/'voice-classifier-35-case.json'),'--family','all']
        output['benchmark_started']=True
        output['benchmark_command']=cmd
        with console.open('xb') as out:
            benchproc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
                     stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                     env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            output['own_benchmark_child_pid']=benchproc.pid
            benchproc.wait(timeout=400)
            output['benchmark_exit_code']=benchproc.returncode
        if (ROOT/'voice-classifier-35-case.json').is_file():
            tested=json.loads((ROOT/'voice-classifier-35-case.json').read_text())
            output['actual_tested_cases']=len(tested.get('cases') or [])
            output['actual_passed_cases']=sum(x.get('passed') is True for x in tested.get('cases') or [])
            output['actual_invalid_classifier_responses']=sum(x.get('classifier_valid') is False for x in tested.get('cases') or [])
            output['actual_exceptions']=sum('exception' in x for x in tested.get('cases') or [])
            output['benchmark_file_sha256']=sha(ROOT/'voice-classifier-35-case.json')
            output['observed_benchmark_status']=tested.get('status')
            output['observed_model_case_elapsed_seconds']=[x.get('elapsed_seconds') for x in tested.get('cases') or []]
            output['status']=('completed_full_35_case_benchmark' if output['actual_tested_cases']==35 else 'incomplete_benchmark')
        else:output['status']='benchmark_missing_output'
    except Exception as exc:
        output['error_type']=type(exc).__name__
        output['error_description_safe']=str(exc)[:210]
    finally:
        # Must not terminate any other process. Both are our own Popen kids.
        for child,name in ((benchproc,'benchmark'),(server,'model')):
            if child is None:continue
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=8)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=8)
            output[f'{name}_exit_code']=child.returncode
            output[f'{name}_stopped']=child.poll() is not None
        output['elapsed_seconds']=round(time.monotonic()-started,3)
        output['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        output['no_model_replay_attempts']=True
        if model_log.is_file():output['model_log_sha256']=sha(model_log)
        if console.is_file():output['benchmark_console_sha256']=sha(console)
        exclusive_json(ROOT/'result.json',output)
    print('ISOLATED_REAL_LOCAL_CLASSIFIER_BENCHMARK',json.dumps({
      k:output.get(k) for k in ('status','elapsed_seconds','model_ready_elapsed_seconds',
          'warmup_model_response_seconds','actual_tested_cases','actual_passed_cases',
          'actual_invalid_classifier_responses','benchmark_exit_code',
          'model_stopped','benchmark_stopped','error_type')}),flush=True)
    return 0 if output['status']=='completed_full_35_case_benchmark' else 3


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--broker-lease-id',default='')
    args=p.parse_args(argv)
    return run(args)

if __name__=='__main__':raise SystemExit(main())
