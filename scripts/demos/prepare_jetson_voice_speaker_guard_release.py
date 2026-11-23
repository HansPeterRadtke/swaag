#!/usr/bin/env python3
"""Strict source-pinned Jetson voice release PREVIEW, not an installer.

This tool NEVER writes to /etc, changes systemd, sends speech, or executes a
SWAAG, model, TTS, or agent request. It snapshots exact currently effective
unit identity, deployed source hashes, config SHA and read-only health. With
explicit opt-in it creates a private, immutable-hash-verified candidate tree,
preview-only systemd drop-in and a rollback specification. The preview is
NOT an authorization to deploy, and never edits the live unit. Existing active
sessions, altered source/config/unit or degraded agent gateway always mean
NO-GO. Any deployment later needs separate authorization and checks again.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import urllib.error
import urllib.request

from verify_isolated_voice_release_server import (
    EXPECTED_SOURCES,EXPECTED_TEMPLATE_SHA,inspect as inspect_candidate,
)

OPT_IN='SWAAG_SPEAKER_RELEASE_PREVIEW_NO_INSTALL'
EXPECTED_LIVE_ROOT='/data/src/worktrees/voice-sse-durable-fastpath-20261022/voice_agent_live_runtime'
EXPECTED_CURRENT_FILES={
 'cloudflare_ws_voice_server.py':'561183f64ae76a9aa92e18b8f2a8c1192b7f9fecc7597233bb363bb64e85415f',
 'voice_ws_config.py':'63111aaff04f5e0bf57abf7689fff97a5ac5a612204487f5e17682aea04c450b',
 **{k:v for k,v in EXPECTED_SOURCES.items() if k not in ('cloudflare_ws_voice_server.py','voice_ws_config.py')},
}
EXPECTED_UNIT='jetson-voice-ws.service'
EXPECTED_CONFIG='/etc/voice-agent/ws.json'
EXPECTED_HEALTH='http://thor:13484/health'
EXPECTED_SERVICE_HEALTH='http://127.0.0.1:13483/health'
EXPECTED_DROPIN_OLD='/etc/systemd/system/jetson-voice-ws.service.d/60-sse-durable-status.conf'
PREVIEW_DROPIN_NAME='70-swaag-verified-speaker-review.conf'
PYTHON='/data/src/github/devtests/voice_agent_live_runtime/native_streaming_stt/.venv/bin/python'


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def check_unchanged_deployed_source(live:Path)->dict:
    if str(live)!=EXPECTED_LIVE_ROOT or live.is_symlink() or not live.is_dir():
        raise ValueError('Not the original pinned installed voice source directory')
    result={}
    for name,want in EXPECTED_CURRENT_FILES.items():
        p=live/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size>260000:
            raise ValueError('Missing or symlinked original deployed voice module: '+name)
        observed=sha(p.read_bytes())
        if observed!=want:raise ValueError('Existing production voice source changed: '+name)
        result[name]=observed
    return result


def effective_unit_view(service:str=EXPECTED_UNIT)->dict:
    if service!=EXPECTED_UNIT:raise ValueError('Only the exact named Jetson voice service may be inspected')
    result=subprocess.run(['systemctl','show','--no-pager',service,'--property=ActiveState,SubState,MainPID,WorkingDirectory,ExecStart,FragmentPath,DropInPaths,ExecMainStartTimestamp,NeedDaemonReload'],
                          text=True,capture_output=True,timeout=6,check=True)
    data={}
    for line in result.stdout.splitlines():
        if '=' in line:
            k,v=line.split('=',1)
            data[k]=v.strip()
    for k in ('ActiveState','MainPID','WorkingDirectory','ExecStart','FragmentPath','DropInPaths','NeedDaemonReload'):
        if k not in data:raise ValueError('Incomplete effective systemd unit: '+k)
    return data


def verify_unit(unit:dict,live:Path)->dict:
    if unit['ActiveState']!='active' or int(unit['MainPID'])<=0:
        raise ValueError('Installed service is not actually active')
    if unit['NeedDaemonReload']!='no':
        raise ValueError('Effective systemd unit has unloaded on-disk changes; no preview admission')
    if unit['WorkingDirectory']!=str(live):
        raise ValueError('Effective service directory differs from verified original')
    start=unit['ExecStart']
    executable=str(live/'cloudflare_ws_voice_server.py')
    if (PYTHON not in start or executable not in start
        or '--config '+EXPECTED_CONFIG not in start):
        raise ValueError('Effective systemd ExecStart is not the pinned production server')
    if unit['FragmentPath']!='/etc/systemd/system/'+EXPECTED_UNIT:
        raise ValueError('Unexpected actual installed systemd unit fragment')
    drops=unit['DropInPaths'].split()
    if EXPECTED_DROPIN_OLD not in drops:
        raise ValueError('The expected previous 60-override is not in the effective unit')
    if any(p.endswith('/'+PREVIEW_DROPIN_NAME) for p in drops):
        raise ValueError('The preview override is already deployed; do not stage or replay')
    if any(not p.startswith('/etc/systemd/system/jetson-voice-ws.service.d/') for p in drops):
        raise ValueError('Unexpected effective systemd drop-in outside the voice service')
    return {'name':EXPECTED_UNIT,'actual_main_pid':int(unit['MainPID']),
            'effective_source_root':str(live),'source_version':'4.0_durable_voice_commit',
            'unit_fragment':unit['FragmentPath'],'overrides':drops,
            'systemd_effective_exec_start_sha256':sha(start.encode()),
            'systemd_effective_fragment_sha256':sha(json.dumps(unit,sort_keys=True).encode())}


def verify_locked_config(template:dict)->None:
    server=template.get('server') or {}
    interaction=template.get('interaction') or {}
    llm=template.get('llm') or {}
    if (server.get('host')!='127.0.0.1' or server.get('port')!=13483
        or server.get('websocket_path')!='/ws'):
        raise ValueError('Production listener is not pinned to localhost 13483/ws')
    if llm.get('health_url')!=EXPECTED_HEALTH:
        raise ValueError('Production agent dependency endpoint changed')
    if interaction.get('require_verified_speaker_for_auto_send',True) is not True:
        raise ValueError('Config disables unknown-speaker review; deployment forbidden')
    if interaction.get('auto_send_default') is not True:
        raise ValueError('Expected legacy automatic-send mode changed; repeat safety acceptance')
    if server.get('auth_token_file')!='/etc/voice-agent/ws.token':
        raise ValueError('Production owner credential path differs from expected protected configuration')


def verify_private_token_stat(template:dict)->dict:
    # Only stat metadata, NEVER read or copy bearer bytes into evidence.
    path=Path(template['server']['auth_token_file'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('Original production token missing or symlinked')
    meta=path.stat()
    permission=stat.S_IMODE(meta.st_mode)
    if permission not in (0o400,0o600) or not os.access(path,os.R_OK):
        raise ValueError('Original private bearer is not exclusively readable by service user')
    return {'credential_exists':True,'credential_mode':oct(permission),
            'bearer_contents_read_or_saved':False}


def health_one(url:str)->dict:
    if url not in (EXPECTED_HEALTH,EXPECTED_SERVICE_HEALTH):
        raise ValueError('Not one of the two fixed read-only health endpoints')
    request=urllib.request.Request(url,headers={'Accept':'application/json'},method='GET')
    try:
        response=urllib.request.urlopen(request,timeout=4)
    except urllib.error.HTTPError as exc:
        response=exc
    with response as stream:
        status=int(stream.status)
        data=stream.read(64000)
    obj=json.loads(data)
    if not isinstance(obj,dict):raise ValueError('Health was not a JSON object')
    return {'http_status':status,'body':obj}


def classify_health(gateway:dict,live:dict,unit:dict)->dict:
    reasons=[]
    gw=gateway.get('body') or {}
    components=live.get('body') or {}
    if gateway.get('http_status')!=200 or gw.get('ok') is not True:
        if isinstance(gw.get('classifier_model'),dict) and gw['classifier_model'].get('ok') is False:
            reasons.append('dedicated_turn_classifier_offline')
        else:reasons.append('thor_agent_gateway_not_ready')
    if gw.get('swaag',{}).get('ok') is False:
        reasons.append('swaag_runtime_unavailable')
    if live.get('http_status')!=200 or components.get('live') is not True:
        reasons.append('voice_server_liveness_not_confirmed')
    if components.get('ready') is not True or components.get('components',{}).get('agent') is not True:
        reasons.append('voice_server_end_to_end_readiness_false')
    if components.get('config_sha256')!=EXPECTED_TEMPLATE_SHA:
        reasons.append('running_server_config_drift')
    for k in ('active_sessions','active_conversations','active_answer_jobs'):
        if not isinstance(components.get(k),int):
            reasons.append('voice_'+k+'_not_measurable')
        elif components[k]!=0:
            reasons.append('voice_'+k+'_not_idle')
    if unit['MainPID'].strip()!=str(components.get('main_pid',unit['MainPID'])):
        reasons.append('service_pid_identity_unexpected')
    return {'release_gate':'NO_GO' if reasons else 'TECHNICAL_PREFLIGHT_PASS_NOT_AUTHORIZED',
            'reasons':sorted(set(reasons)),'voice_active_sessions':components.get('active_sessions'),
            'voice_active_conversations':components.get('active_conversations'),
            'voice_active_answer_jobs':components.get('active_answer_jobs'),
            'backend_summary':{'foreground_available':bool(gw.get('fast_model',{}).get('ok')),
                 'background_available':bool(gw.get('strong_model',{}).get('ok')),
                 'SWAAG_reachable':bool(gw.get('swaag',{}).get('ok')),
                 'turn_classifier_available':bool(gw.get('classifier_model',{}).get('ok'))}}


def preview_override(candidate:Path)->str:
    if not candidate.is_absolute() or not str(candidate).startswith('/data/var/swaag-benchmarks/'):
        raise ValueError('Preview source must reside in private SWAAG staging storage')
    if not re.fullmatch(r'/data/var/swaag-benchmarks/[a-zA-Z0-9_/-]+',str(candidate)):
        raise ValueError('Unsafe or ambiguous candidate path in systemd preview')
    return ('# PREVIEW ONLY. NOT INSTALLED. Never deploy from temporary evidence storage.\n'
            '# Before any release, move the SHA-verified source to an approved persistent revisioned path.\n'
            '[Unit]\nRequiresMountsFor='+str(candidate)+'\n'
            '[Service]\nWorkingDirectory='+str(candidate)+'\n'
            'ExecStart=\nExecStart='+PYTHON+' '+str(candidate/'cloudflare_ws_voice_server.py')+' --config '+EXPECTED_CONFIG+'\n')


def verify_staged_preview_unit(staged:Path,evidence:Path)->dict:
    # Checks a standalone PREVIEW unit grammar, NOT the installed service.
    # No systemctl daemon-reload/install/restart or binding to live ports.
    unit=evidence/'private-preview-grammar-only.service'
    body=('[Unit]\nDescription=SWAAG private candidate parser verification only\n'
          +'RequiresMountsFor='+str(staged)+'\n'
          +'[Service]\nType=simple\nUser=hans\nGroup=hans\n'
          +'WorkingDirectory='+str(staged)+'\n'
          +'Environment=PYTHONDONTWRITEBYTECODE=1\n'
          +'ExecStart='+PYTHON+' '+str(staged/'cloudflare_ws_voice_server.py')
             +' --config '+EXPECTED_CONFIG+'\n'
          +'NoNewPrivileges=true\nPrivateTmp=true\nProtectHome=true\n')
    _write_unique(unit,body.encode())
    check=subprocess.run(['systemd-analyze','verify',str(unit)],
                         capture_output=True,text=True,timeout=18)
    _write_unique(evidence/'private-unit-parser-output.log',
                  (check.stdout+check.stderr)[:25000].encode())
    if check.returncode!=0:
        raise ValueError('The independent, noninstalled systemd preview syntax check failed')
    return {'offline_systemd_unit_parser_returncode':check.returncode,
            'standalone_preview_sha256':sha(body.encode()),
            'service_installed':False}


def stage_preview(candidate:Path,evidence:Path,template:dict,unit:dict,
                  health_gate:dict,source_hashes:dict)->dict:
    staged=evidence/'candidate';staged.mkdir(mode=0o700)
    for filename,want in EXPECTED_SOURCES.items():
        raw=(candidate/filename).read_bytes()
        if sha(raw)!=want:raise ValueError('Source changed during private stage')
        _write_unique(staged/filename,raw)
        if sha((staged/filename).read_bytes())!=want:
            raise ValueError('Staged candidate bytes mismatch')
    text=preview_override(staged)
    _write_unique(evidence/PREVIEW_DROPIN_NAME,text.encode())
    rollback={'not_executable_or_authorizing':True,
      'previous_live_source_root':EXPECTED_LIVE_ROOT,
      'previous_live_source_sha256':source_hashes,
      'previous_effective_working_directory':unit['WorkingDirectory'],
      'previous_effective_exec_start_sha256':sha(unit['ExecStart'].encode()),
      'config_sha256':EXPECTED_TEMPLATE_SHA,
      'existing_verified_last_override':EXPECTED_DROPIN_OLD,
      'no_unit_or_live_source_files_modified':True,
      'manual_rollback_contract':[
        'After separately authorized deployment, verify no active user session, submitted operation or queued response exists.',
        'Do not touch still-needed Qwen, ASR, TTS, broker or other services.',
        'Remove ONLY the future owned 70-swaag-verified-speaker-review.conf override if present and exact SHA verified.',
        'After separate authorization use systemctl daemon-reload and restart ONLY jetson-voice-ws.service with explicit idle gate.',
        'Independently confirm effective unit returns to the verified 60-sse-durable-status.conf source, prior config hash and expected process health.',
        'Never infer that the private preview was installed or executed.'
       ]}
    _write_unique(evidence/'rollback-manual.json',json.dumps(rollback,indent=2).encode())
    syntax=verify_staged_preview_unit(staged,evidence)
    return {'private_candidate_root':str(staged),'candidate_modules':len(EXPECTED_SOURCES),
      'independent_offline_systemd_parser':syntax,
      'candidate_source_hashes':EXPECTED_SOURCES,
      'preview_only_dropin':str(evidence/PREVIEW_DROPIN_NAME),
      'preview_only_dropin_sha256':sha(text.encode()),
      'rollback_spec':str(evidence/'rollback-manual.json'),
      'configured_legacy_auto_send':True,
      'guard_default_on_under_proposed_patch':True,
      'production_unit_config_or_service_changed':False,
      'release_gate':health_gate['release_gate']}


def _write_unique(path:Path,data:bytes):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def inspect_all(candidate:Path,live:Path,config:Path,service:str)->tuple[dict,dict,dict]:
    if str(config)!=EXPECTED_CONFIG or service!=EXPECTED_UNIT:
        raise ValueError('Live config or service identity not supported')
    template=inspect_candidate(candidate,config)
    verify_locked_config(template)
    verify_private_token_stat(template)
    source_hashes=check_unchanged_deployed_source(live)
    unit=effective_unit_view(service)
    identity=verify_unit(unit,live)
    return template,source_hashes,{'unit':unit,'identity':identity}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--live',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--service',default=EXPECTED_UNIT)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    template,hashes,unit=inspect_all(a.candidate,a.live,a.config,a.service)
    if not a.run:
        print(json.dumps({'status':'read_only_exact_service_source_preflight',
          'live_source_hashes_match':len(hashes),'candidate_hashes_match':len(EXPECTED_SOURCES),
          'effective_systemd_main_pid':unit['identity']['actual_main_pid'],
          'config_sha256':EXPECTED_TEMPLATE_SHA,'no_network_or_external_effects':True}))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('Explicit one-use PRIVATE preview opt-in required')
    if a.evidence.exists() or a.evidence.is_symlink():
        raise FileExistsError('Existing or uncertain release preview must never be overwritten')
    a.evidence.mkdir(mode=0o700,parents=True,exist_ok=False)
    os.chmod(a.evidence,0o700)
    _write_unique(a.evidence/'attempt.json',json.dumps({
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'source_pinned':True,'production_install_or_restart_allowed':False,
        'previous_service_main_pid':unit['identity']['actual_main_pid']}).encode())
    try:
        gateway=health_one(EXPECTED_HEALTH)
        server=health_one(EXPECTED_SERVICE_HEALTH)
        gate=classify_health(gateway,server,unit['unit'])
        staged=stage_preview(a.candidate,a.evidence,template,unit['unit'],gate,hashes)
        answer={'status':'private_release_preview_only','source_verification':True,
            'service_identity':unit['identity'],'health':gate,'staged':staged,
            'read_only_remote_health_get_count':2,'production_unchanged':True,
            'actual_release_deployed':False,'separate_owner_release_authorization_required':True}
    except BaseException as exc:
        answer={'status':'failed_or_uncertain','error_type':type(exc).__name__,
                'error_detail_private':str(exc)[:200],
                'actual_release_deployed':False,'production_unchanged':True}
    _write_unique(a.evidence/'result.json',json.dumps(answer,indent=2).encode())
    print(json.dumps({'status':answer['status'],'release_gate':answer.get('health',{}).get('release_gate'),
           'reasons':answer.get('health',{}).get('reasons'),
           'production_unchanged':answer['production_unchanged'],
           'error_type':answer.get('error_type')}))
    return 0 if answer['status']=='private_release_preview_only' else 2

if __name__=='__main__':raise SystemExit(main())
