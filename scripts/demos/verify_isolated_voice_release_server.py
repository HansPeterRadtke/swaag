#!/usr/bin/env python3
"""Source-pinned private-process voice server release gate (zero user actions).

Starts an UNDEPLOYED candidate as a one-use child on an ephemeral localhost
port, with a fake bearer, private storage and all external model/ASR/TTS URLs
redirected to an unused localhost port. Checks actual HTTP liveness, denied
unauthenticated WebSocket access, the real authenticated hello speaker guard,
and an empty durable history. Stops only its own child; never touches the
running Jetson systemd service, production configuration or owner phone.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

EXPECTED_SOURCES={
 'cloudflare_ws_voice_server.py':'cbcc77fb36ce8cf72cc6442ec0f35d84526aec4868a27b810b2c38bde52bb302',
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
EXPECTED_TEMPLATE_SHA='f6cec4a73e96521cbedc085e44ccd298d0707c630ccd44869b5dce1d9aca07de'
OPT_IN='SWAAG_PRIVATE_RELEASE_SMOKE_NOACTION'


def sha(data: bytes) -> str:return hashlib.sha256(data).hexdigest()


def inspect(source: Path, template: Path) -> dict:
    if source.is_symlink() or template.is_symlink():
        raise ValueError('Symlinked candidate or template is not permitted')
    for name,want in EXPECTED_SOURCES.items():
        p=source/name
        if (not p.is_file() or p.is_symlink() or p.stat().st_size>250000
            or sha(p.read_bytes())!=want):
            raise ValueError('Candidate file differs from source-proven release: '+name)
    if (not template.is_file() or template.stat().st_size>50000 or
            sha(template.read_bytes())!=EXPECTED_TEMPLATE_SHA):
        raise ValueError('Production template drifted; no unattended test admission')
    return json.loads(template.read_text())


def scrub_config(original: dict, root: Path, port: int) -> dict:
    """An isolated child must have NO route to any model or production state."""
    v=json.loads(json.dumps(original))
    v['server'].update({'host':'127.0.0.1','port':port,
        'auth_token_file':str(root/'fake-only-bearer')})
    v['storage'].update({'runtime_root':str(root/'state'),
        'conversation_root':str(root/'conversations'),
        'turn_audio_root':str(root/'turn-audio'),
        'reply_audio_root':str(root/'reply-audio'),
        'correction_path':str(root/'corrections.json'),
        'log_path':str(root/'voice-private.jsonl')})
    v['tts'].update({'service_url':'http://127.0.0.1:9/synthesize',
        'service_health_url':'http://127.0.0.1:9/health',
        'piper_bin':str(root/'inert-piper'),
        'piper_model':str(root/'inert-piper.onnx')})
    for key in ('service_url','secondary_service_url','final_quality_url','reconcile_url'):
        if key in v['asr']:v['asr'][key]='http://127.0.0.1:9/unavailable'
    for key in ('health_url','secondary_health_url'):
        if key in v['asr']:v['asr'][key]='http://127.0.0.1:9/health'
    v['asr']['final_quality_enabled']=False
    v['llm'].update({'url':'http://127.0.0.1:9/run','health_url':'http://127.0.0.1:9/health'})
    v['worker_events'].update({'host':'127.0.0.1','port':9})
    v['interaction'].update({'auto_send_default':True,'require_client_commit':True,
        'require_verified_speaker_for_auto_send':True,
        'semantic_interrupt_url':'http://127.0.0.1:9/v1/voice/interrupt'})
    v['logging']['include_content']=False
    return v


def _write_unique(path: Path,raw: bytes,mode=0o600):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())


async def probe(port: int, child) -> dict:
    from websockets.legacy.client import connect
    from websockets.exceptions import InvalidStatusCode
    endpoint=f'ws://127.0.0.1:{port}/ws?conversation=private-guard-smoke&generation=1'
    bad_status=None
    for _ in range(50):
        if child.poll() is not None:
            raise RuntimeError('Private candidate process exited before ready')
        try:
            async with connect(endpoint,extra_headers={'Authorization':'Bearer test-only-private-voice-key'},
                               open_timeout=0.8,close_timeout=1,max_size=2**20) as websocket:
                hello=json.loads(await asyncio.wait_for(websocket.recv(),4))
                if hello.get('type')!='hello':raise AssertionError('No actual server hello')
                if hello.get('unverified_speaker_requires_review') is not True:
                    raise AssertionError('Candidate service did not advertise the speaker gate')
                if hello.get('auto_send_default') is not True:
                    raise AssertionError('Auto-send admission was not tested under an active default')
                await websocket.send(json.dumps({'type':'client_state',
                    'build':'Voice Agent 1.7.16','auto_send':True,
                    'speaker_identity':'owner','speaker_verified':True}))
                await websocket.send(json.dumps({'type':'history_get'}))
                deadline=asyncio.get_running_loop().time()+5
                while True:
                    if asyncio.get_running_loop().time()>deadline:
                        raise RuntimeError('No private history response')
                    event=json.loads(await asyncio.wait_for(websocket.recv(),5))
                    if event.get('type')=='turn' and event.get('event')=='submitted':
                        raise AssertionError('No voice input should submit a turn')
                    if event.get('type')=='turn_proposal':
                        raise AssertionError('No model action should be proposed')
                    if event.get('type')=='history':
                        if event.get('turns')!=[]:
                            raise AssertionError('Private fresh history was not empty')
                        break
            break
        except (OSError,ConnectionError):
            await asyncio.sleep(.12)
    else:
        raise RuntimeError('Private voice server never opened its own listener')
    # Liveness is independent of the intentionally disabled service readiness.
    def local_health():
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as reply:
            return reply.status,json.loads(reply.read(200000))
    http_status,health=await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(None,local_health),5)
    if (http_status!=200 or health.get('live') is not True or
            health.get('service')!='cloudflare_ws_voice_agent'):
        raise AssertionError('The real private process did not satisfy health liveness')
    try:
        async with connect(endpoint,open_timeout=1,close_timeout=1):
            raise AssertionError('Unauthenticated client was accepted')
    except InvalidStatusCode as e:
        bad_status=e.status_code
    if bad_status!=401:
        raise AssertionError('Unauthenticated WebSocket was not rejected with 401')
    return {'authenticated_real_hello':True,'private_http_health_liveness':True,
        'speaker_review_guard_default_on':True,
        'auto_send_default_exercised':True,'new_history_empty':True,
        'no_voice_or_user_turn_submitted':True,
        'unauthenticated_connection_http_status':bad_status}


def runtime_python() -> str:
    # A venv often uses a symlink to the base interpreter. Resolving it loses
    # the virtualenv's site-packages; retain the original executable pathname.
    executable=sys.executable
    if not executable or not Path(executable).is_file():
        raise RuntimeError('Current virtualenv interpreter is unavailable')
    return executable


def run_once(source: Path, original: dict, evidence: Path) -> dict:
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    config=scrub_config(original,evidence,port)
    _write_unique(evidence/'fake-only-bearer',b'test-only-private-voice-key\n')
    _write_unique(evidence/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(evidence/'inert-piper.onnx',b'no-test-model')
    _write_unique(evidence/'inert-piper.onnx.json',b'{}')
    _write_unique(evidence/'private-config.json',json.dumps(config,sort_keys=True).encode())
    # Verify nothing points to owner data, remote model, ASR, or TTS endpoints.
    for kind in ('asr','llm','tts'):
        for key,value in config[kind].items():
            if key.endswith('_url') or key=='url':
                if value and not str(value).startswith('http://127.0.0.1:9/'):
                    raise ValueError('Unexpected external provider route in private config')
    python=runtime_python()
    logfile=evidence/'child-private-console.log'
    t0=time.monotonic()
    child=None
    with logfile.open('xb') as sink:
        try:
            child=subprocess.Popen([str(python),'-u',str(source/'cloudflare_ws_voice_server.py'),
                '--config',str(evidence/'private-config.json')],cwd=str(source),
                stdin=subprocess.DEVNULL,stdout=sink,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source)})
            result=asyncio.run(probe(port,child))
            result['status']='private_unreleased_speaker_safe_server_boot_and_handshake_verified'
            return result
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait(timeout=3)
                if child.poll() is None:raise RuntimeError('Own child did not stop')
            os.chmod(logfile,0o600)


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    config=inspect(a.source,a.template)
    if not a.run:
        print(json.dumps({'status':'source_preflight_no_network_or_writes',
            'candidate_files_valid':len(EXPECTED_SOURCES),
            'template_sha256':EXPECTED_TEMPLATE_SHA}))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('Private process test needs explicit environment opt-in')
    if a.evidence.is_symlink() or a.evidence.exists():
        raise FileExistsError('Existing attempt or uncertain result must not be overwritten')
    a.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(a.evidence,0o700)
    _write_unique(a.evidence/'attempt.json',json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'single_private_child':True,'no_live_process_interference':True,
        'exact_candidate_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py']}).encode())
    t0=time.monotonic()
    try:result=run_once(a.source,config,a.evidence)
    except BaseException as exc:
        result={'status':'failed_or_unknown','error_type':type(exc).__name__,
                'error_detail_internal':str(exc)[:400]}
    result.update({'utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':round(time.monotonic()-t0,3),
        'source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
        'production_service_touched':False,'production_owner_auth_used':False,
        'real_model_asr_tts_or_swaag_actions_requested':False})
    _write_unique(a.evidence/'result.json',json.dumps(result,ensure_ascii=False,indent=2).encode())
    print(json.dumps({k:result.get(k) for k in ('status','error_type','elapsed_seconds',
        'speaker_review_guard_default_on','no_voice_or_user_turn_submitted',
        'unauthenticated_connection_http_status')}))
    return 0 if result['status']=='private_unreleased_speaker_safe_server_boot_and_handshake_verified' else 2


if __name__=='__main__':sys.exit(main())
