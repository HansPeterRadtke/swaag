#!/usr/bin/env python3
"""One fresh original ASR transcript -> real SWAAG fast foreground, no tools.

This one-use test client accepts no paths or commands from stdin. It requires
an existing exclusive private session claim, a source-pinned canonical Git
revision, and zero effective tools. Delegation never starts deep work. It
writes one truthful response and durable journal, or fails closed without
calling an external tool or retrying an uncertain inference.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import redirect_stdout
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.communication import CommunicationService
from verify_real_voice_phonon_reconnect import (
    validate_pinned_swaag_paths, validate_short_voice_clarification,
)
from claim_fresh_voice_noaction_session import CLAIM_NAME


def check_admitted_source(config_path: Path, expected_config_sha256: str,
                          expected_source_revision: str, repository: Path) -> tuple:
    source,config=validate_pinned_swaag_paths(str(repository),str(config_path),
                expected_source_revision,expected_config_sha256)
    if config_path.is_symlink() or repository.is_symlink():
        raise ValueError('Source root and configuration must not be symlinks')
    observed=hashlib.sha256(config_path.read_bytes()).hexdigest()
    if observed!=expected_config_sha256:
        raise ValueError('No-action model config does not match published source evidence')
    head=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
    if head!=expected_source_revision:
        raise ValueError('Canonical SWAAG Git source changed since admission')
    cfg=load_config([config_path],env={})
    if (cfg.tools.enabled != [] or cfg.tools.staged_discovery
        or cfg.tools.allow_stateful_tools or cfg.tools.allow_side_effect_tools
        or not cfg.communication.fast_initial_reply
        or cfg.model.base_url!='http://10.8.0.7:14835'
        or cfg.model.profile_name!='qwen38-27b-q8-full-262k'
        or cfg.model.cache_enabled or cfg.model.max_retries!=0
        or cfg.model.chat_template_kwargs.get('enable_thinking') is not False):
        raise ValueError('Unapproved model, action scope or fast-stage configuration')
    if (cfg.sessions.root != config_path.parent/'sessions'
        or cfg.agent_data.root!=config_path.parent/'agent_data'
        or cfg.logging.file_path!=config_path.parent/'operations.jsonl'):
        raise ValueError('No-action model storage must be entirely private and isolated')
    marker=cfg.sessions.root/CLAIM_NAME
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('Expected a prior single-use session claim')
    claim=json.loads(marker.read_text())
    if (claim.get('admitted') is not True or claim.get('one_use_only') is not True
            or claim.get('prior_session_directories')!=[]
            or claim.get('configuration_sha256')!=expected_config_sha256
            or Path(claim.get('private_session_root',''))!=cfg.sessions.root):
        raise ValueError('Source session not independently claimed for a single run')
    if list(cfg.sessions.root.glob('session_*')):
        raise ValueError('Existing model session would be reused: refuse')
    return cfg,marker


def claim_one_inference(cfg, original_input_sha256: str) -> Path:
    root=cfg.sessions.root
    marker=root/'.fast_noaction_inference_started.json'
    fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        row={'started_utc':datetime.now(timezone.utc).isoformat(),
             'source_text_sha256':original_input_sha256,
             'one_model_request_maximum':True,
             'never_retry_same_private_session':True}
        data=(json.dumps(row,sort_keys=True,indent=2)+'\n').encode()
        os.write(fd,data);os.fsync(fd)
    finally:os.close(fd)
    return marker


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--config-sha256',required=True)
    p.add_argument('--transcript-sha256',required=True)
    p.add_argument('--source-root',required=True,type=Path)
    args=p.parse_args(argv)
    # No unsafe arbitrary text or unbounded file read; speech travels ONLY
    # through the stdin pipe of a fixed authenticated private SSH command.
    raw=sys.stdin.buffer.read(3001)
    if len(raw)>3000:
        raise ValueError('Source ASR input exceeds bounded acceptance')
    text=raw.decode('utf8','strict')
    if (not text.strip() or text!=text.strip() or
            hashlib.sha256(raw).hexdigest()!=args.transcript_sha256):
        raise ValueError('Observed ASR text differs from exact reviewed source')
    cfg,_=check_admitted_source(args.config,args.config_sha256,
                               args.source_revision,args.source_root)
    # If an invocation attempt fails during the model call, the marker remains;
    # this is intentional, not a temporary cache to clear and re-run.
    marker=claim_one_inference(cfg,hashlib.sha256(raw).hexdigest())
    root=args.config.parent
    output=root/'fast-oneuse-canonical-result.json'
    log_path=root/'fast-oneuse-canonical-model-console.log'
    model_console=io.StringIO()
    report={'status':'failed_or_incomplete','source_revision':args.source_revision,
            'source_transcript_sha256':args.transcript_sha256,
            'model':'qwen38-27b-q8-full-262k','strict_zero_tools':True,
            'claim_marker_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),
            'model_generation_not_replayed':True}
    try:
        with redirect_stdout(model_console):
            runtime=AgentRuntime(cfg)
            service=CommunicationService(runtime,orchestrator_runtime=runtime)
            try:
                state=service._orchestrator_state()
                assert not runtime.tools.enabled_tools(cfg)
                if runtime.history.list_pending_control_messages(state.session_id):
                    raise RuntimeError('Nonempty previous control queue')
                start=time.monotonic()
                receipt=service._receive_orchestrator_message(text,
                    control_id='one_fresh_reviewed_recorded_voice_fast_noaction')
                decision=service._try_fast_foreground_reply(state,receipt)
                report['fast_stage_wall_seconds']=round(time.monotonic()-start,3)
                rows=runtime.history.read_history(state.session_id)
                counts=collections.Counter(x.event_type for x in rows)
                responses=[x.payload for x in rows if x.event_type=='model_response_received']
                report.update({'fast_decision_mode':decision.get('mode') if decision else None,
                               'model_requests':counts['model_request_sent'],
                               'model_responses':counts['model_response_received'],
                               'tools_executed':counts['tool_called'],
                               'worker_starts':counts['worker_started'],
                               'turn_finished':counts['turn_finished'],
                               'session_id':state.session_id,
                               'semantic_fast_input_tokens':decision.get('input_tokens') if decision else None,
                               'model_elapsed_seconds':responses[-1].get('elapsed_seconds') if responses else None,
                               'model_first_token_seconds':responses[-1].get('first_token_seconds') if responses else None})
                if (not decision or decision.get('mode')!='answer'
                        or counts['model_request_sent']!=1 or counts['model_response_received']!=1
                        or counts['turn_finished']!=1 or counts['tool_called']!=0
                        or counts['worker_started']!=0):
                    raise RuntimeError('Fast stage unavailable/delegated or non-action evidence incomplete')
                result=validate_short_voice_clarification(str(decision['reply']))
                report.update({'status':'verified_one_fast_natural_clarification',
                               'reply':result,'reply_sha256':hashlib.sha256(result.encode()).hexdigest(),
                               'full_orchestrator_called':False})
            finally:service.close()
    except BaseException as exc:
        report['error_type']=type(exc).__name__
        report['error_message']=str(exc)[:400]
        raise
    finally:
        log_path.write_text(model_console.getvalue());log_path.chmod(0o600)
        sid=report.get('session_id')
        if isinstance(sid,str) and sid.startswith('session_'):
            journal=cfg.sessions.root/sid/'complete_history.jsonl'
            if journal.is_file():
                report['journal_sha256']=hashlib.sha256(journal.read_bytes()).hexdigest()
                report['journal_bytes']=journal.stat().st_size
        output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');output.chmod(0o600)
    # Only this machine-readable object goes to the test WebSocket. No private
    # model prompt or runtime keys are printed or sent to other services.
    print(json.dumps(report,sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
