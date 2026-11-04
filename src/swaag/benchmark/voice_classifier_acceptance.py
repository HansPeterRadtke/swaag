"""Opt-in acceptance of the real voice gateway's classifier on an isolated local model.

Never sends to SWAAG, executes audio/media operations, or changes production
configuration. An unknown outcome is recorded, not counted as a successful no.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from typing import Any, Sequence


DEFAULT_ORACLE = Path(__file__).resolve().parent / 'fixtures/voice_classifier_oracle_20261104.json'


def check_oracle_result(family: str, expected: bool, result: dict[str, Any]) -> dict[str, Any]:
    """Fail closed: classifier_invalid is not a successful negative prediction."""
    kind = str(result.get('kind') or '')
    field = 'complete' if family == 'turn_completion' else 'interrupt'
    observed = result.get(field)
    allowed = ({'finished', 'continuing', 'non_speech'} if family == 'turn_completion'
               else {'take_floor', 'keep_listening', 'explicit_interrupt', 'explicit_stop'})
    trusted = (
        type(observed) is bool and kind in allowed
        and type(result.get('confidence')) in (float, int)
        and 0 <= result['confidence'] <= 1
    )
    return {'passed': trusted and observed is expected,
            'classifier_valid': trusted, 'observed': observed, 'kind': kind,
            'confidence': result.get('confidence')}


def validate_isolated_endpoint(url: str) -> str:
    endpoint = urlparse(url)
    if endpoint.scheme != 'http' or endpoint.hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('voice classifier acceptance requires a loopback HTTP model')
    if endpoint.port is None or endpoint.port in {14920, 13484, 13401}:
        raise ValueError('never test against the deployed voice gateway or classifier port')
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ('', '/'):
        raise ValueError('model URL must be just the loopback origin')
    return f'http://127.0.0.1:{endpoint.port}'


def run(*, gateway_source_dir: Path, gateway_config: Path, model_origin: str,
        model_identity: str, oracle_path: Path, output: Path, family: str = 'all') -> dict:
    base = validate_isolated_endpoint(model_origin)
    if family not in {'all','turn_completion','interruption'}:
        raise ValueError('invalid family')
    if output.exists():
        raise FileExistsError('existing or uncertain classifier test receipt; never replay')
    oracle = json.loads(oracle_path.read_text(encoding='utf-8'))
    cases = [case for case in oracle['cases'] if family == 'all' or case['family'] == family]
    if not cases:
        raise ValueError('no classifier cases selected')
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status':'started', 'started_utc':datetime.now(timezone.utc).isoformat(),
               'model_origin':base,'model_identity':model_identity,
               'source_dir':str(gateway_source_dir),'selected_family':family,
               'selected_cases':len(cases),'cases':[],'no_production_change':True}
    # Persist the unique receipt BEFORE the first model request; a crash leaves
    # unknown outcome visible rather than inviting a blind duplicate run.
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    sys.path.insert(0,str(gateway_source_dir.resolve()))
    try:
        gateway = importlib.import_module('voice_agent_gateway')
        module = importlib.import_module('voice_agent_gateway_config')
        gateway_file = (gateway_source_dir.resolve() / 'voice_agent_gateway.py')
        config_file = (gateway_source_dir.resolve() / 'voice_agent_gateway_config.py')
        if (Path(gateway.__file__).resolve() != gateway_file.resolve()
                or Path(module.__file__).resolve() != config_file.resolve()):
            raise RuntimeError('voice gateway import did not resolve to selected source revision')
        receipt['gateway_sha256'] = hashlib.sha256(gateway_file.read_bytes()).hexdigest()
        receipt['gateway_config_module_sha256'] = hashlib.sha256(config_file.read_bytes()).hexdigest()
        config=module.load_gateway_config(gateway_config)
        gateway.CONFIG = replace(config,classifier=module.ClassifierConfig(
            chat_url=base+'/v1/chat/completions',health_url=base+'/health',
            model=model_identity,request_timeout_seconds=8.0,enable_thinking=False))
        for c in cases:
            start=time.perf_counter()
            try:
                if c['family']=='turn_completion':answer=gateway.classify_turn_completion(c['text'])
                else:answer=gateway.classify_interruption(c['text'],c.get('assistant_text') or '')
                check=check_oracle_result(c['family'],c['expected'],answer)
                check['elapsed_seconds']=round(time.perf_counter()-start,3)
            except Exception as e:
                check={'passed':False,'classifier_valid':False,'exception':type(e).__name__,
                       'elapsed_seconds':round(time.perf_counter()-start,3)}
            receipt['cases'].append({'case_id':c['case_id'],'expected':c['expected'],**check})
            output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
            print(c['case_id'],check['passed'],check.get('kind'),check['elapsed_seconds'],flush=True)
        receipt['status']='finished'
        receipt['passed']=sum(bool(c['passed']) for c in receipt['cases'])
        receipt['failed']=len(receipt['cases'])-receipt['passed']
    except BaseException as e:
        receipt['status']='interrupted_or_failed'
        receipt['error_type']=type(e).__name__
        raise
    finally:
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--gateway-source-dir',type=Path,required=True)
    parser.add_argument('--gateway-config',type=Path,required=True)
    parser.add_argument('--model-origin',required=True)
    parser.add_argument('--model-identity',required=True)
    parser.add_argument('--oracle',type=Path,default=DEFAULT_ORACLE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--family',choices=['all','turn_completion','interruption'],default='all')
    args=parser.parse_args(argv)
    result=run(gateway_source_dir=args.gateway_source_dir,gateway_config=args.gateway_config,
        model_origin=args.model_origin,model_identity=args.model_identity,
        oracle_path=args.oracle,output=args.output,family=args.family)
    print(f"voice_classifier_passed={result['passed']}/{len(result['cases'])}")
    return 0 if result['failed']==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
