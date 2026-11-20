#!/usr/bin/env python3
"""Generate an immutable private SWAAG live-model read-only demo workspace.

Does NOT execute a model, send messages, install packages, or configure SWAAG
production services. All live model calls require a distinct explicit
operator command and the resource broker's admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

INPUTS=('tests/test_history_replay.py','tests/test_storage_boundaries.py')
PROMPTS=('source_audit_prompt_v1.txt','source_audit_correction_v2.txt')


def digest(data:bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(repo:Path,evidence:Path,model_base_url:str) -> dict:
    repo=repo.resolve()
    evidence=evidence.resolve()
    if repo==evidence or repo in evidence.parents:
        raise ValueError('Private model run evidence cannot be in the source repository')
    endpoint=urlparse(model_base_url)
    if (endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','10.8.0.7')
        or not endpoint.port):
        raise ValueError('Model backend must use a preauthorized loopback/private endpoint')
    if evidence.exists():
        raise FileExistsError('Do not overwrite any previous test run or correction')
    for path in INPUTS:
        if not (repo/path).is_file():
            raise FileNotFoundError(path)
    prompt_root=repo/'scripts/demos'
    for path in PROMPTS:
        if not (prompt_root/path).is_file():
            raise FileNotFoundError(path)
    evidence.mkdir(parents=True,mode=0o700)
    config=f'''# Fixed minimal read-only SWAAG agent demo; generated private config.
[model]
base_url = {json.dumps(model_base_url)}
completion_endpoint = "/completion"
profile_name = "qwen38-27b-q8-full-262k"
cache_enabled = false
timeout_seconds = 120
structured_timeout_seconds = 120
verification_timeout_seconds = 120
max_generation_wall_seconds = 160
max_retries = 0
chat_template_kwargs = {{ enable_thinking = false }}

[tools]
enabled = ["list_files", "read_file", "calculator"]
read_roots = [{json.dumps(str(repo))}]
allow_stateful_tools = true
allow_side_effect_tools = false
staged_discovery = false

[runtime]
completion_evaluation_enabled = true
max_validation_recovery_cycles = 2

[sessions]
root = {json.dumps(str(evidence/'sessions'))}
write_projections = true

[agent_data]
root = {json.dumps(str(evidence/'agent-data'))}

[logging]
file_path = {json.dumps(str(evidence/'operations.jsonl'))}
'''
    correction=f'''# Explicitly new, non-replayed correction. All side effects remain OFF.
[model]
max_generation_wall_seconds = 375
structured_timeout_seconds = 390
timeout_seconds = 390
max_retries = 0

[tools]
enabled = ["calculator"]
read_roots = [{json.dumps(str(repo))}]
allow_stateful_tools = false
allow_side_effect_tools = false
staged_discovery = false

[runtime]
completion_evaluation_enabled = false
'''
    sources={
        name:digest((repo/name).read_bytes()) for name in INPUTS
    }
    config_files={
        'demo-readonly-config.toml':config,
        'demo-correction-config.toml':correction,
    }
    for name,content in config_files.items():
        path=evidence/name
        path.write_text(content,encoding='utf8')
        path.chmod(0o600)
    for source,name in zip(PROMPTS,('demo-task-v1.txt','demo-correction-v2.txt')):
        output=evidence/name
        output.write_bytes((prompt_root/source).read_bytes())
        output.chmod(0o600)
    repo_rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],
                                     text=True).strip()
    manifest={
        'purpose':'real model-controlled read-only multi-file security audit',
        'source_revision':repo_rev,
        'model_endpoint':model_base_url,
        'model_profile':'qwen38-27b-q8-full-262k',
        'real_sources_sha256':sources,
        'config_file_sha256':{name:digest((evidence/name).read_bytes())
                              for name in config_files},
        'prompt_file_sha256':{name:digest((evidence/name).read_bytes())
                              for name in ('demo-task-v1.txt','demo-correction-v2.txt')},
        'agent_tools':['list_files','read_file','calculator'],
        'allow_side_effect_tools':False,
        'stateful_file_reads_allowed_only_inside_repo':str(repo),
        'no_model_called_by_prepare':True,
        'prior_evidence_not_overwritten':True,
    }
    (evidence/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (evidence/'manifest.json').chmod(0o600)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--model-url',required=True)
    args=p.parse_args()
    report=prepare(args.repo,args.evidence,args.model_url)
    print('PREPARED_READONLY_AGENT_DEMO',
          json.dumps({'source_revision':report['source_revision'],
                      'model_profile':report['model_profile'],
                      'tools':report['agent_tools'],
                      'no_model_called':True,
                      'output':str(args.evidence)},sort_keys=True))


if __name__=='__main__':
    main()
