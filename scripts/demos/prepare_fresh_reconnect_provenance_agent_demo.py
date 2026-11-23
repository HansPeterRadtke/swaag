#!/usr/bin/env python3
"""One-use, genuinely NEW read-only SWAAG reconnect/provenance model test.

Only prepares pinned input and private evidence. It performs no inference,
model change, user Send, external action, shell command as a model tool,
or production service modification. Run with separate one-use harness.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ORIGINAL_SOURCE_REVISION='c2a9e55008c8872abed290ac4e34e08c473083dc'
ORIGINAL_SOURCES={
 'tests/test_orchestrator_indexed_receipt_replay.py':(
  '675779990610a4b56d581e626ed64ea1a908987eb96d4696bcdbb93f43c1f9ff',
  'test_reconnect_replays_exact_answer_without_materializing_giant_journal',2),
 'tests/test_tool_result_provenance.py':(
  'e44c724b7cf102f1c22a88775e41c6f7e8b1962df2f4ee1c7cbbab8a55b79aed',
  'test_tool_message_is_separate_accounting_component_with_source_event',3),
}
BAD_PATH='tests/test_replay_connection_legacy.py'
MODEL_ALIAS='qwen3-coder-30b-a3b-instruct-q4-k-m'
MODEL_BASE='http://10.8.0.7:14833'
TOOLS=['read_file','list_files','calculator']

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def prepare(repo:Path,private:Path)->dict:
    if (not repo.is_absolute() or repo.is_symlink() or not repo.is_dir()
        or not private.is_absolute() or private.is_symlink() or private.exists()
        or repo==private or repo in private.parents):
        raise ValueError('Use a new absolute private evidence root outside the source checkout')
    checked={}
    for name,(sha0,want_test,want_count) in ORIGINAL_SOURCES.items():
        path=repo/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or symlinked fixed source '+name)
        raw=path.read_bytes()
        if sha(raw)!=sha0:raise ValueError('Actual frozen source SHA mismatch '+name)
        functions=[n.name for n in ast.parse(raw).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
        if len(functions)!=want_count or want_test not in functions:
            raise ValueError('Independent test function oracle differs '+name)
        checked[name]={'sha256':sha0,'required_exact_name':want_test,'top_level_test_count':want_count}
    if (repo/BAD_PATH).exists():raise ValueError('The deliberately missing historical path unexpectedly exists')
    rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(repo),'cat-file','-e',ORIGINAL_SOURCE_REVISION+'^{commit}'],
         check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=8)
    task='''A new READ-ONLY SWAAG source audit of actual reconnect and tool provenance tests.
Use the permitted REAL tools. No shell, file writes, external effects or user Sends.
Tool history is authoritative; never claim a success without its tool result.

1. Make exactly ONE `read_file` attempt with `{"path":"tests/test_replay_connection_legacy.py"}`.
This path is intentionally obsolete. Observe its REAL error. It will not be
created in this fixed workspace. NEVER repeat that same failed read again.
2. Once the missing-path error is visible, your NEXT tool call MUST be
`list_files` with `{"path":"tests"}`. This observes the actual directory.
3. Then use `read_file` separately on the two CURRENT checked-in files
`tests/test_orchestrator_indexed_receipt_replay.py` and
`tests/test_tool_result_provenance.py`.
4. From the observed actual source text, identify the exact test functions
`test_reconnect_replays_exact_answer_without_materializing_giant_journal`
and `test_tool_message_is_separate_accounting_component_with_source_event`.
Count how many TOP-LEVEL `def test_...` functions each actual file contains.
A parametrized test function counts as ONE declaration, not its repetitions.
5. CALL the real `calculator` tool to ADD the two counts. Do not mentally guess.
6. Finish with both actual source paths and test names, the observed missing
file error and directory-listing recovery, each number, exact calculator
expression and real result. Report incomplete if any operation fails.
The tool schema for each step is already loaded. Do not add another user turn.
'''
    config=f'''# Private read-only voice resilience source audit; no external effects.
[model]
base_url = "{MODEL_BASE}"
completion_endpoint = "/completion"
profile_name = "{MODEL_ALIAS}"
cache_enabled = false
timeout_seconds = 120
structured_timeout_seconds = 120
verification_timeout_seconds = 120
max_generation_wall_seconds = 160
max_retries = 0
chat_template_kwargs = {{ enable_thinking = false }}

[tools]
enabled = ["read_file", "list_files", "calculator"]
read_roots = [{json.dumps(str(repo))}]
allow_stateful_tools = true
allow_side_effect_tools = false
staged_discovery = false

[runtime]
completion_evaluation_enabled = false
max_validation_recovery_cycles = 1

[sessions]
root = {json.dumps(str(private/'sessions'))}
write_projections = true

[agent_data]
root = {json.dumps(str(private/'agent-data'))}

[logging]
file_path = {json.dumps(str(private/'operations.jsonl'))}
'''
    private.mkdir(mode=0o700,parents=True,exist_ok=False)
    os.chmod(private,0o700)
    for name in ('sessions','agent-data'):(private/name).mkdir(mode=0o700)
    for name,body in (('fixed-new-reconnect-task.txt',task),('read-only-config.toml',config)):
        path=private/name
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'w') as f:f.write(body);f.flush();os.fsync(f.fileno())
    manifest={'source_revision':rev,'original_source_blob_revision':ORIGINAL_SOURCE_REVISION,
        'original_source_hash_and_oracle':checked,
        'controlled_obsolete_missing_file':BAD_PATH,
        'read_only_tools':TOOLS,'no_side_effect_tools':True,
        'model_profile':MODEL_ALIAS,'model_endpoint':MODEL_BASE,
        'configuration_sha256':sha((private/'read-only-config.toml').read_bytes()),
        'fixed_task_sha256':sha((private/'fixed-new-reconnect-task.txt').read_bytes()),
        'distinct_from_all_previous_swaag_demonstration_source_inputs':True,
        'model_calls_or_tools_executed_during_preparation':0,
        'do_not_repeat_unknown_model_or_tool_effects':True}
    path=private/'manifest.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(manifest,f,indent=2);f.flush();os.fsync(f.fileno())
    return manifest

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    args=p.parse_args(argv)
    data=prepare(args.repo,args.evidence)
    print('NEW_PINNED_READONLY_RECONNECT_SOURCE_PREPARED',json.dumps({
      'source_rev':data['source_revision'],'counts':[v['top_level_test_count'] for v in data['original_source_hash_and_oracle'].values()],
      'allowed_tools':data['read_only_tools'],'model_calls':0,'evidence':str(args.evidence)}))
    return 0

if __name__=='__main__':raise SystemExit(main())
