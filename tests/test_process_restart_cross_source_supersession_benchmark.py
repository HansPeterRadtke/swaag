from __future__ import annotations
import json, os, subprocess
from pathlib import Path
from typing import Any
from swaag.benchmark.process_restart_cross_source_supersession import OLD_VALUE, NEW_VALUE, run_process_restart_cross_source_supersession_benchmark
from swaag.model import CompletionRequestPolicy
from swaag.types import CompletionResult, ContractSpec

class _Client:
    is_deterministic_test_client=True
    def tokenize(self,text): return len(text.split()) if text.strip() else 0
    def tokenize_selection(self,text): return self.tokenize(text)
    def cache_identity(self): return "cross-source-supersession-test"
    def context_limit_resolution(self): return 12000,"test"
    def select_request_policy(self,*,contract:ContractSpec,**_kwargs): return CompletionRequestPolicy("test","server_schema",contract.mode,30,0.01)
    def resolve_contract(self,contract:ContractSpec,**kwargs): return contract,self.select_request_policy(contract=contract,**kwargs)
    def build_completion_request(self,prompt,*,max_tokens,contract,**_kwargs): return {"prompt":prompt,"n_predict":max_tokens,"contract":contract.name,"json_schema":contract.json_schema}
    def send_completion(self,payload:dict[str,Any],**_kwargs):
        if payload["contract"]=="history_compaction_selection": text=json.dumps({"criticality":"compressible","reason":"routine"})
        elif payload["contract"]=="summary_refinement": text=json.dumps({"summary":"compressed"})
        else:
            spans=[v for v in (OLD_VALUE,NEW_VALUE) if v in str(payload["prompt"])]
            text=json.dumps({"summary":"compressed","preserve_recent_messages":0,"verbatim_spans":spans})
        return CompletionResult(text=text,raw_request=payload,raw_response={"content":text},prompt_tokens=None,completion_tokens=None,finish_reason="stop")

def test_cross_source_parent_requires_tool_event_and_distinct_child(make_config,tmp_path):
    observed={}
    def fake_run(command,**kwargs):
        observed['command']=list(command); observed['kwargs']=kwargs
        out=Path(command[command.index('--output')+1]); out.parent.mkdir(parents=True,exist_ok=True)
        tool_seq=int(command[command.index('--old-tool-event-sequence')+1])
        out.write_text(json.dumps({"passed":True,"child_pid":os.getpid()+55,"old_tool_event_present":True,"old_tool_event_sequence":tool_seq,"exact_supersession_retrieval":True}),encoding='utf-8')
        return subprocess.CompletedProcess(command,0,stdout='ok\n',stderr='')
    report=run_process_restart_cross_source_supersession_benchmark(output_dir=tmp_path/'run',config=make_config(model__context_limit=12000),model_client=_Client(),subprocess_runner=fake_run,clean=True)
    assert report['passed'] is True
    assert report['first_compaction'] is True
    assert report['different_process'] is True
    assert report['old_tool_event_sequence'] > 0
    assert report['old_message_sequence'] > report['old_tool_event_sequence']
    assert report['update_sequence'] > report['old_message_sequence']
    cmd=observed['command']
    assert '--old-tool-event-sequence' in cmd
    assert OLD_VALUE in cmd and NEW_VALUE in cmd
