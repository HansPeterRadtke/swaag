from __future__ import annotations
import json, os, subprocess
from pathlib import Path
from typing import Any
from swaag.benchmark.process_restart_chain import run_process_restart_chain_benchmark
from swaag.benchmark.process_restart_multisource import USER_FACT, TOOL_FACT
from swaag.model import CompletionRequestPolicy
from swaag.types import CompletionResult, ContractSpec

class _Client:
    is_deterministic_test_client=True
    def tokenize(self,text): return len(text.split()) if text.strip() else 0
    def tokenize_selection(self,text): return self.tokenize(text)
    def cache_identity(self): return "chain-test"
    def context_limit_resolution(self): return 12000,"test"
    def select_request_policy(self,*,contract:ContractSpec,**_kwargs): return CompletionRequestPolicy("test","server_schema",contract.mode,30,0.01)
    def resolve_contract(self,contract:ContractSpec,**kwargs): return contract,self.select_request_policy(contract=contract,**kwargs)
    def build_completion_request(self,prompt,*,max_tokens,contract,**_kwargs): return {"prompt":prompt,"n_predict":max_tokens,"contract":contract.name,"json_schema":contract.json_schema}
    def send_completion(self,payload:dict[str,Any],**_kwargs):
        if payload["contract"]=="history_compaction_selection": text=json.dumps({"criticality":"compressible","reason":"routine"})
        elif payload["contract"]=="summary_refinement": text=json.dumps({"summary":"compressed"})
        else:
            spans=[v for v in (USER_FACT,TOOL_FACT) if v in str(payload["prompt"])]
            text=json.dumps({"summary":"compressed","preserve_recent_messages":0,"verbatim_spans":spans})
        return CompletionResult(text=text,raw_request=payload,raw_response={"content":text},prompt_tokens=None,completion_tokens=None,finish_reason="stop")

def test_chain_requires_three_distinct_processes(make_config,tmp_path):
    calls=[]
    def fake_run(command,**_kwargs):
        calls.append(command)
        out=Path(command[command.index("--output")+1]); out.parent.mkdir(parents=True,exist_ok=True)
        idx=len(calls); query="--skip-query" not in command
        out.write_text(json.dumps({"passed":True,"child_pid":os.getpid()+idx,"query_performed":query,"exact_delayed_retrieval":True if query else None}),encoding="utf-8")
        return subprocess.CompletedProcess(command,0,stdout=f"phase-{idx}\n",stderr="")
    report=run_process_restart_chain_benchmark(output_dir=tmp_path/"chain",config=make_config(model__context_limit=12000),model_client=_Client(),subprocess_runner=fake_run,clean=True)
    assert report["passed"] is True
    assert report["distinct_processes"] is True
    assert report["phase_two_report"]["query_performed"] is False
    assert report["phase_three_report"]["query_performed"] is True
    assert "--skip-query" in report["phase_two_command"]
    assert "--skip-query" not in report["phase_three_command"]
