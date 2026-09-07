from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from swaag.benchmark.process_restart_multisource import (
    ATTACHMENT_FACT, TOOL_FACT, USER_FACT, run_process_restart_multisource_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.types import CompletionResult, ContractSpec


class _Client:
    is_deterministic_test_client = True
    def tokenize(self, text: str) -> int: return len(text.split()) if text.strip() else 0
    def tokenize_selection(self, text: str) -> int: return self.tokenize(text)
    def cache_identity(self): return "multisource-parent-test"
    def context_limit_resolution(self): return 12_000, "test"
    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy("test", "server_schema", contract.mode, 30, 0.01)
    def resolve_contract(self, contract: ContractSpec, **kwargs): return contract, self.select_request_policy(contract=contract, **kwargs)
    def build_completion_request(self, prompt: str, *, max_tokens: int, contract: ContractSpec, **_kwargs):
        return {"prompt": prompt, "n_predict": max_tokens, "contract": contract.name, "json_schema": contract.json_schema}
    def send_completion(self, payload: dict[str, Any], **_kwargs):
        if payload["contract"] == "history_compaction_selection":
            text=json.dumps({"criticality":"compressible","reason":"routine window"})
        elif payload["contract"] == "summary_refinement":
            text=json.dumps({"summary":"Routine progress compressed."})
        else:
            spans=[v for v in (USER_FACT, TOOL_FACT) if v in str(payload["prompt"])]
            text=json.dumps({"summary":"Routine progress compressed.","preserve_recent_messages":0,"verbatim_spans":spans})
        return CompletionResult(text=text,raw_request=payload,raw_response={"content":text},prompt_tokens=None,completion_tokens=None,finish_reason="stop")


def test_multisource_parent_handoff(tmp_path, make_config) -> None:
    def fake_run(command, **_kwargs):
        out=Path(command[command.index("--output")+1]); out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps({"passed":True,"child_pid":os.getpid()+10,"exact_delayed_retrieval":True}),encoding="utf-8")
        return subprocess.CompletedProcess(command,0,stdout="ok\n",stderr="")
    report=run_process_restart_multisource_benchmark(
        output_dir=tmp_path/"run",config=make_config(model__context_limit=12_000),
        model_client=_Client(),subprocess_runner=fake_run,clean=True,
    )
    assert report["passed"] is True
    assert report["first_compaction"] is True
    assert report["different_process"] is True
    assert report["attachment_id"]
    assert len(report["attachment_sha256"]) == 64
    cmd=report["command"]
    assert "swaag.benchmark.process_restart_multisource_probe" in cmd
    assert USER_FACT in cmd and TOOL_FACT in cmd and ATTACHMENT_FACT in cmd
