from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from swaag.benchmark.process_restart_delayed import (
    PROCESS_DELAYED_RELEVANCE_FACT,
    run_process_restart_delayed_relevance_benchmark,
)
from swaag.model import CompletionRequestPolicy
from swaag.types import CompletionResult, ContractSpec


class _CompactionClient:
    is_deterministic_test_client = True

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def cache_identity(self):
        return "process-restart-parent-test"

    def context_limit_resolution(self) -> tuple[int, str]:
        return 12_000, "test"

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy("test", "server_schema", contract.mode, 30, 0.01)

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self, prompt: str, *, max_tokens: int, contract: ContractSpec, temperature=None, **_kwargs
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        if payload["contract"] == "history_compaction_selection":
            text = json.dumps({"criticality": "compressible", "reason": "routine window"})
        elif payload["contract"] == "summary_refinement":
            text = json.dumps({"summary": "Routine progress compressed."})
        else:
            assert payload["contract"] == "summary"
            spans = (
                [PROCESS_DELAYED_RELEVANCE_FACT]
                if PROCESS_DELAYED_RELEVANCE_FACT in str(payload["prompt"])
                else []
            )
            text = json.dumps(
                {
                    "summary": "Routine progress compressed.",
                    "preserve_recent_messages": 0,
                    "verbatim_spans": spans,
                }
            )
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_parent_process_handoff_requires_distinct_child_and_passed_report(
    make_config, tmp_path
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "passed": True,
                    "child_pid": os.getpid() + 1000,
                    "exact_delayed_retrieval": True,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="child-ok\n", stderr="")

    config = make_config(model__context_limit=12_000)
    report = run_process_restart_delayed_relevance_benchmark(
        output_dir=tmp_path / "process",
        config=config,
        model_client=_CompactionClient(),
        subprocess_runner=fake_run,
        clean=True,
    )
    assert report["passed"] is True
    assert report["first_compaction"] is True
    assert report["different_process"] is True
    assert report["subprocess_returncode"] == 0
    command = observed["command"]
    assert command[1:3] == ["-m", "swaag.benchmark.process_restart_probe"]
    assert PROCESS_DELAYED_RELEVANCE_FACT in command
    assert "--session-id" in command
    assert "--sessions-root" in command
    assert "--context-limit" in command
    assert observed["kwargs"]["capture_output"] is True
    assert observed["kwargs"]["text"] is True
    assert observed["kwargs"]["check"] is False
    assert (tmp_path / "process" / "child_stdout.txt").read_text() == "child-ok\n"


def test_parent_process_handoff_rejects_same_pid_child(make_config, tmp_path) -> None:
    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"passed": True, "child_pid": os.getpid()}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    report = run_process_restart_delayed_relevance_benchmark(
        output_dir=tmp_path / "process",
        config=make_config(model__context_limit=12_000),
        model_client=_CompactionClient(),
        subprocess_runner=fake_run,
    )
    assert report["different_process"] is False
    assert report["passed"] is False
