#!/usr/bin/env python3
"""Independent source-backed verifier for the NEW AST-tool one-use agent task.

Reads retained original source blobs and durable event history, never calls
LLMs or executes source code. A completed model answer is not proof.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess

from swaag.events import verify_event_integrity
from swaag.types import HistoryEvent
from scripts.demos.nov28_ast_reliability_oneuse import SOURCES, MISSING, ALLOWLIST
from scripts.demos.nov28_core_agent_oneuse import sha


def _blob(repo: Path, revision: str, name: str) -> bytes:
    return subprocess.check_output(["git","-C",str(repo),"cat-file","blob",
                                    revision + ":" + name],timeout=12)


def verify(repo: Path, folder: Path) -> dict:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Missing private source-backed run")
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("schema") != "swaag-core-ast-holdout-oneuse-v1"
            or manifest.get("repo") != str(repo)
            or manifest.get("tool_names") != ALLOWLIST
            or manifest.get("controlled_error") != MISSING
            or manifest.get("tool_side_effects_enabled") is not False):
        raise ValueError("Unauthorized or changed test identity")
    if (manifest["task_sha256"] != sha((folder / "task.txt").read_bytes())
            or manifest["config_sha256"] != sha((folder / "config.toml").read_bytes())):
        raise ValueError("Frozen source task/config was modified")
    originals = {}
    expected = {}
    for name in SOURCES:
        raw = _blob(repo, manifest["source_revision"], name)
        if sha(raw) != manifest["sources"][name]["sha256"]:
            raise ValueError("Frozen original Git source hash did not match manifest")
        funcs = [node for node in ast.parse(raw).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        test_names = [node.name for node in funcs if node.name.startswith("test_")]
        if len(test_names) != manifest["sources"][name]["test_declarations"] or test_names[0] != manifest["sources"][name]["first_exact_test_name"]:
            raise ValueError("Original AST source count/identity mismatch")
        expected[name] = {"test_count":len(test_names),"functions":funcs,
                          "test_names":test_names,"sha256":sha(raw)}
        originals[name] = raw
    result = json.loads((folder / "run-result.json").read_text())
    marker = json.loads((folder / ".one_use_started.json").read_text())
    if (result.get("source_revision") != manifest["source_revision"]
            or result.get("console_sha256") != sha((folder / "console.log").read_bytes())
            or marker.get("owner") != "owner-20261128-swaag-core"
            or not str(marker.get("lease_id", "")).startswith("rb_")):
        raise ValueError("Missing real process/broker/source provenance")
    journals = list((folder / "sessions").glob("session_*/complete_history.jsonl"))
    if len(journals) != 1 or journals[0].is_symlink() or journals[0].stat().st_size > 75_000_000:
        raise ValueError("Expected exactly one bounded genuine source event journal")
    raw_lines = [json.loads(line) for line in journals[0].read_text().splitlines() if line.strip()]
    if not raw_lines or len(raw_lines) > 7000 or len({row.get("session_id") for row in raw_lines}) != 1:
        raise ValueError("Missing/mixed original source events")
    prev=None
    for row in raw_lines:
        event=HistoryEvent(**row)
        verify_event_integrity(event,prev)
        prev=event.hash
    if any(a["sequence"] >= b["sequence"] for a,b in zip(raw_lines,raw_lines[1:])):
        raise ValueError("Original event ordering changed")
    kinds=Counter(row["type"] for row in raw_lines)
    user_messages=[row["payload"]["message"]["content"] for row in raw_lines
        if row["type"]=="message_added" and row["payload"].get("message",{}).get("role")=="user"]
    one_user=len(user_messages)==1 and user_messages[0]==(folder / "task.txt").read_text()
    starts={};results={};failures={};tool_timing=[];issues=[]
    for row in raw_lines:
        p=row.get("payload",{});cid=p.get("call_id");kind=row["type"]
        if kind=="tool_called":
            if not isinstance(cid,str) or not cid or cid in starts:
                issues.append("duplicate_or_missing_tool_call_id")
            else:starts[cid]=row
        if kind not in ("tool_result","tool_error"):continue
        prior=starts.get(cid)
        if prior is None or cid in results or cid in failures:
            issues.append("unmatched_tool_receipt");continue
        args=p.get("validated_input") if kind=="tool_result" else p.get("tool_input")
        if (p.get("tool_name") != prior["payload"].get("tool_name")
                or args != prior["payload"].get("tool_input")
                or row["sequence"]<=prior["sequence"]):
            issues.append("tool_receipt_does_not_match_call");continue
        (results if kind=="tool_result" else failures)[cid]=row
        tool_timing.append({"tool":p["tool_name"], "elapsed_seconds":round((
            datetime.fromisoformat(row["timestamp"])-datetime.fromisoformat(prior["timestamp"])
        ).total_seconds(),6),"success":kind=="tool_result"})
    if set(starts)!=set(results)|set(failures):
        issues.append("unreconciled_tool_call")
    actual_calls=[{"sequence":row["sequence"],"tool":row["payload"].get("tool_name"),
                   "input":row["payload"].get("tool_input")} for row in raw_lines
                  if row["type"]=="tool_called"]
    unsafe=[item for item in actual_calls if item["tool"] not in ALLOWLIST]
    controlled=[row for row in failures.values()
                if row["payload"].get("tool_name")=="python_source_structure"
                and row["payload"].get("tool_input")=={"path":MISSING}
                and row["payload"].get("error_type")=="FilesystemError"]
    exactly_one_failure=(len(controlled)==1 and sum(
        c["tool"]=="python_source_structure" and c["input"]=={"path":MISSING}
        for c in actual_calls)==1)
    failure_seq=controlled[0]["sequence"] if exactly_one_failure else None
    observed_stats={}
    for row in results.values():
        p=row["payload"]
        if p.get("tool_name") != "python_source_structure":continue
        path=p.get("validated_input",{}).get("path")
        if path not in SOURCES:continue
        out=p.get("output") or {}
        expected_source=expected[path]
        functions=[{"name":n.name,"line":n.lineno,"async":isinstance(n,ast.AsyncFunctionDef)}
                   for n in expected_source["functions"]]
        exact=(out.get("source_sha256")==expected_source["sha256"]
               and out.get("top_level_functions")==functions
               and out.get("counts",{}).get("top_level_functions")==len(functions)
               and out.get("counts",{}).get("top_level_test_functions")==expected_source["test_count"])
        observed_stats[path]={"sequence":row["sequence"],"exact":exact,
                               "test_count":out.get("counts",{}).get("top_level_test_functions")}
    stats_verified=bool(failure_seq and set(observed_stats)==set(SOURCES)
        and all(row["exact"] and row["sequence"]>failure_seq for row in observed_stats.values()))
    calculator=[]
    for row in results.values():
        p=row["payload"]
        if p.get("tool_name")=="calculator":calculator.append(row)
    counts=[expected[name]["test_count"] for name in SOURCES]
    expression={f"{counts[0]}+{counts[1]}",f"{counts[1]}+{counts[0]}"}
    calc_correct=bool(stats_verified and any(
        str(row["payload"].get("output",{}).get("expression","")).replace(" ","") in expression
        and row["payload"].get("output",{}).get("result")==sum(counts)
        and all(row["sequence"]>observed_stats[name]["sequence"] for name in SOURCES)
        for row in calculator))
    requests=[row for row in raw_lines if row["type"]=="model_request_sent"]
    responses=[row for row in raw_lines if row["type"]=="model_response_received"]
    request_ids={row["payload"].get("call_id"):row for row in requests}
    response_ids={row["payload"].get("call_id"):row for row in responses}
    correlated=bool(requests and len(request_ids)==len(requests) and
        len(response_ids)==len(responses) and request_ids.keys()==response_ids.keys()
        and all(response_ids[k]["sequence"]>r["sequence"] for k,r in request_ids.items()))
    role_integrity=True
    saw_native_assistant=False
    prompt_rows=[row for row in raw_lines if row["type"]=="prompt_built"]
    for req in requests:
        messages=req["payload"].get("request",{}).get("messages",[])
        if any(item.get("role")=="assistant" for item in messages): saw_native_assistant=True
        prior=next((p for p in reversed(prompt_rows) if p["sequence"]<req["sequence"] and
                    p["payload"].get("kind")==req["payload"].get("kind")),None)
        if prior is None or not messages or messages[0].get("role")!="system":
            role_integrity=False;continue
        comps=prior["payload"].get("components",[])
        bounds=prior["payload"].get("message_ranges",[])
        if len(bounds)!=len(messages):role_integrity=False;continue
        for item,interval in zip(messages,bounds):
            serialized="".join(c["text"] for c in comps[interval["component_start"]:interval["component_end"]])
            if item.get("role")!=interval["role"] or item.get("content")!=serialized:
                role_integrity=False
    model_selected=True
    selected=[r for r in raw_lines if r["type"]=="agent_action_selected"]
    for call in actual_calls:
        last=next((r for r in reversed(selected) if r["sequence"]<call["sequence"]),None)
        if last is None or not any(isinstance(tool,dict) and tool.get("tool_name")==call["tool"]
                and tool.get("arguments")==call["input"] for tool in
                (last.get("payload",{}).get("action") or {}).get("tool_calls",[])):
            model_selected=False
    observed_error_in_later_model=False
    if exactly_one_failure and stats_verified:
        first_success=min(row["sequence"] for row in observed_stats.values())
        for req in requests:
            if not failure_seq<req["sequence"]<first_success:continue
            text="\n".join(m.get("content","") for m in req["payload"]["request"].get("messages",[]))
            if MISSING in text and "FilesystemError" in text and req["payload"].get("call_id") in response_ids:
                observed_error_in_later_model=True
                break
    assistant_messages=[row["payload"]["message"]["content"] for row in raw_lines
        if row["type"]=="message_added" and row["payload"].get("message",{}).get("role")=="assistant"]
    final=assistant_messages[-1] if assistant_messages else ""
    final_verified=bool(isinstance(final,str)
        and all(name in final for name in SOURCES)
        and all(expected[name]["test_names"][0] in final for name in SOURCES)
        and re.search(r"\b"+str(counts[0])+r"\b",final)
        and re.search(r"\b"+str(counts[1])+r"\b",final)
        and re.search(r"\b"+str(sum(counts))+r"\b",final))
    evaluations=[row for row in raw_lines if row["type"]=="completion_evaluated"]
    reviewer_valid=bool(evaluations and evaluations[-1]["payload"].get("complete") is True
                         and evaluations[-1]["payload"].get("completion_claim_inconsistent") is False)
    checks={
        "hash_linked_original_source_events":True,
        "original_task_unchanged_one_user_turn":one_user and kinds["turn_started"]==1,
        "process_exited_successfully":result.get("exit_code")==0 and kinds["turn_finished"]==1,
        "original_requests_and_responses_correlated":correlated,
        "exact_native_message_roles_and_contents":role_integrity and saw_native_assistant,
        "real_allowlisted_model_selected_tools":bool(actual_calls) and not issues and not unsafe and model_selected,
        "one_deliberate_real_filesystem_error":exactly_one_failure,
        "error_observed_before_model_recovery":observed_error_in_later_model,
        "both_exact_python_source_AST_tool_results":stats_verified,
        "correct_source_derived_real_calculator":calc_correct,
        "supported_complete_final_answer":final_verified,
        "consistent_completion_evaluator_success":reviewer_valid,
        "no_extra_model_transport_retries":kinds["model_retry_scheduled"]==0 and kinds["model_call_failed"]==0,
        "no_outside_authority_effect_tools":not unsafe,
    }
    return {"status":"verified_first_pass" if all(checks.values()) else "failed_or_unverified",
            "checks":checks,"passed":sum(checks.values()),"total":len(checks),
            "source_revision":manifest["source_revision"],
            "expected_top_level_test_counts":counts,
            "original_journal_sha256":sha(journals[0].read_bytes()),
            "original_final_answer_sha256":sha(str(final).encode()),
            "real_model_request_count":len(requests),
            "actual_tool_calls":[call["tool"] for call in actual_calls],
            "reported_model_input_tokens":[resp["payload"].get("completion",{}).get("prompt_tokens") for resp in responses],
            "real_model_timing_seconds":[resp["payload"].get("completion",{}).get("elapsed_seconds") for resp in responses],
            "real_tool_receipt_timing_seconds":tool_timing,
            "original_elapsed_seconds":result.get("elapsed_seconds"),
            "no_automated_rerun_or_model_effect_from_this_verifier":True}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError("Never overwrite an independent source verdict")
    result=verify(args.repo,args.evidence)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("INDEPENDENT_AST_AGENT_VERDICT",json.dumps({
        "status":result["status"],"checks":f"{result['passed']}/{result['total']}",
        "model_requests":result["real_model_request_count"],
        "tool_names":result["actual_tool_calls"]}))
    return 0 if result["status"]=="verified_first_pass" else 3


if __name__=="__main__":
    raise SystemExit(main())
