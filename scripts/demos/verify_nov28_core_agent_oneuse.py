#!/usr/bin/env python3
"""Independent fail-closed source-and-event oracle for one genuine SWAAG run.

No model, tool or browser action. Reads only the frozen original journal,
Git source blobs at the recorded commit, and the one-use process receipt.
Synthetic fixtures do not qualify as a real agent demonstration.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

from swaag.events import verify_event_integrity
from swaag.types import HistoryEvent
from scripts.demos.nov28_core_agent_oneuse import SOURCES, MISSING, ALLOWED_TOOLS, sha


def read_recorded(repo: Path, folder: Path) -> tuple[dict, dict, list[dict], dict]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Private genuine evidence directory missing or symlinked")
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "swaag-core-native-roles-live-v1"
            or manifest.get("tool_names") != ALLOWED_TOOLS
            or manifest.get("allow_side_effect_tools") is not False
            or manifest.get("expected_missing") != MISSING
            or manifest.get("source_repo") != str(repo)
            or manifest.get("model") != "qwen3-coder-30b-a3b-instruct-q4-k-m"
            or manifest.get("native_transport") != "/chat/completions"):
        raise ValueError("Private source/tool authority does not match the fixed one-use task")
    task = folder / "fixed-task.txt"
    cfg = folder / "read-only-config.toml"
    if (sha(task.read_bytes()) != manifest.get("task_sha256")
            or sha(cfg.read_bytes()) != manifest.get("config_sha256")):
        raise ValueError("Original task/config bytes changed")
    for name, required in SOURCES.items():
        if manifest["source_checks"][name]["required"] != required:
            raise ValueError("Frozen source function changed in manifest")
        blob = subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob",
                                       manifest["source_revision"] + ":" + name], timeout=10)
        if sha(blob) != manifest["source_checks"][name]["sha256"]:
            raise ValueError("Frozen source blob mismatches its original SHA")
        names = [node.name for node in ast.parse(blob).body if isinstance(node,
                    (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]
        if len(names) != manifest["source_checks"][name]["test_functions"] or required not in names:
            raise ValueError("Frozen AST oracle differs from actual original source")
    original_sources = {name: subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "blob", manifest["source_revision"] + ":" + name],
        timeout=10).decode() for name in SOURCES}
    receipt_path = folder / "result.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("One-use actual model process has not finished")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_revision") != manifest["source_revision"]:
        raise ValueError("Process source revision differs from frozen source")
    if receipt.get("console_sha256") != sha((folder / "real-cli-console.log").read_bytes()):
        raise ValueError("CLI console transcript has been modified")
    marker = json.loads((folder / ".one_use_started.json").read_text())
    if not marker.get("lease_id") or marker.get("owner") != "owner-20261128-swaag-core":
        raise ValueError("Live task lacks a verified resource-broker identity")
    journals = list((folder / "sessions").glob("session_*/complete_history.jsonl"))
    if len(journals) != 1 or journals[0].is_symlink() or journals[0].stat().st_size > 100_000_000:
        raise ValueError("Exactly one bounded genuine SWAAG history journal is required")
    rows = [json.loads(line) for line in journals[0].read_text().splitlines() if line.strip()]
    if not rows or len(rows) > 6000 or len({row.get("session_id") for row in rows}) != 1:
        raise ValueError("Missing, mixed or excessive original journal")
    previous = None
    for row in rows:
        event = HistoryEvent(**row)
        verify_event_integrity(event, previous)
        previous = event.hash
    if any(a["sequence"] >= b["sequence"] for a,b in zip(rows, rows[1:])):
        raise ValueError("Original durable history out of order")
    return manifest, receipt, rows, original_sources


def oracle(repo: Path, folder: Path) -> dict:
    manifest, receipt, rows, original_sources = read_recorded(repo, folder)
    types = Counter(row["type"] for row in rows)
    prompts = [r for r in rows if r["type"] == "message_added"
        and (r.get("payload",{}).get("message") or {}).get("role") == "user"]
    exact_user = len(prompts) == 1 and prompts[0]["payload"]["message"]["content"] == (folder / "fixed-task.txt").read_text()
    starts: dict[str, dict] = {}
    results: dict[str, dict] = {}
    errors: dict[str, dict] = {}
    tool_durations = []
    integrity_issues = []
    calls = []
    for row in rows:
        typ, payload = row["type"], row.get("payload") or {}
        call_id = payload.get("call_id")
        if typ == "tool_called":
            calls.append({"sequence": row["sequence"], "name": payload.get("tool_name"),
                          "input": payload.get("tool_input"), "call_id": call_id})
            if not isinstance(call_id, str) or call_id in starts:
                integrity_issues.append("duplicate_tool_call_id")
            else: starts[call_id] = row
        if typ not in ("tool_result", "tool_error"):
            continue
        src = starts.get(call_id)
        if src is None or call_id in results or call_id in errors:
            integrity_issues.append("unmatched_or_duplicate_tool_result");continue
        original = src["payload"]
        given = payload.get("validated_input") if typ == "tool_result" else payload.get("tool_input")
        if (payload.get("tool_name") != original.get("tool_name")
                or given != original.get("tool_input")
                or row["sequence"] <= src["sequence"]):
            integrity_issues.append("tool_result_input_mismatch");continue
        tool_durations.append({"tool": original.get("tool_name"),
                               "success": typ == "tool_result",
                               "seconds": round((datetime.fromisoformat(row["timestamp"])
                                - datetime.fromisoformat(src["timestamp"])).total_seconds(), 6)})
        (results if typ == "tool_result" else errors)[call_id] = row
    if set(starts) != set(results) | set(errors):
        integrity_issues.append("unreconciled_tool_action")
    unauthorized = [call for call in calls if call["name"] not in ALLOWED_TOOLS]
    failures = [row for row in errors.values() if row["payload"].get("tool_name") == "read_file"
             and row["payload"].get("tool_input",{}).get("path") == MISSING
             and row["payload"].get("error_type") == "FilesystemError"]
    failed_one = len(failures) == 1 and sum(call["name"] == "read_file" and call["input"].get("path") == MISSING
                                            for call in calls) == 1
    fail_seq = failures[0]["sequence"] if failed_one else None
    listing = next((row for row in results.values() if row["payload"].get("tool_name") == "list_files"
                   and row["payload"].get("validated_input") == {"path": "tests"}
                   and fail_seq is not None and row["sequence"] > fail_seq
                   and all(name in row["payload"].get("output",{}).get("entries",[]) for name in SOURCES)), None)
    read_map = {}
    for row in results.values():
        payload = row["payload"]
        if payload.get("tool_name") == "read_file":
            name = payload.get("validated_input",{}).get("path")
            if name in SOURCES and payload.get("output",{}).get("text") == original_sources[name]:
                read_map[name] = row
    valid_reads = bool(listing and len(read_map) == len(SOURCES)
                       and all(row["sequence"] > listing["sequence"] for row in read_map.values()))
    counts = [manifest["source_checks"][name]["test_functions"] for name in SOURCES]
    calculators = [row for row in results.values() if row["payload"].get("tool_name") == "calculator"]
    expected_sum = sum(counts)
    correct_calc = any(row["payload"].get("output",{}).get("result") == expected_sum
         and str(row["payload"].get("output",{}).get("expression", "")).replace(" ", "")
           in (f"{counts[0]}+{counts[1]}", f"{counts[1]}+{counts[0]}")
         and valid_reads and all(row["sequence"] > read["sequence"] for read in read_map.values())
         for row in calculators)
    requests = [row for row in rows if row["type"] == "model_request_sent"]
    responses = [row for row in rows if row["type"] == "model_response_received"]
    request_by_id = {row["payload"].get("call_id"): row for row in requests}
    response_by_id = {row["payload"].get("call_id"): row for row in responses}
    correct_call_pairs = bool(requests and len(request_by_id) == len(requests)
        and len(response_by_id) == len(responses)
        and request_by_id.keys() == response_by_id.keys()
        and all(response_by_id[cid]["sequence"] > row["sequence"] for cid,row in request_by_id.items()))
    native_role_valid = True
    native_role_lists = []
    source_prompt_match = True
    prompt_built = [row for row in rows if row["type"] == "prompt_built"]
    for request in requests:
        r = request["payload"]["request"]
        messages = r.get("messages", [])
        native_role_lists.append([message.get("role") for message in messages])
        if (not messages or messages[0].get("role") != "system"
                or any(m.get("role") not in {"system", "user", "assistant"} for m in messages)):
            native_role_valid = False
        if "prompt" in r and not messages:
            native_role_valid = False
        preceding = next((event for event in reversed(prompt_built)
                          if event["sequence"] < request["sequence"]
                          and event["payload"].get("kind") == request["payload"].get("kind")), None)
        if preceding is None:
            source_prompt_match = False
        else:
            original = preceding["payload"]
            ranges, components = original.get("message_ranges", []), original.get("components", [])
            if len(ranges) != len(messages):
                source_prompt_match = False
            else:
                for rng, item in zip(ranges, messages):
                    body = "".join(c["text"] for c in components[
                        rng["component_start"]:rng["component_end"]])
                    if item.get("role") != rng.get("role") or item.get("content") != body:
                        source_prompt_match = False
    # An actual model step after the missing-path error must see the real
    # recorded tool error BEFORE the directory listing; same-action planning
    # is insufficient to establish model-driven recovery.
    error_seen_by_model = False
    if fail_seq is not None and listing is not None:
        for req in requests:
            if not fail_seq < req["sequence"] < listing["sequence"]: continue
            payload = req["payload"].get("request", {})
            text = "\n".join(m.get("content", "") for m in payload.get("messages", []))
            cid = req["payload"].get("call_id")
            if (MISSING in text and "FilesystemError" in text
                    and cid in response_by_id
                    and response_by_id[cid]["sequence"] < listing["sequence"]):
                error_seen_by_model = True
                break
    selected_events = [row for row in rows if row["type"] == "agent_action_selected"]
    model_action_receipts = True
    for call in calls:
        selected = next((x for x in reversed(selected_events) if x["sequence"] < call["sequence"]),None)
        if selected is None or not any(isinstance(item, dict)
            and item.get("tool_name") == call["name"] and item.get("arguments") == call["input"]
            for item in ((selected.get("payload",{}).get("action") or {}).get("tool_calls") or [])):
            model_action_receipts = False
    completions = [row for row in rows if row["type"] == "message_added"
                   and (row.get("payload",{}).get("message") or {}).get("role") == "assistant"]
    final_text = completions[-1]["payload"]["message"]["content"] if completions else ""
    final_source_accuracy = (isinstance(final_text, str)
        and all(name in final_text for name in SOURCES)
        and all(name in final_text for name in SOURCES.values())
        and re.search(r"\b" + str(expected_sum) + r"\b", final_text) is not None)
    duration = [round(row["payload"].get("completion",{}).get("elapsed_seconds",0),3) for row in responses]
    input_tokens = [row["payload"].get("completion",{}).get("prompt_tokens") for row in responses]
    checks = {
        "immutable_original_input": exact_user,
        "genuine_sourced_event_chain": True, # checked above, raises if invalid
        "process_zero_and_complete": receipt.get("child_returncode") == 0 and receipt.get("terminal_turns") == 1,
        "all_model_request_response_ids_correlated": correct_call_pairs,
        "provider_native_roles_in_actual_requests": native_role_valid,
        "source_components_equal_transmitted_messages": source_prompt_match,
        "model_selected_real_allowlisted_tool_calls": bool(calls) and not unauthorized and model_action_receipts,
        "one_true_missing_file_error": failed_one,
        "model_saw_error_before_recovery": error_seen_by_model,
        "real_directory_recovery": listing is not None,
        "two_exact_source_reads_after_listing": valid_reads,
        "real_calculator_result_after_reads": correct_calc,
        "accurate_supported_final_answer": bool(final_source_accuracy),
        "no_model_retry_or_live_tool_failure": types["model_retry_scheduled"] == 0 and types["model_call_failed"] == 0,
        "no_unsafe_external_tools": not unauthorized,
        "one_original_user_turn": types["turn_started"] == 1,
    }
    result = {"status": "verified_first_pass" if all(checks.values()) else "failed_or_unverified",
              "checks": checks, "passed": sum(checks.values()), "checked": len(checks),
              "source_revision": manifest["source_revision"], "source_sha256": {
                  name: manifest["source_checks"][name]["sha256"] for name in SOURCES},
              "prompt_tokens_observed_by_provider": input_tokens,
              "model_elapsed_seconds_per_response": duration,
              "tool_elapsed_seconds_per_receipt": tool_durations,
              "total_process_elapsed_seconds": receipt.get("elapsed_seconds"),
              "input_roles_per_request": native_role_lists,
              "tool_calls": [{"name": call["name"], "input": call["input"]} for call in calls],
              "tool_error_count": len(errors), "model_request_count": len(requests),
              "model_response_count": len(responses), "final_answer_sha256": sha(str(final_text).encode()),
              "original_journal_sha256": sha(next((folder / "sessions").glob("session_*/complete_history.jsonl")).read_bytes()),
              "frozen_test_count_sum": expected_sum,
              "not_a_voice_or_product_release": True}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError("Refusing to overwrite earlier independent evidence")
    verdict = oracle(args.repo, args.evidence)
    args.out.write_text(json.dumps(verdict, indent=2) + "\n")
    print("INDEPENDENT_SWAAG_CORE_VERDICT", json.dumps({
        "status": verdict["status"], "passed": verdict["passed"], "checked": verdict["checked"],
        "model_requests": verdict["model_request_count"], "tool_calls": len(verdict["tool_calls"])}))
    return 0 if verdict["status"] == "verified_first_pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
