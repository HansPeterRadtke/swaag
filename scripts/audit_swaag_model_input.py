#!/usr/bin/env python3
"""Read-only, content-redacted audit of one durable SWAAG model request.

Print aggregates only. Never copy private source text or model output into Git.
Usage: python scripts/audit_swaag_model_input.py /path/to/complete_history.jsonl --request-sequence 4484
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(path: Path, request_sequence: int) -> dict:
    previous_prompt = None
    selected = None
    recorded_response = None
    call_id = None
    with path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            sequence = event.get("sequence")
            event_type = event.get("type", event.get("event_type"))
            payload = event.get("payload") or {}
            if selected is None:
                if event_type == "prompt_built":
                    previous_prompt = event
                if sequence == request_sequence:
                    if event_type != "model_request_sent":
                        raise ValueError("Selected event is not a model_request_sent")
                    if previous_prompt is None or previous_prompt["payload"].get("kind") != payload.get("kind"):
                        raise ValueError("No matching preceding prompt-built event")
                    selected = (previous_prompt, event)
                    call_id = payload.get("call_id")
            elif (event_type == "model_response_received"
                  and payload.get("call_id") == call_id):
                recorded_response = event
                break
    if selected is None:
        raise ValueError("Requested model call was not found in history")
    prompt, request = selected
    prompt_data = prompt["payload"]
    request_data = request["payload"]
    components = prompt_data["components"]
    request_messages = request_data["request"].get("messages", [])
    ranges = prompt_data["message_ranges"]
    if len(request_messages) != len(ranges):
        raise ValueError("Recorded transport roles do not match stored message ranges")
    for rng, msg in zip(ranges, request_messages):
        if rng["role"] != msg["role"]:
            raise ValueError("Native role mismatch in recorded request")
        text = "".join(c["text"] for c in components[rng["component_start"]:rng["component_end"]])
        if text != msg["content"]:
            raise ValueError("Native content disagrees with prompt component ranges")
    raw_prompt = prompt_data["prompt"]
    if raw_prompt != "".join(c["text"] for c in components):
        raise ValueError("Prompt component concatenation differs from original")
    seen: dict[tuple[str, str], int] = {}
    duplicate_count = 0
    hypothetical_savings = 0
    tool_chars = 0
    for component in components:
        text = component["text"]
        if component.get("category") != "tool_result":
            continue
        tool_chars += len(text)
        match = re.fullmatch(r"\[TOOL:([^\] ]+) event (\d+)\]\n(.*)\n\n", text, re.S)
        if not match or len(match.group(3)) < 512:
            continue
        tool_name, source_seq, body = match.group(1), int(match.group(2)), match.group(3)
        key = (tool_name, hashlib.sha256(body.encode("utf-8")).hexdigest())
        first_seq = seen.get(key)
        if first_seq is None:
            seen[key] = source_seq
        elif source_seq != first_seq:
            reference = (
                "[BYTE-IDENTICAL REPEATED TOOL OUTPUT; full source retained at "
                f"event sequence={first_seq}; this source event sequence={source_seq}; sha256={key[1]}. "
                "Both originals remain authoritative and retrievable using history_window. "
                "No repeated tool action is implied.]"
            )
            hypothetical_savings += len(body) - len(reference)
            duplicate_count += 1
    observed = (recorded_response or {}).get("payload", {}).get("completion", {})
    input_tokens = observed.get("prompt_tokens")
    budget_tokens = request_data.get("budget_report", {}).get("input_tokens")
    return {
        "request_event_sequence": request["sequence"],
        "prompt_event_sequence": prompt["sequence"],
        "response_event_sequence": recorded_response["sequence"] if recorded_response else None,
        "transport_roles": [message["role"] for message in request_messages],
        "transport_content_chars": [len(message["content"]) for message in request_messages],
        "prompt_chars": len(raw_prompt),
        "tool_component_chars": tool_chars,
        "provider_reported_input_tokens": input_tokens,
        "conservative_budget_input_tokens": budget_tokens,
        "provider_elapsed_seconds": observed.get("elapsed_seconds"),
        "provider_first_token_seconds": observed.get("first_token_seconds"),
        "source_verified_duplicate_tool_events": duplicate_count,
        "hypothetical_deduplication_saved_chars": hypothetical_savings,
        "hypothetical_deduplicated_prompt_chars": len(raw_prompt) - hypothetical_savings,
        "hypothetical_deduplication_tokens_verified": False,
        "contains_private_source_text": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_file", type=Path)
    parser.add_argument("--request-sequence", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.history_file, args.request_sequence), indent=2))


if __name__ == "__main__":
    main()
