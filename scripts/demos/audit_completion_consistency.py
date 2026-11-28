#!/usr/bin/env python3
"""Read-only source-backed audit of a SWAAG completion-contract contradiction.

No model calls, no tool execution, no raw user text or prompts in output. The
original append-only journal is checked against every event hash and schema.
Use the exact private journal path only in an owner-private environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swaag.events import verify_event_integrity
from swaag.types import HistoryEvent


def audit(journal: Path) -> dict:
    if not journal.is_file() or journal.is_symlink():
        raise ValueError("Expected one actual nonsymlink durable SWAAG event journal")
    responses: list[dict] = []
    evaluated: list[dict] = []
    previous = None
    event_count = 0
    with journal.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            raw = json.loads(line)
            event = HistoryEvent(**raw)
            verify_event_integrity(event, previous)
            if event.sequence != event_count + 1:
                raise ValueError("Unexpected missing/out-of-order source events")
            previous = event.hash
            event_count += 1
            if event.event_type == "model_response_received" and event.payload.get("kind") == "completion_evaluation":
                completion = event.payload.get("completion") or {}
                text = completion.get("text", "")
                try:
                    parsed = json.loads(text)
                except (ValueError, TypeError):
                    continue
                if isinstance(parsed, dict) and isinstance(parsed.get("complete"), bool):
                    responses.append({"sequence": event.sequence, "complete": parsed["complete"],
                                      "remaining": parsed.get("remaining_work", []),
                                      "call_id": event.payload.get("call_id")})
            if event.event_type == "completion_evaluated":
                evaluated.append({"sequence": event.sequence,
                                  "complete": event.payload.get("complete")})
    inconsistent = []
    for response in responses:
        if response["complete"] is not True or not response["remaining"]:
            continue
        next_verdict = next((row for row in evaluated if row["sequence"] > response["sequence"]), None)
        inconsistent.append({
            "model_response_sequence": response["sequence"],
            "runtime_verdict_sequence": next_verdict["sequence"] if next_verdict else None,
            "runtime_accepted_complete": next_verdict["complete"] if next_verdict else None,
            "raw_claim_complete": True,
            "raw_remaining_items": len(response["remaining"]),
            "contradictory_model_verdict": True,
        })
    return {"event_count": event_count, "source_integrity_verified": True,
            "completion_model_responses": len(responses),
            "completion_verdict_events": len(evaluated),
            "contradictory_complete_verdicts": inconsistent,
            "no_inference_or_tool_execution": True,
            "raw_prompts_or_user_text_emitted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.journal), indent=2))


if __name__ == "__main__":
    main()
