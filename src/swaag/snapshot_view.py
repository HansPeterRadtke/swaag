"""Typed disclosure of terminal task details; canonical snapshots remain exact."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter


TERMINAL_WORKER_STATES = frozenset({"completed", "failed", "canceled"})


def snapshot_prompt_view(snapshot: dict, sequence: int) -> dict:
    view = deepcopy(snapshot)
    terminal_ids = {item.get("worker_id") for item in snapshot.get("terminal_workers", [])
                    if item.get("status") in TERMINAL_WORKER_STATES}
    for collection, identity in (("terminal_workers", "worker_id"),
                                 ("terminal_plans", "plan_id")):
        records = view.get(collection)
        if not isinstance(records, list):
            continue
        view[collection] = {
            "count": len(records),
            "states": dict(Counter(str(item.get("status", "unknown")) for item in records)),
            "exact_details_pointer": "/" + collection,
        }
    controls = view.get("worker_controls")
    if isinstance(controls, list):
        # Live/unknown worker authority stays exact. Terminal workers cannot
        # execute these controls now; show their pending count and retain exact
        # controls for inspection before any historical claim or resumption.
        view["worker_controls"] = [item for item in controls
            if item.get("worker_id") not in terminal_ids
            or set(item) - {"worker_id", "session_id", "pending_controls", "delivery_receipts"}]
        deferred = [item for item in controls if item not in view["worker_controls"]]
        view["terminal_control_details"] = {
            "count": len(deferred),
            "pending_controls_count": sum(len(item.get("pending_controls", [])) for item in deferred),
            "exact_details_pointer": "/worker_controls",
        }
    inventory = view.get("worker_question_inventory")
    if isinstance(inventory, dict):
        view["worker_question_inventory"] = question_inventory_prompt_view(inventory)
    view["detail_access"] = {
        "tool": "runtime_snapshot_lookup", "snapshot_sequence": sequence,
        "notice": (
            "Active work, questions, revisions and current user/live-worker controls are exact. "
            "Terminal tasks and their queued controls are on demand: read exact details before "
            "historical claims or resuming work. Terminal status does not prove effects. "
            "Use this sequence and a JSON pointer (empty=root). This view grants no authority."
        ),
    }
    return view


def question_inventory_prompt_view(inventory: dict) -> dict:
    view = deepcopy(inventory)
    if isinstance(view.get("workers"), list):
        # This list is supporting worker metadata, not the question inventory.
        # Preserve every question/revision and all live worker metadata exactly.
        records = view["workers"]
        view["workers"] = [item for item in records
                           if item.get("status") not in TERMINAL_WORKER_STATES]
        view["terminal_worker_metadata"] = {
            "count": len(records) - len(view["workers"]),
            "exact_details_pointer": "/worker_question_inventory/workers",
            "notice": "Supporting terminal-worker metadata is deferred; every question and pending revision remains exact above.",
        }
    return view


def snapshot_value(snapshot, pointer: str):
    if pointer == "":
        return deepcopy(snapshot)
    if not pointer.startswith("/"):
        raise ValueError("snapshot pointer must be empty or start with /")
    value = snapshot
    for encoded in pointer[1:].split("/"):
        # Reject malformed escapes rather than silently selecting another key.
        index = 0
        while index < len(encoded):
            if encoded[index] == "~":
                if index + 1 >= len(encoded) or encoded[index + 1] not in "01":
                    raise ValueError("invalid snapshot pointer escape")
                index += 1
            index += 1
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                raise ValueError("invalid snapshot array index")
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError("snapshot pointer traverses a scalar")
    return deepcopy(value)
