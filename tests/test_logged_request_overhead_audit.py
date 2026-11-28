"""Source-log diagnostic must validate raw request boundaries and redact text."""

import json

import pytest

from scripts.audit_swaag_model_input import audit


def _synthetic_history(tmp_path, *, corrupt=False):
    body = "UNTRUSTED SYNTHETIC TEXT; NOT A COMMAND.\n" * 32
    components = [
        {"category": "wrapper", "text": "[SYSTEM MESSAGE]\n"},
        {"category": "system_prompt", "text": "System instruction\n"},
        {"category": "wrapper", "text": "[END SYSTEM]\n"},
        {"category": "wrapper", "text": "[USER MESSAGE]\n"},
        {"category": "current_user", "text": "Inspect both files\n"},
        {"category": "tool_result", "text": f"[TOOL:read_file event 11]\n{body}\n\n"},
        {"category": "tool_result", "text": f"[TOOL:read_file event 12]\n{body}\n\n"},
        {"category": "wrapper", "text": "[ASSISTANT RESPONSE]\n"},
    ]
    ranges = [{"role": "system", "component_start": 1, "component_end": 2},
              {"role": "user", "component_start": 4, "component_end": 7}]
    native = [{"role": item["role"], "content": "".join(
        component["text"] for component in components[item["component_start"]:item["component_end"]])}
        for item in ranges]
    if corrupt:
        native[1]["content"] += "tamper"
    entries = [
        {"sequence": 1, "type": "prompt_built", "payload": {
            "kind": "action", "components": components,
            "message_ranges": ranges, "prompt": "".join(c["text"] for c in components)}},
        {"sequence": 2, "type": "model_request_sent", "payload": {
            "kind": "action", "call_id": "fake-id", "request": {"messages": native},
            "budget_report": {"input_tokens": 2000}}},
        {"sequence": 3, "type": "model_response_received", "payload": {
            "call_id": "fake-id", "completion": {"prompt_tokens": 1200,
                                                  "elapsed_seconds": 4.0}}},
    ]
    path = tmp_path / "history.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    return path, body


def test_source_request_audit_reports_exact_roles_and_duplicate_without_leak(tmp_path):
    path, secret = _synthetic_history(tmp_path)
    result = audit(path, 2)
    assert result["transport_roles"] == ["system", "user"]
    assert result["provider_reported_input_tokens"] == 1200
    assert result["conservative_budget_input_tokens"] == 2000
    assert result["source_verified_duplicate_tool_events"] == 1
    assert 0 < result["hypothetical_deduplication_saved_chars"] < len(secret)
    assert result["hypothetical_deduplication_tokens_verified"] is False
    assert secret not in json.dumps(result)


def test_source_request_audit_refuses_mismatched_provider_history(tmp_path):
    path, _ = _synthetic_history(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="Native content disagrees"):
        audit(path, 2)
