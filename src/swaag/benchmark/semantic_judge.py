from __future__ import annotations

import json
from typing import Any

from swaag.schema_portability import assert_portable_json_schema
from swaag.types import ContractSpec, HistoryEvent
from swaag.utils import stable_json_dumps


def _contract() -> ContractSpec:
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "evidence": {"type": "string"},
        },
        "required": ["passed", "evidence"],
        "additionalProperties": False,
    }
    assert_portable_json_schema(schema, schema_name="benchmark_quality_judge")
    return ContractSpec(name="benchmark_quality_judge", mode="json_schema", json_schema=schema)


def judge_prompt_quality(
    *,
    client: Any,
    context_limit: int,
    prompt: str,
    oracle: dict[str, Any],
    assistant_text: str,
    events: list[HistoryEvent],
    timeout_seconds: int,
) -> dict[str, Any]:
    observable_events = [
        {"type": event.event_type, "payload": event.payload}
        for event in events
        if event.event_type in {"tool_called", "tool_result", "tool_error", "agent_action_selected", "turn_finished"}
    ]
    judge_prompt = (
        "Judge only the observable behavior against the benchmark oracle. Do not require hidden planning, classification, or strategy state. "
        "Return passed=true only when the final answer and actual tool behavior satisfy the user request and the oracle semantically.\n\n"
        f"Original user request:\n{prompt}\n\n"
        f"Oracle:\n{stable_json_dumps(oracle, indent=2)}\n\n"
        f"Final assistant answer:\n{assistant_text}\n\n"
        f"Observable events:\n{stable_json_dumps(observable_events, indent=2)}"
    )
    contract = _contract()
    max_tokens = 512
    schema_text = stable_json_dumps(contract.json_schema or {}, indent=None)
    prompt_tokens = int(client.tokenize(judge_prompt))
    schema_tokens = int(client.tokenize(schema_text))
    if prompt_tokens + schema_tokens + max_tokens + 256 > int(context_limit):
        raise ValueError("benchmark quality judge prompt does not fit model context")
    resolved, policy = client.resolve_contract(contract, kind="benchmark_quality_judge", prompt=judge_prompt, max_tokens=max_tokens)
    request = client.build_completion_request(judge_prompt, max_tokens=max_tokens, contract=resolved, temperature=0.0)
    completion = client.send_completion(request, timeout_seconds=min(timeout_seconds, int(policy.effective_timeout_seconds)))
    payload = json.loads(completion.text)
    if not isinstance(payload, dict) or not isinstance(payload.get("passed"), bool) or not isinstance(payload.get("evidence"), str):
        raise ValueError("benchmark quality judge returned invalid constrained payload")
    return payload
