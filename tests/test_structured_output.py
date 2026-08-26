from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from swaag.model import CompletionRequestPolicy
from swaag.preemption import ModelCallPreempted
from swaag.runtime import AgentRuntime
from swaag.structured_output import merge_caller_output, prepare_caller_output_spec
from swaag.task_api import TaskApi
from swaag.types import CompletionResult, ContractSpec
from swaag.workers import WorkerManager


def _closed(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class _StructuredOutputClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
        return CompletionRequestPolicy(
            "test", "server_schema", contract.mode, 10, 0.01
        )

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        self.requests.append(payload)
        if payload["contract"] == "agent_action":
            text = json.dumps(
                {
                    "assistant_message": "The audit found no remaining defects.",
                    "tool_calls": [],
                    "continue_loop": False,
                    "silent_completion": False,
                    "questions": [],
                    "status": {
                        "situation": "The audit is complete.",
                        "action": "Report the verified result.",
                        "reason": "The available evidence supports completion.",
                        "importance": "normal",
                    },
                }
            )
        else:
            assert payload["contract"] == "caller_structured_output"
            text = json.dumps({"finding": "no remaining defects", "confidence": "high"})
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


class _CancellableStructuredOutputClient(_StructuredOutputClient):
    def __init__(self) -> None:
        super().__init__()
        self.structured_started = threading.Event()

    def send_completion(self, payload: dict[str, Any], *, cancel_check=None, **kwargs):
        if payload["contract"] == "agent_action":
            return super().send_completion(payload, **kwargs)
        self.requests.append(payload)
        self.structured_started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cancel_check is not None and cancel_check():
                raise ModelCallPreempted("structured output cancellation observed")
            time.sleep(0.005)
        raise AssertionError("structured output cancellation was not observed")


class _OutputLimitedStructuredOutputClient(_StructuredOutputClient):
    def __init__(self) -> None:
        super().__init__()
        self.limited = False

    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        if payload["contract"] == "caller_structured_output" and not self.limited:
            self.limited = True
            self.requests.append(payload)
            return CompletionResult(
                text="{",
                raw_request=payload,
                raw_response={"content": "{", "stop_type": "limit"},
                prompt_tokens=None,
                completion_tokens=payload["n_predict"],
                finish_reason="length",
            )
        return super().send_completion(payload, **kwargs)


def test_caller_structured_output_recompiles_after_output_starvation(
    make_config,
) -> None:
    client = _OutputLimitedStructuredOutputClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=12_000, model__max_retries=1),
        model_client=client,
    )
    state = runtime.create_or_load_session()
    output = runtime.generate_caller_structured_output(
        state,
        original_request="Return the audit finding.",
        assistant_message="The audit found no remaining defects.",
        tool_results=[],
        semantic_schema=_closed(
            {
                "finding": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "high"]},
            }
        ),
    )

    assert output == {"finding": "no remaining defects", "confidence": "high"}
    assert len(client.requests) == 2
    assert client.requests[1]["n_predict"] > client.requests[0]["n_predict"]


def test_output_spec_separates_semantic_and_mechanical_fields() -> None:
    schema = _closed(
        {
            "finding": {"type": "string"},
            "worker": {"type": "string"},
            "attempts": {"type": "integer"},
        }
    )
    spec = prepare_caller_output_spec(
        schema,
        {"worker": "worker_id", "attempts": "run_count"},
    )

    assert spec is not None
    assert list(spec.semantic_schema["properties"]) == ["finding"]
    assert merge_caller_output(
        spec,
        {"finding": "verified"},
        {"worker_id": "worker_1", "run_count": 2},
    ) == {"finding": "verified", "worker": "worker_1", "attempts": 2}


def test_output_spec_rejects_unknown_mechanical_sources() -> None:
    with pytest.raises(ValueError, match="unsupported mechanical"):
        prepare_caller_output_spec(
            _closed({"value": {"type": "string"}}),
            {"value": "model_guess"},
        )


def test_task_api_generates_semantic_fields_and_fills_mechanics(
    make_config,
) -> None:
    client = _StructuredOutputClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=32_000), model_client=client
    )
    manager = WorkerManager(runtime)
    api = TaskApi(manager)
    schema = _closed(
        {
            "finding": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "high"]},
            "worker": {"type": "string"},
            "state": {"type": "string", "enum": ["completed"]},
            "attempts": {"type": "integer"},
        }
    )
    created = api.execute(
        "create",
        {
            "objective": "Audit the implementation and report the verified finding.",
            "output_schema": schema,
            "mechanical_fields": {
                "worker": "worker_id",
                "state": "status",
                "attempts": "run_count",
            },
            "start": True,
        },
    )
    worker_id = created["worker"]["worker_id"]

    finished = manager.wait(worker_id, timeout_seconds=10)
    inspected = api.execute("get", {"worker_id": worker_id})
    events = manager.events(worker_id)
    state = runtime.history.rebuild_from_history(
        finished.session_id, write_projections=False
    )
    manager.shutdown()

    output = inspected["structured_output"]
    assert output == {
        "finding": "no remaining defects",
        "confidence": "high",
        "worker": worker_id,
        "state": "completed",
        "attempts": 1,
    }
    caller_request = next(
        request
        for request in client.requests
        if request["contract"] == "caller_structured_output"
    )
    assert set(caller_request["json_schema"]["properties"]) == {
        "finding",
        "confidence",
    }
    assert "The audit found no remaining defects." in caller_request["prompt"]
    assert events[-1].payload["structured_output"] == output
    assert any(
        event.event_type == "caller_structured_output_created"
        for event in runtime.history.read_history(state.session_id)
    )


def test_caller_output_generation_is_cancellable_worker_inference(make_config) -> None:
    client = _CancellableStructuredOutputClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=32_000), model_client=client
    )
    manager = WorkerManager(runtime)
    worker = manager.create(
        "Return a finding, then wait while its structured form is generated.",
        output_schema=_closed({"finding": {"type": "string"}}),
    )
    manager.start(worker.worker_id)
    assert client.structured_started.wait(timeout=10)

    requested = manager.cancel(worker.worker_id, reason="stop structured output")
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert requested.status == "cancellation_requested"
    assert finished.status == "canceled"
    assert runtime.history.read_active_run(worker.session_id) is None
