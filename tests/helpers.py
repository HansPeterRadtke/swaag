from __future__ import annotations

import json
import re
from typing import Any

from swaag.failure import classify_failure_from_payload
from swaag.model import CompletionRequestPolicy
from swaag.strategy import strategy_from_payload
from swaag.types import CompletionResult, ContractSpec


class FakeModelClient:
    is_deterministic_test_client = True

    def __init__(self, responses: list[Any] | None = None, *, contract_responses: dict[str, list[Any]] | None = None):
        self._responses = list(responses or [])
        self._contract_responses = {key: list(value) for key, value in (contract_responses or {}).items()}
        self._pending_tool_inputs: dict[str, list[str]] = {}
        self.requests: list[dict[str, Any]] = []
        self.tokenize_requests: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tokenize(self, text: str) -> int:
        self.tokenize_requests.append(text)
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def build_completion_request(self, prompt: str, *, max_tokens: int, contract, temperature: float | None = None) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "contract": contract.name,
        }
        if contract.json_schema:
            payload["json_schema"] = contract.json_schema
        return payload

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> CompletionRequestPolicy:
        return CompletionRequestPolicy(
            profile_name="test",
            structured_output_mode="server_schema",
            effective_contract_mode=contract.mode,
            effective_timeout_seconds=30,
            progress_poll_seconds=0.05,
        )

    def resolve_contract(
        self,
        contract: ContractSpec,
        *,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> tuple[ContractSpec, CompletionRequestPolicy]:
        return contract, self.select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
    ) -> CompletionResult:
        self.requests.append(payload)
        contract_name = str(payload.get("contract", ""))
        response = None
        contract_queue = self._contract_responses.get(contract_name)
        if contract_queue:
            response = contract_queue.pop(0)
        elif contract_name in {"yes_no", "summary", "agent_action"} and not self._responses:
            response = self._auto_frontend_response(payload)
        elif self._responses:
            response = self._responses.pop(0)
        else:
            raise AssertionError("No fake model responses left")
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(payload=payload)
        if isinstance(response, CompletionResult):
            return response
        if not isinstance(response, str):
            raise TypeError(f"Unsupported fake response: {response!r}")
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )

    def complete(self, prompt: str, *, max_tokens: int, contract, temperature: float | None = None) -> CompletionResult:
        return self.send_completion(self.build_completion_request(prompt, max_tokens=max_tokens, contract=contract, temperature=temperature))

    def _auto_frontend_response(self, payload: dict[str, Any]) -> str:
        contract_name = str(payload.get("contract", ""))
        if contract_name == "yes_no":
            return json.dumps({"answer": "yes"})
        if contract_name == "summary":
            return json.dumps({"summary": "test summary", "preserve_recent_messages": 0})
        if contract_name == "agent_action":
            return json.dumps({"assistant_message": "done", "tool_calls": [], "continue_loop": False})
        raise AssertionError(f"Unhandled current contract in test model: {contract_name}")
