from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from swaag.config import AgentConfig
from swaag.types import CompletionResult, ContractSpec


class ModelClientError(RuntimeError):
    pass


@dataclass(slots=True)
class CompletionRequestPolicy:
    profile_name: str
    structured_output_mode: str
    effective_contract_mode: str
    effective_timeout_seconds: int
    progress_poll_seconds: float


@dataclass(slots=True)
class LlamaCppClient:
    config: AgentConfig

    @property
    def _base(self) -> str:
        return self.config.model.base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(
            f"{self._base}{self.config.model.health_endpoint}",
            timeout=(self.config.model.connect_timeout_seconds, self.config.model.timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ModelClientError(f"Unexpected health response: {payload!r}")
        return payload

    def tokenize(self, text: str) -> int:
        response = requests.post(
            f"{self._base}{self.config.model.tokenize_endpoint}",
            json={"content": text},
            timeout=(self.config.model.connect_timeout_seconds, self.config.model.timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload.get("tokens"), list):
            return len(payload["tokens"])
        if isinstance(payload.get("token_ids"), list):
            return len(payload["token_ids"])
        if isinstance(payload.get("n_tokens"), int):
            return int(payload["n_tokens"])
        raise ModelClientError(f"Unexpected tokenize response: {payload!r}")

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> CompletionRequestPolicy:
        mode = contract.mode
        structured_output_mode = self.config.model.structured_output_mode
        # `post_validate` now means "use generation-time contract
        # enforcement and then validate locally as an additional guard".
        # Core semantic calls must not silently downgrade to plain output.
        if kind == "verification":
            timeout_seconds = self.config.model.verification_timeout_seconds
        elif live_mode and (len(prompt) > 1200 or max_tokens > 192):
            timeout_seconds = self.config.model.benchmark_timeout_seconds
        elif mode in {"json_schema", "gbnf"}:
            timeout_seconds = self.config.model.structured_timeout_seconds
        else:
            timeout_seconds = self.config.model.simple_timeout_seconds
        timeout_seconds = max(timeout_seconds, self.config.model.timeout_seconds)
        return CompletionRequestPolicy(
            profile_name=self.config.model.profile_name,
            structured_output_mode=structured_output_mode,
            effective_contract_mode=mode,
            effective_timeout_seconds=timeout_seconds,
            progress_poll_seconds=self.config.model.progress_poll_seconds,
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
        policy = self.select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )
        return contract, policy

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.config.model.temperature if temperature is None else temperature,
            "top_p": self.config.model.top_p,
            "seed": self.config.model.seed,
            "stop": list(self.config.model.stop),
        }
        if contract.mode == "gbnf":
            if not contract.grammar:
                raise ModelClientError(f"GBNF contract {contract.name} is missing grammar text")
            payload["grammar"] = contract.grammar
        if contract.mode == "json_schema":
            if not contract.json_schema:
                raise ModelClientError(f"JSON schema contract {contract.name} is missing schema")
            payload["json_schema"] = contract.json_schema
        return payload

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> CompletionResult:
        response = requests.post(
            f"{self._base}{self.config.model.completion_endpoint}",
            json=payload,
            timeout=(self.config.model.connect_timeout_seconds, timeout_seconds or self.config.model.timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ModelClientError(f"Unexpected completion response: {body!r}")
        if "content" not in body:
            raise ModelClientError(f"Completion response missing 'content': {body!r}")
        return CompletionResult(
            text=str(body.get("content", "")),
            raw_request=payload,
            raw_response=body,
            prompt_tokens=body.get("tokens_evaluated"),
            completion_tokens=body.get("tokens_predicted"),
            finish_reason="stop" if body.get("stop") else None,
        )

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
        kind: str = "answer",
        live_mode: bool = False,
    ) -> CompletionResult:
        resolved_contract, policy = self.resolve_contract(
            contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )
        request = self.build_completion_request(
            prompt,
            max_tokens=max_tokens,
            contract=resolved_contract,
            temperature=temperature,
        )
        return self.send_completion(request, timeout_seconds=policy.effective_timeout_seconds)
