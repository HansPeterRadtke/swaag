from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable

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

    def _token_timeout_seconds(self, timeout_seconds: int | None = None) -> float:
        del timeout_seconds
        raw = os.environ.get("SWAAG_MODEL_TOKEN_TIMEOUT_SECONDS", "60")
        try:
            value = float(raw)
        except ValueError:
            value = 60.0
        return max(1.0, value)

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CompletionResult:
        token_timeout_seconds = self._token_timeout_seconds(timeout_seconds)
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        started = time.monotonic()
        response = requests.post(
            f"{self._base}{self.config.model.completion_endpoint}",
            json=stream_payload,
            timeout=(self.config.model.connect_timeout_seconds, token_timeout_seconds),
            stream=True,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = _http_error_detail(response)
            raise requests.HTTPError(
                f"{exc} :: {detail}",
                request=exc.request,
                response=exc.response,
            ) from exc
        content_parts: list[str] = []
        last_body: dict[str, Any] = {}
        completion_events = 0
        reported_tokens = 0
        first_token_seconds: float | None = None
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ModelClientError(f"Unexpected streamed completion line: {raw_line!r}") from exc
                if not isinstance(item, dict):
                    raise ModelClientError(f"Unexpected streamed completion item: {item!r}")
                last_body = item
                piece = str(item.get("content", "")) if "content" in item else ""
                token_count_changed = False
                if piece:
                    content_parts.append(piece)
                    completion_events += 1
                    if first_token_seconds is None:
                        first_token_seconds = round(time.monotonic() - started, 3)
                    token_count_changed = True
                raw_predicted = item.get("tokens_predicted")
                if isinstance(raw_predicted, int) and raw_predicted >= reported_tokens:
                    reported_tokens = raw_predicted
                    token_count_changed = True
                elif completion_events > reported_tokens:
                    reported_tokens = completion_events
                if token_count_changed and progress_callback is not None:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    progress_callback(
                        {
                            "completion_tokens": reported_tokens,
                            "elapsed_seconds": round(elapsed, 3),
                            "tokens_per_second": round(reported_tokens / elapsed, 3),
                            "first_token_seconds": first_token_seconds,
                            "token_timeout_seconds": token_timeout_seconds,
                        }
                    )
                if item.get("stop"):
                    break
        except requests.Timeout as exc:
            raise requests.ReadTimeout(f"No streamed model token/event for {token_timeout_seconds:.1f} seconds") from exc
        if not content_parts and "content" not in last_body:
            raise ModelClientError(f"Streamed completion response missing content: {last_body!r}")
        elapsed_seconds = round(time.monotonic() - started, 3)
        body = dict(last_body)
        body["content"] = "".join(content_parts)
        body["stream"] = True
        body["token_timeout_seconds"] = token_timeout_seconds
        completion_tokens = body.get("tokens_predicted")
        if not isinstance(completion_tokens, int):
            completion_tokens = reported_tokens
        tokens_per_second = round(completion_tokens / max(elapsed_seconds, 1e-9), 3) if completion_tokens else 0.0
        body["elapsed_seconds"] = elapsed_seconds
        body["tokens_per_second"] = tokens_per_second
        body["first_token_seconds"] = first_token_seconds
        return CompletionResult(
            text=str(body.get("content", "")),
            raw_request=stream_payload,
            raw_response=body,
            prompt_tokens=body.get("tokens_evaluated"),
            completion_tokens=completion_tokens,
            finish_reason="stop" if body.get("stop") else None,
            elapsed_seconds=elapsed_seconds,
            tokens_per_second=tokens_per_second,
            first_token_seconds=first_token_seconds,
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


def _http_error_detail(response: requests.Response) -> str:
    text = response.text.strip()
    if not text:
        return f"http_status={response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            error_type = str(error.get("type", "")).strip()
            message = str(error.get("message", "")).strip()
            parts = [part for part in (f"http_status={response.status_code}", error_type, message) if part]
            return " | ".join(parts)
    trimmed = text[:400].replace("\n", " ").strip()
    return f"http_status={response.status_code} | body={trimmed}"
