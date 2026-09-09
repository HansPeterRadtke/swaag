from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import json
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
            "stream": True,
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

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CompletionResult:
        request_payload = {**payload, "stream": True}
        read_timeout = float(timeout_seconds) if timeout_seconds is not None else float(self.config.model.timeout_seconds)
        token_inactivity_timeout = max(float(self.config.model.progress_poll_seconds), read_timeout)
        with requests.post(
            f"{self._base}{self.config.model.completion_endpoint}",
            json=request_payload,
            stream=True,
            timeout=(self.config.model.connect_timeout_seconds, token_inactivity_timeout),
        ) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = _http_error_detail(response)
                raise requests.HTTPError(
                    f"{exc} :: {detail}",
                    request=exc.request,
                    response=exc.response,
                ) from exc

            chunks: list[str] = []
            raw_chunks: list[dict[str, Any]] = []
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            finish_reason: str | None = None
            saw_chunk = False

            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                text_line = str(line).strip()
                if not text_line:
                    continue
                if text_line.startswith("data:"):
                    text_line = text_line[5:].strip()
                if text_line == "[DONE]":
                    break
                try:
                    chunk = json.loads(text_line)
                except ValueError as exc:
                    raise ModelClientError(f"Malformed streaming completion chunk: {text_line[:400]!r}") from exc
                if not isinstance(chunk, dict):
                    raise ModelClientError(f"Unexpected streaming completion chunk: {chunk!r}")
                saw_chunk = True
                raw_chunks.append(chunk)
                if stream_callback is not None:
                    stream_callback({
                        "chunk_index": len(raw_chunks),
                        "chunk": chunk,
                        "content": str(chunk.get("content", "")) if chunk.get("content") is not None else "",
                        "stop": bool(chunk.get("stop")),
                    })
                content = chunk.get("content")
                if content is not None:
                    chunks.append(str(content))
                if chunk.get("tokens_evaluated") is not None:
                    prompt_tokens = chunk.get("tokens_evaluated")
                if chunk.get("tokens_predicted") is not None:
                    completion_tokens = chunk.get("tokens_predicted")
                if chunk.get("stop"):
                    finish_reason = "stop"

            if not saw_chunk:
                raise ModelClientError("Streaming completion response produced no chunks")
            full_text = "".join(chunks)
            if not full_text and not any(chunk.get("stop") for chunk in raw_chunks):
                raise ModelClientError(f"Streaming completion response missing content: {raw_chunks[-1] if raw_chunks else {}}")
            raw_response: dict[str, Any] = {
                "content": full_text,
                "chunks": raw_chunks,
                "stream": True,
                "stop": finish_reason == "stop",
            }
            if prompt_tokens is not None:
                raw_response["tokens_evaluated"] = prompt_tokens
            if completion_tokens is not None:
                raw_response["tokens_predicted"] = completion_tokens
            return CompletionResult(
                text=full_text,
                raw_request=request_payload,
                raw_response=raw_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
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
