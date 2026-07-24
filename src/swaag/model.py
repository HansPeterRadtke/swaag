from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests

from swaag.config import AgentConfig
from swaag.schema_portability import PortableSchemaError, assert_portable_json_schema
from swaag.types import CompletionResult, ContractSpec
from swaag.utils import sha256_text, stable_json_dumps


class ModelClientError(RuntimeError):
    pass


@dataclass(slots=True)
class CompletionRequestPolicy:
    profile_name: str
    structured_output_mode: str
    effective_contract_mode: str
    effective_timeout_seconds: int
    progress_poll_seconds: float


def uses_chat_completions_transport(base_url: str, completion_endpoint: str) -> bool:
    base = base_url.rstrip("/").lower()
    endpoint = completion_endpoint.rstrip("/").lower()
    return (
        endpoint.endswith("/chat/completions")
        or "openrouter.ai" in base
        or base.endswith("/v1")
    )


def completion_url(base_url: str, completion_endpoint: str) -> str:
    base = base_url.rstrip("/")
    endpoint = completion_endpoint
    if uses_chat_completions_transport(base_url, completion_endpoint) and completion_endpoint.rstrip("/") == "/completion":
        endpoint = "/chat/completions"
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base}{endpoint}"


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

    def cache_identity(self) -> dict[str, Any]:
        """Return a stable identity for every model/server fact that can affect output."""
        configured_identity = self.config.model.model_identity.strip()
        identity: dict[str, Any] = {
            "configured_model_identity": configured_identity,
            "base_url": self.config.model.base_url.rstrip("/"),
            "completion_endpoint": self.config.model.completion_endpoint,
            "profile_name": self.config.model.profile_name,
        }
        try:
            response = requests.get(
                f"{self._base}/props",
                timeout=(self.config.model.connect_timeout_seconds, min(self.config.model.timeout_seconds, 15)),
            )
            response.raise_for_status()
            props = response.json()
            if not isinstance(props, dict):
                raise ModelClientError(f"Unexpected model props response: {props!r}")
            model_path = str(props.get("model_path", "")).strip()
            model_file: dict[str, Any] = {"path": model_path}
            if model_path:
                try:
                    stat = Path(model_path).stat()
                    model_file.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                except OSError as exc:
                    model_file["stat_error"] = exc.__class__.__name__
            output_affecting_props = {
                "model_alias": props.get("model_alias"),
                "model_file": model_file,
                "build_info": props.get("build_info"),
                "bos_token": props.get("bos_token"),
                "eos_token": props.get("eos_token"),
                "chat_template": props.get("chat_template"),
                "default_generation_settings": props.get("default_generation_settings"),
                "modalities": props.get("modalities"),
            }
            identity["server_properties_sha256"] = sha256_text(
                stable_json_dumps(output_affecting_props, indent=None)
            )
            identity["model_alias"] = props.get("model_alias")
            identity["model_file"] = model_file
            identity["server_build_info"] = props.get("build_info")
            identity["status"] = "resolved"
        except Exception as exc:
            identity["status"] = "configured_only" if configured_identity else "unresolved"
            identity["probe_error_type"] = exc.__class__.__name__
        return identity

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
        if structured_output_mode != "server_schema":
            raise ModelClientError("Every live model call must use server_schema structured output mode")
        if kind == "verification":
            timeout_seconds = self.config.model.verification_timeout_seconds
        elif live_mode and (len(prompt) > 1200 or max_tokens > 192):
            timeout_seconds = self.config.model.benchmark_timeout_seconds
        elif mode == "json_schema":
            timeout_seconds = self.config.model.structured_timeout_seconds
        else:
            raise ModelClientError(f"Unsupported unconstrained contract mode for live model call: {mode}")
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

    def _require_portable_schema(self, contract: ContractSpec) -> dict[str, Any]:
        if contract.mode != "json_schema":
            raise ModelClientError(f"Every live model call must use json_schema, got {contract.mode!r} for {contract.name}")
        if not contract.json_schema:
            raise ModelClientError(f"JSON schema contract {contract.name} is missing schema")
        try:
            assert_portable_json_schema(contract.json_schema, schema_name=contract.name)
        except PortableSchemaError as exc:
            raise ModelClientError(f"Contract {contract.name} is not portable: {exc}") from exc
        return contract.json_schema

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        schema = self._require_portable_schema(contract)
        effective_temperature = self.config.model.temperature if temperature is None else temperature
        if uses_chat_completions_transport(self.config.model.base_url, self.config.model.completion_endpoint):
            return {
                "model": self.config.model.profile_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": effective_temperature,
                "top_p": self.config.model.top_p,
                "seed": self.config.model.seed,
                "stop": list(self.config.model.stop),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": contract.name.replace(":", "_"),
                        "strict": True,
                        "schema": schema,
                    },
                },
                "provider": {"require_parameters": True},
            }
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": effective_temperature,
            "top_p": self.config.model.top_p,
            "seed": self.config.model.seed,
            "stop": list(self.config.model.stop),
            "json_schema": schema,
        }

    def _token_timeout_seconds(self, timeout_seconds: int | None = None) -> float:
        raw = os.environ.get("SWAAG_MODEL_TOKEN_TIMEOUT_SECONDS")
        if raw is None or raw.strip() == "":
            return max(1.0, float(timeout_seconds if timeout_seconds is not None else self.config.model.timeout_seconds))
        try:
            value = float(raw)
        except ValueError:
            value = float(timeout_seconds if timeout_seconds is not None else self.config.model.timeout_seconds)
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
            completion_url(self.config.model.base_url, self.config.model.completion_endpoint),
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
                piece = _completion_piece(item)
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
                if item.get("stop") or _chat_finished(item):
                    break
        except requests.Timeout as exc:
            raise requests.ReadTimeout(f"No streamed model token/event for {token_timeout_seconds:.1f} seconds") from exc
        if not content_parts and "content" not in last_body and not _chat_content(last_body):
            raise ModelClientError(f"Streamed completion response missing content: {last_body!r}")
        elapsed_seconds = round(time.monotonic() - started, 3)
        body = dict(last_body)
        body["content"] = "".join(content_parts) or _chat_content(last_body)
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


def _completion_piece(item: dict[str, Any]) -> str:
    if "content" in item:
        return str(item.get("content", ""))
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return str(delta["content"])
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"])
    return ""


def _chat_content(item: dict[str, Any]) -> str:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"])
    return ""


def _chat_finished(item: dict[str, Any]) -> bool:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    return isinstance(first, dict) and bool(first.get("finish_reason"))
