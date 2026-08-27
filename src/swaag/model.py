from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from urllib.parse import urlparse
import time
import threading
from pathlib import Path
from typing import Any, Callable

import requests

from swaag.config import AgentConfig
from swaag.schema_portability import PortableSchemaError, assert_portable_json_schema
from swaag.preemption import ModelCallPreempted
from swaag.tokens import ConservativeEstimator, CountResult
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


def stable_llama_server_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Select output-affecting /props fields and exclude runtime-instance noise."""
    generation = props.get("default_generation_settings")
    return {
        "model_alias": props.get("model_alias"),
        "model_path": props.get("model_path"),
        "model_ftype": props.get("model_ftype"),
        "default_generation_settings": (
            generation if isinstance(generation, dict) else {}
        ),
        "chat_template_sha256": sha256_text(str(props.get("chat_template", ""))),
        "chat_template_caps": props.get("chat_template_caps"),
        "bos_token": props.get("bos_token"),
        "eos_token": props.get("eos_token"),
        "modalities": props.get("modalities"),
        "build_info": props.get("build_info"),
    }


def stable_openai_model_metadata(model: dict[str, Any]) -> dict[str, Any]:
    """Keep model/protocol facts while excluding prices and request-time noise."""
    top_provider = model.get("top_provider")
    return {
        "id": model.get("id"),
        "canonical_slug": model.get("canonical_slug"),
        "owned_by": model.get("owned_by"),
        "root": model.get("root"),
        "parent": model.get("parent"),
        "context_length": model.get("context_length"),
        "context_window": model.get("context_window"),
        "max_context_length": model.get("max_context_length"),
        "max_model_len": model.get("max_model_len"),
        "max_position_embeddings": model.get("max_position_embeddings"),
        "architecture": model.get("architecture"),
        "default_parameters": model.get("default_parameters"),
        "supported_parameters": model.get("supported_parameters"),
        "top_provider": (
            {
                "context_length": top_provider.get("context_length"),
                "max_completion_tokens": top_provider.get("max_completion_tokens"),
            }
            if isinstance(top_provider, dict)
            else None
        ),
        "shutdown_date": model.get("shutdown_date"),
        "expiration_date": model.get("expiration_date"),
    }


@dataclass(slots=True)
class LlamaCppClient:
    config: AgentConfig
    _remote_prompt_tokens: dict[str, int] = field(default_factory=dict, init=False)

    @property
    def _base(self) -> str:
        return self.config.model.base_url.rstrip("/")

    @property
    def _uses_chat_transport(self) -> bool:
        return uses_chat_completions_transport(
            self.config.model.base_url,
            self.config.model.completion_endpoint,
        )

    @property
    def _is_openrouter(self) -> bool:
        return "openrouter.ai" in self._base.lower()

    def _authorization_headers(self) -> dict[str, str]:
        env_name = self.config.model.api_key_env.strip()
        if not env_name:
            return {}
        token = os.environ.get(env_name, "").strip()
        if not token:
            raise ModelClientError(
                f"Configured model bearer-token environment variable {env_name!r} is empty"
            )
        return {"Authorization": f"Bearer {token}"}

    def _request_headers_kwargs(self) -> dict[str, dict[str, str]]:
        headers = self._authorization_headers()
        return {"headers": headers} if headers else {}

    def _remote_models_payload(self) -> dict[str, Any]:
        response = requests.get(
            f"{self._base}/models",
            timeout=(
                self.config.model.connect_timeout_seconds,
                min(self.config.model.timeout_seconds, 15),
            ),
            **self._request_headers_kwargs(),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ModelClientError(f"Unexpected OpenAI models response: {payload!r}")
        return payload

    def _remote_model(self) -> dict[str, Any]:
        models = [
            item
            for item in self._remote_models_payload()["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        configured = self.config.model.profile_name.strip()
        matches = [item for item in models if item.get("id") == configured]
        if len(matches) == 1:
            return matches[0]
        available = sorted(str(item["id"]) for item in models)
        raise ModelClientError(
            "Configured model profile is not uniquely available from GET /models: "
            f"configured={configured!r} available={available!r}"
        )

    @staticmethod
    def _remote_context_from_model(model: dict[str, Any]) -> tuple[int, str] | None:
        candidates: list[tuple[str, Any]] = [
            ("context_length", model.get("context_length")),
            ("max_model_len", model.get("max_model_len")),
            ("max_context_length", model.get("max_context_length")),
            ("context_window", model.get("context_window")),
            ("max_position_embeddings", model.get("max_position_embeddings")),
            ("n_ctx", model.get("n_ctx")),
        ]
        top_provider = model.get("top_provider")
        if isinstance(top_provider, dict):
            candidates.append(
                ("top_provider.context_length", top_provider.get("context_length"))
            )
        for source, value in candidates:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value), f"openai_models:{source}"
        return None

    @staticmethod
    def _remote_accounting_envelope(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, int]]]:
        pieces: list[str] = []
        offsets: list[dict[str, int]] = []
        cursor = 0
        for index, message in enumerate(messages, start=1):
            content = message["content"]
            prefix = (
                f"<swaag-openai-message index={index} role={message['role']} bytes={len(content.encode('utf-8'))}>\n"
            )
            pieces.append(prefix)
            cursor += len(prefix)
            start = cursor
            pieces.append(content)
            cursor += len(content)
            offsets.append({"start": start, "end": cursor})
            suffix = "\n</swaag-openai-message>\n"
            pieces.append(suffix)
            cursor += len(suffix)
        pieces.append("<swaag-openai-generation/>\n")
        return "".join(pieces), offsets

    def _remote_tokenize_url(self) -> str:
        endpoint = self.config.model.tokenize_endpoint.strip()
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        parsed = urlparse(self._base)
        if endpoint.startswith("/"):
            return f"{parsed.scheme}://{parsed.netloc}{endpoint}"
        return f"{self._base}/{endpoint}"

    def _remote_tokenize_messages(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[int, int | None]:
        response = requests.post(
            self._remote_tokenize_url(),
            json={
                "model": self.config.model.profile_name,
                "messages": messages,
                "add_generation_prompt": True,
            },
            timeout=(
                self.config.model.connect_timeout_seconds,
                min(self.config.model.timeout_seconds, 30),
            ),
            **self._request_headers_kwargs(),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ModelClientError(f"Unexpected remote tokenize response: {payload!r}")
        count = payload.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            tokens = payload.get("tokens")
            if isinstance(tokens, list):
                count = len(tokens)
            else:
                raise ModelClientError(
                    f"Remote tokenize response lacks an exact count: {payload!r}"
                )
        capacity = payload.get("max_model_len")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            capacity = None
        return int(count), capacity

    def health(self) -> dict[str, Any]:
        if self._uses_chat_transport:
            model = self._remote_model()
            return {
                "status": "ok",
                "transport": "openai_chat_completions",
                "model": str(model.get("id", "")),
            }
        response = requests.get(
            f"{self._base}{self.config.model.health_endpoint}",
            timeout=(self.config.model.connect_timeout_seconds, self.config.model.timeout_seconds),
            **self._request_headers_kwargs(),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ModelClientError(f"Unexpected health response: {payload!r}")
        return payload

    def _local_server_process_identity(self) -> dict[str, Any] | None:
        parsed = urlparse(self.config.model.base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        proc_root = Path("/proc")
        try:
            candidates = []
            for child in proc_root.iterdir():
                if not child.name.isdigit():
                    continue
                cmdline_path = child / "cmdline"
                try:
                    raw = cmdline_path.read_bytes()
                except OSError:
                    continue
                if not raw:
                    continue
                args = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
                if not args or "llama-server" not in Path(args[0]).name:
                    continue
                matches_port = any(
                    arg == str(port) and index > 0 and args[index - 1] in {"--port", "-p"}
                    for index, arg in enumerate(args)
                )
                if not matches_port:
                    continue
                exe = Path(args[0])
                executable: dict[str, Any] = {"path": str(exe)}
                try:
                    stat = exe.stat()
                    executable.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                except OSError as exc:
                    executable["stat_error"] = exc.__class__.__name__
                candidates.append({"argv": args, "executable": executable})
            if not candidates:
                return None
            return sorted(candidates, key=lambda item: stable_json_dumps(item, indent=None))[0]
        except OSError:
            return None

    def cache_identity(self) -> dict[str, Any]:
        """Return a stable identity for every model/server fact that can affect output."""
        configured_identity = self.config.model.model_identity.strip()
        identity: dict[str, Any] = {
            "configured_model_identity": configured_identity,
            "base_url": self.config.model.base_url.rstrip("/"),
            "completion_endpoint": self.config.model.completion_endpoint,
            "profile_name": self.config.model.profile_name,
        }
        if self._uses_chat_transport:
            try:
                model = self._remote_model()
                stable_model = stable_openai_model_metadata(model)
                identity["server_properties_sha256"] = sha256_text(
                    stable_json_dumps(stable_model, indent=None)
                )
                identity["model_alias"] = model.get("id")
                identity["remote_model_metadata"] = stable_model
                identity["transport"] = "openai_chat_completions"
                identity["status"] = "resolved"
            except Exception as exc:
                identity["status"] = "configured_only" if configured_identity else "unresolved"
                identity["probe_error_type"] = exc.__class__.__name__
            return identity
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
            stable_props = stable_llama_server_properties(props)
            stable_props["model_file"] = model_file
            identity["server_properties_sha256"] = sha256_text(
                stable_json_dumps(stable_props, indent=None)
            )
            identity["model_alias"] = props.get("model_alias")
            identity["model_file"] = model_file
            identity["server_build_info"] = props.get("build_info")
            local_process_identity = self._local_server_process_identity()
            if local_process_identity is not None:
                identity["local_server_process_sha256"] = sha256_text(
                    stable_json_dumps(local_process_identity, indent=None)
                )
                identity["local_server_process"] = local_process_identity
            identity["status"] = "resolved"
        except Exception as exc:
            identity["status"] = "configured_only" if configured_identity else "unresolved"
            identity["probe_error_type"] = exc.__class__.__name__
        return identity

    def server_context_limit(self) -> int:
        if self._uses_chat_transport:
            return self.context_limit_resolution()[0]
        response = requests.get(
            f"{self._base}/props",
            timeout=(
                self.config.model.connect_timeout_seconds,
                min(self.config.model.timeout_seconds, 15),
            ),
        )
        response.raise_for_status()
        props = response.json()
        if not isinstance(props, dict):
            raise ModelClientError(f"Unexpected model props response: {props!r}")
        generation = props.get("default_generation_settings")
        if not isinstance(generation, dict):
            raise ModelClientError("Model props are missing default_generation_settings")
        n_ctx = generation.get("n_ctx")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ModelClientError(f"Model props contain invalid n_ctx: {n_ctx!r}")
        return int(n_ctx)

    def _prompt_protocol_identity(self) -> dict[str, str]:
        if self._uses_chat_transport:
            model = self._remote_model()
            stable_model = stable_openai_model_metadata(model)
            return {
                "serialization": "provider_opaque_openai_chat_v1",
                "model_metadata_sha256": sha256_text(
                    stable_json_dumps(stable_model, indent=None)
                ),
                "model_alias": str(model.get("id", "")),
                "configured_model_identity": self.config.model.model_identity.strip(),
                "completion_endpoint": completion_url(
                    self.config.model.base_url,
                    self.config.model.completion_endpoint,
                ),
            }
        response = requests.get(
            f"{self._base}/props",
            timeout=(
                self.config.model.connect_timeout_seconds,
                min(self.config.model.timeout_seconds, 15),
            ),
        )
        response.raise_for_status()
        props = response.json()
        if not isinstance(props, dict):
            raise ModelClientError(f"Unexpected model props response: {props!r}")
        template = props.get("chat_template")
        if not isinstance(template, str) or not template.strip():
            raise ModelClientError("Model props are missing a usable chat_template")
        model_path = str(props.get("model_path", ""))
        model_file: dict[str, Any] = {"path": model_path}
        if model_path:
            try:
                stat = Path(model_path).stat()
                model_file.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            except OSError as exc:
                model_file["stat_error"] = exc.__class__.__name__
        generation = props.get("default_generation_settings")
        n_ctx = generation.get("n_ctx") if isinstance(generation, dict) else None
        return {
            "chat_template_sha256": sha256_text(template),
            "chat_template_caps_sha256": sha256_text(
                stable_json_dumps(props.get("chat_template_caps"), indent=None)
            ),
            "bos_token_sha256": sha256_text(str(props.get("bos_token", ""))),
            "eos_token_sha256": sha256_text(str(props.get("eos_token", ""))),
            "context_limit": str(n_ctx if isinstance(n_ctx, int) else ""),
            "model_file_sha256": sha256_text(
                stable_json_dumps(model_file, indent=None)
            ),
            "model_alias": str(props.get("model_alias", "")),
            "server_build_info": str(props.get("build_info", "")),
        }

    def render_chat_prompt(self, messages: list[dict[str, str]]) -> dict[str, str]:
        if not messages or any(
            not isinstance(item, dict)
            or item.get("role") not in {"system", "user", "assistant"}
            or not isinstance(item.get("content"), str)
            for item in messages
        ):
            raise ModelClientError("Chat-template messages are invalid")
        if self._uses_chat_transport:
            before = self._prompt_protocol_identity()
            prompt, message_offsets = self._remote_accounting_envelope(messages)
            token_count = ""
            tokenizer_context_limit = ""
            token_strategy = "conservative_estimator_required"
            try:
                exact_count, discovered_limit = self._remote_tokenize_messages(messages)
            except Exception:
                pass
            else:
                self._remote_prompt_tokens[sha256_text(prompt)] = exact_count
                token_count = str(exact_count)
                tokenizer_context_limit = str(discovered_limit or "")
                token_strategy = "provider_tokenize_messages"
            after = self._prompt_protocol_identity()
            if before != after:
                raise ModelClientError(
                    "Model metadata changed while serializing the request"
                )
            return {
                "prompt": prompt,
                "prompt_protocol_sha256": sha256_text(
                    stable_json_dumps(before, indent=None)
                ),
                "prompt_serialization_exact": "false",
                "prompt_serialization_strategy": "provider_opaque_openai_chat_v1",
                "message_content_offsets": stable_json_dumps(
                    message_offsets,
                    indent=None,
                ),
                "input_token_count": token_count,
                "input_token_strategy": token_strategy,
                "tokenizer_context_limit": tokenizer_context_limit,
                **before,
            }
        before = self._prompt_protocol_identity()
        response = requests.post(
            f"{self._base}/apply-template",
            json={"messages": messages},
            timeout=(
                self.config.model.connect_timeout_seconds,
                min(self.config.model.timeout_seconds, 30),
            ),
        )
        response.raise_for_status()
        payload = response.json()
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ModelClientError(f"Unexpected apply-template response: {payload!r}")
        after = self._prompt_protocol_identity()
        if before != after:
            raise ModelClientError(
                "Model or chat template changed while serializing the request"
            )
        return {
            "prompt": prompt,
            "prompt_protocol_sha256": sha256_text(
                stable_json_dumps(before, indent=None)
            ),
            **before,
        }

    def verify_prompt_protocol(self, prompt_protocol_sha256: str) -> None:
        current = sha256_text(
            stable_json_dumps(self._prompt_protocol_identity(), indent=None)
        )
        if current != prompt_protocol_sha256:
            raise ModelClientError(
                "Model or chat template changed after context compilation"
            )

    def server_slot_count(self) -> int:
        if self._uses_chat_transport:
            return 1
        response = requests.get(
            f"{self._base}/props",
            timeout=(
                self.config.model.connect_timeout_seconds,
                min(self.config.model.timeout_seconds, 15),
            ),
        )
        response.raise_for_status()
        props = response.json()
        if not isinstance(props, dict):
            raise ModelClientError(f"Unexpected model props response: {props!r}")
        total_slots = props.get("total_slots")
        if (
            not isinstance(total_slots, int)
            or isinstance(total_slots, bool)
            or total_slots <= 0
        ):
            raise ModelClientError(
                f"Model props contain invalid total_slots: {total_slots!r}"
            )
        return int(total_slots)

    def context_limit_resolution(self) -> tuple[int, str]:
        if self._uses_chat_transport:
            model = self._remote_model()
            discovered = self._remote_context_from_model(model)
            if discovered is not None:
                return discovered
            try:
                _count, tokenizer_limit = self._remote_tokenize_messages(
                    [
                        {"role": "system", "content": "Capacity probe."},
                        {"role": "user", "content": "Count this request."},
                    ]
                )
            except Exception:
                tokenizer_limit = None
            if tokenizer_limit is not None:
                return tokenizer_limit, "provider_tokenize:max_model_len"
            fallback = self.config.model.remote_context_limit_fallback
            if fallback > 0:
                return fallback, "configured:model.remote_context_limit_fallback"
            raise ModelClientError(
                "Remote OpenAI-compatible backend exposed no context capacity; "
                "set model.remote_context_limit_fallback explicitly only when the deployed limit is known"
            )
        return self.server_context_limit(), "server_props:n_ctx"

    def tokenize(self, text: str) -> int:
        if self._uses_chat_transport:
            recorded = self._remote_prompt_tokens.get(sha256_text(text))
            if recorded is not None:
                return recorded
            raise ModelClientError(
                "Exact tokenization is unavailable for this opaque OpenAI-compatible prompt fragment"
            )
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

    def count_text(self, text: str) -> CountResult:
        if self._uses_chat_transport:
            recorded = self._remote_prompt_tokens.get(sha256_text(text))
            if recorded is not None:
                return CountResult(
                    tokens=recorded,
                    exact=True,
                    strategy="provider_tokenize_messages",
                )
            return ConservativeEstimator().count_text(text)
        return CountResult(
            tokens=self.tokenize(text),
            exact=True,
            strategy="llama_cpp_server",
        )

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
        if live_mode and (len(prompt) > 1200 or max_tokens > 192):
            timeout_seconds = self.config.model.benchmark_timeout_seconds
        elif kind == "verification":
            timeout_seconds = self.config.model.verification_timeout_seconds
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
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        schema = self._require_portable_schema(contract)
        effective_temperature = self.config.model.temperature if temperature is None else temperature
        if uses_chat_completions_transport(self.config.model.base_url, self.config.model.completion_endpoint):
            request = {
                "model": self.config.model.profile_name,
                "messages": messages or [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": effective_temperature,
                "top_p": self.config.model.top_p,
                "seed": self.config.model.seed,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": contract.name.replace(":", "_"),
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            if self._is_openrouter:
                request["provider"] = {"require_parameters": True}
            if self.config.model.stop:
                request["stop"] = list(self.config.model.stop)
            return request
        request = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": effective_temperature,
            "top_p": self.config.model.top_p,
            "seed": self.config.model.seed,
            "json_schema": schema,
        }
        if self.config.model.stop:
            request["stop"] = list(self.config.model.stop)
        return request

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
        cancel_check: Callable[[], bool] | None = None,
        cancel_poll_seconds: float = 0.05,
    ) -> CompletionResult:
        token_timeout_seconds = self._token_timeout_seconds(timeout_seconds)
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        started = time.monotonic()
        request_kwargs: dict[str, Any] = {
            "json": stream_payload,
            "timeout": (
                self.config.model.connect_timeout_seconds,
                token_timeout_seconds,
            ),
            "stream": True,
        }
        request_kwargs.update(self._request_headers_kwargs())
        response = requests.post(
            completion_url(self.config.model.base_url, self.config.model.completion_endpoint),
            **request_kwargs,
        )
        cancel_observed = threading.Event()
        stop_watcher = threading.Event()
        watcher: threading.Thread | None = None
        if cancel_check is not None:
            def watch_for_cancel() -> None:
                while not stop_watcher.wait(max(0.005, float(cancel_poll_seconds))):
                    try:
                        should_cancel = bool(cancel_check())
                    except Exception:
                        should_cancel = False
                    if should_cancel:
                        cancel_observed.set()
                        close = getattr(response, "close", None)
                        if callable(close):
                            close()
                        return
            watcher = threading.Thread(target=watch_for_cancel, name="swaag-model-cancel", daemon=True)
            watcher.start()
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            stop_watcher.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            close = getattr(response, "close", None)
            if callable(close):
                close()
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
                if cancel_observed.is_set():
                    raise ModelCallPreempted("model call preempted for communication")
                if not raw_line:
                    continue
                line = raw_line.strip()
                # Server-Sent Events permit comment/keepalive lines beginning with
                # a colon. llama.cpp/proxies may emit bare ':' heartbeats while
                # long constrained generations are in flight; these are transport
                # metadata, not completion payloads.
                if line.startswith(":"):
                    continue
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
            if cancel_observed.is_set():
                raise ModelCallPreempted("model call preempted for communication") from exc
            raise requests.ReadTimeout(f"No streamed model token/event for {token_timeout_seconds:.1f} seconds") from exc
        except (requests.RequestException, OSError, ValueError) as exc:
            if cancel_observed.is_set():
                raise ModelCallPreempted("model call preempted for communication") from exc
            raise
        except Exception as exc:
            # Closing a live urllib3 stream can surface transport-internal
            # exceptions outside requests' public hierarchy. Once cancellation
            # is observed, the close-induced exception is a preemption outcome.
            if cancel_observed.is_set():
                raise ModelCallPreempted("model call preempted for communication") from exc
            raise
        finally:
            stop_watcher.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if cancel_observed.is_set():
            raise ModelCallPreempted("model call preempted for communication")
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
            finish_reason=_completion_finish_reason(body),
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
        messages: list[dict[str, str]] | None = None,
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
            messages=messages,
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


def _completion_finish_reason(body: dict[str, Any]) -> str | None:
    if body.get("truncated") is True:
        return "context_overflow"
    stop_type = str(body.get("stop_type", "")).strip().lower()
    if stop_type == "limit":
        return "length"
    if stop_type in {"eos", "word"}:
        return "stop"
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return "stop" if body.get("stop") else None
