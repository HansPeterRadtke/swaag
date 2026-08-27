from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import inspect
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Iterator

try:  # Linux/Unix production path.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX compatibility fallback
    fcntl = None

from swaag.fsops import atomic_replace, ensure_dir, remove_file
from swaag.types import CompletionResult, ContractSpec
from swaag.utils import sha256_text, stable_json_dumps


class MissingReplayEntryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecordReplayEntry:
    request_hash: str
    request: dict[str, Any]
    response: dict[str, Any]


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize cache mutation across threads and processes."""
    ensure_dir(path.parent)
    process_lock = _thread_lock(path)
    with process_lock:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        with os.fdopen(fd, "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_json(value: Any) -> Any:
    """Benchmark-only normalization for intentionally replayable dynamic run state."""
    if isinstance(value, dict):
        return {str(key): _normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        normalized = re.sub(r"\b[a-z]+_[0-9a-f]{12}\b", "<generated-id>", value)
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)", "<timestamp>", normalized)
        normalized = re.sub(r"elapsed=\d+(?:\.\d+)?s", "elapsed=<duration>", normalized)
        normalized = re.sub(r"avg_tps=(?:None|\d+(?:\.\d+)?)", "avg_tps=<rate>", normalized)
        normalized = re.sub(r"Workspace: [^\n]+", "Workspace: <workspace>", normalized)
        return normalized
    return value


def _stable_json(value: Any) -> Any:
    """Convert values to stable JSON without changing model-visible strings."""
    if isinstance(value, dict):
        return {str(key): _stable_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _completion_result_payload(result: CompletionResult, canonicalize) -> dict[str, Any]:
    return {
        "text": result.text,
        "raw_request": canonicalize(result.raw_request),
        "raw_response": canonicalize(result.raw_response),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
        "first_token_seconds": result.first_token_seconds,
    }


def default_model_cache_path(config: Any) -> Path:
    configured = str(config.model.cache_path).strip()
    if configured:
        return Path(configured).expanduser()
    return Path(config.sessions.root).parent / "llm-model-cache.json"


def build_model_client(
    config: Any,
    *,
    delegate: Any | None = None,
    request_metadata: dict[str, Any] | None = None,
    canonicalize_dynamic_values: bool = False,
) -> Any:
    """Build the configured live model client, cache-first by default."""
    if delegate is None:
        from swaag.model import LlamaCppClient

        delegate = LlamaCppClient(config)
    if getattr(delegate, "is_record_replay_client", False) or not config.model.cache_enabled:
        return delegate
    return RecordReplayModelClient(
        cassette_path=default_model_cache_path(config),
        mode=config.model.cache_mode,
        delegate=delegate,
        request_metadata=request_metadata,
        canonicalize_dynamic_values=canonicalize_dynamic_values,
    )


class RecordReplayModelClient:
    def __init__(
        self,
        *,
        cassette_path: Path,
        mode: str,
        delegate: Any,
        request_metadata: dict[str, Any] | None = None,
        canonicalize_dynamic_values: bool = False,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"record", "replay"}:
            raise ValueError(f"Unsupported record/replay mode: {mode!r}")
        self.mode = normalized_mode
        self.delegate = delegate
        self.cassette_path = Path(cassette_path)
        self.canonicalize_dynamic_values = bool(canonicalize_dynamic_values)
        self._canonicalize = _normalize_json if self.canonicalize_dynamic_values else _stable_json
        self._caller_request_metadata = dict(request_metadata or {})
        self._recorded_request_metadata = self._load_recorded_request_metadata()
        self.request_metadata = {}
        self._refresh_request_metadata()
        self._entries = self._load_entries()
        self._prompt_renderings = self._load_prompt_renderings()
        self._token_counts = self._load_token_counts()
        self._replayed_count = 0
        self._recorded_count = 0

    @property
    def is_record_replay_client(self) -> bool:
        return True

    @property
    def recorded_count(self) -> int:
        return self._recorded_count

    @property
    def replayed_count(self) -> int:
        return self._replayed_count

    @property
    def _cache_lock_path(self) -> Path:
        return self.cassette_path.with_name(f"{self.cassette_path.name}.lock")

    def _request_lock_path(self, request_hash: str) -> Path:
        return self.cassette_path.with_name(f"{self.cassette_path.name}.{request_hash}.request.lock")

    def _default_request_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "client_class": type(self.delegate).__name__,
            "replay_contract_version": "2026-08-26-server-prompt-protocol-v3",
            "model_transport": "streaming_token_timeout",
            "canonicalize_dynamic_values": self.canonicalize_dynamic_values,
        }
        identity_provider = getattr(self.delegate, "cache_identity", None)
        if self.mode == "replay":
            metadata["model_identity"] = {
                "status": "unresolved",
                "probe_skipped": "offline_replay",
            }
        elif callable(identity_provider):
            try:
                metadata["model_identity"] = identity_provider()
            except Exception as exc:  # never reuse a possibly different unresolved model
                metadata["model_identity"] = {
                    "status": "unresolved",
                    "error_type": exc.__class__.__name__,
                    "nonce": os.urandom(16).hex(),
                }
        config = getattr(self.delegate, "config", None)
        model = getattr(config, "model", None)
        if model is not None:
            metadata["model_base_url"] = getattr(model, "base_url", "")
            metadata["completion_endpoint"] = getattr(model, "completion_endpoint", "")
            metadata["model_profile"] = getattr(model, "profile_name", "")
            metadata["structured_output_mode"] = getattr(model, "structured_output_mode", "")
            metadata["configured_seed"] = getattr(model, "seed", None)
        return metadata


    def _refresh_request_metadata(self) -> None:
        # Refresh both the cassette metadata and live model/server identity before
        # every lookup. This prevents a long-lived client from replaying an entry
        # after the model, server build/properties, or local server launch flags change.
        self._recorded_request_metadata = self._load_recorded_request_metadata()
        requested_metadata = self._default_request_metadata() | dict(self._caller_request_metadata)
        self.request_metadata = self._canonicalize(
            self._resolve_model_identity(requested_metadata)
        )

    def _load_recorded_request_metadata(self) -> dict[str, Any]:
        if not self.cassette_path.exists():
            return {}
        try:
            payload = json.loads(self.cassette_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        metadata = payload.get("request_metadata", {}) if isinstance(payload, dict) else {}
        return metadata if isinstance(metadata, dict) else {}

    def _resolve_model_identity(self, requested: dict[str, Any]) -> dict[str, Any]:
        """Reuse a recorded fingerprint offline only for the identical client configuration."""
        resolved = dict(requested)
        current_identity = resolved.get("model_identity")
        if not isinstance(current_identity, dict) or current_identity.get("status") != "unresolved":
            return resolved
        recorded = self._recorded_request_metadata
        recorded_identity = recorded.get("model_identity") if isinstance(recorded, dict) else None
        compatibility_keys = (
            "client_class",
            "replay_contract_version",
            "model_transport",
            "canonicalize_dynamic_values",
            "model_base_url",
            "completion_endpoint",
            "model_profile",
            "structured_output_mode",
            "configured_seed",
        )
        compatible = isinstance(recorded_identity, dict) and all(
            recorded.get(key) == resolved.get(key) for key in compatibility_keys
        )
        # Caller-supplied cache scope/task metadata must also match exactly.
        if compatible:
            for key, value in resolved.items():
                if key == "model_identity":
                    continue
                if recorded.get(key) != value:
                    compatible = False
                    break
        if compatible and recorded_identity.get("status") in {"resolved", "configured_only"}:
            resolved["model_identity"] = recorded_identity
            return resolved
        # A fresh unresolved identity is intentionally non-reusable. This avoids
        # cross-model hits when /props is unavailable and no compatible cassette
        # can prove which model generated an answer.
        isolated = dict(current_identity)
        isolated["nonce"] = os.urandom(16).hex()
        resolved["model_identity"] = isolated
        return resolved

    def _load_entries(self) -> dict[str, RecordReplayEntry]:
        if not self.cassette_path.exists():
            return {}
        payload = json.loads(self.cassette_path.read_text(encoding="utf-8"))
        entries: dict[str, RecordReplayEntry] = {}
        for item in payload.get("entries", []):
            if not isinstance(item, dict):
                continue
            request_hash = str(item.get("request_hash", "")).strip()
            request = item.get("request", {})
            response = item.get("response", {})
            if request_hash:
                entries[request_hash] = RecordReplayEntry(
                    request_hash=request_hash,
                    request=request if isinstance(request, dict) else {},
                    response=response if isinstance(response, dict) else {},
                )
        return entries

    def _load_prompt_renderings(self) -> dict[str, dict[str, str]]:
        if not self.cassette_path.exists():
            return {}
        payload = json.loads(self.cassette_path.read_text(encoding="utf-8"))
        raw = payload.get("prompt_renderings", {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): dict(value)
            for key, value in raw.items()
            if isinstance(value, dict)
            and all(isinstance(name, str) and isinstance(item, str) for name, item in value.items())
            and isinstance(value.get("prompt"), str)
            and isinstance(value.get("prompt_protocol_sha256"), str)
        }

    def _load_token_counts(self) -> dict[str, int]:
        if not self.cassette_path.exists():
            return {}
        payload = json.loads(self.cassette_path.read_text(encoding="utf-8"))
        raw = payload.get("token_counts", {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): int(value)
            for key, value in raw.items()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }

    def _serialized_entries(self) -> str:
        payload = {
            "mode": "record_replay",
            "request_metadata": self._canonicalize(self.request_metadata),
            "hash_basis": "request_metadata_plus_payload",
            "transport_metadata_not_in_hash": ["send_completion_timeout_seconds"],
            "prompt_renderings": {
                key: self._prompt_renderings[key]
                for key in sorted(self._prompt_renderings)
            },
            "token_counts": {
                key: self._token_counts[key]
                for key in sorted(self._token_counts)
            },
            "entries": [asdict(entry) for entry in sorted(self._entries.values(), key=lambda item: item.request_hash)],
        }
        return stable_json_dumps(payload, indent=2) + "\n"

    def _write_entries_atomic(self) -> None:
        """Write a complete cassette atomically; caller must hold the cache lock."""
        parent = self.cassette_path.parent
        ensure_dir(parent)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.cassette_path.name}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(self._serialized_entries())
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(temp_path, self.cassette_path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            remove_file(temp_path, missing_ok=True)

    def _request_envelope(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> dict[str, Any]:
        return {
            "request_metadata": self._canonicalize(self.request_metadata),
            "send_completion_timeout_seconds": timeout_seconds,
            "payload": self._canonicalize(payload),
        }

    def _hash_envelope(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The full generation payload includes prompt/messages, schema, model,
        # seed, temperature, top-p, token limit, and stop sequences. Transport
        # timeouts are deliberately excluded because they cannot change a
        # successfully completed model output.
        return {
            "request_metadata": self._canonicalize(self.request_metadata),
            "payload": self._canonicalize(payload),
        }

    def _request_hash(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> tuple[str, dict[str, Any]]:
        envelope = self._request_envelope(payload, timeout_seconds=timeout_seconds)
        request_hash = sha256_text(stable_json_dumps(self._hash_envelope(payload), indent=None))
        return request_hash, envelope

    def health(self) -> dict[str, Any]:
        health = getattr(self.delegate, "health", None)
        if callable(health):
            return health()
        return {"status": "ok", "mode": self.mode}

    def tokenize(self, text: str) -> int:
        return self._tokenize_record_replay(text, operation="tokenize")

    def tokenize_selection(self, text: str) -> int:
        return self._tokenize_record_replay(text, operation="tokenize_selection")

    def _tokenize_record_replay(self, text: str, *, operation: str) -> int:
        self._refresh_request_metadata()
        self._token_counts = self._load_token_counts()
        key = sha256_text(
            stable_json_dumps(
                {
                    "request_metadata": self.request_metadata,
                    "operation": operation,
                    "text": text,
                },
                indent=None,
            )
        )
        recorded = self._token_counts.get(key)
        if recorded is not None:
            return int(recorded)
        if self.mode == "replay":
            raise MissingReplayEntryError(
                f"No exact recorded {operation} result for {key}"
            )
        tokenizer = getattr(self.delegate, operation, None)
        if not callable(tokenizer) and operation == "tokenize_selection":
            tokenizer = getattr(self.delegate, "tokenize", None)
        count = int(tokenizer(text)) if callable(tokenizer) else len(text.split()) if text.strip() else 0
        if count < 0:
            raise RuntimeError(f"Model client returned a negative {operation} count")
        with _exclusive_file_lock(self._cache_lock_path):
            self._entries = self._load_entries()
            self._prompt_renderings = self._load_prompt_renderings()
            self._token_counts = self._load_token_counts()
            existing = self._token_counts.get(key)
            if existing is not None and existing != count:
                raise RuntimeError(
                    f"The same model/text produced different {operation} counts"
                )
            self._token_counts[key] = count
            self._write_entries_atomic()
        return count

    def render_chat_prompt(self, messages: list[dict[str, str]]) -> dict[str, str]:
        self._refresh_request_metadata()
        self._prompt_renderings = self._load_prompt_renderings()
        key = sha256_text(
            stable_json_dumps(
                {
                    "request_metadata": self.request_metadata,
                    "messages": self._canonicalize(messages),
                },
                indent=None,
            )
        )
        recorded = self._prompt_renderings.get(key)
        if recorded is not None:
            return dict(recorded)
        if self.mode == "replay":
            raise MissingReplayEntryError(
                f"No exact recorded chat-template rendering for {key}"
            )
        renderer = getattr(self.delegate, "render_chat_prompt", None)
        if not callable(renderer):
            raise RuntimeError("Model client cannot render its chat template")
        rendered = renderer(messages)
        if (
            not isinstance(rendered, dict)
            or not isinstance(rendered.get("prompt"), str)
            or not rendered["prompt"]
            or not isinstance(rendered.get("prompt_protocol_sha256"), str)
            or len(rendered["prompt_protocol_sha256"]) != 64
            or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in rendered.items()
            )
        ):
            raise RuntimeError("Model client returned an invalid chat-template rendering")
        normalized = dict(rendered)
        with _exclusive_file_lock(self._cache_lock_path):
            self._entries = self._load_entries()
            self._prompt_renderings = self._load_prompt_renderings()
            self._token_counts = self._load_token_counts()
            existing = self._prompt_renderings.get(key)
            if existing is not None and existing != normalized:
                raise RuntimeError(
                    "The same model/messages produced different chat-template renderings"
                )
            self._prompt_renderings[key] = normalized
            self._write_entries_atomic()
        return dict(normalized)

    def verify_prompt_protocol(self, prompt_protocol_sha256: str) -> None:
        if self.mode == "replay":
            return
        verifier = getattr(self.delegate, "verify_prompt_protocol", None)
        if callable(verifier):
            verifier(prompt_protocol_sha256)

    def context_limit_resolution(self) -> tuple[int, str]:
        """Resolve capacity without making replay-only execution depend on a server."""
        if self.mode == "replay":
            configured = getattr(getattr(self.delegate, "config", None), "model", None)
            value = getattr(configured, "context_limit", None)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Replay mode requires a positive configured model.context_limit")
            return int(value), "configured:replay"

        resolver = getattr(self.delegate, "context_limit_resolution", None)
        if callable(resolver):
            return resolver()
        server_resolver = getattr(self.delegate, "server_context_limit", None)
        if callable(server_resolver):
            return int(server_resolver()), "server_props:n_ctx"
        configured = getattr(getattr(self.delegate, "config", None), "model", None)
        value = getattr(configured, "context_limit", None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("Model client has no context-capacity resolver or configured fallback")
        return int(value), "configured"

    def server_slot_count(self) -> int:
        if self.mode == "replay":
            return 128
        resolver = getattr(self.delegate, "server_slot_count", None)
        return int(resolver()) if callable(resolver) else 1

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        builder = self.delegate.build_completion_request
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "contract": contract,
            "temperature": temperature,
        }
        parameters = inspect.signature(builder).parameters.values()
        if messages is not None and any(
            parameter.name == "messages"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            kwargs["messages"] = messages
        return builder(prompt, **kwargs)

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ):
        return self.delegate.select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )

    def resolve_contract(
        self,
        contract: ContractSpec,
        *,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ):
        return self.delegate.resolve_contract(
            contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )

    def _return_from_entry(self, entry: RecordReplayEntry, payload: dict[str, Any]) -> CompletionResult:
        response_payload = dict(entry.response)
        self._replayed_count += 1
        return CompletionResult(
            text=str(response_payload.get("text", "")),
            raw_request=payload,
            raw_response=response_payload.get("raw_response", {}) if isinstance(response_payload.get("raw_response", {}), dict) else {},
            prompt_tokens=response_payload.get("prompt_tokens"),
            completion_tokens=response_payload.get("completion_tokens"),
            finish_reason=response_payload.get("finish_reason"),
            elapsed_seconds=response_payload.get("elapsed_seconds"),
            tokens_per_second=response_payload.get("tokens_per_second"),
            first_token_seconds=response_payload.get("first_token_seconds"),
        )

    def _reload_and_find(self, request_hash: str) -> RecordReplayEntry | None:
        self._entries = self._load_entries()
        return self._entries.get(request_hash)

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> CompletionResult:
        self._refresh_request_metadata()
        request_hash, request_envelope = self._request_hash(payload, timeout_seconds=timeout_seconds)
        with _exclusive_file_lock(self._cache_lock_path):
            entry = self._reload_and_find(request_hash)
        if entry is not None:
            return self._return_from_entry(entry, payload)
        if self.mode == "replay":
            raise MissingReplayEntryError(
                f"No replay entry for request hash {request_hash}; record a cassette for the current full request payload first."
            )

        # A per-request lock prevents duplicate calls for the same hash while
        # allowing unrelated model requests to execute concurrently.
        with _exclusive_file_lock(self._request_lock_path(request_hash)):
            with _exclusive_file_lock(self._cache_lock_path):
                entry = self._reload_and_find(request_hash)
            if entry is not None:
                return self._return_from_entry(entry, payload)

            send = self.delegate.send_completion
            kwargs = {
                "timeout_seconds": timeout_seconds,
                "progress_callback": progress_callback,
            }
            try:
                signature = inspect.signature(send)
                supports_cancel = "cancel_check" in signature.parameters or any(
                    item.kind == inspect.Parameter.VAR_KEYWORD
                    for item in signature.parameters.values()
                )
            except (TypeError, ValueError):
                supports_cancel = False
            if supports_cancel:
                kwargs["cancel_check"] = cancel_check
            result = send(payload, **kwargs)
            new_entry = RecordReplayEntry(
                request_hash=request_hash,
                request=request_envelope,
                response=_completion_result_payload(result, self._canonicalize),
            )
            with _exclusive_file_lock(self._cache_lock_path):
                concurrent_entry = self._reload_and_find(request_hash)
                if concurrent_entry is not None:
                    return self._return_from_entry(concurrent_entry, payload)
                self._prompt_renderings = self._load_prompt_renderings()
                self._token_counts = self._load_token_counts()
                self._entries[request_hash] = new_entry
                self._write_entries_atomic()
            self._recorded_count += 1
            return result

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
        payload = self.build_completion_request(
            prompt,
            max_tokens=max_tokens,
            contract=resolved_contract,
            temperature=temperature,
            messages=messages,
        )
        return self.send_completion(payload, timeout_seconds=policy.effective_timeout_seconds)
