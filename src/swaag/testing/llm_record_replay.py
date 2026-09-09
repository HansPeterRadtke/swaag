from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from swaag.fsops import write_text
from swaag.types import CompletionResult, ContractSpec
from swaag.utils import sha256_text, stable_json_dumps


class MissingReplayEntryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecordReplayEntry:
    request_hash: str
    request: dict[str, Any]
    response: dict[str, Any]


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _completion_result_payload(result: CompletionResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "raw_request": _normalize_json(result.raw_request),
        "raw_response": _normalize_json(result.raw_response),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
    }


class RecordReplayModelClient:
    def __init__(
        self,
        *,
        cassette_path: Path,
        mode: str,
        delegate: Any,
        request_metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"record", "replay"}:
            raise ValueError(f"Unsupported record/replay mode: {mode!r}")
        self.mode = normalized_mode
        self.delegate = delegate
        self.cassette_path = Path(cassette_path)
        self.request_metadata = self._default_request_metadata() | dict(request_metadata or {})
        self._entries = self._load_entries()
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

    def _default_request_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"client_class": type(self.delegate).__name__}
        config = getattr(self.delegate, "config", None)
        model = getattr(config, "model", None)
        if model is not None:
            # Include every known model setting that can influence generated text
            # or request shape. Transport-only send timeouts stay outside the hash.
            for name in (
                "base_url",
                "completion_endpoint",
                "profile_name",
                "structured_output_mode",
                "seed",
                "temperature",
                "top_p",
                "stop",
                "context_limit",
                "timeout_seconds",
                "connect_timeout_seconds",
                "simple_timeout_seconds",
                "structured_timeout_seconds",
                "verification_timeout_seconds",
                "benchmark_timeout_seconds",
                "progress_poll_seconds",
            ):
                metadata[f"model_{name}"] = _normalize_json(getattr(model, name, None))
            metadata["request_timeout_affects_generation"] = False
        return metadata

    def _load_entries(self) -> dict[str, RecordReplayEntry]:
        if not self.cassette_path.exists():
            return {}
        raw = self.cassette_path.read_text(encoding="utf-8")
        payload = _normalize_json(json.loads(raw))
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

    def _write_entries(self) -> None:
        payload = {
            "mode": "record_replay",
            "request_metadata": _normalize_json(self.request_metadata),
            "hash_basis": "request_metadata_plus_payload",
            "transport_metadata_not_in_hash": ["send_completion_timeout_seconds"],
            "entries": [asdict(entry) for entry in sorted(self._entries.values(), key=lambda item: item.request_hash)],
        }
        write_text(self.cassette_path, stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _request_envelope(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> dict[str, Any]:
        # send_completion timeout is stored for debugging, not hashed. The model
        # config timeout is in request_metadata when configured.
        return {
            "request_metadata": _normalize_json(self.request_metadata),
            "send_completion_timeout_seconds": timeout_seconds,
            "payload": _normalize_json(payload),
        }

    def _hash_envelope(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Hash key covers only request_metadata + payload.
        # timeout_seconds is excluded because it is a transport parameter and
        # does not affect what the model produces; including it would make cassettes
        # unnecessarily brittle across environments with different timeout configs.
        return {
            "request_metadata": _normalize_json(self.request_metadata),
            "payload": _normalize_json(payload),
        }

    def _request_hash(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> tuple[str, dict[str, Any]]:
        envelope = self._request_envelope(payload, timeout_seconds=timeout_seconds)
        hash_input = self._hash_envelope(payload)
        request_hash = sha256_text(stable_json_dumps(hash_input, indent=None))
        return request_hash, envelope

    def health(self) -> dict[str, Any]:
        health = getattr(self.delegate, "health", None)
        if callable(health):
            return health()
        return {"status": "ok", "mode": self.mode}

    def tokenize(self, text: str) -> int:
        tokenize = getattr(self.delegate, "tokenize", None)
        if callable(tokenize):
            return int(tokenize(text))
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        tokenize_selection = getattr(self.delegate, "tokenize_selection", None)
        if callable(tokenize_selection):
            return int(tokenize_selection(text))
        return self.tokenize(text)

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return self.delegate.build_completion_request(
            prompt,
            max_tokens=max_tokens,
            contract=contract,
            temperature=temperature,
        )

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

    def _return_from_entry(
        self,
        entry: RecordReplayEntry,
        payload: dict[str, Any],
        *,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CompletionResult:
        response_payload = dict(entry.response)
        raw_response = response_payload.get("raw_response", {})
        if not isinstance(raw_response, dict):
            raw_response = {}
        raw_response = {
            **raw_response,
            "replay_cache": {
                "source": "cache",
                "request_hash": entry.request_hash,
                "cassette_path": str(self.cassette_path),
                "mode": self.mode,
            },
        }
        if stream_callback is not None:
            chunks = raw_response.get("chunks", [])
            if isinstance(chunks, list):
                for index, chunk in enumerate(chunks, start=1):
                    if isinstance(chunk, dict):
                        stream_callback({
                            "chunk_index": index,
                            "chunk": chunk,
                            "content": str(chunk.get("content", "")) if chunk.get("content") is not None else "",
                            "stop": bool(chunk.get("stop")),
                            "replayed": True,
                        })
        self._replayed_count += 1
        return CompletionResult(
            text=str(response_payload.get("text", "")),
            raw_request=payload,
            raw_response=raw_response,
            prompt_tokens=response_payload.get("prompt_tokens"),
            completion_tokens=response_payload.get("completion_tokens"),
            finish_reason=response_payload.get("finish_reason"),
        )

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CompletionResult:
        request_hash, request_envelope = self._request_hash(payload, timeout_seconds=timeout_seconds)
        if self.mode == "replay":
            entry = self._entries.get(request_hash)
            if entry is None:
                raise MissingReplayEntryError(
                    f"No replay entry for request hash {request_hash}; record a cassette for the current full request payload first."
                )
            return self._return_from_entry(entry, payload, stream_callback=stream_callback)
        # "record" mode: replay from existing cassette entry if present (avoids unnecessary model calls),
        # otherwise call the real delegate and record the new entry.
        existing = self._entries.get(request_hash)
        if existing is not None:
            return self._return_from_entry(existing, payload, stream_callback=stream_callback)
        try:
            result = self.delegate.send_completion(payload, timeout_seconds=timeout_seconds, stream_callback=stream_callback)
        except TypeError as exc:
            if "stream_callback" not in str(exc):
                raise
            result = self.delegate.send_completion(payload, timeout_seconds=timeout_seconds)
        response_payload = _completion_result_payload(result)
        self._entries[request_hash] = RecordReplayEntry(
            request_hash=request_hash,
            request=request_envelope,
            response=response_payload,
        )
        self._write_entries()
        self._recorded_count += 1
        raw_response = dict(result.raw_response or {})
        raw_response["replay_cache"] = {
            "source": "live_recorded",
            "request_hash": request_hash,
            "cassette_path": str(self.cassette_path),
            "mode": self.mode,
        }
        return CompletionResult(
            text=result.text,
            raw_request=result.raw_request,
            raw_response=raw_response,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
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
        payload = self.build_completion_request(
            prompt,
            max_tokens=max_tokens,
            contract=resolved_contract,
            temperature=temperature,
        )
        return self.send_completion(payload, timeout_seconds=policy.effective_timeout_seconds)
