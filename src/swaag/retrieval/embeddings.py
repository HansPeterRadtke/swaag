from __future__ import annotations

import json
import os
import time
from functools import lru_cache

import requests
from pathlib import Path
from typing import Any

from swaag.grammar import relevance_scoring_contract
from swaag.model import completion_url, uses_chat_completions_transport

try:  # pragma: no cover - exercised only when transformers/torch are installed
    import torch as _torch
    from transformers import AutoModel as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer
except Exception:  # pragma: no cover - fallback path is unit-tested without transformers
    _torch = None
    _AutoModel = None
    _AutoTokenizer = None


class EmbeddingBackend:
    mode = "llm_scoring"
    degraded = False

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def similarity(self, left: str, right: str) -> float:
        scores = self.score_query(left, [right])
        return scores[0] if scores else 0.0


class UnavailableEmbeddingBackend(EmbeddingBackend):
    """Neutral offline backend.

    This backend is for tests and offline structural runs only. It performs
    no lexical, regex, TF-IDF, embedding, filename, or keyword scoring. All
    candidates receive the same zero relevance score, so callers can still
    exercise retrieval mechanics without Python making a semantic decision.
    """

    mode = "unavailable"
    degraded = True

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        del query
        return [0.0 for _ in texts]


def _candidate_model_paths() -> list[Path]:
    """Look up T5 model paths from explicit env var only.

    No silent home-folder caches: the user must opt in by exporting
    SWAAG_SEMANTIC_MODEL_PATH. T5 is an optional secondary backend; the
    primary semantic backend is the LLM via :class:`LlmScoringBackend`.
    """

    paths: list[Path] = []
    env_path = os.environ.get("SWAAG_SEMANTIC_MODEL_PATH", "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser())
    return paths


def discover_transformer_model_path() -> Path | None:
    for path in _candidate_model_paths():
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_transformer_components(model_path_text: str):
    if _AutoTokenizer is None or _AutoModel is None or _torch is None:
        raise RuntimeError("transformers backend is unavailable")
    tokenizer = _AutoTokenizer.from_pretrained(model_path_text, local_files_only=True)
    model = _AutoModel.from_pretrained(model_path_text, local_files_only=True)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=4096)
def _encode_transformer_text(model_path_text: str, text: str) -> tuple[float, ...]:
    tokenizer, model = _load_transformer_components(model_path_text)
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with _torch.inference_mode():
        # Use the encoder forward pass so we obtain contextual representations
        # rather than raw embedding-table lookups. T5 is an encoder-decoder so
        # we ask the encoder directly. Other models expose the same input
        # signature.
        if hasattr(model, "encoder") and callable(getattr(model, "encoder", None)):
            encoder_inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            outputs = model.encoder(**encoder_inputs)
        else:
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = _torch.nn.functional.normalize(pooled, p=2, dim=1)
    return tuple(float(value) for value in pooled[0].tolist())


class TransformerEmbeddingBackend(EmbeddingBackend):
    """Optional T5/transformer encoder embeddings.

    Only used when ``SWAAG_SEMANTIC_MODEL_PATH`` is set to a model
    directory. Treated as a fallback to :class:`LlmScoringBackend`.
    """

    mode = "transformer_local"
    degraded = False

    def __init__(self, model_path: Path):
        self._model_path = str(model_path)

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        query_vector = _torch.tensor(_encode_transformer_text(self._model_path, query))
        return [
            max(
                0.0,
                min(
                    1.0,
                    float(
                        _torch.dot(
                            query_vector,
                            _torch.tensor(_encode_transformer_text(self._model_path, text)),
                        )
                    ),
                ),
            )
            for text in texts
        ]


class SemanticBackendUnavailableError(RuntimeError):
    pass


class SemanticBackendProtocolError(RuntimeError):
    pass


class LlmScoringBackend(EmbeddingBackend):
    """Primary semantic backend: ask the LLM to rate relevance.

    The LLM behind the configured llama.cpp endpoint is asked, via a
    structured JSON contract, to rate each candidate's relevance to the
    query on a 0..1 scale. This replaces formula-based and embedding-based
    scoring for skills, retrieval and guidance selection.

    This backend must remain semantic-first. It therefore does NOT silently
    degrade to lexical similarity when the semantic engine is unavailable.
    Unavailability is treated as an explicit blocked state and retried with
    backoff until the server returns. Malformed structured responses are
    surfaced as protocol errors rather than silently replaced with a fallback
    score.
    """

    mode = "llm_scoring"
    degraded = False

    def __init__(
        self,
        *,
        base_url: str,
        completion_endpoint: str = "/completion",
        connect_timeout_seconds: int = 10,
        read_timeout_seconds: int = 60,
        max_text_chars: int = 280,
        max_items_per_call: int = 12,
        seed: int = 11,
        sleep_func=time.sleep,
        max_unavailable_attempts: int | None = None,
        max_protocol_attempts: int = 2,
        model_client: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._endpoint = completion_endpoint
        self._connect_timeout = connect_timeout_seconds
        self._read_timeout = read_timeout_seconds
        self._max_text_chars = max_text_chars
        self._max_items_per_call = max_items_per_call
        self._seed = seed
        self._sleep = sleep_func
        self._max_unavailable_attempts = max_unavailable_attempts
        self._max_protocol_attempts = max(1, int(max_protocol_attempts))
        self._model_client = model_client

    def _truncate(self, text: str) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= self._max_text_chars:
            return text
        return text[: self._max_text_chars - 1].rstrip() + "…"

    def _build_prompt(self, query: str, texts: list[str]) -> str:
        lines = [
            "You are a relevance scorer. Rate how relevant each candidate is to the query.",
            "Return JSON only with exactly one numeric field per candidate: score_0, score_1, and so on.",
            "Each score must be between 0.0 and 1.0. 0.0 means unrelated; 1.0 means perfectly relevant.",
            "",
            f"Query:\n{self._truncate(query)}",
            "",
            "Candidates:",
        ]
        for index, text in enumerate(texts):
            lines.append(f"[{index}] {self._truncate(text)}")
        lines.append("")
        lines.append("JSON:")
        return "\n".join(lines)

    def _call_model_client(self, prompt: str, item_count: int) -> list[float]:
        contract = relevance_scoring_contract(item_count)
        max_tokens = max(64, item_count * 12)
        unavailable_attempts = 0
        protocol_attempts = 0
        while True:
            effective_prompt = prompt
            if protocol_attempts:
                effective_prompt += (
                    "\n\nCorrection: the previous response violated the strict relevance schema. "
                    "Return exactly the required score_N fields and no extra text."
                )
            try:
                resolved_contract, policy = self._model_client.resolve_contract(
                    contract,
                    kind="verification",
                    prompt=effective_prompt,
                    max_tokens=max_tokens,
                    live_mode=False,
                )
                payload = self._model_client.build_completion_request(
                    effective_prompt,
                    max_tokens=max_tokens,
                    contract=resolved_contract,
                    temperature=0.0,
                )
                # Semantic scoring owns its configured seed; the complete payload
                # (including this seed) is part of the shared cache key.
                payload["seed"] = self._seed
                result = self._model_client.send_completion(
                    payload,
                    timeout_seconds=policy.effective_timeout_seconds,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if self._max_unavailable_attempts is not None and unavailable_attempts >= self._max_unavailable_attempts:
                    raise SemanticBackendUnavailableError(str(exc)) from exc
                self.degraded = True
                self._sleep(min(60.0, float(2**min(unavailable_attempts, 6))))
                unavailable_attempts += 1
                continue
            except requests.HTTPError as exc:
                response = getattr(exc, "response", None)
                if response is not None and getattr(response, "status_code", None) in {502, 503, 504}:
                    if self._max_unavailable_attempts is not None and unavailable_attempts >= self._max_unavailable_attempts:
                        raise SemanticBackendUnavailableError(str(exc)) from exc
                    self.degraded = True
                    self._sleep(min(60.0, float(2**min(unavailable_attempts, 6))))
                    unavailable_attempts += 1
                    continue
                raise SemanticBackendProtocolError(str(exc)) from exc
            except Exception as exc:
                raise SemanticBackendProtocolError(str(exc)) from exc
            parsed = _parse_score_payload(result.text, item_count)
            if parsed is not None:
                return parsed
            protocol_attempts += 1
            if protocol_attempts >= self._max_protocol_attempts:
                raise SemanticBackendProtocolError("Structured relevance response violated the requested schema")

    def _call_llm(self, prompt: str, item_count: int) -> list[float]:
        if self._model_client is not None:
            return self._call_model_client(prompt, item_count)
        try:
            import requests as requests_module
        except Exception:  # pragma: no cover - requests is a hard dep
            raise SemanticBackendUnavailableError("requests dependency is unavailable")
        contract = relevance_scoring_contract(item_count)
        assert contract.json_schema is not None
        max_tokens = max(64, item_count * 12)
        if uses_chat_completions_transport(self._base_url, self._endpoint):
            payload: dict[str, Any] = {
                "model": os.environ.get("SWAAG_LLM_SCORING_MODEL", "local"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": self._seed,
                "stop": ["<|eot_id|>", "<|end_of_text|>"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": contract.name,
                        "strict": True,
                        "schema": contract.json_schema,
                    },
                },
                "provider": {"require_parameters": True},
            }
        else:
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": self._seed,
                "stop": ["<|eot_id|>", "<|end_of_text|>"],
                "json_schema": contract.json_schema,
            }
        unavailable_attempts = 0
        protocol_attempts = 0
        while True:
            while True:
                try:
                    response = requests_module.post(
                        completion_url(self._base_url, self._endpoint),
                        json=payload,
                        timeout=(self._connect_timeout, self._read_timeout),
                    )
                    response.raise_for_status()
                    body = response.json()
                    break
                except requests_module.ConnectionError as exc:
                    if self._max_unavailable_attempts is not None and unavailable_attempts >= self._max_unavailable_attempts:
                        raise SemanticBackendUnavailableError(str(exc)) from exc
                    self.degraded = True
                    self._sleep(min(60.0, float(2**min(unavailable_attempts, 6))))
                    unavailable_attempts += 1
                    continue
                except requests_module.HTTPError as exc:
                    response = getattr(exc, "response", None)
                    if response is not None and getattr(response, "status_code", None) in {502, 503, 504}:
                        if self._max_unavailable_attempts is not None and unavailable_attempts >= self._max_unavailable_attempts:
                            raise SemanticBackendUnavailableError(str(exc)) from exc
                        self.degraded = True
                        self._sleep(min(60.0, float(2**min(unavailable_attempts, 6))))
                        unavailable_attempts += 1
                        continue
                    raise SemanticBackendProtocolError(str(exc)) from exc
                except requests_module.Timeout as exc:
                    if self._max_unavailable_attempts is not None and unavailable_attempts >= self._max_unavailable_attempts:
                        raise SemanticBackendUnavailableError(str(exc)) from exc
                    self.degraded = True
                    self._sleep(min(60.0, float(2**min(unavailable_attempts, 6))))
                    unavailable_attempts += 1
                    continue
                except Exception as exc:
                    raise SemanticBackendProtocolError(str(exc)) from exc
            if not isinstance(body, dict):
                raise SemanticBackendProtocolError(f"Expected JSON object body, got {body!r}")
            text = body.get("content", "")
            if not isinstance(text, str):
                text = _chat_response_content(body)
            if not isinstance(text, str):
                raise SemanticBackendProtocolError(f"Completion response missing string content: {body!r}")
            parsed = _parse_score_payload(text, item_count)
            if parsed is not None:
                return parsed
            protocol_attempts += 1
            if protocol_attempts >= self._max_protocol_attempts:
                raise SemanticBackendProtocolError("Structured relevance response violated the requested schema")

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self.degraded = False
        if not query.strip():
            return [0.0] * len(texts)
        scores: list[float] = []
        for offset in range(0, len(texts), self._max_items_per_call):
            chunk = texts[offset : offset + self._max_items_per_call]
            prompt = self._build_prompt(query, chunk)
            chunk_scores = self._call_llm(prompt, len(chunk))
            scores.extend(chunk_scores)
        return scores


def _parse_score_payload(text: str, expected_count: int) -> list[float] | None:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    fixed_keys = [f"score_{index}" for index in range(expected_count)]
    if set(payload) == set(fixed_keys):
        raw_scores = [payload[key] for key in fixed_keys]
    else:
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != expected_count:
            return None
    parsed: list[float] = []
    for value in raw_scores:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        parsed.append(max(0.0, min(1.0, number)))
    return parsed

def _chat_response_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"])
    return ""


def build_backend(
    mode: str,
    *,
    base_url: str | None = None,
    seed: int = 11,
    connect_timeout_seconds: int = 10,
    read_timeout_seconds: int = 60,
    max_text_chars: int | None = None,
    model_client: Any | None = None,
) -> EmbeddingBackend:
    """Construct a semantic scoring backend.

    Selection order:
        - ``llm_scoring`` returns :class:`LlmScoringBackend` (the default
          primary backend) when ``base_url`` is provided. Without ``base_url``
          it fails explicitly because semantic relevance is unavailable.
        - ``transformer_local`` is opt-in and only loads when
          ``SWAAG_SEMANTIC_MODEL_PATH`` points to a real model directory
          and the optional transformer dependencies are installed. If the
          transformer backend is requested explicitly and unavailable, that is
          an explicit configuration error rather than a silent fallback.
        - ``unavailable`` returns a neutral degraded backend for tests and
          offline structural runs. It never scores semantic relevance.
    """

    if mode == "unavailable":
        return UnavailableEmbeddingBackend()
    if mode == "llm_scoring":
        if base_url or model_client is not None:
            kwargs: dict = {
                "base_url": base_url or "",
                "seed": seed,
                "connect_timeout_seconds": connect_timeout_seconds,
                "read_timeout_seconds": read_timeout_seconds,
                "model_client": model_client,
            }
            if max_text_chars is not None:
                kwargs["max_text_chars"] = max_text_chars
            return LlmScoringBackend(**kwargs)
        raise RuntimeError("llm_scoring backend requires an explicit base_url")
    if mode == "transformer_local":
        model_path = discover_transformer_model_path()
        if model_path is None:
            raise RuntimeError(
                "transformer_local backend requires SWAAG_SEMANTIC_MODEL_PATH to point to a local model directory"
            )
        if _AutoTokenizer is None or _AutoModel is None or _torch is None:
            raise RuntimeError("transformer_local backend requires optional transformer dependencies")
        return TransformerEmbeddingBackend(model_path)
    raise RuntimeError(f"Unknown retrieval backend mode: {mode}")
