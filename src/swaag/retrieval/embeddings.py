from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised in environments with numpy available
    import numpy as _np
except Exception:  # pragma: no cover - fallback path is unit-tested without numpy
    _np = None

try:  # pragma: no cover - exercised only when transformers/torch are installed
    import torch as _torch
    from transformers import AutoModel as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer
except Exception:  # pragma: no cover - fallback path is unit-tested without transformers
    _torch = None
    _AutoModel = None
    _AutoTokenizer = None


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", lowered)
    lowered = re.sub(r"[^a-z0-9_./:-]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def semantic_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in _TOKEN_RE.findall(_normalize_text(text)):
        parts = [part for part in re.split(r"[._/:-]+", raw) if part]
        for part in parts:
            if len(part) >= 2:
                terms.append(part)
        for index in range(len(parts) - 1):
            combined = f"{parts[index]}_{parts[index + 1]}"
            if len(combined) >= 5:
                terms.append(combined)
    return terms


def _char_ngrams(text: str) -> list[str]:
    normalized = f" {_normalize_text(text)} "
    grams: list[str] = []
    for size in (3, 4, 5):
        for index in range(max(len(normalized) - size + 1, 0)):
            grams.append(f"c{size}:{normalized[index:index + size]}")
    return grams


def _build_idf(feature_lists: list[list[str]]) -> dict[str, float]:
    doc_count = max(len(feature_lists), 1)
    document_frequency: Counter[str] = Counter()
    for features in feature_lists:
        document_frequency.update(set(features))
    return {
        feature: math.log((1.0 + doc_count) / (1.0 + frequency)) + 1.0
        for feature, frequency in document_frequency.items()
    }


def _tfidf_vector(features: list[str], idf: dict[str, float], *, weight: float) -> dict[str, float]:
    counts = Counter(features)
    length = max(sum(counts.values()), 1)
    return {
        feature: weight * (count / length) * idf.get(feature, 1.0)
        for feature, count in counts.items()
    }


def _merge_vectors(*vectors: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for vector in vectors:
        for key, value in vector.items():
            merged[key] = merged.get(key, 0.0) + value
    return merged


def _cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = 0.0
    for key, value in left.items():
        numerator += value * right.get(key, 0.0)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_dense(left, right) -> float:
    if _np is None:
        return 0.0
    left_norm = float(_np.linalg.norm(left))
    right_norm = float(_np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(_np.dot(left, right) / (left_norm * right_norm))


class EmbeddingBackend:
    mode = "local_semantic"
    degraded = False

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def similarity(self, left: str, right: str) -> float:
        scores = self.score_query(left, [right])
        return scores[0] if scores else 0.0


class LocalSemanticBackend(EmbeddingBackend):
    """TF-IDF + LSA fallback. Marked degraded so callers can react."""

    mode = "heuristic_fallback"
    degraded = True

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        corpus = [query, *texts]
        word_features = [semantic_terms(item) for item in corpus]
        char_features = [_char_ngrams(item) for item in corpus]
        word_idf = _build_idf(word_features)
        char_idf = _build_idf(char_features)
        sparse_vectors = [
            _merge_vectors(
                _tfidf_vector(words, word_idf, weight=1.0),
                _tfidf_vector(chars, char_idf, weight=0.75),
            )
            for words, chars in zip(word_features, char_features, strict=True)
        ]
        baseline_scores = [_cosine_sparse(sparse_vectors[0], vector) for vector in sparse_vectors[1:]]
        if _np is None:
            return baseline_scores
        feature_names = sorted({key for vector in sparse_vectors for key in vector})
        rich_documents = sum(1 for features in word_features if len(features) >= 4)
        if (
            len(feature_names) < 4
            or len(sparse_vectors) < 3
            or len(word_features[0]) < 3
            or rich_documents < max(len(corpus) // 2, 2)
        ):
            return baseline_scores
        feature_index = {name: index for index, name in enumerate(feature_names)}
        matrix = _np.zeros((len(sparse_vectors), len(feature_names)), dtype=float)
        for row_index, vector in enumerate(sparse_vectors):
            for feature, value in vector.items():
                matrix[row_index, feature_index[feature]] = value
        try:
            u, singular_values, _vt = _np.linalg.svd(matrix, full_matrices=False)
        except _np.linalg.LinAlgError:
            return baseline_scores
        rank = min(len(singular_values), max(min(matrix.shape) - 1, 0), 16)
        if rank <= 0:
            return baseline_scores
        latent = u[:, :rank] * singular_values[:rank]
        latent_scores = [_cosine_dense(latent[0], row) for row in latent[1:]]
        return [
            max(0.0, min(1.0, 0.7 * latent_score + 0.3 * baseline_score))
            for latent_score, baseline_score in zip(latent_scores, baseline_scores, strict=True)
        ]


class DegradedLexicalBackend(EmbeddingBackend):
    mode = "degraded_lexical"
    degraded = True

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        query_terms = Counter(semantic_terms(query))
        scores: list[float] = []
        for text in texts:
            doc_terms = Counter(semantic_terms(text))
            if not query_terms or not doc_terms:
                scores.append(0.0)
                continue
            numerator = 0.0
            for key, value in query_terms.items():
                numerator += value * doc_terms.get(key, 0.0)
            query_norm = math.sqrt(sum(value * value for value in query_terms.values()))
            doc_norm = math.sqrt(sum(value * value for value in doc_terms.values()))
            if query_norm == 0.0 or doc_norm == 0.0:
                scores.append(0.0)
            else:
                scores.append(numerator / (query_norm * doc_norm))
        return scores


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

    def _truncate(self, text: str) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= self._max_text_chars:
            return text
        return text[: self._max_text_chars - 1].rstrip() + "…"

    def _build_prompt(self, query: str, texts: list[str]) -> str:
        lines = [
            "You are a relevance scorer. Rate how relevant each candidate is to the query.",
            "Return JSON only: {\"scores\": [s0, s1, ...]} where each score is between 0.0 and 1.0.",
            "0.0 means unrelated; 1.0 means perfectly relevant. Use the order of candidates.",
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

    def _call_llm(self, prompt: str, item_count: int) -> list[float]:
        try:
            import requests
        except Exception:  # pragma: no cover - requests is a hard dep
            raise SemanticBackendUnavailableError("requests dependency is unavailable")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max(64, item_count * 12),
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self._seed,
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
            "json_schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": item_count,
                        "maxItems": item_count,
                    }
                },
                "required": ["scores"],
                "additionalProperties": False,
            },
        }
        unavailable_attempts = 0
        while True:
            try:
                response = requests.post(
                    f"{self._base_url}{self._endpoint}",
                    json=payload,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
                response.raise_for_status()
                body = response.json()
                break
            except requests.ConnectionError as exc:
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
            except requests.Timeout as exc:
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
            raise SemanticBackendProtocolError(f"Completion response missing string content: {body!r}")
        parsed = _parse_score_payload(text, item_count)
        if parsed is None:
            raise SemanticBackendProtocolError("Structured relevance response violated the requested schema")
        return parsed

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
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        return None
    if len(raw_scores) != expected_count:
        return None
    parsed: list[float] = []
    for value in raw_scores:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        parsed.append(max(0.0, min(1.0, number)))
    return parsed


def build_backend(
    mode: str,
    *,
    base_url: str | None = None,
    seed: int = 11,
    connect_timeout_seconds: int = 10,
    read_timeout_seconds: int = 60,
    max_text_chars: int | None = None,
) -> EmbeddingBackend:
    """Construct a semantic scoring backend.

    Selection order:
        - ``degraded_lexical`` always returns the lightweight fallback used in
          tests and offline environments.
        - ``llm_scoring`` returns :class:`LlmScoringBackend` (the default
          primary backend) when ``base_url`` is provided. Without ``base_url``
          it fails explicitly because semantic relevance is unavailable.
        - ``transformer_local`` is opt-in and only loads when
          ``SWAAG_SEMANTIC_MODEL_PATH`` points to a real model directory
          and the optional transformer dependencies are installed. If the
          transformer backend is requested explicitly and unavailable, that is
          an explicit configuration error rather than a silent fallback.
        - ``local_semantic`` is the deterministic offline fallback.
    """

    if mode == "degraded_lexical":
        return DegradedLexicalBackend()
    if mode == "llm_scoring":
        if base_url:
            kwargs: dict = {
                "base_url": base_url,
                "seed": seed,
                "connect_timeout_seconds": connect_timeout_seconds,
                "read_timeout_seconds": read_timeout_seconds,
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
    if mode == "local_semantic":
        return LocalSemanticBackend()
    raise RuntimeError(f"Unknown retrieval backend mode: {mode}")
