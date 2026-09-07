from __future__ import annotations

from typing import Protocol

from swaag.embedding_index import EmbeddingProvider, OpenAICompatibleEmbeddingProvider


class EmbeddingProviderConfig(Protocol):
    base_url: str
    endpoint: str
    model: str
    timeout_seconds: float


def build_embedding_provider(config: EmbeddingProviderConfig) -> EmbeddingProvider:
    """Build the configured embedding transport behind the neutral provider interface."""
    return OpenAICompatibleEmbeddingProvider(
        config.base_url,
        config.endpoint,
        config.model,
        config.timeout_seconds,
    )
