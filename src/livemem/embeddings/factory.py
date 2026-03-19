"""Factory functions for embedding and reranking implementations."""
from __future__ import annotations

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embeddings.base import BaseEmbedder, BaseReranker
from livemem.embeddings.fastembed_cross_encoder import FastEmbedCrossEncoderReranker
from livemem.embeddings.fastembed_text import FastEmbedTextEmbedder
from livemem.embeddings.mock import MockEmbedder


_EMBEDDER_IMPLEMENTATIONS: dict[str, type[BaseEmbedder]] = {
    "mock": MockEmbedder,
    "fastembed": FastEmbedTextEmbedder,
    "fastembed_text": FastEmbedTextEmbedder,
}

_RERANKER_IMPLEMENTATIONS: dict[str, type[BaseReranker]] = {
    "fastembed_cross_encoder": FastEmbedCrossEncoderReranker,
}


def make_embedder(
    cfg: LiveConfig = DEFAULT_CONFIG,
    *,
    mock: bool = False,
) -> BaseEmbedder:
    """Build the configured embedder implementation."""
    implementation = "mock" if mock else cfg.embedder_implementation
    try:
        embedder_cls = _EMBEDDER_IMPLEMENTATIONS[implementation]
    except KeyError as exc:
        available = ", ".join(sorted(_EMBEDDER_IMPLEMENTATIONS))
        raise ValueError(
            f"Unknown embedder implementation '{implementation}'. "
            f"Available: {available}"
        ) from exc
    return embedder_cls(cfg)


def make_reranker(cfg: LiveConfig = DEFAULT_CONFIG) -> BaseReranker:
    """Build the configured reranker implementation."""
    try:
        reranker_cls = _RERANKER_IMPLEMENTATIONS[cfg.reranker_implementation]
    except KeyError as exc:
        available = ", ".join(sorted(_RERANKER_IMPLEMENTATIONS))
        raise ValueError(
            f"Unknown reranker implementation '{cfg.reranker_implementation}'. "
            f"Available: {available}"
        ) from exc
    return reranker_cls(cfg)
