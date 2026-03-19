"""Backward-compatible facade for the embeddings package."""
from livemem.embeddings import (
    BaseEmbedder,
    BaseReranker,
    FastEmbedCrossEncoderReranker,
    FastEmbedTextEmbedder,
    MockEmbedder,
    make_embedder,
    make_reranker,
)

CrossEncoderReranker = FastEmbedCrossEncoderReranker
RealEmbedder = FastEmbedTextEmbedder

__all__ = [
    "BaseEmbedder",
    "BaseReranker",
    "MockEmbedder",
    "RealEmbedder",
    "CrossEncoderReranker",
    "FastEmbedTextEmbedder",
    "FastEmbedCrossEncoderReranker",
    "make_embedder",
    "make_reranker",
]
