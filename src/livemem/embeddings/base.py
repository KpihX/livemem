"""Base contracts for embedders and rerankers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from livemem.config import DEFAULT_CONFIG, LiveConfig


class BaseEmbedder(ABC):
    """Abstract interface for text embedding implementations."""

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        self._cfg = cfg

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed one text string into a unit-norm vector."""

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(text) for text in texts]


class BaseReranker(ABC):
    """Abstract interface for retrieval rerankers."""

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        self._cfg = cfg

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[float, str]]:
        """Return candidates sorted by decreasing reranker score."""
