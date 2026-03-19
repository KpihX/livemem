"""Deterministic mock embedding implementation."""
from __future__ import annotations

import hashlib

import numpy as np

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embeddings.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Deterministic unit-vector embedder for tests and offline demos."""

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        super().__init__(cfg)

    def embed(self, text: str) -> np.ndarray:
        sha = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(sha[:4], byteorder="big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self._cfg.d).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            vector = np.zeros(self._cfg.d, dtype=np.float32)
            vector[0] = 1.0
            return vector
        vector /= norm
        return vector
