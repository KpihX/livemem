"""FastEmbed-backed text embedding implementation."""
from __future__ import annotations

import logging

import numpy as np

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class FastEmbedTextEmbedder(BaseEmbedder):
    """Production embedder using fastembed TextEmbedding."""

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        super().__init__(cfg)
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required for FastEmbedTextEmbedder. "
                    "Install with: pip install fastembed"
                ) from exc
            logger.info("Loading fastembed model: %s", self._cfg.model_name)
            self._model = TextEmbedding(model_name=self._cfg.model_name)
            probe = list(self._model.embed(["probe"]))  # type: ignore[attr-defined]
            actual_d = len(probe[0]) if probe else 0
            if actual_d and actual_d != self._cfg.d:
                logger.warning(
                    "FastEmbedTextEmbedder: model '%s' outputs d=%d but cfg.d=%d. "
                    "Update config.yaml or override LiveConfig(d=%d).",
                    self._cfg.model_name,
                    actual_d,
                    self._cfg.d,
                    actual_d,
                )
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        vectors = list(model.embed([text]))  # type: ignore[attr-defined]
        vector = np.array(vectors[0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vector /= norm
        return vector

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        model = self._get_model()
        vectors = list(model.embed(texts))  # type: ignore[attr-defined]
        result: list[np.ndarray] = []
        for raw_vector in vectors:
            vector = np.array(raw_vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12:
                vector /= norm
            result.append(vector)
        return result
