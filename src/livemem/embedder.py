"""
embedder.py — Text-to-vector embedding abstraction layer.

WHY an abstract base class:
    Tests must not depend on a real embedding model (network download,
    inference latency, non-determinism). The MockEmbedder provides
    deterministic unit vectors derived from the text's hash, making
    tests fast and reproducible. RealEmbedder wraps fastembed for
    production use. make_embedder() is the single factory used throughout
    the codebase, switching behaviour via a flag.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from livemem.config import DEFAULT_CONFIG, LiveConfig

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface for text embedding.

    Any concrete embedder must produce unit-norm vectors of shape (d,).
    The dimension d is set at construction and must match the LiveConfig
    passed to the rest of the system.
    """

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        self._cfg = cfg

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single string, returning a unit-norm vector (shape (d,)).

        WHY unit-norm contract:
            Node.__post_init__ re-normalises anyway, but having the
            embedder normalise first avoids any floating-point drift
            accumulating through multiple transforms.
        """

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of strings.

        Default implementation: loop over embed(). Concrete subclasses
        may override for more efficient batch inference.
        """
        return [self.embed(t) for t in texts]


class MockEmbedder(BaseEmbedder):
    """Deterministic unit-vector embedder for testing.

    WHY deterministic:
        Test assertions about cosine similarity, tier transitions, and
        graph structure require predictable vectors. Using a seeded RNG
        derived from the SHA-256 hash of the text guarantees that:
          1. The same text ALWAYS returns the same vector across runs.
          2. Different texts (with high probability) return different vectors.
          3. No network access or model download is required.

    Implementation detail:
        SHA256(text) → first 4 bytes → uint32 seed →
        np.random.default_rng(seed).standard_normal(d) → L2-normalize.
    """

    def embed(self, text: str) -> np.ndarray:
        """Return a deterministic unit-norm vector derived from text hash."""
        sha = hashlib.sha256(text.encode("utf-8")).digest()
        # Take first 4 bytes as a uint32 seed for the RNG.
        seed = int.from_bytes(sha[:4], byteorder="big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._cfg.d).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            # Extremely unlikely edge case: zero vector → fall back to e_0.
            v = np.zeros(self._cfg.d, dtype=np.float32)
            v[0] = 1.0
        else:
            v /= norm
        return v


class RealEmbedder(BaseEmbedder):
    """Production embedder using fastembed TextEmbedding.

    WHY lazy loading:
        fastembed downloads a ONNX model (~120 MB for bge-small) on first
        use. Lazy loading in __call__/embed defers this download until an
        actual embedding is requested, so importing livemem does not
        trigger a network request.

    WHY fastembed over sentence-transformers:
        fastembed uses ONNX Runtime (no PyTorch overhead), making it
        ~3× faster on CPU for single-sentence inference — critical for
        low-latency awake ingestion.
    """

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        super().__init__(cfg)
        self._model: object | None = None  # Lazy-loaded TextEmbedding.

    def _get_model(self) -> object:
        """Lazy-load the fastembed model on first call."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required for RealEmbedder. "
                    "Install with: pip install fastembed"
                ) from exc
            logger.info("Loading fastembed model: %s", self._cfg.model_name)
            self._model = TextEmbedding(model_name=self._cfg.model_name)
            logger.info("Model loaded.")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string, returning a unit-norm vector.

        fastembed.TextEmbedding.embed() is a generator; we materialise
        the first (and only) result.
        """
        model = self._get_model()
        # embed() returns a generator of numpy arrays.
        vectors = list(model.embed([text]))  # type: ignore[attr-defined]
        v = np.array(vectors[0], dtype=np.float32)
        # Normalise (fastembed already normalises bge models, but be safe).
        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            v /= norm
        return v

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Efficient batch embedding via fastembed's passage_embed.

        WHY batch:
            ONNX Runtime can process a padded batch in one forward pass,
            giving ~N× throughput compared to N sequential calls.
        """
        if not texts:
            return []
        model = self._get_model()
        vectors = list(model.embed(texts))  # type: ignore[attr-defined]
        result: list[np.ndarray] = []
        for v_raw in vectors:
            v = np.array(v_raw, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 1e-12:
                v /= norm
            result.append(v)
        return result


def make_embedder(
    cfg: LiveConfig = DEFAULT_CONFIG, *, mock: bool = False
) -> BaseEmbedder:
    """Factory function: return a MockEmbedder or RealEmbedder.

    Parameters
    ----------
    cfg  : LiveConfig — configuration (d, model_name).
    mock : bool       — if True, return MockEmbedder (tests / offline use).
    """
    if mock:
        return MockEmbedder(cfg)
    return RealEmbedder(cfg)
