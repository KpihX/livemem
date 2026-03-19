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
        """Lazy-load the fastembed model on first call.

        WHY dimension check:
            If the caller passes d=384 but model_name points to bge-base
            (768d), every vector will be silently truncated or padded by
            the HNSW index, producing garbage retrieval. We log a warning
            so the mismatch surfaces immediately at first embed() call
            rather than manifesting as mysterious low recall.
        """
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
            # Sanity-check: emit a probe vector and compare its length to cfg.d.
            probe = list(self._model.embed(["probe"]))  # type: ignore[attr-defined]
            actual_d = len(probe[0]) if probe else 0
            if actual_d and actual_d != self._cfg.d:
                logger.warning(
                    "RealEmbedder: model '%s' outputs d=%d but cfg.d=%d. "
                    "Vectors will be mis-sized — update LiveConfig(d=%d).",
                    self._cfg.model_name,
                    actual_d,
                    self._cfg.d,
                    actual_d,
                )
            logger.info("Model loaded (d=%d).", actual_d or self._cfg.d)
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


class CrossEncoderReranker:
    """Re-ranks bi-encoder HNSW candidates using a cross-encoder model.

    WHY cross-encoder post-HNSW (the two-stage retrieval paradigm):
        Bi-encoder HNSW is fast (O(log N) ANN lookup) but encodes query
        and document independently — there is no token-level attention
        between them. Cross-encoders jointly encode (query, document) pairs,
        enabling full attention across both texts. This produces dramatically
        more accurate relevance scores at the cost of O(reranker_k)
        sequential inference calls per query.

        Two-stage pipeline:
          ┌──────────────────────────────────┐
          │ HNSW bi-encoder                  │  fast, approximate
          │ query vector → top-reranker_k    │  O(log N)
          └──────────────┬───────────────────┘
                         │ candidate (node_id, summary) pairs
          ┌──────────────▼───────────────────┐
          │ Cross-encoder re-ranker          │  precise, O(reranker_k)
          │ (query, summary) → score per pair│
          └──────────────┬───────────────────┘
                         │ final top-k, true relevance order
                         ▼

    WHY lazy loading (same rationale as RealEmbedder):
        Cross-encoder ONNX models are ~100-400 MB. Deferring the download
        until the first rerank() call avoids blocking import-time.

    Note: CrossEncoderReranker is stateless between calls — it only
    carries the loaded model and its config. Thread-safety follows from
    fastembed's ONNX session being read-only during inference.
    """

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        self._cfg = cfg
        self._model: object | None = None  # Lazy-loaded TextCrossEncoder.

    def _get_model(self) -> object:
        """Lazy-load the fastembed cross-encoder model on first call."""
        if self._model is None:
            try:
                from fastembed import TextCrossEncoder  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "fastembed >= 0.3 with TextCrossEncoder is required for "
                    "CrossEncoderReranker. Install with: pip install fastembed"
                ) from exc
            logger.info(
                "Loading cross-encoder model: %s", self._cfg.reranker_model_name
            )
            self._model = TextCrossEncoder(
                model_name=self._cfg.reranker_model_name
            )
            logger.info("Cross-encoder loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[float, str]]:
        """Re-rank (node_id, summary) pairs for a query.

        Parameters
        ----------
        query      : str — the retrieval query text (same as passed to embed).
        candidates : list of (node_id, summary) — bi-encoder top candidates.

        Returns
        -------
        list of (cross_encoder_score, node_id) sorted descending by score.

        WHY use summary, not full content:
            Nodes store a compact summary (≤200 chars) rather than raw
            content. The cross-encoder re-ranks on this summary — which is
            already a distilled semantic representation of the full content.
            Using the full ref_uri content would require loading potentially
            large blobs from disk on every retrieval, breaking latency.
        """
        if not candidates:
            return []
        model = self._get_model()
        passages = [summary for _, summary in candidates]
        node_ids = [nid for nid, _ in candidates]
        # fastembed TextCrossEncoder.rerank() returns an iterable of floats.
        raw_scores: list[float] = list(
            model.rerank(query, passages)  # type: ignore[attr-defined]
        )
        ranked = sorted(
            zip(raw_scores, node_ids),
            key=lambda x: x[0],
            reverse=True,
        )
        return list(ranked)


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
