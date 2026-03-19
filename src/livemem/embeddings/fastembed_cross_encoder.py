"""FastEmbed-backed cross-encoder reranker."""
from __future__ import annotations

import logging

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embeddings.base import BaseReranker

logger = logging.getLogger(__name__)


class FastEmbedCrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using fastembed TextCrossEncoder."""

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        super().__init__(cfg)
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextCrossEncoder  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "fastembed >= 0.3 with TextCrossEncoder is required for "
                    "FastEmbedCrossEncoderReranker. Install with: pip install fastembed"
                ) from exc
            logger.info(
                "Loading cross-encoder model: %s", self._cfg.reranker_model_name
            )
            self._model = TextCrossEncoder(
                model_name=self._cfg.reranker_model_name
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[float, str]]:
        if not candidates:
            return []
        model = self._get_model()
        passages = [summary for _, summary in candidates]
        node_ids = [node_id for node_id, _ in candidates]
        raw_scores: list[float] = list(
            model.rerank(query, passages)  # type: ignore[attr-defined]
        )
        ranked = sorted(
            zip(raw_scores, node_ids),
            key=lambda item: item[0],
            reverse=True,
        )
        return list(ranked)
