"""
test_urgency_retrieval.py — Tests for urgency sweep in retrieve().

WHY a dedicated test file:
    The urgency sweep is a safety-critical path: an urgency-pinned node
    MUST always appear in retrieval results regardless of its cosine
    similarity to the query. This is the "Eisenhower urgent" guarantee —
    a deadline node must not be buried by low cosine.

    These tests verify:
    1. urgent_forced bypass: u_eff ≥ theta_urgent → node in results even
       for completely unrelated queries (low cosine).
    2. Multiple unrelated queries all surface the urgent node.
    3. Non-urgent nodes are NOT force-injected (no false positives).
    4. Urgency epsilon component raises rank for mildly urgent nodes.
    5. Score sort order is preserved after urgency merge.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livemem.config import LiveConfig
from livemem.embedder import CrossEncoderReranker, MockEmbedder
from livemem.memory import LiveMem
from livemem.types import Tier, urgency_effective

from .conftest import make_node


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def urgency_config() -> LiveConfig:
    """Config designed to maximise urgency-sweep visibility.

    WHY these values:
        theta_min=0.01  : allow edges between nearly-orthogonal MockEmbedder
                          vectors so the graph is connected.
        theta_urgent=0.7: realistic threshold — a node with urgency=0.95
                          and urgency_lambda=1e-5 stays pinned for hours.
        urgency_lambda=1e-5: slow enough that within a test run (< 1s) the
                          urgency does not decay below theta_urgent.
        epsilon_score=0.08: same as DEFAULT_CONFIG.
    """
    return LiveConfig(
        d=16,
        T1=1.0,
        T2=10.0,
        theta_min=0.01,
        theta_sleep=0.01,
        theta_promote=0.01,
        theta_compress=0.95,
        theta_urgent=0.7,
        urgency_lambda=1e-5,
        importance_medium_floor=0.6,
        decay_lambda=1e-5,
        delta_reinforce=0.10,
        tau_long=5.0,
        idle_ttl=300.0,
        daemon_check_interval=30.0,
        max_nodes=100,
        long_compress_fraction=0.7,
        alpha_score=0.45,
        beta_score=0.20,
        gamma_score=0.15,
        delta_score=0.12,
        epsilon_score=0.08,
        s_base_init=0.5,
        hnsw_max_elements=1000,
        hnsw_ef_construction=50,
        hnsw_M=8,
        hnsw_ef_search=50,
    )


@pytest.fixture
def mem_with_deadline(urgency_config: LiveConfig) -> LiveMem:
    """LiveMem with 8 nodes including one DEADLINE node (urgency=0.95).

    All 8 summaries produce orthogonal-ish MockEmbedder vectors (SHA256-seeded).
    The DEADLINE node has urgency=0.95 >> theta_urgent=0.7.
    The other 7 nodes have urgency=0.0.
    """
    mem = LiveMem(cfg=urgency_config, mock=True)
    for summary, importance, urgency in [
        ("Python is great for data science", 0.8, 0.0),
        ("The Eiffel Tower is in Paris", 0.5, 0.0),
        ("Coffee helps with productivity", 0.3, 0.0),
        ("Stars shine in the night sky", 0.2, 0.0),
        ("DEADLINE: submit report by 5pm", 0.7, 0.95),
        ("Deep learning beats classical ML", 0.9, 0.0),
        ("France is in Western Europe", 0.55, 0.0),
        ("Weather is sunny today", 0.2, 0.0),
    ]:
        mem.ingest_awake(summary, importance=importance, urgency=urgency)
        time.sleep(0.001)  # ensure distinct t values
    return mem


# ── Core: urgent_forced bypass ─────────────────────────────────────────────────


def test_urgent_node_in_results_for_unrelated_query(
    mem_with_deadline: LiveMem,
) -> None:
    """DEADLINE must appear in top-5 even for a semantically unrelated query.

    WHY this matters:
        The urgency sweep in retrieve() collects u_eff ≥ theta_urgent nodes
        via a direct graph scan and merges them into all_candidates before
        scoring. Without this, a low-cosine DEADLINE node would never enter
        the ANN seed pool and be silently dropped.
    """
    results = mem_with_deadline.retrieve("coffee weather sky today", k=5)
    summaries = [r.summary for r in results]
    assert any("DEADLINE" in s for s in summaries), (
        "Urgency sweep failed: DEADLINE node not found in top-5 for unrelated query."
    )


def test_urgent_node_surfaces_for_multiple_unrelated_queries(
    mem_with_deadline: LiveMem,
) -> None:
    """DEADLINE must surface regardless of which unrelated query is used."""
    queries = [
        "coffee weather sky today",
        "Paris Eiffel Tower France",
        "machine learning gradient descent",
    ]
    for query in queries:
        results = mem_with_deadline.retrieve(query, k=5)
        found = any("DEADLINE" in r.summary for r in results)
        assert found, f"DEADLINE not found for query: {query!r}"


def test_urgent_node_not_present_when_urgency_zero(
    urgency_config: LiveConfig,
) -> None:
    """A node with urgency=0.0 must NOT be force-injected.

    The urgency sweep only injects nodes with u_eff ≥ theta_urgent.
    A node with urgency=0.0 has u_eff=0.0 at all times.
    """
    mem = LiveMem(cfg=urgency_config, mock=True)
    # Only non-urgent nodes.
    for summary, importance in [
        ("Python is great for data science", 0.8),
        ("Coffee helps with productivity", 0.3),
        ("Weather is sunny today", 0.2),
    ]:
        mem.ingest_awake(summary, importance=importance, urgency=0.0)
        time.sleep(0.001)

    results = mem.retrieve("astronomy space stars", k=3)
    # No force-injection: results must be only the 3 nodes we have.
    assert len(results) <= 3
    # None should have urgency > 0.
    assert all(r.urgency == 0.0 for r in results)


# ── Sort order after urgency merge ────────────────────────────────────────────


def test_scores_descending_with_urgent_node(mem_with_deadline: LiveMem) -> None:
    """Results must be sorted descending by score even after urgency merge."""
    for query in ["coffee weather sky today", "Paris Eiffel France"]:
        results = mem_with_deadline.retrieve(query, k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not descending for query {query!r}: {scores}"
        )


# ── Epsilon score component ────────────────────────────────────────────────────


def test_urgency_epsilon_raises_rank(urgency_config: LiveConfig) -> None:
    """A moderately urgent node should rank higher than an identical non-urgent node.

    We manually add two nodes with the same vector and importance, differing
    only in urgency, then retrieve and verify the urgent one ranks higher.
    WHY same vector: eliminates cosine and traversal variation — only
    s_eff, importance, and urgency_effective differ.
    """
    mem = LiveMem(cfg=urgency_config, mock=True)
    rng = np.random.default_rng(42)
    shared_v = rng.standard_normal(urgency_config.d).astype(np.float32)
    shared_v /= np.linalg.norm(shared_v)

    now = time.time()

    # Inject two nodes with identical vectors but different urgency.
    urgent_node = make_node(
        summary="urgent task",
        d=urgency_config.d,
        importance=0.5,
        urgency=0.8,
        s_base=0.5,
        seed=42,
    )
    # Overwrite v with shared_v so vectors are identical.
    object.__setattr__(urgent_node, "v", shared_v.copy())

    calm_node = make_node(
        summary="calm task",
        d=urgency_config.d,
        importance=0.5,
        urgency=0.0,
        s_base=0.5,
        seed=42,
    )
    object.__setattr__(calm_node, "v", shared_v.copy())

    # Manually add to graph + index (bypassing ingest_awake for full control).
    mem.graph.add_node(urgent_node)
    mem.index.add(urgent_node.id, urgent_node.v, Tier.SHORT)
    mem.graph.add_node(calm_node)
    mem.index.add(calm_node.id, calm_node.v, Tier.SHORT)

    results = mem.retrieve("urgent task", k=2)
    assert len(results) == 2
    ids = [r.node_id for r in results]
    # Urgent node must rank #1 (higher score via epsilon component).
    assert ids[0] == urgent_node.id, (
        f"urgent_node should rank #1 but got: {[r.summary for r in results]}"
    )


# ── urgency_effective helper ───────────────────────────────────────────────────


def test_urgency_effective_at_creation_equals_urgency(
    urgency_config: LiveConfig,
) -> None:
    """u_eff at creation time should equal node.urgency."""
    node = make_node(urgency=0.8, d=urgency_config.d, seed=99)
    u = urgency_effective(node, node.t, urgency_config)
    assert abs(u - 0.8) < 1e-6


def test_urgency_effective_decays_below_threshold_after_long_time(
    urgency_config: LiveConfig,
) -> None:
    """After sufficient time, u_eff must fall below theta_urgent (no eternal pin).

    With urgency_lambda=1e-5 and urgency=0.95, the half-life is ~19h.
    After 200_000 seconds (~2.3 days), u_eff should be well below 0.7.
    """
    node = make_node(urgency=0.95, d=urgency_config.d, seed=100)
    u = urgency_effective(node, node.t + 200_000, urgency_config)
    assert u < urgency_config.theta_urgent, (
        f"Node should no longer be pinned after 200k seconds, but u_eff={u:.4f}"
    )


# ── CrossEncoderReranker unit tests ────────────────────────────────────────────


def test_cross_encoder_reranker_empty_candidates(
    urgency_config: LiveConfig,
) -> None:
    """CrossEncoderReranker.rerank() with empty candidates returns []."""
    reranker = CrossEncoderReranker(urgency_config)
    result = reranker.rerank("some query", [])
    assert result == []


def test_cross_encoder_reranker_sorts_descending(
    urgency_config: LiveConfig,
) -> None:
    """CrossEncoderReranker.rerank() must return (score, node_id) sorted descending.

    We mock fastembed.TextCrossEncoder to return controlled scores
    without any network download.
    """
    mock_model = MagicMock()
    # Model returns scores [0.3, 0.9, 0.1] for 3 candidates.
    mock_model.rerank.return_value = iter([0.3, 0.9, 0.1])

    reranker = CrossEncoderReranker(urgency_config)
    reranker._model = mock_model  # inject mock directly (lazy-load already "done")

    candidates = [("node_a", "summary a"), ("node_b", "summary b"), ("node_c", "summary c")]
    result = reranker.rerank("query", candidates)

    assert len(result) == 3
    scores = [s for s, _ in result]
    node_ids = [nid for _, nid in result]
    # Sorted descending.
    assert scores == sorted(scores, reverse=True)
    # Highest score corresponds to node_b (score 0.9).
    assert node_ids[0] == "node_b"


def test_cross_encoder_reranker_disabled_uses_biencoder_score(
    urgency_config: LiveConfig,
) -> None:
    """When reranker_enabled=False, retrieve() must use bi-encoder scores, not CE.

    We verify by checking that no cross-encoder model is ever called.
    """
    cfg = LiveConfig(**{**urgency_config.__dict__, "reranker_enabled": False})
    mem = LiveMem(cfg=cfg, mock=True)
    for s in ["Python ML models", "Paris Eiffel Tower", "Coffee morning"]:
        mem.ingest_awake(s, importance=0.5)
        time.sleep(0.001)

    with patch.object(mem._reranker, "rerank") as mock_rerank:
        results = mem.retrieve("Python code", k=2)
        mock_rerank.assert_not_called()

    assert len(results) <= 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_cross_encoder_reranker_enabled_calls_rerank(
    urgency_config: LiveConfig,
) -> None:
    """When reranker_enabled=True, retrieve() must call CrossEncoderReranker.rerank().

    We inject a mock model so no download occurs.
    """
    cfg = LiveConfig(**{**urgency_config.__dict__, "reranker_enabled": True, "reranker_k": 3})
    mem = LiveMem(cfg=cfg, mock=True)
    for s in ["Python ML models", "Paris Eiffel Tower", "Coffee morning"]:
        mem.ingest_awake(s, importance=0.5)
        time.sleep(0.001)

    # Mock the CrossEncoderReranker.rerank to return a controlled ranking.
    def fake_rerank(query: str, candidates: list) -> list:
        # Return candidates in reverse order with fake scores.
        return [(1.0 - i * 0.1, nid) for i, (nid, _) in enumerate(candidates)]

    with patch.object(mem._reranker, "rerank", side_effect=fake_rerank) as mock_rerank:
        results = mem.retrieve("Python code", k=2)
        mock_rerank.assert_called_once()

    assert len(results) <= 2
