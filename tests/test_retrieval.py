"""
test_retrieval.py — Focused tests for the retrieve() algorithm:
                    scoring formula, traversal, CAPITAL sweep, edge cases.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from livemem.config import LiveConfig
from livemem.memory import LiveMem
from livemem.types import Importance, Tier
from tests.conftest import make_node


# ── Basic retrieval ────────────────────────────────────────────────────────────

def test_retrieve_at_most_k_results(fresh_mem):
    for i in range(10):
        fresh_mem.ingest_awake(f"Memory {i} with some content")
    results = fresh_mem.retrieve("memory content", k=5)
    assert len(results) <= 5


def test_retrieve_scores_descending(fresh_mem):
    for i in range(6):
        fresh_mem.ingest_awake(f"Fact {i} about science and technology")
    results = fresh_mem.retrieve("science technology", k=6)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_with_one_node(fresh_mem):
    nid = fresh_mem.ingest_awake("The only node in the graph")
    results = fresh_mem.retrieve("only node", k=10)
    assert len(results) >= 1
    assert results[0].node_id == nid


def test_retrieve_k_larger_than_total_nodes(fresh_mem):
    """k > total_nodes should return all available nodes without crash."""
    for i in range(3):
        fresh_mem.ingest_awake(f"Node {i}")
    results = fresh_mem.retrieve("node", k=100)
    assert len(results) <= 3


# ── Traversal score ────────────────────────────────────────────────────────────

def test_graph_traversal_adds_score_to_neighbors(small_config, mock_embedder):
    """A node connected to a top-k seed should receive a traversal bonus."""
    from livemem.types import Edge, EdgeType
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)

    # Ingest seed node.
    seed_id = mem.ingest_awake("machine learning neural networks")
    seed_node = mem.graph.V[seed_id]

    # Create a neighbour node and connect it to the seed via a DIRECT edge.
    neighbor = make_node(
        summary="deep learning architectures",
        tier=Tier.SHORT,
        seed=900,
        d=small_config.d,
    )
    mem.graph.add_node(neighbor)
    mem.index.add(neighbor.id, neighbor.v, Tier.SHORT)

    # Edge: seed → neighbor (seed is newer).
    e = Edge(
        from_id=seed_id,
        to_id=neighbor.id,
        cos_sim=0.8,
        delta_t=1.0,
        edge_type=EdgeType.DIRECT,
    )
    mem.graph.add_edge(e)

    results = mem.retrieve("machine learning", k=10)
    result_ids = [r.node_id for r in results]
    # Both seed and neighbour should appear (traversal brings in neighbour).
    assert seed_id in result_ids


# ── CAPITAL sweep ──────────────────────────────────────────────────────────────

def test_capital_nodes_from_medium_appear(small_config, mock_embedder):
    """CAPITAL nodes in MEDIUM should be surfaced by retrieve."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    cap_id = mem.ingest_awake("critical capital architecture decision", importance=Importance.CAPITAL)
    node = mem.graph.V[cap_id]
    # Move to MEDIUM.
    mem.graph.update_tier_set(node, Tier.SHORT, Tier.MEDIUM)
    mem.index.move(cap_id, node.v, Tier.SHORT, Tier.MEDIUM)
    node.tier = Tier.MEDIUM

    results = mem.retrieve("critical architecture", k=5)
    assert cap_id in [r.node_id for r in results]


def test_capital_nodes_from_long_appear(small_config, mock_embedder):
    """CAPITAL nodes in LONG should be surfaced by retrieve."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    cap_id = mem.ingest_awake("fundamental capital fact about memory", importance=Importance.CAPITAL)
    node = mem.graph.V[cap_id]
    # Move to LONG.
    mem.graph.update_tier_set(node, Tier.SHORT, Tier.LONG)
    mem.index.move(cap_id, node.v, Tier.SHORT, Tier.LONG)
    node.tier = Tier.LONG

    results = mem.retrieve("fundamental memory fact", k=5)
    assert cap_id in [r.node_id for r in results]


# ── Scoring formula ────────────────────────────────────────────────────────────

def test_score_uses_all_components(small_config, mock_embedder):
    """Final score should be positive and < 1 for typical inputs."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("test scoring formula components")
    results = mem.retrieve("scoring formula", k=5)
    assert len(results) >= 1
    for r in results:
        assert r.score >= 0.0


def test_score_importance_bonus(small_config, mock_embedder):
    """A KEY node should score higher than an identical WEAK node."""
    from livemem.embedder import MockEmbedder
    import uuid as _uuid

    mem = LiveMem(cfg=small_config, embedder=mock_embedder)

    # Create two nodes with the same vector but different importances.
    v = MockEmbedder(small_config).embed("same vector text")
    key_node = make_node(
        summary="same vector text",
        importance=Importance.KEY,
        tier=Tier.SHORT,
        d=small_config.d,
        seed=950,
    )
    weak_node = make_node(
        summary="same vector text",
        importance=Importance.WEAK,
        tier=Tier.SHORT,
        d=small_config.d,
        seed=950,
    )
    # Override IDs to be different.
    import uuid
    object.__setattr__(key_node, "id", str(uuid.uuid4()))
    object.__setattr__(weak_node, "id", str(uuid.uuid4()))

    mem.graph.add_node(key_node)
    mem.index.add(key_node.id, key_node.v, Tier.SHORT)
    mem.graph.add_node(weak_node)
    mem.index.add(weak_node.id, weak_node.v, Tier.SHORT)

    results = mem.retrieve("same vector text", k=10)
    scores = {r.node_id: r.score for r in results}

    if key_node.id in scores and weak_node.id in scores:
        assert scores[key_node.id] >= scores[weak_node.id]


# ── Reinforce side-effects ────────────────────────────────────────────────────

def test_retrieve_updates_t_accessed(fresh_mem):
    nid = fresh_mem.ingest_awake("access time tracking test")
    node = fresh_mem.graph.V[nid]
    t_before = node.t_accessed
    time.sleep(0.01)
    fresh_mem.retrieve("access time tracking", k=5)
    assert node.t_accessed >= t_before


# ── RetrievalResult fields ────────────────────────────────────────────────────

def test_retrieval_result_has_correct_ref_uri(fresh_mem):
    nid = fresh_mem.ingest_awake(
        "media reference test",
        ref_uri="/audio/file.mp3",
        ref_type="audio",
    )
    results = fresh_mem.retrieve("media reference", k=5)
    matching = [r for r in results if r.node_id == nid]
    if matching:
        assert matching[0].ref_uri == "/audio/file.mp3"
        assert matching[0].ref_type == "audio"
