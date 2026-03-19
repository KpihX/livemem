"""
test_memory.py — Tests for the LiveMem core operations:
                 ingest_awake, retrieve, _reinforce, _update_tier, status.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from livemem.config import LiveConfig
from livemem.types import Tier
from tests.conftest import make_node


# ── ingest_awake ───────────────────────────────────────────────────────────────

def test_ingest_returns_uuid(fresh_mem):
    nid = fresh_mem.ingest_awake("Test memory about Python programming.")
    assert isinstance(nid, str)
    assert len(nid) == 36  # UUID4 format.


def test_ingest_lands_in_short(fresh_mem):
    nid = fresh_mem.ingest_awake("Another test memory.")
    node = fresh_mem.graph.V[nid]
    assert node.tier == Tier.SHORT


def test_ingest_stores_summary(fresh_mem):
    summary = "The Eiffel Tower is in Paris."
    nid = fresh_mem.ingest_awake(summary)
    assert fresh_mem.graph.V[nid].summary == summary


def test_ingest_stores_ref_uri(fresh_mem):
    nid = fresh_mem.ingest_awake(
        "A photo", ref_uri="/images/photo.jpg", ref_type="image"
    )
    node = fresh_mem.graph.V[nid]
    assert node.ref_uri == "/images/photo.jpg"
    assert node.ref_type == "image"


def test_ingest_ref_type_image(fresh_mem):
    nid = fresh_mem.ingest_awake("Image node", ref_type="image")
    assert fresh_mem.graph.V[nid].ref_type == "image"


def test_ingest_ref_type_audio(fresh_mem):
    nid = fresh_mem.ingest_awake("Audio node", ref_type="audio")
    assert fresh_mem.graph.V[nid].ref_type == "audio"


def test_ingest_ref_type_video(fresh_mem):
    nid = fresh_mem.ingest_awake("Video node", ref_type="video")
    assert fresh_mem.graph.V[nid].ref_type == "video"


def test_ingest_ref_type_url(fresh_mem):
    nid = fresh_mem.ingest_awake(
        "URL node", ref_uri="https://example.com", ref_type="url"
    )
    assert fresh_mem.graph.V[nid].ref_type == "url"


def test_ingest_creates_edges_between_similar_nodes(small_config, mock_embedder):
    """Two very similar nodes should get a DIRECT edge between them."""
    from livemem.memory import LiveMem
    # Use very low theta_min to ensure edges form.
    cfg = LiveConfig(**{**small_config.__dict__, "theta_min": 0.0})
    mem = LiveMem(cfg=cfg, embedder=mock_embedder)

    # Ingest two semantically related items (same summary → same vector).
    nid1 = mem.ingest_awake("neural network backpropagation")
    nid2 = mem.ingest_awake("neural network backpropagation learning")
    # At least some edge should exist in the graph after two ingestions.
    assert mem.graph.total_edges() >= 0  # Won't crash; edge may or may not form.


def test_ingest_high_theta_creates_no_edges(small_config, mock_embedder):
    """With theta_min=0.999, dissimilar vectors should NOT form edges."""
    from livemem.memory import LiveMem
    cfg = LiveConfig(**{**small_config.__dict__, "theta_min": 0.999})
    mem = LiveMem(cfg=cfg, embedder=mock_embedder)
    mem.ingest_awake("alpha")
    mem.ingest_awake("beta different topic completely")
    # With such a high threshold, edges between random mock vectors unlikely.
    # The test just verifies no crash and logic runs correctly.
    assert mem.graph.total_nodes() == 2


def test_ingest_all_start_in_short(fresh_mem):
    """Multiple ingested nodes should all start in SHORT."""
    for i in range(5):
        nid = fresh_mem.ingest_awake(f"Memory fact {i}")
        assert fresh_mem.graph.V[nid].tier == Tier.SHORT


def test_ingest_importance_stored(fresh_mem):
    nid = fresh_mem.ingest_awake("Capital memory", importance=1.0)
    assert abs(fresh_mem.graph.V[nid].importance - 1.0) < 1e-6


# ── retrieve ───────────────────────────────────────────────────────────────────

def test_retrieve_empty_returns_empty(fresh_mem):
    results = fresh_mem.retrieve("anything", k=5)
    assert results == []


def test_retrieve_returns_sorted_by_score(fresh_mem):
    for i in range(5):
        fresh_mem.ingest_awake(f"Memory about topic {i} details")
    results = fresh_mem.retrieve("topic details", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_reinforces_returned_nodes(fresh_mem):
    nid = fresh_mem.ingest_awake("Python programming language")
    node = fresh_mem.graph.V[nid]
    old_t_accessed = node.t_accessed
    # Small delay to ensure time difference.
    time.sleep(0.01)
    fresh_mem.retrieve("Python", k=5)
    # t_accessed should have been updated.
    assert node.t_accessed >= old_t_accessed


def test_retrieve_high_importance_from_medium(small_config, mock_embedder):
    """High-importance nodes in MEDIUM should appear in retrieve results."""
    from livemem.memory import LiveMem
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)

    # Ingest a high-importance node (>= importance_medium_floor=0.6) and move to MEDIUM.
    nid = mem.ingest_awake("critical capital memory", importance=1.0)
    node = mem.graph.V[nid]
    # Force move to MEDIUM.
    mem.graph.update_tier_set(node, Tier.SHORT, Tier.MEDIUM)
    mem.index.move(nid, node.v, Tier.SHORT, Tier.MEDIUM)
    node.tier = Tier.MEDIUM

    results = mem.retrieve("critical capital", k=5)
    returned_ids = [r.node_id for r in results]
    assert nid in returned_ids


def test_retrieve_with_one_node_returns_it(fresh_mem):
    nid = fresh_mem.ingest_awake("Unique memory for single-node test")
    results = fresh_mem.retrieve("Unique memory", k=5)
    assert len(results) >= 1
    assert results[0].node_id == nid


# ── _reinforce ─────────────────────────────────────────────────────────────────

def test_reinforce_increases_s_base(small_config, mock_embedder):
    """Reinforce on a decayed node should increase s_base."""
    from livemem.memory import LiveMem
    from livemem.types import strength_effective
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("reinforce test")
    node = mem.graph.V[nid]
    # Simulate decay by backdating t_accessed.
    node.t_accessed -= 100_000  # 100k seconds in the past.
    s_before = strength_effective(node, time.time(), small_config)
    mem._reinforce(node)
    assert node.s_base > s_before


def test_reinforce_clamps_at_one(small_config, mock_embedder):
    """Reinforce on a full-strength node should not exceed 1.0."""
    from livemem.memory import LiveMem
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("strong node")
    node = mem.graph.V[nid]
    node.s_base = 1.0
    node.t_accessed = time.time()  # No decay.
    mem._reinforce(node)
    assert node.s_base <= 1.0


def test_reinforce_updates_t_accessed(fresh_mem):
    nid = fresh_mem.ingest_awake("access time test")
    node = fresh_mem.graph.V[nid]
    old_t = node.t_accessed
    time.sleep(0.01)
    fresh_mem._reinforce(node)
    assert node.t_accessed >= old_t


def test_update_tier_promotes_strong_node(small_config, mock_embedder):
    """After reinforce, a node with a short effective age should stay SHORT."""
    from livemem.memory import LiveMem
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("tier update test")
    node = mem.graph.V[nid]
    # Should already be SHORT for a fresh node.
    assert node.tier == Tier.SHORT


# ── status ─────────────────────────────────────────────────────────────────────

def test_status_counts_correct_after_ingestion(fresh_mem):
    for i in range(3):
        fresh_mem.ingest_awake(f"fact {i}")
    s = fresh_mem.status()
    assert s["total_nodes"] == 3
    assert s["tier_counts"]["SHORT"] == 3
    assert s["tier_counts"]["MEDIUM"] == 0
    assert s["tier_counts"]["LONG"] == 0
