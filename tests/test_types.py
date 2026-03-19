"""
test_types.py — Tests for types.py: enums, Node, Edge, RetrievalResult,
                strength_effective, and tier_fn.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from livemem.types import (
    Edge,
    EdgeType,
    Importance,
    Node,
    RetrievalResult,
    Tier,
    strength_effective,
    tier_fn,
)
from tests.conftest import make_node


# ── Importance ordering ────────────────────────────────────────────────────────

def test_importance_ordering():
    assert Importance.CAPITAL > Importance.KEY > Importance.NORMAL > Importance.WEAK


def test_importance_values():
    assert Importance.WEAK == 0
    assert Importance.NORMAL == 1
    assert Importance.KEY == 2
    assert Importance.CAPITAL == 3


# ── strength_effective ─────────────────────────────────────────────────────────

def test_strength_effective_no_decay():
    """At t == t_accessed, s_eff should equal s_base."""
    node = make_node(s_base=0.7, seed=42)
    s = strength_effective(node, node.t_accessed)
    assert abs(s - 0.7) < 1e-6


def test_strength_effective_decays_over_time():
    """After some time, s_eff should be less than s_base."""
    node = make_node(s_base=1.0, seed=42)
    past = node.t_accessed
    future = past + 100_000  # 100k seconds later
    s = strength_effective(node, future)
    assert s < 1.0


def test_strength_effective_approaches_zero():
    """After very long time, strength should be negligible."""
    node = make_node(s_base=1.0, seed=1)
    huge_delta = 1e10  # ~317 years
    s = strength_effective(node, node.t_accessed + huge_delta)
    assert s < 1e-6


def test_strength_effective_negative_delta_clipped():
    """Negative elapsed time (t < t_accessed) should be treated as 0."""
    node = make_node(s_base=0.5, seed=2)
    # t before t_accessed: should return s_base unchanged.
    s = strength_effective(node, node.t_accessed - 1000)
    assert abs(s - 0.5) < 1e-6


# ── tier_fn ────────────────────────────────────────────────────────────────────

def test_tier_fn_fresh_node_is_short(small_config):
    """A brand new node should always be SHORT."""
    node = make_node(seed=3)
    assert tier_fn(node, time.time(), small_config) == Tier.SHORT


def test_tier_fn_old_node_is_long(small_config):
    """A node created far in the past with no access should be LONG."""
    node = make_node(s_base=0.001, t_offset=-10000.0, seed=4)
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result == Tier.LONG


def test_tier_fn_capital_floor(small_config):
    """A CAPITAL node should never reach LONG — at most MEDIUM."""
    node = make_node(
        importance=Importance.CAPITAL,
        s_base=0.001,
        t_offset=-10000.0,
        tier=Tier.LONG,
        seed=5,
    )
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result != Tier.LONG
    # Should be MEDIUM at most.
    assert result in (Tier.SHORT, Tier.MEDIUM)


def test_tier_fn_medium_tier(small_config):
    """Node between T1 and T2 effective age → MEDIUM."""
    node = make_node(s_base=0.001, t_offset=-5.0, seed=6)
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result == Tier.MEDIUM


def test_tier_fn_strength_slows_aging(small_config):
    """High-strength node stays in SHORT longer than weak node."""
    # Both created at the same "old" offset, but different strengths.
    strong_node = make_node(s_base=1.0, t_offset=-1.5, seed=7)
    weak_node = make_node(s_base=0.0, t_offset=-1.5, seed=7)
    # They have the same v but different s_base.
    weak_node.s_base = 0.0

    t_now = time.time()
    strong_tier = tier_fn(strong_node, t_now, small_config)
    weak_tier = tier_fn(weak_node, t_now, small_config)
    # Strong node should be in an equal or lower-numbered tier (≤ weak).
    assert strong_tier <= weak_tier


# ── Node dataclass ─────────────────────────────────────────────────────────────

def test_node_unit_norm_enforced():
    """Node __post_init__ should normalise a non-unit vector."""
    v_raw = np.array([3.0, 4.0] + [0.0] * 14, dtype=np.float32)
    node = Node(v=v_raw, summary="test")
    assert abs(np.linalg.norm(node.v) - 1.0) < 1e-5


def test_node_unique_ids():
    """Two nodes created without explicit id should have different UUIDs."""
    n1 = make_node(seed=10)
    n2 = make_node(seed=11)
    assert n1.id != n2.id


def test_node_defaults():
    """Node defaults: tier=SHORT, diffused=False, consolidated=False, sources=[]."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node = Node(v=v, summary="defaults")
    assert node.tier == Tier.SHORT
    assert node.diffused is False
    assert node.consolidated is False
    assert node.sources == []


def test_node_fields_stored_correctly():
    """Node should store summary, ref_uri, ref_type correctly."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node = Node(
        v=v,
        summary="my summary",
        ref_uri="/path/to/file.txt",
        ref_type="text",
        importance=Importance.KEY,
    )
    assert node.summary == "my summary"
    assert node.ref_uri == "/path/to/file.txt"
    assert node.ref_type == "text"
    assert node.importance == Importance.KEY


def test_node_zero_vector_raises():
    """A near-zero vector should raise ValueError."""
    v = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError, match="near-zero norm"):
        Node(v=v, summary="bad")


def test_node_hash_by_id():
    """Two Node objects with same id should hash equally."""
    import uuid as _uuid
    shared_id = str(_uuid.uuid4())
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    n1 = Node(v=v.copy(), summary="a", id=shared_id)
    n2 = Node(v=v.copy(), summary="b", id=shared_id)
    assert hash(n1) == hash(n2)
    assert n1 == n2


# ── Edge dataclass ─────────────────────────────────────────────────────────────

def test_edge_valid_construction():
    """A valid edge should be created without error."""
    e = Edge(from_id="a", to_id="b", cos_sim=0.85, delta_t=100.0)
    assert e.from_id == "a"
    assert e.to_id == "b"
    assert e.cos_sim == 0.85
    assert e.delta_t == 100.0
    assert e.edge_type == EdgeType.DIRECT


def test_edge_negative_delta_t_raises():
    """Edge with delta_t < 0 should raise ValueError."""
    with pytest.raises(ValueError, match="delta_t must be ≥ 0"):
        Edge(from_id="a", to_id="b", cos_sim=0.5, delta_t=-1.0)


def test_edge_default_type():
    """Edge default edge_type should be DIRECT."""
    e = Edge(from_id="x", to_id="y", cos_sim=0.5, delta_t=0.0)
    assert e.edge_type == EdgeType.DIRECT


def test_edge_zero_delta_t_is_valid():
    """delta_t == 0 (simultaneous creation) should be valid."""
    e = Edge(from_id="a", to_id="b", cos_sim=0.5, delta_t=0.0)
    assert e.delta_t == 0.0


# ── RetrievalResult ────────────────────────────────────────────────────────────

def test_retrieval_result_fields():
    """RetrievalResult should store all fields correctly."""
    r = RetrievalResult(
        node_id="abc",
        score=0.95,
        summary="test summary",
        ref_uri="/some/path.mp3",
        ref_type="audio",
        tier=Tier.MEDIUM,
        importance=Importance.KEY,
        cos_direct=0.88,
    )
    assert r.node_id == "abc"
    assert r.score == 0.95
    assert r.ref_type == "audio"
    assert r.tier == Tier.MEDIUM
    assert r.importance == Importance.KEY
    assert r.cos_direct == 0.88
