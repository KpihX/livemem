"""
test_types.py — Tests for types.py: Node, Edge, RetrievalResult,
                strength_effective, urgency_effective, and tier_fn.

CONTINUITY PRINCIPLE:
    importance and urgency are continuous floats in [0, 1].
    No Importance enum — no discrete enum tests.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from livemem.config import LiveConfig
from livemem.types import (
    Edge,
    EdgeType,
    Node,
    RetrievalResult,
    Tier,
    strength_effective,
    tier_fn,
    urgency_effective,
)
from tests.conftest import make_node


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


# ── urgency_effective ──────────────────────────────────────────────────────────

def test_urgency_effective_no_decay_at_creation():
    """At t == node.t (creation time), u_eff should equal node.urgency."""
    node = make_node(urgency=0.8, seed=10)
    u = urgency_effective(node, node.t)
    assert abs(u - 0.8) < 1e-6


def test_urgency_effective_decays_over_time():
    """After some time, u_eff should be less than node.urgency."""
    node = make_node(urgency=1.0, seed=11)
    u = urgency_effective(node, node.t + 100_000)
    assert u < 1.0


def test_urgency_effective_decays_from_creation_not_access():
    """Urgency decays from node.t (creation), not t_accessed."""
    node = make_node(urgency=1.0, s_base=0.5, seed=12)
    # Simulate t_accessed being newer than node.t (reinforced node).
    node.t_accessed = node.t + 50_000  # accessed recently
    # Urgency should still reflect elapsed time from creation.
    u = urgency_effective(node, node.t + 100_000)
    # Should be lower than initial urgency (100k elapsed from creation).
    assert u < 1.0


def test_urgency_effective_zero_urgency():
    """A node with urgency=0.0 should have u_eff=0.0 at any time."""
    node = make_node(urgency=0.0, seed=13)
    u = urgency_effective(node, node.t + 1_000_000)
    assert u == 0.0


def test_urgency_effective_faster_than_strength():
    """Urgency should decay faster than strength (urgency_lambda > decay_lambda)."""
    from livemem.config import LiveConfig
    cfg = LiveConfig(urgency_lambda=5e-5, decay_lambda=1e-5)
    node = make_node(urgency=1.0, s_base=1.0, seed=14)
    elapsed = 10_000  # 2.7 hours
    t = node.t + elapsed

    u_eff = urgency_effective(node, t, cfg)
    s_eff = strength_effective(node, t, cfg)  # uses t_accessed = node.t
    # Urgency should have decayed more than strength.
    assert u_eff < s_eff


def test_urgency_effective_negative_delta_clipped():
    """t < node.t should be treated as elapsed=0."""
    node = make_node(urgency=0.6, seed=15)
    u = urgency_effective(node, node.t - 1000)
    assert abs(u - 0.6) < 1e-6


# ── tier_fn — urgency pin ──────────────────────────────────────────────────────

def test_tier_fn_fresh_node_is_short(small_config):
    """A brand new node should always be SHORT."""
    node = make_node(seed=3)
    assert tier_fn(node, time.time(), small_config) == Tier.SHORT


def test_tier_fn_old_node_is_long(small_config):
    """A node created far in the past with no access should be LONG."""
    node = make_node(s_base=0.001, t_offset=-10000.0, seed=4, importance=0.1)
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result == Tier.LONG


def test_tier_fn_urgency_pins_to_short(small_config):
    """A node with high urgency should be pinned to SHORT regardless of age."""
    # Very old, very weak, but highly urgent.
    node = make_node(
        urgency=1.0,
        s_base=0.001,
        t_offset=-10000.0,
        importance=0.1,
        tier=Tier.LONG,
        seed=5,
    )
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result == Tier.SHORT


def test_tier_fn_urgency_pin_requires_threshold(small_config):
    """Urgency below theta_urgent should NOT pin to SHORT."""
    # urgency_effective at creation = node.urgency (no elapsed time).
    # We want u_eff < theta_urgent (0.7) so we use urgency=0.3.
    node = make_node(
        urgency=0.3,
        s_base=0.001,
        t_offset=-10000.0,
        importance=0.1,
        tier=Tier.LONG,
        seed=6,
    )
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    # With low urgency, the node should NOT be pinned to SHORT.
    assert result != Tier.SHORT


def test_tier_fn_importance_floor(small_config):
    """A high-importance node should never reach LONG — at most MEDIUM."""
    node = make_node(
        importance=0.9,  # >= importance_medium_floor (0.6)
        urgency=0.0,
        s_base=0.001,
        t_offset=-10000.0,
        tier=Tier.LONG,
        seed=7,
    )
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result != Tier.LONG
    assert result in (Tier.SHORT, Tier.MEDIUM)


def test_tier_fn_low_importance_can_be_long(small_config):
    """A low-importance, low-urgency old node should reach LONG."""
    node = make_node(
        importance=0.1,  # < importance_medium_floor (0.6)
        urgency=0.0,
        s_base=0.001,
        t_offset=-10000.0,
        seed=8,
    )
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result == Tier.LONG


def test_tier_fn_medium_tier(small_config):
    """Node between T1 and T2 effective age → MEDIUM."""
    node = make_node(s_base=0.001, t_offset=-5.0, urgency=0.0, importance=0.1, seed=9)
    t_now = time.time()
    result = tier_fn(node, t_now, small_config)
    assert result == Tier.MEDIUM


def test_tier_fn_strength_slows_aging(small_config):
    """High-strength node stays in SHORT longer than weak node."""
    strong_node = make_node(s_base=1.0, t_offset=-1.5, urgency=0.0, seed=17)
    weak_node = make_node(s_base=0.001, t_offset=-1.5, urgency=0.0, seed=17)
    # Same time offset but different strength.
    t_now = time.time()
    strong_tier = tier_fn(strong_node, t_now, small_config)
    weak_tier = tier_fn(weak_node, t_now, small_config)
    # Strong node should be in an equal or lower-numbered tier (≤ weak).
    assert strong_tier <= weak_tier


def test_tier_fn_importance_slows_aging(small_config):
    """High-importance node ages more slowly than low-importance node."""
    high_imp = make_node(importance=1.0, s_base=0.001, t_offset=-2.0, urgency=0.0, seed=18)
    low_imp = make_node(importance=0.0, s_base=0.001, t_offset=-2.0, urgency=0.0, seed=18)
    t_now = time.time()
    tier_high = tier_fn(high_imp, t_now, small_config)
    tier_low = tier_fn(low_imp, t_now, small_config)
    # High importance ages more slowly → tier ≤ low importance tier.
    assert tier_high <= tier_low


# ── Node dataclass — continuity ───────────────────────────────────────────────

def test_node_importance_is_float():
    """Node importance should be a float, not an enum."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node = Node(v=v, summary="test", importance=0.75)
    assert isinstance(node.importance, float)
    assert abs(node.importance - 0.75) < 1e-6


def test_node_urgency_is_float():
    """Node urgency should be a float in [0, 1]."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node = Node(v=v, summary="test", urgency=0.9)
    assert isinstance(node.urgency, float)
    assert abs(node.urgency - 0.9) < 1e-6


def test_node_importance_clamped():
    """Out-of-range importance should be clamped to [0, 1]."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node_high = Node(v=v.copy(), summary="high", importance=2.5)
    node_low = Node(v=v.copy(), summary="low", importance=-0.3)
    assert 0.0 <= node_high.importance <= 1.0
    assert 0.0 <= node_low.importance <= 1.0


def test_node_urgency_clamped():
    """Out-of-range urgency should be clamped to [0, 1]."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node = Node(v=v, summary="test", urgency=5.0)
    assert 0.0 <= node.urgency <= 1.0


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
    assert node.urgency == 0.0
    assert node.importance == 0.5


def test_node_fields_stored_correctly():
    """Node should store summary, ref_uri, ref_type, importance (float) correctly."""
    v = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    node = Node(
        v=v,
        summary="my summary",
        ref_uri="/path/to/file.txt",
        ref_type="text",
        importance=0.7,
        urgency=0.3,
    )
    assert node.summary == "my summary"
    assert node.ref_uri == "/path/to/file.txt"
    assert node.ref_type == "text"
    assert abs(node.importance - 0.7) < 1e-6
    assert abs(node.urgency - 0.3) < 1e-6


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
    """RetrievalResult should store all fields correctly, with float importance/urgency."""
    r = RetrievalResult(
        node_id="abc",
        score=0.95,
        summary="test summary",
        ref_uri="/some/path.mp3",
        ref_type="audio",
        tier=Tier.MEDIUM,
        importance=0.7,
        urgency=0.3,
        cos_direct=0.88,
    )
    assert r.node_id == "abc"
    assert r.score == 0.95
    assert r.ref_type == "audio"
    assert r.tier == Tier.MEDIUM
    assert abs(r.importance - 0.7) < 1e-6
    assert abs(r.urgency - 0.3) < 1e-6
    assert r.cos_direct == 0.88
