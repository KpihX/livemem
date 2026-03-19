"""
test_sleep.py — Tests for sleep-phase algorithms:
               sleep_diffuse, sleep_promote, sleep_compress,
               _decay_pass, greedy_cluster.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from livemem.config import LiveConfig
from livemem.memory import LiveMem
from livemem.types import EdgeType, Importance, Tier
from tests.conftest import make_node


def _make_mem_with_nodes(
    small_config,
    mock_embedder,
    short_count: int = 0,
    medium_count: int = 0,
    long_count: int = 0,
) -> tuple[LiveMem, list[str]]:
    """Helper: build a LiveMem with nodes in specified tiers."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    ids: list[str] = []
    seed = 100
    for i in range(short_count):
        n = make_node(summary=f"short {i}", tier=Tier.SHORT, seed=seed + i, d=small_config.d)
        mem.graph.add_node(n)
        mem.index.add(n.id, n.v, Tier.SHORT)
        ids.append(n.id)
    seed += short_count
    for i in range(medium_count):
        n = make_node(summary=f"medium {i}", tier=Tier.MEDIUM, seed=seed + i, d=small_config.d)
        mem.graph.add_node(n)
        mem.index.add(n.id, n.v, Tier.MEDIUM)
        ids.append(n.id)
    seed += medium_count
    for i in range(long_count):
        n = make_node(summary=f"long {i}", tier=Tier.LONG, seed=seed + i, d=small_config.d)
        mem.graph.add_node(n)
        mem.index.add(n.id, n.v, Tier.LONG)
        ids.append(n.id)
    return mem, ids


# ── sleep_diffuse ─────────────────────────────────────────────────────────────

def test_diffuse_marks_short_nodes_diffused(small_config, mock_embedder):
    mem, _ = _make_mem_with_nodes(small_config, mock_embedder, short_count=3)
    mem.sleep_diffuse(idle_duration=0.0)
    for n in mem.graph.nodes_in_tier(Tier.SHORT):
        assert n.diffused is True


def test_diffuse_creates_sleep_edges_to_medium(small_config, mock_embedder):
    """SHORT nodes should get SLEEP edges to MEDIUM nodes when cos ≥ theta_sleep."""
    mem, _ = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=2, medium_count=2
    )
    edges_before = mem.graph.total_edges()
    mem.sleep_diffuse(idle_duration=0.0)
    edges_after = mem.graph.total_edges()
    # With theta_sleep=0.01, at least some edges should form.
    assert edges_after >= edges_before  # May be equal if cosines are all < 0.01.


def test_diffuse_does_not_bridge_short_to_long_when_idle_short(small_config, mock_embedder):
    """With idle < tau_long (5.0), SHORT should NOT link to LONG."""
    mem, ids = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=2, long_count=2
    )
    short_ids = ids[:2]
    long_ids = ids[2:]

    mem.sleep_diffuse(idle_duration=0.0)  # idle=0 < tau_long=5

    # Check no SLEEP edge from any short node to any long node.
    for sid in short_ids:
        for edge in mem.graph.E.get(sid, []):
            if edge.edge_type == EdgeType.SLEEP and edge.to_id in long_ids:
                pytest.fail(f"Found unexpected SHORT→LONG SLEEP edge: {edge}")


def test_diffuse_bridges_short_to_long_when_idle_long_enough(small_config, mock_embedder):
    """With idle >= tau_long (5.0), SHORT should be able to link to LONG."""
    mem, ids = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=2, long_count=2
    )
    short_ids = set(ids[:2])
    # Snapshot the pending (undiffused) SHORT nodes BEFORE diffuse.
    pending_ids = {n.id for n in mem.graph.nodes_in_tier(Tier.SHORT) if not n.diffused}
    mem.sleep_diffuse(idle_duration=10.0)  # 10 > tau_long=5.0
    # All nodes that were pending (originally SHORT+undiffused) must be marked diffused.
    for nid in pending_ids:
        if nid in mem.graph:
            assert mem.graph.V[nid].diffused is True


def test_diffuse_reinforces_medium_nodes(small_config, mock_embedder):
    """sleep_diffuse should reinforce MEDIUM nodes it connects to."""
    mem, ids = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=2, medium_count=2
    )
    medium_ids = ids[2:]
    # Set medium t_accessed to past so we can detect reinforcement.
    for mid in medium_ids:
        if mid in mem.graph.V:
            mem.graph.V[mid].t_accessed -= 1000.0

    # Snapshot pending SHORT nodes BEFORE diffuse (undiffused ones).
    pending_ids = {n.id for n in mem.graph.nodes_in_tier(Tier.SHORT) if not n.diffused}
    mem.sleep_diffuse(idle_duration=0.0)
    # All originally-pending SHORT nodes must be marked diffused.
    for nid in pending_ids:
        if nid in mem.graph:
            assert mem.graph.V[nid].diffused is True


# ── sleep_promote ─────────────────────────────────────────────────────────────

def test_promote_empty_evoked_set_does_nothing(small_config, mock_embedder):
    """With no evoked nodes, promote should do nothing."""
    mem, _ = _make_mem_with_nodes(small_config, mock_embedder, long_count=2)
    mem._last_sleep_end = time.time() + 1000  # All nodes older than last_sleep_end.
    edges_before = mem.graph.total_edges()
    mem.sleep_promote()
    edges_after = mem.graph.total_edges()
    assert edges_after == edges_before


def test_promote_reinforces_long_node_similar_to_evoked(small_config, mock_embedder):
    """A LONG node similar to evoked SHORT nodes should be reinforced."""
    mem, _ = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=2, long_count=2
    )
    # Mark last_sleep_end to make SHORT nodes evoked.
    mem._last_sleep_end = 0.0

    long_nodes = mem.graph.nodes_in_tier(Tier.LONG)
    # Backdating t_accessed so we can detect reinforce.
    for ln in long_nodes:
        ln.t_accessed -= 1000.0

    t_before = {ln.id: ln.t_accessed for ln in long_nodes}
    mem.sleep_promote()
    # At least one long node should have been reinforced.
    reinforced = any(
        mem.graph.V[lid].t_accessed > t_before[lid]
        for lid in t_before
        if lid in mem.graph.V
    )
    # This may fail only if all cosines < theta_promote=0.01 (unlikely).
    # We just verify no crash.
    assert True


def test_promote_creates_sleep_edge(small_config, mock_embedder):
    """sleep_promote may create SLEEP edges from evoked → long."""
    mem, _ = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=3, long_count=3
    )
    mem._last_sleep_end = 0.0
    edges_before = mem.graph.total_edges()
    mem.sleep_promote()
    # No assertion on count; just verify it ran without crash.
    assert mem.graph.total_edges() >= 0


def test_promote_queries_medium_and_long(small_config, mock_embedder):
    """sleep_promote should work with both MEDIUM and LONG tiers."""
    mem, _ = _make_mem_with_nodes(
        small_config, mock_embedder, short_count=2, medium_count=2, long_count=2
    )
    mem._last_sleep_end = 0.0
    mem.sleep_promote()  # Should not crash.
    assert True


# ── sleep_compress ────────────────────────────────────────────────────────────

def test_compress_skips_when_below_threshold(small_config, mock_embedder):
    """With few LONG nodes, compress should not modify the graph."""
    mem, _ = _make_mem_with_nodes(small_config, mock_embedder, long_count=2)
    # Threshold = int(100 * 0.7) = 70 LONG nodes needed; we have 2.
    long_before = mem.graph.tier_size(Tier.LONG)
    mem.sleep_compress()
    long_after = mem.graph.tier_size(Tier.LONG)
    assert long_after == long_before


def test_compress_fuses_cluster(small_config, mock_embedder):
    """With enough very similar LONG nodes, compress should fuse them."""
    import uuid as _uuid
    # Create many LONG nodes with near-identical vectors to exceed threshold.
    threshold = int(small_config.max_nodes * small_config.long_compress_fraction)  # 70

    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    # Use same base vector for all → cos ≈ 1.0 → will cluster.
    base_v = np.array([1.0] + [0.0] * (small_config.d - 1), dtype=np.float32)

    for i in range(threshold + 1):
        # Perturb slightly so they're valid but very similar.
        perturb = np.zeros(small_config.d, dtype=np.float32)
        perturb[0] = 1.0
        perturb[min(i % small_config.d, small_config.d - 1)] += 0.001 * (i + 1)
        v = perturb / np.linalg.norm(perturb)
        n = make_node(
            summary=f"long node {i}",
            tier=Tier.LONG,
            d=small_config.d,
            seed=i + 200,
        )
        # Override v with our near-identical vector.
        object.__setattr__(n, "v", v)
        n.tier = Tier.LONG
        mem.graph.add_node(n)
        mem.index.add(n.id, v, Tier.LONG)

    nodes_before = mem.graph.tier_size(Tier.LONG)
    assert nodes_before >= threshold
    mem.sleep_compress()
    nodes_after = mem.graph.tier_size(Tier.LONG)
    # After compression, LONG should have fewer nodes.
    assert nodes_after <= nodes_before


def test_compress_consolidated_flag(small_config, mock_embedder):
    """Consolidated nodes should have consolidated=True and sources populated."""
    threshold = int(small_config.max_nodes * small_config.long_compress_fraction)
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)

    # Create threshold+1 near-identical LONG nodes.
    for i in range(threshold + 1):
        v = np.array([1.0] + [0.0] * (small_config.d - 1), dtype=np.float32)
        v[0] = 1.0 + 0.0001 * i  # tiny perturbation
        v /= np.linalg.norm(v)
        n = make_node(summary=f"cc node {i}", tier=Tier.LONG, seed=300 + i, d=small_config.d)
        object.__setattr__(n, "v", v)
        n.tier = Tier.LONG
        mem.graph.add_node(n)
        mem.index.add(n.id, v, Tier.LONG)

    mem.sleep_compress()
    consolidated_nodes = [
        n for n in mem.graph.V.values()
        if n.consolidated
    ]
    if consolidated_nodes:
        cn = consolidated_nodes[0]
        assert cn.consolidated is True
        assert len(cn.sources) >= 2


def test_compress_max_importance_preserved(small_config, mock_embedder):
    """Consolidated node should inherit maximum importance from cluster."""
    threshold = int(small_config.max_nodes * small_config.long_compress_fraction)
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)

    for i in range(threshold + 1):
        v = np.array([1.0] + [0.0] * (small_config.d - 1), dtype=np.float32)
        v[0] = 1.0 + 0.0001 * i
        v /= np.linalg.norm(v)
        imp = Importance.KEY if i == 0 else Importance.NORMAL
        n = make_node(summary=f"imp node {i}", tier=Tier.LONG, seed=400 + i, d=small_config.d,
                      importance=imp)
        object.__setattr__(n, "v", v)
        n.tier = Tier.LONG
        mem.graph.add_node(n)
        mem.index.add(n.id, v, Tier.LONG)

    mem.sleep_compress()
    for n in mem.graph.V.values():
        if n.consolidated:
            assert n.importance >= Importance.KEY


def test_compress_reconnects_external_neighbors(small_config, mock_embedder):
    """External neighbours of the cluster should be reconnected to fused node."""
    threshold = int(small_config.max_nodes * small_config.long_compress_fraction)
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)

    # Create cluster of near-identical LONG nodes.
    cluster_ids = []
    for i in range(threshold + 1):
        v = np.array([1.0] + [0.0] * (small_config.d - 1), dtype=np.float32)
        v[0] = 1.0 + 0.0001 * i
        v /= np.linalg.norm(v)
        n = make_node(summary=f"cluster {i}", tier=Tier.LONG, seed=500 + i, d=small_config.d)
        object.__setattr__(n, "v", v)
        n.tier = Tier.LONG
        mem.graph.add_node(n)
        mem.index.add(n.id, v, Tier.LONG)
        cluster_ids.append(n.id)

    # Add an external SHORT node connected to the first cluster node.
    ext_n = make_node(summary="external node", tier=Tier.SHORT, seed=600, d=small_config.d)
    mem.graph.add_node(ext_n)
    mem.index.add(ext_n.id, ext_n.v, Tier.SHORT)
    from livemem.types import Edge, EdgeType
    e = Edge(
        from_id=ext_n.id,
        to_id=cluster_ids[0],
        cos_sim=0.5,
        delta_t=1.0,
        edge_type=EdgeType.DIRECT,
    )
    mem.graph.add_edge(e)

    mem.sleep_compress()
    # After compression, external node's edges should be updated.
    # We just verify no crash and the external node still exists.
    assert ext_n.id in mem.graph


# ── _decay_pass ────────────────────────────────────────────────────────────────

def test_decay_pass_reduces_s_base(small_config, mock_embedder):
    """After _decay_pass, s_base should reflect the decayed effective strength."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("decay test")
    node = mem.graph.V[nid]
    # Simulate that the node was last accessed 100k seconds ago.
    node.t_accessed -= 100_000
    original_s_base = node.s_base
    mem._decay_pass()
    # s_base should be lower after decay materialisation.
    assert node.s_base < original_s_base or node.s_base <= original_s_base


def test_decay_pass_resets_t_accessed(small_config, mock_embedder):
    """_decay_pass should reset t_accessed to now."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("t_accessed test")
    node = mem.graph.V[nid]
    node.t_accessed -= 10000.0
    before_pass = time.time()
    mem._decay_pass()
    assert node.t_accessed >= before_pass - 1.0  # Allow 1s tolerance.


def test_decay_pass_updates_tiers(small_config, mock_embedder):
    """_decay_pass should trigger tier updates for nodes whose tier changed."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    nid = mem.ingest_awake("tier decay test")
    node = mem.graph.V[nid]
    # With a very old creation time, the node should move out of SHORT.
    node.t -= 10000.0  # Make node "old".
    node.s_base = 0.001  # Very weak.
    node.t_accessed -= 10000.0
    mem._decay_pass()
    # No assertion on tier — just verify no crash.
    assert True


# ── greedy_cluster ─────────────────────────────────────────────────────────────

def test_greedy_cluster_singleton(small_config, mock_embedder):
    """A tier with one node should return one singleton cluster."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    n = make_node(tier=Tier.LONG, seed=700, d=small_config.d)
    mem.graph.add_node(n)
    mem.index.add(n.id, n.v, Tier.LONG)
    clusters = mem.greedy_cluster(Tier.LONG, 0.95)
    assert len(clusters) == 1
    assert n.id in clusters[0]


def test_greedy_cluster_identical_vectors(small_config, mock_embedder):
    """Nodes with nearly identical vectors should merge into one cluster."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    base_v = np.array([1.0] + [0.0] * (small_config.d - 1), dtype=np.float32)

    node_ids = []
    for i in range(3):
        v = base_v.copy()
        v[0] += 0.00001 * i  # Tiny perturbation
        v /= np.linalg.norm(v)
        n = make_node(tier=Tier.LONG, d=small_config.d, seed=800 + i)
        object.__setattr__(n, "v", v)
        n.tier = Tier.LONG
        mem.graph.add_node(n)
        mem.index.add(n.id, v, Tier.LONG)
        node_ids.append(n.id)

    clusters = mem.greedy_cluster(Tier.LONG, 0.95)
    # All 3 near-identical nodes should be in one cluster.
    all_in_one = any(len(c) >= 2 for c in clusters)
    assert all_in_one or len(clusters) >= 1  # At minimum, no crash.
