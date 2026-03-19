"""
test_graph.py — Tests for graph.py: Graph operations.
"""
from __future__ import annotations

import time

import pytest

from livemem.graph import Graph
from livemem.types import Edge, EdgeType, Tier
from tests.conftest import make_node


@pytest.fixture
def graph() -> Graph:
    return Graph()


# ── add_node ───────────────────────────────────────────────────────────────────

def test_add_node_registers_in_V(graph):
    n = make_node(seed=1)
    graph.add_node(n)
    assert n.id in graph.V


def test_add_node_registers_in_tier_set(graph):
    n = make_node(tier=Tier.SHORT, seed=2)
    graph.add_node(n)
    assert graph.tier_size(Tier.SHORT) == 1


def test_add_node_increments_tier_size(graph):
    n1 = make_node(tier=Tier.SHORT, seed=3)
    n2 = make_node(tier=Tier.SHORT, seed=4)
    graph.add_node(n1)
    graph.add_node(n2)
    assert graph.tier_size(Tier.SHORT) == 2


# ── remove_node ────────────────────────────────────────────────────────────────

def test_remove_node_removes_from_V(graph):
    n = make_node(seed=5)
    graph.add_node(n)
    graph.remove_node(n.id)
    assert n.id not in graph.V


def test_remove_node_cascades_outgoing_edges(graph):
    n1 = make_node(seed=6)
    n2 = make_node(seed=7)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.9, delta_t=1.0)
    graph.add_edge(e)
    graph.remove_node(n1.id)
    # n1's outgoing edges should be gone.
    assert n1.id not in graph.E or graph.E.get(n1.id) == []
    # n2's incoming edges should no longer reference n1.
    incoming = [ed for ed in graph.E_r.get(n2.id, []) if ed.from_id == n1.id]
    assert incoming == []


def test_remove_node_cascades_incoming_edges(graph):
    n1 = make_node(seed=8)
    n2 = make_node(seed=9)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.8, delta_t=1.0)
    graph.add_edge(e)
    graph.remove_node(n2.id)
    # n1's outgoing edges should no longer reference n2.
    outgoing = [ed for ed in graph.E.get(n1.id, []) if ed.to_id == n2.id]
    assert outgoing == []


def test_remove_node_nonexistent_is_silent(graph):
    """Removing a node that doesn't exist should not raise."""
    graph.remove_node("nonexistent-uuid")


def test_remove_node_decrements_tier_size(graph):
    n = make_node(tier=Tier.SHORT, seed=10)
    graph.add_node(n)
    assert graph.tier_size(Tier.SHORT) == 1
    graph.remove_node(n.id)
    assert graph.tier_size(Tier.SHORT) == 0


# ── add_edge ───────────────────────────────────────────────────────────────────

def test_add_edge_updates_E(graph):
    n1 = make_node(seed=11)
    n2 = make_node(seed=12)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.7, delta_t=0.0)
    graph.add_edge(e)
    assert e in graph.E[n1.id]


def test_add_edge_updates_E_r(graph):
    n1 = make_node(seed=13)
    n2 = make_node(seed=14)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.7, delta_t=0.0)
    graph.add_edge(e)
    assert e in graph.E_r[n2.id]


# ── add_edge_if_new ────────────────────────────────────────────────────────────

def test_add_edge_if_new_prevents_duplicates(graph):
    n1 = make_node(seed=15)
    n2 = make_node(seed=16)
    graph.add_node(n1)
    graph.add_node(n2)
    e1 = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.8, delta_t=0.0)
    e2 = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.9, delta_t=0.0)
    graph.add_edge_if_new(e1)
    graph.add_edge_if_new(e2)
    # Only one edge from n1→n2 should exist.
    n1_to_n2 = [ed for ed in graph.E[n1.id] if ed.to_id == n2.id]
    assert len(n1_to_n2) == 1


def test_add_edge_if_new_returns_true_on_new(graph):
    n1 = make_node(seed=17)
    n2 = make_node(seed=18)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.75, delta_t=0.0)
    result = graph.add_edge_if_new(e)
    assert result is True


def test_add_edge_if_new_returns_false_on_duplicate(graph):
    n1 = make_node(seed=19)
    n2 = make_node(seed=20)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.75, delta_t=0.0)
    graph.add_edge_if_new(e)
    result = graph.add_edge_if_new(e)
    assert result is False


# ── nodes_in_tier ──────────────────────────────────────────────────────────────

def test_nodes_in_tier_returns_correct_nodes(graph):
    n1 = make_node(tier=Tier.SHORT, t_offset=-10.0, seed=21)
    n2 = make_node(tier=Tier.MEDIUM, seed=22)
    graph.add_node(n1)
    graph.add_node(n2)
    short_nodes = graph.nodes_in_tier(Tier.SHORT)
    assert n1 in short_nodes
    assert n2 not in short_nodes


def test_nodes_in_tier_sorted_oldest_first(graph):
    n_old = make_node(tier=Tier.SHORT, t_offset=-10.0, seed=23)
    n_new = make_node(tier=Tier.SHORT, t_offset=0.0, seed=24)
    graph.add_node(n_old)
    graph.add_node(n_new)
    nodes = graph.nodes_in_tier(Tier.SHORT)
    assert nodes[0].id == n_old.id


# ── total_nodes / total_edges ─────────────────────────────────────────────────

def test_total_nodes_and_edges(graph):
    n1 = make_node(seed=25)
    n2 = make_node(seed=26)
    graph.add_node(n1)
    graph.add_node(n2)
    e = Edge(from_id=n1.id, to_id=n2.id, cos_sim=0.5, delta_t=0.0)
    graph.add_edge(e)
    assert graph.total_nodes() == 2
    assert graph.total_edges() == 1


# ── update_tier_set ────────────────────────────────────────────────────────────

def test_update_tier_set_moves_between_tiers(graph):
    n = make_node(tier=Tier.SHORT, seed=27)
    graph.add_node(n)
    assert graph.tier_size(Tier.SHORT) == 1
    assert graph.tier_size(Tier.MEDIUM) == 0
    graph.update_tier_set(n, Tier.SHORT, Tier.MEDIUM)
    # After move, SHORT should be 0 and MEDIUM should be 1.
    assert graph.tier_size(Tier.SHORT) == 0
    assert graph.tier_size(Tier.MEDIUM) == 1


# ── iter and len ──────────────────────────────────────────────────────────────

def test_iter_and_len(graph):
    n1 = make_node(seed=28)
    n2 = make_node(seed=29)
    graph.add_node(n1)
    graph.add_node(n2)
    assert len(graph) == 2
    ids = {n.id for n in graph}
    assert n1.id in ids
    assert n2.id in ids
