"""
graph.py — In-memory directed graph of memory nodes and edges.

WHY a custom graph class (not networkx):
    NetworkX is general-purpose and carries significant overhead per node
    (~1 KB Python objects). LiveMem needs:
      1. O(1) node lookup by UUID.
      2. Sorted iteration by creation time (for oldest-first processing).
      3. Fast cascading deletes (remove node + all incident edges).
      4. Tier-bucketed views without a full scan.
    SortedList from sortedcontainers gives O(log n) insert/delete/iter
    while dict gives O(1) access — better than anything networkx offers
    at this scale.
"""
from __future__ import annotations

from typing import Iterator

from sortedcontainers import SortedList

from livemem.types import Edge, EdgeType, Node, Tier


class Graph:
    """Directed graph of Node objects with tier-bucketed sorted views.

    Internal representation
    -----------------------
    V       : dict[uuid → Node]            — primary node store.
    E       : dict[from_id → list[Edge]]   — outgoing edge lists.
    E_r     : dict[to_id   → list[Edge]]   — incoming edge lists.
    _tier_sets : dict[Tier → SortedList]   — nodes sorted by (t, id)
                 for deterministic oldest-first iteration per tier.

    WHY store both E and E_r:
        Cascading deletes (remove_node) need to find all edges incident
        to a node in O(edges) without scanning the full edge set.
        E_r is the reverse index enabling this.
    """

    def __init__(self) -> None:
        self.V: dict[str, Node] = {}
        self.E: dict[str, list[Edge]] = {}
        self.E_r: dict[str, list[Edge]] = {}
        # SortedList key = (creation_time, uuid) for deterministic order.
        self._tier_sets: dict[Tier, SortedList] = {
            Tier.SHORT: SortedList(key=lambda nid: (self.V[nid].t, nid)),
            Tier.MEDIUM: SortedList(key=lambda nid: (self.V[nid].t, nid)),
            Tier.LONG: SortedList(key=lambda nid: (self.V[nid].t, nid)),
        }

    # ── Node operations ────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """Register a node in the graph.

        WHY: also initialises empty edge buckets and inserts into the
        correct tier set — callers should not need to know about these
        internal structures.
        """
        self.V[node.id] = node
        self.E.setdefault(node.id, [])
        self.E_r.setdefault(node.id, [])
        self._tier_sets[node.tier].add(node.id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all incident edges from the graph.

        WHY cascading delete:
            Orphaned edges pointing to a deleted node would cause KeyError
            during graph traversal. Removing all incident edges atomically
            here keeps the graph consistent without needing integrity checks
            throughout the codebase.

        Silently ignores unknown node_id.
        """
        if node_id not in self.V:
            return

        node = self.V[node_id]

        # Collect all edge lists that reference this node BEFORE any mutation.
        # Outgoing edges: stored in E[node_id].
        outgoing = list(self.E.get(node_id, []))
        # Incoming edges: stored in E_r[node_id].
        incoming = list(self.E_r.get(node_id, []))

        # Remove from reverse-index entries of outgoing edges.
        for edge in outgoing:
            if edge.to_id in self.E_r:
                try:
                    self.E_r[edge.to_id].remove(edge)
                except ValueError:
                    pass  # already removed or not present

        # Remove from forward-index entries of incoming edges.
        for edge in incoming:
            if edge.from_id in self.E:
                try:
                    self.E[edge.from_id].remove(edge)
                except ValueError:
                    pass

        # Remove edge buckets for this node.
        self.E.pop(node_id, None)
        self.E_r.pop(node_id, None)

        # Remove from tier set.
        try:
            self._tier_sets[node.tier].remove(node_id)
        except ValueError:
            pass  # not present (can happen if tier was updated without sync)

        # Remove from primary store.
        del self.V[node_id]

    def update_tier_set(
        self, node: Node, old_tier: Tier, new_tier: Tier
    ) -> None:
        """Move a node's entry between tier SortedLists.

        WHY: must be called whenever node.tier changes so that
        nodes_in_tier() and tier_size() reflect the new state.
        The caller is responsible for updating node.tier AFTER this call.
        """
        try:
            self._tier_sets[old_tier].remove(node.id)
        except ValueError:
            pass  # Tolerate if not found (e.g., fresh node pre-insert).
        self._tier_sets[new_tier].add(node.id)

    # ── Edge operations ────────────────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        """Append an edge to both E and E_r.

        WHY no duplicate check here:
            add_edge_if_new handles deduplication. add_edge is the raw
            primitive used internally when we know the edge is new.
        """
        self.E.setdefault(edge.from_id, []).append(edge)
        self.E_r.setdefault(edge.to_id, []).append(edge)

    def add_edge_if_new(self, edge: Edge) -> bool:
        """Add an edge only if the (from_id, to_id) pair does not yet exist.

        WHY: prevents multi-edges between the same node pair, which would
        inflate traversal scores and waste memory.

        Returns
        -------
        bool
            True if the edge was added, False if it already existed.
        """
        existing = self.E.get(edge.from_id, [])
        for e in existing:
            if e.to_id == edge.to_id:
                return False
        self.add_edge(edge)
        return True

    # ── Tier views ─────────────────────────────────────────────────────────────

    def nodes_in_tier(self, tier: Tier) -> list[Node]:
        """Return all nodes in a tier, sorted oldest-first by creation time.

        WHY oldest-first:
            Sleep consolidation and compression process older nodes first
            to avoid losing recently ingested context that may still be
            active in the user's working memory.
        """
        return [self.V[nid] for nid in self._tier_sets[tier] if nid in self.V]

    def tier_size(self, tier: Tier) -> int:
        """Number of live nodes in a given tier."""
        # Count only UUIDs still present in V (SortedList may lag briefly).
        return sum(1 for nid in self._tier_sets[tier] if nid in self.V)

    # ── Global statistics ──────────────────────────────────────────────────────

    def total_nodes(self) -> int:
        """Total number of live nodes across all tiers."""
        return len(self.V)

    def total_edges(self) -> int:
        """Total number of directed edges in the graph."""
        return sum(len(edges) for edges in self.E.values())

    # ── Python protocol ────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Node]:
        """Iterate over all live nodes (arbitrary order)."""
        return iter(self.V.values())

    def __len__(self) -> int:
        """Number of live nodes (same as total_nodes())."""
        return len(self.V)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.V
