"""
index.py — HNSW-based approximate nearest-neighbour index per memory tier.

WHY hnswlib (not FAISS):
    hnswlib is a pure-Python-bindable, dependency-light HNSW implementation
    with O(log n) insertions and O(log n) queries. FAISS would be faster at
    scale (>1M vectors) but adds a heavy C++ dependency. For LiveMem's target
    scale (<50 k vectors) hnswlib is more than sufficient and easier to
    install cross-platform.

WHY one index per tier (TieredIndex):
    Tier-isolated queries prevent SHORT-tier searches from accidentally
    surfacing LONG-tier nodes (which belong to a different memory stratum).
    Each tier can also have different ef_search settings if needed in future.

WHY soft-delete (mark_deleted):
    hnswlib does not support true removal — it only marks elements deleted
    in the graph so they are filtered from results. We track deleted int
    labels in _deleted to filter them on our side as well, ensuring
    __contains__ and all_uuids are accurate.
"""
from __future__ import annotations

import numpy as np

try:
    import hnswlib
except ImportError as exc:
    raise ImportError(
        "hnswlib is required for LiveMem index. "
        "Install with: pip install hnswlib"
    ) from exc

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.types import Tier


class TierIndex:
    """HNSW approximate nearest-neighbour index for a single memory tier.

    Maps UUID strings ↔ monotonic integer labels (hnswlib only accepts
    integer labels). Soft-deletes are tracked in _deleted to filter results
    on our side in addition to hnswlib's internal mark.

    Attributes
    ----------
    _index        : hnswlib.Index  — the underlying HNSW structure.
    _uuid_to_int  : dict[str, int] — UUID → int label mapping.
    _int_to_uuid  : dict[int, str] — int label → UUID reverse mapping.
    _counter      : int            — monotonically increasing label counter.
    _deleted      : set[int]       — int labels that have been soft-deleted.
    _size         : int            — count of live (non-deleted) items.
    """

    def __init__(self, cfg: LiveConfig, tier: Tier) -> None:
        """Initialise the HNSW index with cosine distance space.

        WHY cosine space:
            Embeddings are unit-normalised (enforced by Node.__post_init__).
            For unit vectors, cosine similarity = dot product, and
            hnswlib's 'cosine' space computes 1 - cos_sim as distance,
            which we convert back to cos_sim = 1 - dist in query().
        """
        self._cfg = cfg
        self._tier = tier
        self._index = hnswlib.Index(space="cosine", dim=cfg.d)
        self._index.init_index(
            max_elements=cfg.hnsw_max_elements,
            ef_construction=cfg.hnsw_ef_construction,
            M=cfg.hnsw_M,
        )
        self._index.set_ef(cfg.hnsw_ef_search)

        self._uuid_to_int: dict[str, int] = {}
        self._int_to_uuid: dict[int, str] = {}
        self._counter: int = 0
        self._deleted: set[int] = set()
        self._size: int = 0

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add(self, node_id: str, v: np.ndarray) -> int:
        """Insert a node into the index, assigning a new monotonic int label.

        WHY monotonic counter:
            hnswlib labels must be non-negative integers. Using a counter
            avoids collisions even after soft-deletes (reusing deleted labels
            would be ambiguous in the reverse mapping).

        Parameters
        ----------
        node_id : str   — UUID of the node.
        v       : array — unit-norm embedding (shape (d,)).

        Returns
        -------
        int — the assigned integer label.
        """
        label = self._counter
        self._counter += 1
        self._uuid_to_int[node_id] = label
        self._int_to_uuid[label] = node_id
        self._index.add_items(v.reshape(1, -1).astype(np.float32), [label])
        self._size += 1
        return label

    def remove(self, node_id: str) -> None:
        """Soft-delete a node from the index.

        WHY soft-delete:
            hnswlib does not support hard removal. mark_deleted() tells
            the index to skip this label in future queries. We also track
            deleted labels in _deleted so our own query() filter is accurate.

        Silently ignores unknown node_ids.
        """
        label = self._uuid_to_int.get(node_id)
        if label is None:
            return
        if label not in self._deleted:
            try:
                self._index.mark_deleted(label)
            except Exception:
                pass  # Already deleted internally.
            self._deleted.add(label)
            self._size = max(0, self._size - 1)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self, v: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Find k approximate nearest neighbours by cosine similarity.

        WHY convert distance to similarity:
            hnswlib returns (1 - cos_sim) distances. We convert to cos_sim
            = 1 - dist so callers deal with intuitive similarity scores
            (higher = more similar), not distances.

        WHY filter _deleted on our side:
            hnswlib's mark_deleted is not guaranteed to be perfectly
            synchronous in all builds. Our _deleted filter is a safety net.

        Parameters
        ----------
        v : np.ndarray — query vector (unit-norm, shape (d,)).
        k : int        — number of neighbours requested.

        Returns
        -------
        list of (uuid, cos_sim) sorted descending by cos_sim.
        """
        live = self._size
        if live == 0:
            return []

        k_actual = min(k, live)
        labels, distances = self._index.knn_query(
            v.reshape(1, -1).astype(np.float32), k=k_actual
        )
        results: list[tuple[str, float]] = []
        for label, dist in zip(labels[0], distances[0]):
            if label in self._deleted:
                continue
            node_id = self._int_to_uuid.get(int(label))
            if node_id is None:
                continue
            cos_sim = float(np.clip(1.0 - dist, 0.0, 1.0))
            results.append((node_id, cos_sim))

        # Sort descending by cos_sim.
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Inspection ────────────────────────────────────────────────────────────

    def __contains__(self, node_id: str) -> bool:
        """True if node_id is live (added and not soft-deleted)."""
        label = self._uuid_to_int.get(node_id)
        if label is None:
            return False
        return label not in self._deleted

    @property
    def size(self) -> int:
        """Number of live (non-deleted) items."""
        return self._size

    def all_uuids(self) -> list[str]:
        """Return all live UUID strings (soft-deleted excluded)."""
        return [
            uid for uid, label in self._uuid_to_int.items()
            if label not in self._deleted
        ]


class TieredIndex:
    """Aggregate of three TierIndex instances, one per Tier.

    WHY a dedicated container class:
        Hides the 3-index structure from callers. Memory.py only calls
        tiered_index.add(uuid, v, tier) without knowing which underlying
        index is used. This also enforces tier isolation: you cannot
        accidentally query LONG data from a SHORT search.
    """

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG) -> None:
        self._cfg = cfg
        self._indices: dict[Tier, TierIndex] = {
            Tier.SHORT: TierIndex(cfg, Tier.SHORT),
            Tier.MEDIUM: TierIndex(cfg, Tier.MEDIUM),
            Tier.LONG: TierIndex(cfg, Tier.LONG),
        }

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add(self, node_id: str, v: np.ndarray, tier: Tier) -> None:
        """Insert a node into the index for the given tier."""
        self._indices[tier].add(node_id, v)

    def remove(self, node_id: str, tier: Tier) -> None:
        """Soft-delete a node from the specified tier index."""
        self._indices[tier].remove(node_id)

    def move(
        self, node_id: str, v: np.ndarray, old_tier: Tier, new_tier: Tier
    ) -> None:
        """Atomically move a node from old_tier to new_tier.

        WHY atomically:
            If remove succeeds but add fails, the node would be lost from
            the index entirely. We perform both operations and let any
            exception propagate without partial state.
        """
        self._indices[old_tier].remove(node_id)
        self._indices[new_tier].add(node_id, v)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self, tier: Tier, v: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Query k nearest neighbours from a specific tier only."""
        return self._indices[tier].query(v, k)

    # ── Inspection ────────────────────────────────────────────────────────────

    def size(self, tier: Tier) -> int:
        """Live node count in the specified tier."""
        return self._indices[tier].size

    def total_size(self) -> int:
        """Total live node count across all tiers."""
        return sum(idx.size for idx in self._indices.values())

    def all_uuids(self, tier: Tier) -> list[str]:
        """All live UUIDs in the specified tier."""
        return self._indices[tier].all_uuids()

    def __contains__(self, item: tuple[str, Tier]) -> bool:
        """Check if (uuid, tier) pair is live in the correct tier index."""
        node_id, tier = item
        return node_id in self._indices[tier]
