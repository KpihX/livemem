"""
test_index.py — Tests for index.py: TierIndex and TieredIndex.
"""
from __future__ import annotations

import numpy as np
import pytest

from livemem.index import TierIndex, TieredIndex
from livemem.types import Tier
from tests.conftest import make_node


@pytest.fixture
def tier_idx(small_config) -> TierIndex:
    return TierIndex(small_config, Tier.SHORT)


@pytest.fixture
def tiered_idx(small_config) -> TieredIndex:
    return TieredIndex(small_config)


# ── TierIndex ─────────────────────────────────────────────────────────────────

def test_tierindex_add_returns_int_label(tier_idx):
    n = make_node(seed=1)
    label = tier_idx.add(n.id, n.v)
    assert isinstance(label, int)
    assert label >= 0


def test_tierindex_query_returns_sorted_descending(tier_idx, small_config):
    """Query should return (uuid, cos_sim) pairs sorted descending by cos_sim."""
    from livemem.embedder import MockEmbedder
    emb = MockEmbedder(small_config)

    texts = ["alpha", "beta", "gamma", "delta"]
    ids = []
    for t in texts:
        v = emb.embed(t)
        import uuid as _uuid
        uid = str(_uuid.uuid4())
        tier_idx.add(uid, v)
        ids.append((uid, v))

    query_v = emb.embed("alpha")
    results = tier_idx.query(query_v, k=4)
    assert len(results) <= 4
    if len(results) >= 2:
        # Verify descending order.
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)


def test_tierindex_remove_soft_deletes(tier_idx):
    n = make_node(seed=3)
    tier_idx.add(n.id, n.v)
    assert n.id in tier_idx
    tier_idx.remove(n.id)
    assert n.id not in tier_idx


def test_tierindex_remove_not_returned_by_query(tier_idx, small_config):
    from livemem.embedder import MockEmbedder
    emb = MockEmbedder(small_config)
    import uuid as _uuid
    uid = str(_uuid.uuid4())
    v = emb.embed("unique text for removal test")
    tier_idx.add(uid, v)
    tier_idx.remove(uid)
    results = tier_idx.query(v, k=5)
    returned_ids = [r[0] for r in results]
    assert uid not in returned_ids


def test_tierindex_size_tracks_live_count(tier_idx):
    n1 = make_node(seed=4)
    n2 = make_node(seed=5)
    tier_idx.add(n1.id, n1.v)
    tier_idx.add(n2.id, n2.v)
    assert tier_idx.size == 2
    tier_idx.remove(n1.id)
    assert tier_idx.size == 1


def test_tierindex_contains_works(tier_idx):
    n = make_node(seed=6)
    assert n.id not in tier_idx
    tier_idx.add(n.id, n.v)
    assert n.id in tier_idx
    tier_idx.remove(n.id)
    assert n.id not in tier_idx


def test_tierindex_all_uuids_excludes_deleted(tier_idx):
    n1 = make_node(seed=7)
    n2 = make_node(seed=8)
    tier_idx.add(n1.id, n1.v)
    tier_idx.add(n2.id, n2.v)
    tier_idx.remove(n1.id)
    uuids = tier_idx.all_uuids()
    assert n1.id not in uuids
    assert n2.id in uuids


# ── TieredIndex ───────────────────────────────────────────────────────────────

def test_tieredindex_add_to_specific_tier(tiered_idx):
    n = make_node(seed=9)
    tiered_idx.add(n.id, n.v, Tier.MEDIUM)
    assert tiered_idx.size(Tier.MEDIUM) == 1
    assert tiered_idx.size(Tier.SHORT) == 0
    assert tiered_idx.size(Tier.LONG) == 0


def test_tieredindex_query_tier_isolated(tiered_idx, small_config):
    """A MEDIUM query should not return SHORT nodes."""
    from livemem.embedder import MockEmbedder
    import uuid as _uuid
    emb = MockEmbedder(small_config)
    v = emb.embed("isolation test")
    uid_short = str(_uuid.uuid4())
    uid_medium = str(_uuid.uuid4())
    tiered_idx.add(uid_short, v, Tier.SHORT)
    tiered_idx.add(uid_medium, v, Tier.MEDIUM)

    medium_results = [r[0] for r in tiered_idx.query(Tier.MEDIUM, v, k=10)]
    assert uid_medium in medium_results
    assert uid_short not in medium_results


def test_tieredindex_move_removes_from_old_adds_to_new(tiered_idx):
    n = make_node(seed=10)
    tiered_idx.add(n.id, n.v, Tier.SHORT)
    assert tiered_idx.size(Tier.SHORT) == 1
    tiered_idx.move(n.id, n.v, Tier.SHORT, Tier.MEDIUM)
    assert tiered_idx.size(Tier.SHORT) == 0
    assert tiered_idx.size(Tier.MEDIUM) == 1


def test_tieredindex_total_size_sums_all_tiers(tiered_idx):
    n1 = make_node(seed=11)
    n2 = make_node(seed=12)
    n3 = make_node(seed=13)
    tiered_idx.add(n1.id, n1.v, Tier.SHORT)
    tiered_idx.add(n2.id, n2.v, Tier.MEDIUM)
    tiered_idx.add(n3.id, n3.v, Tier.LONG)
    assert tiered_idx.total_size() == 3
