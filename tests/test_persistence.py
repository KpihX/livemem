"""
test_persistence.py — Tests for JSON serialisation and deserialisation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from livemem.memory import LiveMem
from livemem.persistence import load, save
from livemem.types import EdgeType, Importance, Tier


@pytest.fixture
def tmp_path_json(tmp_path) -> Path:
    return tmp_path / "state.json"


@pytest.fixture
def populated_mem(small_config, mock_embedder) -> LiveMem:
    """A LiveMem with a few ingested nodes."""
    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    mem.ingest_awake("Python is great", ref_uri="/notes.txt", ref_type="text",
                     importance=Importance.KEY)
    mem.ingest_awake("HNSW enables fast ANN search", ref_uri="https://example.com", ref_type="url")
    mem.ingest_awake("Audio from the forest", ref_uri="/audio.mp3", ref_type="audio",
                     importance=Importance.CAPITAL)
    return mem


# ── Basic save/load ────────────────────────────────────────────────────────────

def test_save_creates_file(populated_mem, tmp_path_json):
    save(populated_mem, tmp_path_json)
    assert tmp_path_json.exists()


def test_save_load_round_trip_node_count(populated_mem, tmp_path_json, small_config, mock_embedder):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    assert loaded.graph.total_nodes() == populated_mem.graph.total_nodes()


def test_save_load_preserves_summary(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    original_summaries = {n.summary for n in populated_mem.graph.V.values()}
    loaded_summaries = {n.summary for n in loaded.graph.V.values()}
    assert original_summaries == loaded_summaries


def test_save_load_preserves_ref_uri(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    original_uris = {n.ref_uri for n in populated_mem.graph.V.values()}
    loaded_uris = {n.ref_uri for n in loaded.graph.V.values()}
    assert original_uris == loaded_uris


def test_save_load_preserves_ref_type(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    orig = {n.ref_type for n in populated_mem.graph.V.values()}
    rest = {n.ref_type for n in loaded.graph.V.values()}
    assert orig == rest


def test_save_load_preserves_importance(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    orig_map = {n.id: n.importance for n in populated_mem.graph.V.values()}
    for n in loaded.graph.V.values():
        if n.id in orig_map:
            assert n.importance == orig_map[n.id]


def test_save_load_preserves_tier(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    orig_tiers = {n.id: n.tier for n in populated_mem.graph.V.values()}
    for n in loaded.graph.V.values():
        if n.id in orig_tiers:
            assert n.tier == orig_tiers[n.id]


def test_save_load_preserves_s_base(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    orig_s = {n.id: n.s_base for n in populated_mem.graph.V.values()}
    for n in loaded.graph.V.values():
        if n.id in orig_s:
            assert abs(n.s_base - orig_s[n.id]) < 1e-5


# ── Edge preservation ─────────────────────────────────────────────────────────

def test_save_load_preserves_edges(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    assert loaded.graph.total_edges() == populated_mem.graph.total_edges()


def test_save_load_preserves_edge_type(populated_mem, tmp_path_json, small_config):
    """Edge types should survive JSON round-trip."""
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    orig_types = set()
    for edges in populated_mem.graph.E.values():
        for e in edges:
            orig_types.add(e.edge_type)
    loaded_types = set()
    for edges in loaded.graph.E.values():
        for e in edges:
            loaded_types.add(e.edge_type)
    assert orig_types == loaded_types


# ── last_sleep_end ────────────────────────────────────────────────────────────

def test_save_load_preserves_last_sleep_end(populated_mem, tmp_path_json, small_config):
    populated_mem.last_sleep_end = 1234567.89
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    assert abs(loaded.last_sleep_end - 1234567.89) < 1e-3


# ── Error cases ───────────────────────────────────────────────────────────────

def test_load_nonexistent_raises_file_not_found(tmp_path, small_config):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "does_not_exist.json", cfg=small_config, mock=True)


def test_load_malformed_json_raises_value_error(tmp_path, small_config):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("this is not json {{{")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load(bad_file, cfg=small_config, mock=True)


# ── Vector unit norm after round-trip ────────────────────────────────────────

def test_vectors_unit_norm_after_round_trip(populated_mem, tmp_path_json, small_config):
    save(populated_mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)
    for n in loaded.graph.V.values():
        norm = float(np.linalg.norm(n.v))
        assert abs(norm - 1.0) < 1e-5, f"Node {n.id[:8]} vector norm={norm:.6f}"


# ── Consolidated nodes ────────────────────────────────────────────────────────

def test_consolidated_nodes_survive_round_trip(tmp_path_json, small_config, mock_embedder):
    """Nodes with consolidated=True and sources should survive save/load."""
    from tests.conftest import make_node
    import uuid as _uuid

    mem = LiveMem(cfg=small_config, embedder=mock_embedder)
    v = np.array([1.0] + [0.0] * (small_config.d - 1), dtype=np.float32)
    src_ids = [str(_uuid.uuid4()), str(_uuid.uuid4())]
    fused = make_node(
        summary="Consolidated: node a; node b",
        tier=Tier.LONG,
        d=small_config.d,
        seed=999,
    )
    object.__setattr__(fused, "consolidated", True)
    object.__setattr__(fused, "sources", src_ids)
    fused.tier = Tier.LONG
    mem.graph.add_node(fused)
    mem.index.add(fused.id, fused.v, Tier.LONG)

    save(mem, tmp_path_json)
    loaded = load(tmp_path_json, cfg=small_config, mock=True)

    loaded_fused = loaded.graph.V.get(fused.id)
    assert loaded_fused is not None
    assert loaded_fused.consolidated is True
    assert set(loaded_fused.sources) == set(src_ids)
