"""
conftest.py — Shared pytest fixtures for LiveMem tests.

WHY fixtures here (not in each test file):
    Shared fixtures reduce boilerplate and ensure a consistent test
    environment. Using a small config (d=16, T1=1s, T2=10s) keeps tests
    fast by avoiding large vector allocations and quick tier transitions.
"""
from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from livemem.config import LiveConfig
from livemem.embedder import MockEmbedder
from livemem.memory import LiveMem
from livemem.types import Node, Tier


@pytest.fixture
def small_config() -> LiveConfig:
    """Compact config for fast tests.

    WHY these values:
        d=16      : tiny vectors to keep memory and compute overhead minimal.
        T1=1.0    : 1-second SHORT→MEDIUM boundary (enables tier tests in <1s
                   via time manipulation rather than real waiting).
        T2=10.0   : 10-second MEDIUM→LONG boundary.
        theta_min / theta_sleep / theta_promote = 0.01 : very low threshold
                   so that MockEmbedder vectors (which are orthogonal-ish)
                   still form edges in most tests. Tests that need isolation
                   should set theta_min=0.999 explicitly.
        theta_compress = 0.95 : high threshold for compression tests.
        tau_long=5.0  : 5s idle threshold for SHORT→LONG diffusion.
        max_nodes=100 : small enough to test compression easily.
        Scoring weights sum to 1.0: 0.45+0.20+0.15+0.12+0.08=1.00
    """
    return LiveConfig(
        d=16,
        T1=1.0,
        T2=10.0,
        alpha_tier=0.8,
        beta_tier=0.3,
        k_awake=10,
        k_sleep=20,
        k_promote=15,
        theta_min=0.01,
        theta_sleep=0.01,
        theta_promote=0.01,
        theta_compress=0.95,
        urgency_lambda=1e-5,
        theta_urgent=0.7,
        importance_medium_floor=0.6,
        decay_lambda=1e-5,
        delta_reinforce=0.10,
        tau_long=5.0,
        idle_ttl=300.0,
        daemon_check_interval=30.0,
        max_nodes=100,
        long_compress_fraction=0.7,
        alpha_score=0.45,
        beta_score=0.20,
        gamma_score=0.15,
        delta_score=0.12,
        epsilon_score=0.08,
        s_base_init=0.5,
        hnsw_max_elements=1000,
        hnsw_ef_construction=50,
        hnsw_M=8,
        hnsw_ef_search=50,
    )


@pytest.fixture
def mock_embedder(small_config: LiveConfig) -> MockEmbedder:
    """MockEmbedder with d=16."""
    return MockEmbedder(small_config)


@pytest.fixture
def fresh_mem(small_config: LiveConfig, mock_embedder: MockEmbedder) -> LiveMem:
    """Fresh LiveMem instance with small config and mock embedder."""
    return LiveMem(cfg=small_config, embedder=mock_embedder)


def make_node(
    summary: str = "test node",
    ref_uri: str | None = None,
    ref_type: str = "text",
    d: int = 16,
    importance: float = 0.5,
    urgency: float = 0.0,
    s_base: float = 0.5,
    t_offset: float = 0.0,
    tier: Tier = Tier.SHORT,
    seed: int | None = None,
) -> Node:
    """Helper: create a Node with a controlled random vector.

    Parameters
    ----------
    summary    : node summary text.
    ref_uri    : optional URI.
    ref_type   : content type.
    d          : vector dimension.
    importance : importance ∈ [0, 1] — continuous (no enum).
    urgency    : urgency ∈ [0, 1] — continuous.
    s_base     : initial strength.
    t_offset   : shift creation time by this many seconds (negative = older).
    tier       : initial tier.
    seed       : RNG seed for the vector (None = random).
    """
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1e6) % 2**32)
    v = rng.standard_normal(d).astype(np.float32)
    v /= np.linalg.norm(v)
    now = time.time() + t_offset
    return Node(
        v=v,
        summary=summary,
        ref_uri=ref_uri,
        ref_type=ref_type,
        importance=importance,
        urgency=urgency,
        s_base=s_base,
        t=now,
        t_accessed=now,
        tier=tier,
    )
