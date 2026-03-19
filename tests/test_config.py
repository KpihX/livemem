"""
test_config.py — Tests for structured YAML-backed configuration.
"""
from __future__ import annotations

from pathlib import Path

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embeddings.factory import make_embedder
from livemem.embeddings.mock import MockEmbedder


def test_default_config_loads_from_yaml() -> None:
    assert DEFAULT_CONFIG.config_path.endswith("config.yaml")
    assert Path(DEFAULT_CONFIG.config_path).exists()


def test_config_get_supports_dotted_paths() -> None:
    assert DEFAULT_CONFIG.get("embedding.primary.dimension") == DEFAULT_CONFIG.d
    assert DEFAULT_CONFIG.get("retrieval.weights.urgency") == DEFAULT_CONFIG.epsilon_score
    assert DEFAULT_CONFIG.get("does.not.exist", "fallback") == "fallback"


def test_from_yaml_allows_flat_overrides() -> None:
    cfg = LiveConfig.from_yaml(overrides={"theta_min": 0.12, "embedder_implementation": "mock"})
    assert cfg.theta_min == 0.12
    assert cfg.get("edges.min_similarity") == 0.12
    assert cfg.embedder_implementation == "mock"


def test_make_embedder_uses_configured_implementation() -> None:
    cfg = LiveConfig.from_yaml(overrides={"embedder_implementation": "mock"})
    embedder = make_embedder(cfg)
    assert isinstance(embedder, MockEmbedder)
