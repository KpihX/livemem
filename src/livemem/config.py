"""
config.py — structured configuration loader backed by config.yaml.

WHY this module exists:
    The source of truth for livemem configuration should be external to the
    Python code so the system can swap implementations and tune behavior
    without editing module constants. config.yaml holds the structured config;
    LiveConfig loads it into a strongly typed runtime object and exposes
    `get("section.subsection.key")` for ergonomic read access.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must load to a mapping: {path}")
    return data


def _lookup(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


@dataclass(frozen=True)
class LiveConfig:
    """Immutable runtime config with structured lookup support."""

    d: int = 384
    model_name: str = "BAAI/bge-small-en-v1.5"
    embedder_implementation: str = "fastembed_text"
    reranker_enabled: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_implementation: str = "fastembed_cross_encoder"
    reranker_k: int = 20

    T1: float = 86_400.0
    T2: float = 604_800.0
    alpha_tier: float = 0.8
    beta_tier: float = 0.3

    k_awake: int = 10
    k_sleep: int = 20
    k_promote: int = 15

    theta_min: float = 0.60
    theta_sleep: float = 0.75
    theta_promote: float = 0.70
    theta_compress: float = 0.90

    urgency_lambda: float = 5e-5
    theta_urgent: float = 0.7
    importance_medium_floor: float = 0.6

    decay_lambda: float = 5e-6
    delta_reinforce: float = 0.10
    s_base_init: float = 0.5

    tau_long: float = 1_800.0
    idle_ttl: float = 300.0
    daemon_check_interval: float = 30.0

    max_nodes: int = 10_000
    long_compress_fraction: float = 0.7

    alpha_score: float = 0.45
    beta_score: float = 0.20
    gamma_score: float = 0.15
    delta_score: float = 0.12
    epsilon_score: float = 0.08

    hnsw_max_elements: int = 50_000
    hnsw_ef_construction: int = 200
    hnsw_M: int = 16
    hnsw_ef_search: int = 50

    config_path: str = ""

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        config_path: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "LiveConfig":
        """Build a typed config from the structured YAML mapping."""
        flat = {
            "d": int(_lookup(data, "embedding.primary.dimension")),
            "model_name": str(_lookup(data, "embedding.primary.model_name")),
            "embedder_implementation": str(
                _lookup(data, "embedding.primary.implementation")
            ),
            "reranker_enabled": bool(_lookup(data, "embedding.reranker.enabled")),
            "reranker_model_name": str(
                _lookup(data, "embedding.reranker.model_name")
            ),
            "reranker_implementation": str(
                _lookup(data, "embedding.reranker.implementation")
            ),
            "reranker_k": int(_lookup(data, "embedding.reranker.candidate_pool")),
            "T1": float(_lookup(data, "tiers.short_term_seconds")),
            "T2": float(_lookup(data, "tiers.medium_term_seconds")),
            "alpha_tier": float(_lookup(data, "tiers.alpha_strength")),
            "beta_tier": float(_lookup(data, "tiers.beta_importance")),
            "k_awake": int(_lookup(data, "neighbors.awake")),
            "k_sleep": int(_lookup(data, "neighbors.sleep")),
            "k_promote": int(_lookup(data, "neighbors.promote")),
            "theta_min": float(_lookup(data, "edges.min_similarity")),
            "theta_sleep": float(_lookup(data, "edges.sleep_similarity")),
            "theta_promote": float(_lookup(data, "edges.promote_similarity")),
            "theta_compress": float(_lookup(data, "edges.compression_similarity")),
            "urgency_lambda": float(_lookup(data, "urgency.decay_lambda")),
            "theta_urgent": float(_lookup(data, "urgency.pin_threshold")),
            "importance_medium_floor": float(
                _lookup(data, "urgency.importance_medium_floor")
            ),
            "decay_lambda": float(_lookup(data, "retention.decay_lambda")),
            "delta_reinforce": float(_lookup(data, "retention.reinforce_delta")),
            "s_base_init": float(_lookup(data, "retention.initial_strength")),
            "tau_long": float(_lookup(data, "daemon.long_diffusion_idle_seconds")),
            "idle_ttl": float(_lookup(data, "daemon.idle_ttl_seconds")),
            "daemon_check_interval": float(
                _lookup(data, "daemon.check_interval_seconds")
            ),
            "max_nodes": int(_lookup(data, "compression.max_nodes")),
            "long_compress_fraction": float(
                _lookup(data, "compression.long_term_fraction")
            ),
            "alpha_score": float(_lookup(data, "retrieval.weights.direct_cosine")),
            "beta_score": float(_lookup(data, "retrieval.weights.graph_traversal")),
            "gamma_score": float(
                _lookup(data, "retrieval.weights.effective_strength")
            ),
            "delta_score": float(_lookup(data, "retrieval.weights.importance")),
            "epsilon_score": float(_lookup(data, "retrieval.weights.urgency")),
            "hnsw_max_elements": int(_lookup(data, "index.hnsw.max_elements")),
            "hnsw_ef_construction": int(
                _lookup(data, "index.hnsw.ef_construction")
            ),
            "hnsw_M": int(_lookup(data, "index.hnsw.m")),
            "hnsw_ef_search": int(_lookup(data, "index.hnsw.ef_search")),
            "config_path": str(config_path) if config_path is not None else "",
        }
        if overrides:
            flat.update(overrides)
        return cls(**flat)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "LiveConfig":
        resolved = (
            Path(path).expanduser()
            if path is not None
            else Path(__file__).with_name("config.yaml")
        )
        return cls.from_dict(_read_yaml(resolved), config_path=resolved, overrides=overrides)

    def get(self, path: str, default: Any | None = None) -> Any:
        """Return a config value through dotted-path access."""
        try:
            return _lookup(self.to_nested_dict(), path)
        except KeyError:
            return default

    def to_nested_dict(self) -> dict[str, Any]:
        """Render the flat runtime object back to its structured YAML shape."""
        return {
            "embedding": {
                "primary": {
                    "implementation": self.embedder_implementation,
                    "dimension": self.d,
                    "model_name": self.model_name,
                },
                "reranker": {
                    "enabled": self.reranker_enabled,
                    "implementation": self.reranker_implementation,
                    "model_name": self.reranker_model_name,
                    "candidate_pool": self.reranker_k,
                },
            },
            "tiers": {
                "short_term_seconds": self.T1,
                "medium_term_seconds": self.T2,
                "alpha_strength": self.alpha_tier,
                "beta_importance": self.beta_tier,
            },
            "neighbors": {
                "awake": self.k_awake,
                "sleep": self.k_sleep,
                "promote": self.k_promote,
            },
            "edges": {
                "min_similarity": self.theta_min,
                "sleep_similarity": self.theta_sleep,
                "promote_similarity": self.theta_promote,
                "compression_similarity": self.theta_compress,
            },
            "urgency": {
                "decay_lambda": self.urgency_lambda,
                "pin_threshold": self.theta_urgent,
                "importance_medium_floor": self.importance_medium_floor,
            },
            "retention": {
                "decay_lambda": self.decay_lambda,
                "reinforce_delta": self.delta_reinforce,
                "initial_strength": self.s_base_init,
            },
            "daemon": {
                "long_diffusion_idle_seconds": self.tau_long,
                "idle_ttl_seconds": self.idle_ttl,
                "check_interval_seconds": self.daemon_check_interval,
            },
            "compression": {
                "max_nodes": self.max_nodes,
                "long_term_fraction": self.long_compress_fraction,
            },
            "retrieval": {
                "weights": {
                    "direct_cosine": self.alpha_score,
                    "graph_traversal": self.beta_score,
                    "effective_strength": self.gamma_score,
                    "importance": self.delta_score,
                    "urgency": self.epsilon_score,
                }
            },
            "index": {
                "hnsw": {
                    "max_elements": self.hnsw_max_elements,
                    "ef_construction": self.hnsw_ef_construction,
                    "m": self.hnsw_M,
                    "ef_search": self.hnsw_ef_search,
                }
            },
        }


@lru_cache(maxsize=1)
def load_default_config() -> LiveConfig:
    """Load the package-default config once per process."""
    return LiveConfig.from_yaml()


DEFAULT_CONFIG: LiveConfig = load_default_config()
