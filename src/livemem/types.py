"""
types.py — Core data types for the LiveMem system.

WHY this file exists:
    Centralises all domain types so that every other module imports from
    a single source of truth. Enums, dataclasses, and the three key
    analytical functions (strength_effective, urgency_effective, tier_fn)
    that depend only on a Node and a config live here.

    Keeping analytical functions close to the types they operate on avoids
    circular imports: graph.py and memory.py both import from types.py,
    not from each other.

CONTINUITY PRINCIPLE:
    importance and urgency are continuous floats in [0, 1] — NO discretization.
    This enables smooth gradient-based behaviour, fine-grained tuning, and
    avoids the cliff effects that come from hard enum boundaries.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from math import exp
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from livemem.config import LiveConfig

from livemem.config import DEFAULT_CONFIG


# ── Enumerations ───────────────────────────────────────────────────────────────

class Tier(IntEnum):
    """Memory tier a node currently inhabits.

    WHY 3 tiers:
        SHORT  — working memory: fast ANN, frequently accessed.
        MEDIUM — episodic buffer: moderate access, cross-tier links.
        LONG   — semantic store: stable, compressed, rarely surfaced.
    The tier is NOT stored permanently — it is recomputed by tier_fn
    at access time, making it a dynamic property of the node's age and
    reinforcement history.
    """
    SHORT = 0
    MEDIUM = 1
    LONG = 2


class EdgeType(IntEnum):
    """How an edge was created.

    WHY track edge provenance:
        DIRECT      — created at awake ingest; captures contemporaneous
                      associations (strong semantic signal).
        SLEEP       — created during sleep diffusion/promotion; captures
                      cross-tier thematic links (weaker, exploratory).
        CONSOLIDATED— created when a cluster is fused into one node;
                      connects the fused node to external neighbours.
    """
    DIRECT = 0
    SLEEP = 1
    CONSOLIDATED = 2


class RefType:
    """String constants for the type of content a ref_uri points to.

    WHY a plain class rather than Enum:
        ref_type is serialised as a raw string in JSON, and the set of
        types may grow (e.g., "3d_model"). Using string constants keeps
        the on-disk format stable without migration.
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    URL = "url"

    ALL: tuple[str, ...] = ("text", "image", "audio", "video", "url")


# ── Core dataclasses ───────────────────────────────────────────────────────────

@dataclass
class IngestInput:
    """Typed input payload for one awake-ingest operation.

    WHY a dedicated dataclass:
        The engine now supports both single-item ingest and batch ingest.
        Using a shared typed payload keeps the write-path contract explicit
        across the Python API, REST layer, CLI helpers, and future adapters.
    """

    summary: str
    ref_uri: str | None = None
    ref_type: str = RefType.TEXT
    importance: float = 0.5
    urgency: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "importance", max(0.0, min(1.0, self.importance)))
        object.__setattr__(self, "urgency", max(0.0, min(1.0, self.urgency)))
        if self.ref_type not in RefType.ALL:
            object.__setattr__(self, "ref_type", RefType.TEXT)


@dataclass
class Node:
    """A single memory unit in the graph.

    WHY nodes do NOT store full content:
        Storing raw content (potentially MBs of text/audio) in every
        node would make the graph unmanageable in RAM. Instead, nodes
        hold a compact embedding vector, a short summary (≤200 chars),
        and an optional URI pointing to the actual content on disk or
        network. This follows the pointer-based memory architecture used
        in modern vector databases.

    CONTINUITY PRINCIPLE (importance & urgency):
        Both are continuous floats in [0, 1] — no discretization.
        This enables smooth gradient-based tier transitions and scoring
        without the cliff effects of integer enums.

    Fields
    ------
    v           : Unit-norm embedding (shape (d,)). __post_init__
                  enforces normalization so cosine = dot product.
    summary     : Textual abstract ≤200 chars for human-readable recall.
    ref_uri     : Path or URL to actual content (None if inline text).
    ref_type    : One of RefType constants.
    importance  : Continuous semantic weight ∈ [0, 1].
                  0 = negligible, 0.5 = normal, 1 = absolutely critical.
    urgency     : Continuous time-pressure ∈ [0, 1].
                  Decays from node.t (creation time) via urgency_lambda.
                  0 = no urgency, 1 = extreme deadline pressure.
    s_base      : Base strength ∈ [0,1]. Decays via Ebbinghaus curve.
    t           : Unix timestamp of creation.
    t_accessed  : Unix timestamp of last access / reinforcement.
    tier        : Current tier (dynamic — updated by _update_tier).
    diffused    : True once sleep_diffuse has processed this SHORT node.
    consolidated: True if this node was created by merging a cluster.
    sources     : UUIDs of merged nodes if consolidated=True.
    id          : UUID4 string, primary key.
    """
    v: np.ndarray
    summary: str
    ref_uri: str | None = None
    ref_type: str = RefType.TEXT
    importance: float = 0.5
    urgency: float = 0.0
    s_base: float = 0.5
    t: float = field(default_factory=time.time)
    t_accessed: float = field(default_factory=time.time)
    tier: Tier = Tier.SHORT
    diffused: bool = False
    consolidated: bool = False
    sources: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Enforce invariants on the embedding vector, importance, and urgency.

        WHY: cosine similarity via dot product only equals true cosine
        when both vectors are unit-norm. Enforcing normalization here
        means every caller can safely use np.dot(a.v, b.v) as cos(a,b).

        Clamping importance and urgency to [0, 1] guards against accidental
        out-of-range values while preserving the continuity principle.
        """
        norm = float(np.linalg.norm(self.v))
        if norm < 1e-12:
            raise ValueError(
                f"Node '{self.id}': embedding vector has near-zero norm "
                f"({norm:.2e}); cannot normalize."
            )
        if abs(norm - 1.0) > 1e-5:
            # Normalize in-place — the array may be shared, so copy first.
            object.__setattr__(self, "v", self.v / norm)

        # Clamp s_base to [0, 1].
        if not (0.0 <= self.s_base <= 1.0):
            object.__setattr__(
                self, "s_base", max(0.0, min(1.0, self.s_base))
            )

        # Clamp importance to [0, 1] — continuity principle.
        if not (0.0 <= self.importance <= 1.0):
            object.__setattr__(
                self, "importance", max(0.0, min(1.0, self.importance))
            )

        # Clamp urgency to [0, 1] — continuity principle.
        if not (0.0 <= self.urgency <= 1.0):
            object.__setattr__(
                self, "urgency", max(0.0, min(1.0, self.urgency))
            )

    def __hash__(self) -> int:
        """Hash by id (UUID string) for use in sets / dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id only — two nodes with the same UUID are the same node."""
        if isinstance(other, Node):
            return self.id == other.id
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id[:8]}…, tier={self.tier.name}, "
            f"importance={self.importance:.2f}, urgency={self.urgency:.2f}, "
            f"s_base={self.s_base:.3f}, summary={self.summary[:40]!r})"
        )


@dataclass
class Edge:
    """A directed association between two memory nodes.

    WHY directed:
        Direction encodes temporal precedence — the more recent node
        points to the older one. This mirrors how episodic memory chains
        events together and allows graph traversal to follow causal /
        chronological order.

    Fields
    ------
    from_id   : UUID of the source node (more recent, higher t).
    to_id     : UUID of the target node (older, lower t).
    cos_sim   : Cosine similarity at edge-creation time ∈ [0,1].
    delta_t   : |t_from - t_to| in seconds; must be ≥ 0.
    edge_type : How this edge was created (DIRECT / SLEEP / CONSOLIDATED).
    """
    from_id: str
    to_id: str
    cos_sim: float
    delta_t: float
    edge_type: EdgeType = EdgeType.DIRECT

    def __post_init__(self) -> None:
        """Validate invariants on construction.

        WHY validate delta_t:
            A negative time-delta is physically impossible (from_id is
            always the newer node). Catching it here surfaces bugs in
            edge-creation logic immediately rather than silently
            corrupting the graph structure.
        """
        if self.delta_t < 0:
            raise ValueError(
                f"Edge({self.from_id[:8]}→{self.to_id[:8]}): "
                f"delta_t must be ≥ 0, got {self.delta_t:.3f}"
            )
        # Clamp before validation — HNSW dot products on unit vectors can
        # yield 1.0 + tiny epsilon due to float32 rounding.  Silently
        # clip rather than crash: any value outside (-eps, 1+eps) is still
        # a bug and surfaces as a negative score downstream.
        self.cos_sim = max(0.0, min(1.0, self.cos_sim))
        if not (0.0 <= self.cos_sim <= 1.0):
            raise ValueError(
                f"Edge: cos_sim must be in [0,1], got {self.cos_sim:.4f}"
            )


@dataclass
class RetrievalResult:
    """A single result returned by LiveMem.retrieve().

    WHY a dedicated dataclass:
        Returning raw Node objects would leak internal mutable state to
        callers. RetrievalResult is a read-only snapshot that includes
        the pre-computed score, making it safe to sort, filter, and pass
        across boundaries.
    """
    node_id: str
    score: float
    summary: str
    ref_uri: str | None
    ref_type: str
    tier: Tier
    importance: float
    urgency: float
    cos_direct: float


# ── Analytical functions ───────────────────────────────────────────────────────

def strength_effective(
    node: Node,
    t: float,
    cfg: LiveConfig = DEFAULT_CONFIG,
) -> float:
    """Compute the effective (decayed) strength of a node at time t.

    WHY Ebbinghaus exponential decay:
        The forgetting curve (Ebbinghaus, 1885) shows that memory
        retention decays exponentially with time since last access.
        Modelling this here ensures that nodes which are never accessed
        organically lose relevance, allowing the tier system and
        compression to eventually reclaim space from forgotten memories.

    Formula:
        s_eff = s_base * exp(-λ * max(0, t - t_accessed))

    Parameters
    ----------
    node : Node
        The memory node to evaluate.
    t    : float
        Current unix timestamp (wall-clock "now").
    cfg  : LiveConfig
        Configuration; provides decay_lambda.

    Returns
    -------
    float
        Effective strength ∈ [0, s_base].
    """
    elapsed = max(0.0, t - node.t_accessed)
    return node.s_base * exp(-cfg.decay_lambda * elapsed)


def urgency_effective(
    node: Node,
    t: float,
    cfg: LiveConfig = DEFAULT_CONFIG,
) -> float:
    """Compute the effective (decayed) urgency of a node at time t.

    WHY decay from creation time (node.t), not last access (t_accessed):
        Urgency models time-pressure from an external deadline, not from
        how recently the memory was rehearsed. A task due tomorrow has
        the same urgency whether you looked at it 5 minutes ago or not.
        Using node.t prevents urgency from being inadvertently reset by
        reinforcement, which would make urgent-old nodes immortal.

    WHY faster decay (urgency_lambda >> decay_lambda):
        Urgency is ephemeral by design — a node that was urgent 24 hours
        ago is likely no longer a pressing concern. Strength, on the other
        hand, models long-term memory retention which decays much slower.

    Formula:
        u_eff = urgency * exp(-urgency_lambda * max(0, t - t_creation))

    Parameters
    ----------
    node : Node
        The memory node to evaluate.
    t    : float
        Current unix timestamp (wall-clock "now").
    cfg  : LiveConfig
        Configuration; provides urgency_lambda.

    Returns
    -------
    float
        Effective urgency ∈ [0, node.urgency].
    """
    elapsed = max(0.0, t - node.t)
    return node.urgency * exp(-cfg.urgency_lambda * elapsed)


def tier_fn(
    node: Node,
    t: float,
    cfg: LiveConfig = DEFAULT_CONFIG,
) -> Tier:
    """Compute the current tier of a node using 4-quadrant Eisenhower logic.

    WHY 4-quadrant (urgency × importance):
        Inspired by the Eisenhower matrix, nodes are governed by two
        independent axes: urgency (time-pressure, decays quickly) and
        importance (semantic weight, stable). Together they define four
        behavioural quadrants:

        ┌──────────────────────┬────────────────────────────┐
        │ HIGH urgency         │ HIGH urgency               │
        │ LOW importance       │ HIGH importance            │
        │ → pinned SHORT       │ → pinned SHORT             │
        ├──────────────────────┼────────────────────────────┤
        │ LOW urgency          │ LOW urgency                │
        │ LOW importance       │ HIGH importance            │
        │ → raw tier (ages)    │ → floor at MEDIUM, no LONG │
        └──────────────────────┴────────────────────────────┘

    Algorithm (priority order):
        1. Compute u_eff = urgency_effective(node, t, cfg).
           If u_eff ≥ cfg.theta_urgent → return SHORT (urgency pin).
        2. Compute s_eff = strength_effective(node, t, cfg).
           denominator = 1 + alpha_tier * s_eff + beta_tier * importance
           effective_age = (t - node.t) / max(denominator, 1e-9)
        3. Raw tier from effective_age:
           ea < T1 → SHORT; ea < T2 → MEDIUM; else → LONG
        4. Importance floor: if raw == LONG and importance ≥
           importance_medium_floor → return MEDIUM.
        5. Return raw tier.

    WHY effective age (not raw age):
        Raw wall-clock age would demote frequently-accessed or highly
        important nodes too quickly. The denominator scales down the
        effective age proportionally to the node's strength and
        importance, meaning a high-importance node that is regularly
        reinforced stays in SHORT even days after creation.

    Parameters
    ----------
    node : Node
    t    : float — current unix timestamp
    cfg  : LiveConfig

    Returns
    -------
    Tier
    """
    # ── 1. Urgency pin: HIGH urgency → always SHORT ──────────────────────────
    u_eff = urgency_effective(node, t, cfg)
    if u_eff >= cfg.theta_urgent:
        return Tier.SHORT

    # ── 2. Effective-age computation ─────────────────────────────────────────
    s_eff = strength_effective(node, t, cfg)
    denominator = max(
        1.0
        + cfg.alpha_tier * s_eff
        + cfg.beta_tier * node.importance,
        1e-9,
    )
    effective_age = (t - node.t) / denominator

    # ── 3. Raw tier from effective age ────────────────────────────────────────
    if effective_age < cfg.T1:
        raw = Tier.SHORT
    elif effective_age < cfg.T2:
        raw = Tier.MEDIUM
    else:
        raw = Tier.LONG

    # ── 4. Importance floor: HIGH importance → never LONG ────────────────────
    if raw == Tier.LONG and node.importance >= cfg.importance_medium_floor:
        return Tier.MEDIUM

    return raw
