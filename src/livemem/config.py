"""
config.py — LiveMem configuration dataclass.

WHY this file exists:
    All tunable parameters live here as a single frozen dataclass so that
    every module receives the same immutable configuration object. No magic
    global variables, no scattered constants. The caller can override any
    parameter by constructing a new LiveConfig; the DEFAULT_CONFIG singleton
    is used by helper functions that accept an optional cfg argument.

    Parameters are grouped by concern: embedding, tier boundaries, ANN
    search, edge thresholds, decay/reinforce, daemon timing, and retrieval
    scoring weights.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveConfig:
    """Immutable configuration for the LiveMem system.

    Every field is documented with WHY it exists (not just what it is).
    Frozen = safe to pass around and use as dict keys if needed.
    """

    # ── Embedding ────────────────────────────────────────────────────────────
    d: int = 384
    """Dimensionality of the embedding vector.
    WHY: must match the chosen fastembed model output size.
    BAAI/bge-small-en-v1.5 produces 384-dim vectors — change together."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    """fastembed model identifier used by RealEmbedder.
    WHY: bge-small is a good trade-off between quality (~0.87 BEIR avg)
    and speed (<50ms per query on CPU). Swap for bge-large for higher
    accuracy at the cost of 4x more RAM and latency."""

    # ── Tier boundaries ───────────────────────────────────────────────────────
    T1: float = 86_400.0
    """Effective-age threshold (seconds) below which a node stays SHORT.
    WHY: mirrors human working memory (~24 h before things fade to
    mid-term storage). Effective age is slower than wall-clock time for
    strong/important nodes (see alpha_tier / beta_tier)."""

    T2: float = 604_800.0
    """Effective-age threshold (seconds) below which a node stays MEDIUM.
    WHY: mirrors consolidation into episodic memory (~7 days for human
    hippocampal replay). Beyond this the node becomes LONG (semantic)."""

    alpha_tier: float = 0.8
    """Weight of effective strength on effective-age denominator.
    WHY: a high-strength node ages more slowly — it stays accessible
    longer, like a well-rehearsed memory. Range [0, 1]."""

    beta_tier: float = 0.3
    """Weight of importance [0,1] on effective-age denominator.
    WHY: highly important nodes resist tier demotion because they carry
    higher semantic value, independent of access frequency."""

    # ── ANN neighbor counts ───────────────────────────────────────────────────
    k_awake: int = 10
    """Number of ANN neighbors to query during awake ingestion.
    WHY: edges formed at ingest time establish the initial associative
    web. 10 is enough to wire new nodes into the local cluster without
    creating excessive edge density."""

    k_sleep: int = 20
    """Number of ANN neighbors queried per tier during sleep diffusion.
    WHY: sleep is less time-critical; a wider sweep finds more distant
    associations that would be missed at ingest speed."""

    k_promote: int = 15
    """Number of ANN neighbors queried during sleep promotion pass.
    WHY: promotion pulls relevant LONG/MEDIUM nodes back toward SHORT,
    simulating memory recall during sleep. 15 balances coverage vs cost."""

    # ── Edge cosine thresholds ────────────────────────────────────────────────
    theta_min: float = 0.60
    """Minimum cosine similarity to form any edge (AWAKE or reconnect).
    WHY: below 0.60 two concepts are probably unrelated — edges would add
    noise and slow graph traversal without semantic benefit."""

    theta_sleep: float = 0.75
    """Minimum cosine for SLEEP-type edges (diffusion pass).
    WHY: sleep edges cross tier boundaries and must be higher-confidence
    to avoid polluting LONG memory with spurious short-term associations."""

    theta_promote: float = 0.70
    """Minimum cosine for promotion edges (LONG→MEDIUM reinforce path).
    WHY: slightly looser than theta_sleep because promotion is directional
    (evoked centroid → long node) and the centroid already filters noise."""

    theta_compress: float = 0.90
    """Minimum cosine for merging two LONG nodes during compression.
    WHY: compression is destructive (original nodes are removed). Only
    very similar nodes (≥0.90 cos) should be fused to avoid losing
    semantically distinct memories."""

    # ── Urgency dynamics ──────────────────────────────────────────────────────
    urgency_lambda: float = 5e-5
    """Exponential decay rate for urgency (per second).
    WHY: urgency decays 5× faster than strength — time-pressure is
    ephemeral by nature. A node created 6 hours ago loses ~67% of its
    urgency, matching how human deadline pressure fades quickly.
    Decay starts at creation time (node.t), not last access."""

    theta_urgent: float = 0.7
    """Effective-urgency threshold above which a node is pinned to SHORT.
    WHY: the Eisenhower urgent quadrant — a task with u_eff ≥ 0.7 must
    stay in working memory regardless of its age or reinforcement history.
    This implements the 'urgency pin' independently of importance."""

    importance_medium_floor: float = 0.6
    """Importance floor preventing demotion to LONG.
    WHY: nodes with importance ≥ 0.6 are 'key' or above in the
    Eisenhower importance axis. Burying them in LONG semantic storage
    would make them inaccessible for contextual retrieval. The floor
    allows them to age from SHORT → MEDIUM but never → LONG."""

    # ── Forgetting / reinforcement ────────────────────────────────────────────
    decay_lambda: float = 5e-6
    """Ebbinghaus exponential decay rate (per second).
    WHY: models natural forgetting. At λ=5e-6 the half-life of s_eff
    is ~1.6 days — a fresh node stays above 50% strength for nearly
    two days without reinforcement, then fades over the following week.
    Previous value 1e-5 (half-life ~19h) was too aggressive for agent
    memory: critical facts injected in the morning would be nearly gone
    by evening without a retrieval event."""

    delta_reinforce: float = 0.10
    """Strength boost applied per reinforcement (retrieval or sleep link).
    WHY: each access or sleep-association event adds 0.10 to effective
    strength, mirroring the spacing-effect: repeated exposure
    increases retention."""

    # ── Daemon timing ─────────────────────────────────────────────────────────
    tau_long: float = 1_800.0
    """Idle duration (seconds) that enables SHORT→LONG diffusion during sleep.
    WHY: 30 min of inactivity suggests the user is in a rest state. Only
    then do we bridge the working memory (SHORT) to semantic storage (LONG),
    avoiding premature consolidation during active sessions."""

    idle_ttl: float = 300.0
    """Idle time (seconds) before the daemon triggers a sleep phase.
    WHY: 5 min without ingestion signals the end of an active session.
    The daemon then runs consolidation in the background."""

    daemon_check_interval: float = 30.0
    """How often (seconds) the daemon polls for idle state.
    WHY: frequent polling (e.g., 1 s) wastes CPU; 30 s is imperceptible
    latency for a background process and keeps the event loop light."""

    # ── Compression thresholds ────────────────────────────────────────────────
    max_nodes: int = 10_000
    """Hard ceiling on total live nodes before compression is triggered.
    WHY: unbounded graph growth degrades ANN query speed and RAM usage.
    10 k nodes @ 384-d float32 = ~15 MB for vectors alone — manageable."""

    long_compress_fraction: float = 0.7
    """Fraction of max_nodes that LONG tier must reach to trigger compress.
    WHY: compression is triggered early (70% full) to avoid thrashing at
    the hard ceiling. Only LONG nodes are compressed — SHORT/MEDIUM are
    left intact to preserve recent context."""

    # ── Retrieval scoring weights (must sum to 1.0) ───────────────────────────
    alpha_score: float = 0.45
    """Direct cosine similarity weight in retrieval score.
    WHY: semantic distance to the query is the primary relevance signal,
    so it gets the largest weight. Reduced from 0.50 to make room for
    the urgency component (epsilon_score)."""

    beta_score: float = 0.20
    """Graph traversal score weight in retrieval score.
    WHY: a node strongly connected to the top seed nodes is contextually
    relevant even if its direct cosine is moderate. Reduced from 0.25
    to make room for the urgency component."""

    gamma_score: float = 0.15
    """Effective strength weight in retrieval score.
    WHY: frequently accessed, recently reinforced nodes should surface
    higher — analogous to priming in cognitive science."""

    delta_score: float = 0.12
    """Importance [0,1] weight in retrieval score.
    WHY: highly important nodes carry metadata-level relevance and should
    rank higher than equally similar but low-importance nodes."""

    epsilon_score: float = 0.08
    """Urgency weight in retrieval score.
    WHY: urgent nodes should surface in retrieval even when their cosine
    similarity is moderate. An unread deadline note must not be buried.
    Weight is smaller than importance to avoid overriding semantic signals."""

    # ── Initialization ────────────────────────────────────────────────────────
    s_base_init: float = 0.5
    """Floor strength for new nodes — used as the minimum s_base.
    WHY: actual s_base at ingestion is lerp(s_base_init, 1.0, importance)
    so a critical node (imp=1.0) starts at s_base=1.0, a trivial one
    (imp=0.0) starts at s_base=s_base_init (0.5).  Keeping 0.5 as the
    floor ensures even irrelevant nodes have enough initial strength to
    form edges and be retrievable for at least one sleep cycle before
    they drift to LONG."""

    # ── HNSW index parameters ─────────────────────────────────────────────────
    hnsw_max_elements: int = 50_000
    """Maximum elements the HNSW index pre-allocates space for.
    WHY: hnswlib requires a pre-allocated ceiling. 50 k >> max_nodes,
    giving headroom without wasting memory."""

    hnsw_ef_construction: int = 200
    """ef_construction for HNSW index build quality.
    WHY: higher = better recall at the cost of longer add_items time.
    200 is a standard sweet-spot for <1M vectors."""

    hnsw_M: int = 16
    """M (number of bi-directional links) for HNSW graph.
    WHY: M=16 gives ~0.99 recall@10 for 384-d cosine on typical NLP
    datasets with minimal memory overhead (~64 bytes/node extra)."""

    hnsw_ef_search: int = 50
    """ef_search for HNSW query-time accuracy/speed tradeoff.
    WHY: ef=50 gives excellent recall (>0.99) for k≤20 queries on our
    scale. Increase to 100+ if recall drops below threshold."""


# Singleton used as default argument throughout the codebase.
# WHY a module-level singleton: avoids allocating a new object on every
# function call that uses the default config (Python default-arg semantics
# evaluate once at definition time, so the same object is reused).
DEFAULT_CONFIG: LiveConfig = LiveConfig()
