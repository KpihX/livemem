# LiveMem — Architecture & Design Reference

> **Brain-inspired tiered graph memory with continuous importance/urgency.**
> Read this file instead of asking the AI to display the architecture in the terminal.

---

## 1. Core Philosophy

```
0 Magic — 100% Transparency
0 Hardcoding — 100% Flexibility (all params in LiveConfig)
Continuity Principle: no discretization anywhere
```

Memory is modelled as a **living directed graph** across **3 tiers**,
running two alternating modes: **AWAKE** (fast ingest) and **SLEEP** (background consolidation).

---

## 2. Node Anatomy

```
┌──────────────────────────────────────────────────────────────┐
│  Node                                                        │
│  ─────────────────────────────────────────────────────────── │
│  id          : UUID4            (primary key)                │
│  v           : float32[d]       (unit-norm embedding)        │
│  summary     : str ≤200 chars   (human-readable)             │
│  ref_uri     : str | None       (pointer to content)         │
│  ref_type    : text|image|audio|video|url                    │
│                                                              │
│  importance  : float ∈ [0,1]   (semantic weight, STABLE)    │
│  urgency     : float ∈ [0,1]   (time-pressure, DECAYS)      │
│  s_base      : float ∈ [0,1]   (base strength)              │
│                                                              │
│  t           : float            (creation timestamp)         │
│  t_accessed  : float            (last access/reinforce)      │
│  tier        : SHORT|MEDIUM|LONG  (dynamic, recomputed)      │
│  diffused    : bool             (processed by sleep_diffuse) │
│  consolidated: bool             (created by cluster fusion)  │
│  sources     : list[UUID]       (merged node UUIDs)          │
└──────────────────────────────────────────────────────────────┘
```

**WHY no content stored in nodes:**
Storing raw content (MBs of text/audio/video) in every node would make
the graph unmanageable in RAM. Nodes hold only a compact embedding vector
and a short summary. All actual content is referenced via `ref_uri`.

---

## 3. Two-Axis Node Characterization (Eisenhower)

```
IMPORTANCE [0,1] ──────────────────────────────────►
                  0.0      0.35     0.6      1.0
                  weak   normal     key   critical

URGENCY [0,1]  ─►
  ┌────────────────────────────┬────────────────────────────┐
  │ HIGH urgency (≥ theta_u)   │ HIGH urgency (≥ theta_u)   │
  │ LOW importance              │ HIGH importance            │
  │ "Urgent, not important"    │ "Urgent & important"       │
  │ → PINNED to SHORT          │ → PINNED to SHORT          │
  │   (interrupting tasks)     │   (critical deadline)      │
  ├────────────────────────────┼────────────────────────────┤
  │ LOW urgency                │ LOW urgency                │
  │ LOW importance              │ HIGH importance (≥ floor) │
  │ "Neither"                  │ "Important, not urgent"    │
  │ → ages normally to LONG    │ → floor at MEDIUM, no LONG │
  │   (trivia, ephemera)       │   (architecture decisions) │
  └────────────────────────────┴────────────────────────────┘
```

**Continuity principle:** Both axes are continuous floats in [0, 1].
No discretization, no enum cliffs. The threshold `theta_urgent` (default 0.7)
and `importance_medium_floor` (default 0.6) are tunable config parameters.

---

## 4. Analytical Functions

### 4.1 Ebbinghaus Strength Decay

```
s_eff(node, t) = s_base × exp(−λ × max(0, t − t_accessed))
```

- `λ = decay_lambda` (default 1e-5 s⁻¹ → ~27.7h half-life)
- Decay measured from **last access** (`t_accessed`)
- Reinforcement resets `t_accessed` and raises `s_base`

### 4.2 Urgency Decay

```
u_eff(node, t) = urgency × exp(−μ × max(0, t − t_creation))
```

- `μ = urgency_lambda` (default 5e-5 s⁻¹ → 5× faster than strength)
- Decay measured from **creation time** (`node.t`) — NOT `t_accessed`
- WHY: urgency is a deadline clock, not a rehearsal counter

### 4.3 Tier Function (4-quadrant)

```python
def tier_fn(node, t, cfg):
    u_eff = urgency_effective(node, t, cfg)
    if u_eff >= cfg.theta_urgent:          # ① urgency pin
        return Tier.SHORT

    s_eff = strength_effective(node, t, cfg)
    denom = 1 + α·s_eff + β·importance    # ② slow effective age
    ea    = (t - node.t) / max(denom, ε)

    if ea < T1:   raw = SHORT             # ③ raw tier
    elif ea < T2: raw = MEDIUM
    else:         raw = LONG

    if raw == LONG and importance ≥ floor: # ④ importance floor
        return MEDIUM
    return raw
```

**Configuration knobs:**
| Parameter | Default | Role |
|-----------|---------|------|
| `T1` | 86 400 s (24h) | SHORT → MEDIUM boundary |
| `T2` | 604 800 s (7d) | MEDIUM → LONG boundary |
| `alpha_tier` | 0.8 | Strength weight in denominator |
| `beta_tier` | 0.3 | Importance weight in denominator |
| `theta_urgent` | 0.7 | Urgency pin threshold |
| `importance_medium_floor` | 0.6 | Importance floor to prevent LONG |

---

## 5. Three-Tier Architecture

```
                    ┌─────────────────────────────────────────┐
    AWAKE INGEST ──►│  SHORT (working memory)                 │
                    │  HNSW index — O(log N) ANN              │
                    │  Frequent access, recent context        │
                    └──────────┬──────────────────────────────┘
                               │ sleep_diffuse (SLEEP edges)
                               ▼
                    ┌─────────────────────────────────────────┐
                    │  MEDIUM (episodic buffer)               │
                    │  HNSW index — moderate access           │
                    │  Cross-tier links, sleep promotion      │
                    └──────────┬──────────────────────────────┘
                               │ sleep_diffuse (when idle ≥ tau_long)
                               ▼
                    ┌─────────────────────────────────────────┐
                    │  LONG (semantic store)                  │
                    │  HNSW index — stable, compressed        │
                    │  Greedy cluster fusion                  │
                    └──────────┬──────────────────────────────┘
                               │ sleep_promote (evoked centroid)
                               └──────────────►  SHORT / MEDIUM
```

---

## 6. Graph Model

```
Node V ──DIRECT──► Node V'     (AWAKE: newer → older, cos ≥ theta_min)
Node V ──SLEEP───► Node M      (SLEEP: SHORT → MEDIUM/LONG diffusion)
Node F ──CONSOL──► Node ext    (CONSOLIDATED: fused node → neighbours)
```

**Edge fields:** `(from_id, to_id, cos_sim ∈ [0,1], delta_t ≥ 0, edge_type)`
**Direction:** newer node → older node (encodes temporal precedence)

---

## 7. AWAKE Mode (ingest_awake)

```
Query text / multimodal content
          │
          ▼ embed()
    v ∈ float32[d] (unit-norm)
          │
          ▼ query SHORT HNSW (k_awake neighbours, BEFORE insert)
    candidates = [(node_id, cos_sim), ...]
          │
          ▼ filter cos ≥ theta_min
    for each candidate:
        create DIRECT edge (newer → older)
          │
          ▼
    add Node to graph + SHORT HNSW index
    return UUID
```

**WHY only SHORT at ingest:** Cross-tier links are expensive and
semantically wrong during active sessions. They are established during SLEEP.

---

## 8. SLEEP Mode (sleep_phase)

Triggered by `SleepDaemon` after `idle_ttl` seconds of no ingestion.

```
sleep_phase()
├── 1. _decay_pass()
│       Materialise Ebbinghaus decay: s_eff → s_base
│       Reset t_accessed → now
│       Trigger tier updates (_update_tier)
│
├── 2. sleep_diffuse(idle_duration)
│       For each undiffused SHORT node n:
│         Query MEDIUM HNSW (k_sleep) → SLEEP edges if cos ≥ theta_sleep
│         If idle ≥ tau_long:
│           Query LONG HNSW (k_sleep) → SLEEP edges
│         Mark n.diffused = True
│
├── 3. sleep_promote()
│       evoked = nodes active since last_sleep_end
│       centroid = strength-weighted mean of evoked.v (normalised)
│       Query LONG HNSW with centroid (k_promote)
│         → reinforce matching LONG nodes
│         → add SLEEP edge: most-similar-evoked → LONG node
│       Repeat for MEDIUM
│
└── 4. sleep_compress()
        Trigger: LONG tier size ≥ long_compress_fraction × max_nodes
        clusters = greedy_cluster(LONG, theta_compress)
        For each cluster of size ≥ 2:
          v_f = strength-weighted centroid
          fused = Node(v_f, max(importance), urgency=0.0, ...)
          Remove members from graph + LONG HNSW
          Add fused node (consolidated=True, sources=[...])
          Reconnect external neighbours via CONSOLIDATED edges
```

---

## 9. Retrieval (5-Component Scoring)

```
score(n) = α · cos(q, n.v)           [direct semantic similarity]
         + β · traversal(n)           [1-hop graph bonus]
         + γ · strength_effective(n)  [recency/rehearsal]
         + δ · n.importance           [metadata importance ∈ [0,1]]
         + ε · urgency_effective(n)   [time-pressure bonus]
```

**Default weights (sum = 1.0):**
| Weight | Value | Signal |
|--------|-------|--------|
| α `alpha_score` | 0.45 | Direct cosine similarity |
| β `beta_score` | 0.20 | Graph traversal (1-hop) |
| γ `gamma_score` | 0.15 | Effective strength (Ebbinghaus) |
| δ `delta_score` | 0.12 | Importance [0,1] |
| ε `epsilon_score` | 0.08 | Urgency_effective [0,1] |

**Algorithm:**
```
1. q = embed(query_text)
2. direct = SHORT HNSW query(q, k) → seed candidates
3. High-importance sweep: MEDIUM + LONG HNSW query(q, k//2)
   keep nodes with importance ≥ importance_medium_floor
4. 1-hop traversal from top-5 seeds:
   traversal[nb] += cos(q, seed.v) × edge.cos_sim × s_eff(nb)
5. Score all candidates (5-component formula)
6. Sort descending, take top-k
7. Reinforce all returned nodes
8. Return list[RetrievalResult]
```

---

## 10. Data Structures & Complexity

```
Graph
├── V  : dict[UUID, Node]           O(1) node lookup
├── E  : dict[UUID, list[Edge]]     O(1) forward adjacency
└── E_r: dict[UUID, list[Edge]]     O(1) reverse adjacency (for compress)

TieredIndex
├── SHORT  : hnswlib.Index (M=16, ef_construction=200)  O(log N) add/query
├── MEDIUM : hnswlib.Index                              O(log N)
└── LONG   : hnswlib.Index                              O(log N)

Tier sets (for fast nodes_in_tier):
├── short_ids  : set[UUID]
├── medium_ids : set[UUID]
└── long_ids   : set[UUID]
```

| Operation | Complexity |
|-----------|-----------|
| ingest_awake | O(log N) ANN + O(k) edges |
| retrieve | O(log N) ANN × 3 tiers + O(k·deg) traversal |
| sleep_diffuse | O(N_short · log N) |
| sleep_promote | O(log N_long) |
| sleep_compress | O(N_long · k_query) greedy |

---

## 11. Persistence Format (JSON)

```json
{
  "version": "0.1.0",
  "saved_at": 1742000000.0,
  "last_sleep_end": 1741990000.0,
  "nodes": [
    {
      "id": "uuid4",
      "summary": "short text",
      "ref_uri": "/path/or/url",
      "ref_type": "text|image|audio|video|url",
      "importance": 0.7,
      "urgency": 0.0,
      "s_base": 0.5,
      "t": 1741900000.0,
      "t_accessed": 1741980000.0,
      "tier": 0,
      "diffused": false,
      "consolidated": false,
      "sources": [],
      "v": [0.12, -0.34, ...]
    }
  ],
  "edges": [
    {
      "from_id": "uuid4",
      "to_id": "uuid4",
      "cos_sim": 0.85,
      "delta_t": 120.0,
      "edge_type": 0
    }
  ]
}
```

**WHY JSON over pickle/sqlite:** human-readable, language-agnostic,
forward-compatible (new fields with defaults don't break old files).

---

## 12. Configuration Reference (LiveConfig)

All parameters are immutable after construction (frozen dataclass).

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| Embedding | `d` | 384 | Vector dimension (must match model) |
| | `model_name` | `BAAI/bge-small-en-v1.5` | fastembed model |
| Tier | `T1` | 86 400 s | SHORT→MEDIUM boundary |
| | `T2` | 604 800 s | MEDIUM→LONG boundary |
| | `alpha_tier` | 0.8 | Strength weight in tier denominator |
| | `beta_tier` | 0.3 | Importance weight in tier denominator |
| Urgency | `urgency_lambda` | 5e-5 | Urgency decay rate (s⁻¹) |
| | `theta_urgent` | 0.7 | Urgency pin threshold → SHORT |
| | `importance_medium_floor` | 0.6 | Min importance to avoid LONG |
| ANN | `k_awake` | 10 | Neighbours at ingest |
| | `k_sleep` | 20 | Neighbours at sleep diffuse |
| | `k_promote` | 15 | Neighbours at sleep promote |
| Edges | `theta_min` | 0.60 | Min cos for DIRECT edges |
| | `theta_sleep` | 0.75 | Min cos for SLEEP edges |
| | `theta_promote` | 0.70 | Min cos for promote edges |
| | `theta_compress` | 0.90 | Min cos for cluster fusion |
| Decay | `decay_lambda` | 1e-5 | Strength decay rate (s⁻¹) |
| | `delta_reinforce` | 0.10 | Strength boost per reinforce |
| Daemon | `tau_long` | 1 800 s | Idle → allow SHORT→LONG diffuse |
| | `idle_ttl` | 300 s | Idle → trigger sleep phase |
| | `daemon_check_interval` | 30 s | Daemon poll frequency |
| Compress | `max_nodes` | 10 000 | Hard node ceiling |
| | `long_compress_fraction` | 0.7 | LONG fraction to trigger compress |
| Retrieval | `alpha_score` | 0.45 | Cosine similarity weight |
| | `beta_score` | 0.20 | Graph traversal weight |
| | `gamma_score` | 0.15 | Strength weight |
| | `delta_score` | 0.12 | Importance weight |
| | `epsilon_score` | 0.08 | Urgency weight |
| | `s_base_init` | 0.5 | Initial strength for new nodes |
| HNSW | `hnsw_M` | 16 | Bi-directional links per node |
| | `hnsw_ef_construction` | 200 | Build quality |
| | `hnsw_ef_search` | 50 | Query recall/speed tradeoff |
| | `hnsw_max_elements` | 50 000 | Pre-allocated HNSW capacity |

---

## 13. Module Map

```
src/livemem/
├── __init__.py       Public API surface
├── config.py         LiveConfig frozen dataclass (all params)
├── types.py          Node, Edge, RetrievalResult, Tier, EdgeType, RefType
│                     + strength_effective(), urgency_effective(), tier_fn()
├── graph.py          Graph class: V, E, E_r, tier sets
├── index.py          TieredIndex: 3× hnswlib HNSW wrappers
├── embedder.py       BaseEmbedder, MockEmbedder, RealEmbedder (fastembed)
├── memory.py         LiveMem orchestrator: ingest_awake, sleep_*, retrieve
├── persistence.py    JSON save/load with forward-compatible schema
├── daemon.py         SleepDaemon (asyncio): idle detection + sleep_phase
└── cli.py            Typer + Rich CLI: ingest, retrieve, sleep, status, demo

tests/
├── conftest.py       Fixtures: small_config, mock_embedder, fresh_mem, make_node
├── test_types.py     Node, Edge, strength_effective, urgency_effective, tier_fn
├── test_memory.py    ingest_awake, retrieve, _reinforce, _update_tier
├── test_sleep.py     sleep_diffuse, sleep_promote, sleep_compress, _decay_pass
├── test_retrieval.py scoring, traversal, importance sweep, urgency scoring
└── test_persistence.py JSON round-trip, edge preservation, urgency round-trip
```

---

*Last updated: 2026-03-19 — importance→float[0,1] + urgency field + 4-quadrant tier_fn*
