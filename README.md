# LiveMem — Brain-Inspired Tiered Graph Memory

> **0 Trust - 100% Control | 0 Magic - 100% Transparency | 0 Hardcoding - 100% Flexibility**

A Python prototype of a living, brain-inspired memory system. Nodes are semantic units (embedding + summary + content reference). Edges are directed associations. The system operates in two modes: **AWAKE** (fast ingestion) and **SLEEP** (background consolidation).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     AWAKE SESSION                        │
│                                                         │
│  ingest_awake(...) / ingest_awake_batch([...])          │
│      │                                                  │
│      ▼                                                  │
│  embed(summary) → unit-norm vector v                    │
│      │                                                  │
│      ▼                                                  │
│  ANN query SHORT tier (k_awake neighbours)              │
│      │                                                  │
│      ▼                                                  │
│  Create DIRECT edges (newer → older, cos ≥ theta_min)  │
│      │                                                  │
│      ▼                                                  │
│  Add Node to Graph + SHORT index                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     SLEEP PHASE                          │
│  (triggered by SleepDaemon after idle_ttl seconds)      │
│                                                         │
│  1. _decay_pass  — materialise Ebbinghaus decay         │
│  2. sleep_diffuse — SHORT→MEDIUM SLEEP edges            │
│         if idle ≥ tau_long: SHORT→LONG SLEEP edges      │
│  3. sleep_promote — reinforce LONG/MEDIUM via centroid  │
│  4. sleep_compress — fuse similar LONG nodes            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   MEMORY TIERS                           │
│                                                         │
│  SHORT  (< T1=24h effective age)  — working memory      │
│    │  fast ANN, direct ingest, DIRECT edges             │
│    │                                                    │
│    ▼  sleep_diffuse / time+decay                        │
│  MEDIUM (< T2=7d effective age)  — episodic buffer      │
│    │  cross-tier SLEEP edges, promote targets           │
│    │                                                    │
│    ▼  sleep_diffuse / time+decay                        │
│  LONG  (≥ T2)                   — semantic store        │
│       stable, compressible, CAPITAL floor               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   NODE DESIGN                            │
│                                                         │
│  Node {                                                 │
│    id: UUID4                                            │
│    v: float32[384]   ← unit-norm embedding              │
│    summary: str      ← ≤200 chars                       │
│    ref_uri: str|None ← path or URL to actual content    │
│    ref_type: str     ← text|image|audio|video|url       │
│    importance: float    ← continuous [0,1]              │
│    urgency: float       ← continuous [0,1]              │
│    s_base: float     ← current base strength [0,1]      │
│    t: float          ← creation unix timestamp          │
│    t_accessed: float ← last access unix timestamp       │
│    tier: Tier        ← dynamic (SHORT/MEDIUM/LONG)      │
│    diffused: bool    ← processed by sleep_diffuse?      │
│    consolidated: bool← created by compression?          │
│    sources: list[UUID] ← merged node IDs if consolidated│
│  }                                                      │
│                                                         │
│  Nodes NEVER store full content — only ref_uri + summary│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               RETRIEVAL SCORING                          │
│                                                         │
│  score(n) = α·cos(q,v)   direct semantic match          │
│           + β·traversal  1-hop graph context            │
│           + γ·s_eff      recency/reinforcement          │
│           + δ·imp        importance bonus               │
│           + ε·u_eff      urgency bonus                  │
│                                                         │
│  α=0.45, β=0.20, γ=0.15, δ=0.12, ε=0.08                │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# From project root (editable install for development):
uv sync --dev
uv tool install --editable .

# Or run directly:
uv run livemem --help
```

---

## CLI Commands

### `ingest`
```bash
livemem ingest "Python uses GIL for thread safety" --importance 0.7
livemem ingest "Photo of the lab" --ref-uri /images/lab.jpg --ref-type image
livemem ingest "Podcast episode" --ref-uri /audio/ep1.mp3 --ref-type audio --urgency 0.9 --mock
```

### `retrieve`
```bash
livemem retrieve "thread safety concurrency" --k 5
livemem retrieve "machine learning" --k 10 --mock
```

### `sleep`
```bash
livemem sleep                   # Manual sleep phase
livemem sleep --idle 1800       # Simulate 30min idle (enables SHORT→LONG diffusion)
```

### `status`
```bash
livemem status
```

### `demo`
```bash
livemem demo --mock   # 25 diverse facts, no model download
livemem demo --real   # Uses fastembed BAAI/bge-small-en-v1.5
```

### `serve`
```bash
livemem serve --mock
livemem serve --host 0.0.0.0 --port 8080 --state-path /tmp/livemem.json
```

### `ingest-batch`
```bash
livemem ingest-batch ./batch.json --mock
```

---

## Python API

```python
from livemem import LiveMem, LiveConfig, save, load
from livemem.types import IngestInput

# Create with defaults (uses fastembed for embeddings):
mem = LiveMem()

# Or with mock embedder (no download, deterministic):
mem = LiveMem(mock=True)

# Ingest
node_id = mem.ingest_awake(
    summary="The Eiffel Tower was built in 1889.",
    ref_uri="https://en.wikipedia.org/wiki/Eiffel_Tower",
    ref_type="url",
    importance=0.7,
    urgency=0.2,
)

# Batch ingest
node_ids = mem.ingest_awake_batch(
    [
        IngestInput(summary="Short fact one", importance=0.4),
        IngestInput(summary="DEADLINE: send the report", urgency=0.95),
    ]
)

# Retrieve
results = mem.retrieve("Paris landmark architecture", k=5)
for r in results:
    print(f"[{r.score:.3f}] [{r.tier.name}] {r.summary}")

# Sleep (consolidation)
mem.sleep_phase(idle_duration=300.0)

# Persist
save(mem, "~/.livemem/state.json")
mem2 = load("~/.livemem/state.json", mock=True)

# Daemon (async)
from livemem import SleepDaemon
daemon = SleepDaemon(mem, mem.cfg)
await daemon.start()
# ... application runs ...
await daemon.stop()

# Async wrappers (for service integrations)
node_id = await mem.ingest_awake_async("Urgent task", urgency=0.95)
results = await mem.retrieve_async("urgent", k=5)
await mem.sleep_phase_async(idle_duration=600.0)
```

---

## REST API

Run locally:

```bash
livemem serve --mock
```

Endpoints:

```text
GET  /health
GET  /status
POST /ingest
POST /ingest/batch
POST /retrieve
POST /sleep
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'content-type: application/json' \
  -d '{
    "summary": "DEADLINE: submit report by 5pm",
    "ref_type": "text",
    "importance": 0.7,
    "urgency": 0.95
  }'

curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'content-type: application/json' \
  -d '{"query": "coffee weather sky today", "k": 5}'

curl -X POST http://127.0.0.1:8000/ingest/batch \
  -H 'content-type: application/json' \
  -d '{
    "items": [
      {"summary": "Coffee improves perceived productivity", "importance": 0.3},
      {"summary": "DEADLINE: submit report by 5pm", "importance": 0.7, "urgency": 0.95}
    ]
  }'
```

The API persists state to `~/.livemem/state.json` by default, exactly like the CLI.
`/status` also exposes cumulative compression stats (`runs`, `clusters_fused`,
`nodes_removed`, `nodes_created`, `nodes_saved`, `last_run_at`).

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | 384 | Embedding dimension |
| `model_name` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `T1` | 86400 (24h) | SHORT→MEDIUM boundary (seconds) |
| `T2` | 604800 (7d) | MEDIUM→LONG boundary (seconds) |
| `alpha_tier` | 0.8 | Strength influence on effective age |
| `beta_tier` | 0.3 | Importance influence on effective age |
| `k_awake` | 10 | ANN neighbours at ingest |
| `k_sleep` | 20 | ANN neighbours during diffusion |
| `theta_min` | 0.60 | Min cosine for any edge |
| `theta_sleep` | 0.75 | Min cosine for SLEEP edges |
| `theta_compress` | 0.90 | Min cosine for cluster fusion |
| `decay_lambda` | 5e-6 | Ebbinghaus decay rate (per second) |
| `delta_reinforce` | 0.10 | Strength boost per reinforcement |
| `tau_long` | 1800 (30min) | Idle threshold for SHORT→LONG diffusion |
| `idle_ttl` | 300 (5min) | Daemon sleep trigger threshold |
| `max_nodes` | 10000 | Total node limit |

---

## Tests

```bash
uv run pytest tests/ -v --cov=livemem
```

Test coverage: 10 test files, 157 tests covering types, graph, index, memory, retrieval, urgency sweep, sleep, persistence, daemon, and REST API.

---

## File Structure

```
src/livemem/
├── __init__.py      — public API exports
├── api.py           — FastAPI microservice surface
├── config.py        — LiveConfig frozen dataclass
├── types.py         — Node, Edge, Tier, strength_effective, urgency_effective, tier_fn
├── graph.py         — directed graph with tier-bucketed SortedLists
├── index.py         — hnswlib HNSW per-tier ANN index
├── embedder.py      — MockEmbedder + RealEmbedder + CrossEncoderReranker
├── memory.py        — LiveMem orchestrator (ingest, sleep, retrieve, async wrappers)
├── daemon.py        — asyncio SleepDaemon
├── persistence.py   — JSON save/load
└── cli.py           — Typer CLI
```
