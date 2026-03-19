# CHANGELOG

## [0.1.0] — 2026-03-19

### Added
- [x] `LiveConfig` frozen dataclass with 30+ tunable parameters (tier boundaries, ANN, decay, retrieval weights)
- [x] `Node` dataclass: unit-norm embedding + summary + ref_uri + ref_type + importance + strength + tier
- [x] `Edge` dataclass: directed association with cos_sim, delta_t, EdgeType (DIRECT/SLEEP/CONSOLIDATED)
- [x] `Graph`: directed graph with tier-bucketed SortedLists, cascading deletes, O(1) node lookup
- [x] `TieredIndex`: 3× hnswlib HNSW indices (one per tier), soft-delete, tier isolation
- [x] `MockEmbedder`: deterministic unit vectors via SHA256 seed (no model download)
- [x] `RealEmbedder`: fastembed TextEmbedding (lazy load, batch support)
- [x] `LiveMem.ingest_awake`: embed + ANN query SHORT + DIRECT edges + add node
- [x] `LiveMem.sleep_diffuse`: SHORT→MEDIUM/LONG SLEEP edges based on idle duration
- [x] `LiveMem.sleep_promote`: reinforce LONG/MEDIUM nodes via evoked centroid
- [x] `LiveMem.sleep_compress`: greedy cluster fusion for LONG tier (>= threshold)
- [x] `LiveMem.retrieve`: multi-signal scoring (cosine + traversal + strength + importance)
- [x] `LiveMem._reinforce`: Ebbinghaus spacing-effect strength boost
- [x] `LiveMem._decay_pass`: materialise exponential decay before sleep
- [x] `SleepDaemon`: asyncio background task, idle-triggered sleep, notify_activity
- [x] `save` / `load`: JSON persistence with full round-trip fidelity
- [x] `livemem` CLI: ingest, retrieve, sleep, status, demo commands
- [x] Full test suite: 8 files, ~100 tests (types, graph, index, memory, sleep, retrieval, persistence, daemon)
