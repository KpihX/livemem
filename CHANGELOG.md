# CHANGELOG

## [0.2.0] — 2026-03-19

### Added
- [x] `CrossEncoderReranker`: two-stage retrieval — HNSW bi-encoder candidates → cross-encoder (fastembed TextCrossEncoder, lazy-load, ONNX)
- [x] `LiveConfig.reranker_enabled / reranker_model_name / reranker_k`: opt-in re-ranking (off by default, zero latency impact)
- [x] `LiveConfig`: model-swap documentation (bge-base 768d / bge-large 1024d swap pattern)
- [x] `RealEmbedder`: dimension-mismatch probe warning on model load

### Fixed
- [x] `retrieve()` urgency sweep: urgent nodes now guaranteed to surface via direct O(N_urgent) graph scan into `urgent_forced` set — bypasses cosine-rank seed limit entirely
- [x] `Edge.cos_sim`: clamped to [0, 1] before validation to prevent floating-point drift failures
- [x] `Node.s_base`: importance-proportional initialization (`lerp(s_base_init, 1.0, importance)`) — critical nodes start at full strength
- [x] `LiveConfig.decay_lambda`: tuned from 1e-5 → 5e-6 (half-life ~1.6 days, preventing premature decay during active sessions)

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
