# CHANGELOG

## [0.3.2] — 2026-03-19

### Added
- [x] `src/livemem/config.yaml`: structured external configuration source of truth for embeddings, tiers, retrieval, daemon, compression, and index settings
- [x] `LiveConfig.from_yaml()` and `LiveConfig.get("a.b.c")`: YAML-backed config loading with dotted-path access
- [x] `src/livemem/embeddings/`: dedicated embedding subsystem package split into `base.py`, `mock.py`, `fastembed_text.py`, `fastembed_cross_encoder.py`, and `factory.py`
- [x] `tests/test_config.py`: config loading and implementation-selection coverage

### Changed
- [x] Embedder/reranker selection is now config-driven via `config.yaml` instead of being hardwired in one module
- [x] `src/livemem/embedder.py` is now a thin compatibility facade instead of the primary implementation surface

## [0.3.1] — 2026-03-19

### Added
- [x] `LiveMem.ingest_awake_batch / ingest_awake_batch_async`: typed batch-ingest path reusing the same awake semantics as single-item ingest
- [x] `POST /ingest/batch` and `livemem ingest-batch`: public batch-ingest surfaces for API and CLI workflows
- [x] `compression_stats` in `status()` / `/status`: cumulative runs, fused clusters, removed nodes, created nodes, saved nodes, last run timestamp

### Fixed
- [x] `retrieve()` is now protected by the engine-level `RLock`, so reinforcement/tier updates cannot race with ingest or sleep outside the API wrapper
- [x] JSON persistence now round-trips compression stats alongside nodes, edges, and `last_sleep_end`

## [0.3.0] — 2026-03-19

### Added
- [x] `livemem.api`: FastAPI microservice exposing `/health`, `/status`, `/ingest`, `/retrieve`, and `/sleep`
- [x] `LiveMem.ingest_awake_async / retrieve_async / sleep_phase_async / status_async`: async wrappers using `asyncio.to_thread`
- [x] `livemem serve`: CLI command to run the REST API locally via uvicorn
- [x] `tests/test_api.py`: HTTP contract tests covering health, ingest, retrieve, sleep, and persisted reload
- [x] `tests/test_urgency_retrieval.py`: dedicated retrieval tests for urgency sweep and cross-encoder reranking

### Fixed
- [x] `retrieve()` urgency guarantee is now real, not just candidate-pool admission: urgent nodes are explicitly preserved in final top-k selection
- [x] `retrieve()` ranking remains score-descending after urgent-node forcing via minimal score lift
- [x] `retrieve()` result materialisation no longer exits after the first row when building `RetrievalResult` objects

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
