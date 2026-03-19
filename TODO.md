# TODO

## v0.4.0 Candidates (short-term)

- [ ] **Batch ingest file adapters**: CSV/JSONL/directory adapters on top of `ingest_awake_batch`
- [ ] **Compression observability**: expose per-run cluster sizes and duration, not only cumulative stats
- [ ] **Thread-safety stress tests**: concurrent ingest/retrieve/sleep smoke tests for the engine

## Future Work (long-term)

- [ ] **pgvector backend**: replace in-memory hnswlib with persistent pgvector index for multi-process access
- [ ] **sqlite-vec backend**: lightweight alternative to pgvector for single-machine deployments
- [ ] **Multi-modal embeddings**: CLIP for images, Whisper transcripts for audio/video
- [ ] **Export**: JSON-LD / RDF export for integration with knowledge graphs
- [ ] **Versioned snapshots**: periodic auto-save with rotation (keep last N snapshots)
- [ ] **Metrics**: Prometheus counters for ingest rate, sleep duration, cluster sizes
- [ ] **Web UI**: minimal dashboard showing tier counts, recent ingestions, retrieval queries
- [ ] **Pluggable decay functions**: support power-law decay in addition to Ebbinghaus exponential
