# TODO

## v0.3.0 Candidates (short-term)

- [ ] **Proper test suite**: migrate 12 mini-tests to `src/livemem/tests/` with pytest + fixtures
- [ ] **Async ingest_awake**: non-blocking version using asyncio.to_thread (needed for REST API)
- [ ] **REST API**: FastAPI server exposing ingest/retrieve/sleep/status endpoints
- [ ] **Compression stats**: track cluster sizes, fused counts, and savings in `status()` output
- [ ] **Batch ingest**: `ingest_awake_batch` for efficient multi-document ingestion

## Future Work (long-term)

- [ ] **REST API**: FastAPI server exposing ingest/retrieve/sleep/status endpoints
- [ ] **pgvector backend**: replace in-memory hnswlib with persistent pgvector index for multi-process access
- [ ] **sqlite-vec backend**: lightweight alternative to pgvector for single-machine deployments
- [ ] **Compression stats**: track cluster sizes, fused counts, and savings in `status()` output
- [ ] **Multi-modal embeddings**: CLIP for images, Whisper transcripts for audio/video
- [ ] **Async ingest_awake**: non-blocking version using asyncio.to_thread for embedding
- [ ] **Export**: JSON-LD / RDF export for integration with knowledge graphs
- [ ] **Versioned snapshots**: periodic auto-save with rotation (keep last N snapshots)
- [ ] **Metrics**: Prometheus counters for ingest rate, sleep duration, cluster sizes
- [ ] **Web UI**: minimal dashboard showing tier counts, recent ingestions, retrieval queries
- [ ] **Pluggable decay functions**: support power-law decay in addition to Ebbinghaus exponential
- [ ] **Batch ingest**: `ingest_awake_batch` for efficient multi-document ingestion
