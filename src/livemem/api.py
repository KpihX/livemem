"""
api.py — FastAPI surface for LiveMem.

WHY a dedicated API module:
    LiveMem is a stateful in-process engine. Exposing it as a microservice
    makes it pluggable into agent pipelines, n8n flows, or any other local
    service without forcing direct Python imports. The API is deliberately
    thin: it wraps the existing LiveMem orchestration rather than duplicating
    memory logic.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.memory import LiveMem
from livemem.persistence import load, save
from livemem.types import IngestInput, RetrievalResult


class IngestRequest(BaseModel):
    """Request payload for awake ingestion."""

    summary: str = Field(..., min_length=1, description="Short summary to ingest.")
    ref_uri: str | None = Field(default=None, description="URI/path to the full source.")
    ref_type: str = Field(default="text", description="Reference type: text|image|audio|video|url.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)


class IngestResponse(BaseModel):
    """Response payload after ingestion."""

    node_id: str
    summary: str
    ref_uri: str | None
    ref_type: str
    importance: float
    urgency: float
    tier: str


class BatchIngestRequest(BaseModel):
    """Request payload for batch awake ingestion."""

    items: list[IngestRequest] = Field(..., min_length=1)


class BatchIngestResponse(BaseModel):
    """Response payload after batch ingestion."""

    items: list[IngestResponse]


class RetrieveRequest(BaseModel):
    """Request payload for retrieval."""

    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class RetrievalResultModel(BaseModel):
    """JSON-safe representation of RetrievalResult."""

    node_id: str
    score: float
    summary: str
    ref_uri: str | None
    ref_type: str
    tier: str
    importance: float
    urgency: float
    cos_direct: float

    @classmethod
    def from_domain(cls, result: RetrievalResult) -> "RetrievalResultModel":
        return cls(
            node_id=result.node_id,
            score=result.score,
            summary=result.summary,
            ref_uri=result.ref_uri,
            ref_type=result.ref_type,
            tier=result.tier.name,
            importance=result.importance,
            urgency=result.urgency,
            cos_direct=result.cos_direct,
        )


class RetrieveResponse(BaseModel):
    """Response payload for retrieval."""

    results: list[RetrievalResultModel]


class SleepRequest(BaseModel):
    """Request payload for a manual sleep cycle."""

    idle_duration: float = Field(default=0.0, ge=0.0)


class StatusResponse(BaseModel):
    """Current graph/index status."""

    model_config = ConfigDict(extra="allow")

    total_nodes: int
    total_edges: int
    tier_counts: dict[str, int]
    index_sizes: dict[str, int]
    compression_stats: dict[str, float | int]
    last_sleep_end: float


class HealthResponse(BaseModel):
    """Simple liveness payload."""

    ok: bool
    mock: bool
    model_name: str
    reranker_enabled: bool
    total_nodes: int


class LiveMemApiState:
    """Owns the process-local LiveMem instance and persisted state file.

    WHY serialise API mutations with one lock:
        ingest(), retrieve(), and sleep() all mutate memory state.
        retrieve() reinforces returned nodes, so even read-like queries are
        writes. A single async lock keeps the persisted JSON snapshot aligned
        with the in-memory graph and avoids concurrent save/load races.
    """

    def __init__(
        self,
        cfg: LiveConfig,
        *,
        mock: bool,
        state_path: Path,
    ) -> None:
        self.cfg = cfg
        self.mock = mock
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._op_lock = asyncio.Lock()
        self.mem = self._load_or_new()

    def _load_or_new(self) -> LiveMem:
        if self.state_path.exists():
            return load(self.state_path, mock=self.mock, cfg=self.cfg)
        return LiveMem(cfg=self.cfg, mock=self.mock)

    async def _persist(self) -> None:
        await asyncio.to_thread(save, self.mem, self.state_path)

    async def ingest(self, payload: IngestRequest) -> IngestResponse:
        async with self._op_lock:
            node_id = await self.mem.ingest_awake_async(
                payload.summary,
                payload.ref_uri,
                payload.ref_type,
                payload.importance,
                payload.urgency,
            )
            await self._persist()
            node = self.mem.graph.V[node_id]
            return IngestResponse(
                node_id=node.id,
                summary=node.summary,
                ref_uri=node.ref_uri,
                ref_type=node.ref_type,
                importance=node.importance,
                urgency=node.urgency,
                tier=node.tier.name,
            )

    async def ingest_batch(self, payload: BatchIngestRequest) -> BatchIngestResponse:
        async with self._op_lock:
            inputs = [
                IngestInput(
                    summary=item.summary,
                    ref_uri=item.ref_uri,
                    ref_type=item.ref_type,
                    importance=item.importance,
                    urgency=item.urgency,
                )
                for item in payload.items
            ]
            node_ids = await self.mem.ingest_awake_batch_async(inputs)
            await self._persist()
            responses: list[IngestResponse] = []
            for node_id in node_ids:
                node = self.mem.graph.V[node_id]
                responses.append(
                    IngestResponse(
                        node_id=node.id,
                        summary=node.summary,
                        ref_uri=node.ref_uri,
                        ref_type=node.ref_type,
                        importance=node.importance,
                        urgency=node.urgency,
                        tier=node.tier.name,
                    )
                )
            return BatchIngestResponse(items=responses)

    async def retrieve(self, payload: RetrieveRequest) -> RetrieveResponse:
        async with self._op_lock:
            results = await self.mem.retrieve_async(payload.query, payload.k)
            await self._persist()
            return RetrieveResponse(
                results=[RetrievalResultModel.from_domain(r) for r in results]
            )

    async def sleep(self, payload: SleepRequest) -> StatusResponse:
        async with self._op_lock:
            await self.mem.sleep_phase_async(payload.idle_duration)
            await self._persist()
            return StatusResponse.model_validate(await self.mem.status_async())

    async def status(self) -> StatusResponse:
        async with self._op_lock:
            return StatusResponse.model_validate(await self.mem.status_async())

    async def health(self) -> HealthResponse:
        status = await self.status()
        return HealthResponse(
            ok=True,
            mock=self.mock,
            model_name=self.cfg.model_name,
            reranker_enabled=self.cfg.reranker_enabled,
            total_nodes=status.total_nodes,
        )


def create_app(
    *,
    cfg: LiveConfig = DEFAULT_CONFIG,
    mock: bool = False,
    state_path: Path | str | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to one LiveMem instance."""
    resolved_state_path = (
        Path(state_path).expanduser()
        if state_path is not None
        else Path.home() / ".livemem" / "state.json"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.livemem = LiveMemApiState(
            cfg=cfg,
            mock=mock,
            state_path=resolved_state_path,
        )
        yield

    app = FastAPI(
        title="LiveMem API",
        version="0.3.1",
        summary="Brain-inspired tiered graph memory microservice.",
        lifespan=lifespan,
    )

    def api_state() -> LiveMemApiState:
        return app.state.livemem  # type: ignore[return-value]

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return await api_state().health()

    @app.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        return await api_state().status()

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(payload: IngestRequest) -> IngestResponse:
        return await api_state().ingest(payload)

    @app.post("/ingest/batch", response_model=BatchIngestResponse)
    async def ingest_batch(payload: BatchIngestRequest) -> BatchIngestResponse:
        return await api_state().ingest_batch(payload)

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
        return await api_state().retrieve(payload)

    @app.post("/sleep", response_model=StatusResponse)
    async def sleep(payload: SleepRequest) -> StatusResponse:
        return await api_state().sleep(payload)

    return app


def main() -> None:
    """Run the API with uvicorn for local development."""
    import uvicorn

    uvicorn.run(
        create_app(mock=False),
        host="127.0.0.1",
        port=8000,
    )
