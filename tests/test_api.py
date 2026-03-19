"""
test_api.py — HTTP contract tests for the LiveMem FastAPI surface.

WHY these tests matter:
    The API is the first non-CLI integration surface for livemem. It must
    prove that the in-process memory engine can be exposed safely as a local
    microservice with persisted state and deterministic mock embeddings.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from livemem.api import create_app


def _build_client(state_path: Path) -> TestClient:
    app = create_app(mock=True, state_path=state_path)
    return TestClient(app)


def test_health_starts_clean(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with _build_client(state_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["mock"] is True
    assert body["total_nodes"] == 0


def test_ingest_updates_status_and_persists(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with _build_client(state_path) as client:
        ingest = client.post(
            "/ingest",
            json={
                "summary": "DEADLINE: submit the architecture report",
                "ref_uri": "file:///tmp/report.md",
                "ref_type": "text",
                "importance": 0.8,
                "urgency": 0.95,
            },
        )
        status = client.get("/status")

    assert ingest.status_code == 200
    assert status.status_code == 200
    assert state_path.exists()
    ingest_body = ingest.json()
    status_body = status.json()
    assert ingest_body["summary"].startswith("DEADLINE:")
    assert ingest_body["tier"] == "SHORT"
    assert status_body["total_nodes"] == 1
    assert status_body["tier_counts"]["SHORT"] == 1


def test_retrieve_returns_ingested_memory(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with _build_client(state_path) as client:
        client.post(
            "/ingest",
            json={
                "summary": "Coffee improves perceived productivity in the morning",
                "importance": 0.3,
                "urgency": 0.0,
            },
        )
        client.post(
            "/ingest",
            json={
                "summary": "DEADLINE: submit report by 5pm",
                "importance": 0.7,
                "urgency": 0.95,
            },
        )
        response = client.post(
            "/retrieve",
            json={"query": "coffee weather sky today", "k": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert any("DEADLINE" in item["summary"] for item in body["results"])


def test_sleep_endpoint_returns_updated_status(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with _build_client(state_path) as client:
        client.post(
            "/ingest",
            json={"summary": "One fact", "importance": 0.5, "urgency": 0.0},
        )
        response = client.post("/sleep", json={"idle_duration": 600.0})

    assert response.status_code == 200
    body = response.json()
    assert body["total_nodes"] == 1
    assert body["last_sleep_end"] > 0.0


def test_state_reloads_across_app_instances(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with _build_client(state_path) as client:
        client.post(
            "/ingest",
            json={
                "summary": "The Eiffel Tower is in Paris",
                "importance": 0.5,
                "urgency": 0.0,
            },
        )

    with _build_client(state_path) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["total_nodes"] == 1
