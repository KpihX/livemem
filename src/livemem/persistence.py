"""
persistence.py — JSON serialisation and deserialisation of LiveMem state.

WHY JSON (not pickle):
    pickle is Python-specific, version-sensitive, and insecure (arbitrary
    code execution on load). JSON is human-readable, language-agnostic,
    and forward-compatible: adding new fields with defaults doesn't break
    old files.

WHY not a database (sqlite, redis):
    For the prototype scale (<10 k nodes), a single JSON file is simpler
    and easier to inspect/debug. A pgvector or sqlite-vec backend is
    listed in TODO.md as a future upgrade.

Format
------
{
  "version": "0.1.0",
  "saved_at": <unix timestamp>,
  "last_sleep_end": <float>,
  "nodes": [ { ...node fields... }, ... ],
  "edges": [ { ...edge fields... }, ... ]
}

Vectors are stored as list[float] (JSON native). On load they are
reconstructed as np.float32 arrays and re-normalised to guard against
floating-point drift during serialisation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embedder import BaseEmbedder, make_embedder
from livemem.graph import Graph
from livemem.index import TieredIndex
from livemem.memory import LiveMem
from livemem.types import Edge, EdgeType, Importance, Node, Tier

_FORMAT_VERSION = "0.1.0"


def _node_to_dict(node: Node) -> dict:
    """Serialise a Node to a JSON-compatible dict.

    WHY explicit field listing:
        Using dataclasses.asdict() would recursively convert numpy arrays
        to nested lists in an unpredictable way. Explicit conversion gives
        full control over the format and makes schema evolution easier.
    """
    return {
        "id": node.id,
        "summary": node.summary,
        "ref_uri": node.ref_uri,
        "ref_type": node.ref_type,
        "importance": int(node.importance),
        "s_base": float(node.s_base),
        "t": float(node.t),
        "t_accessed": float(node.t_accessed),
        "tier": int(node.tier),
        "diffused": node.diffused,
        "consolidated": node.consolidated,
        "sources": list(node.sources),
        # Vector stored as list[float]; np.float32 → Python float via tolist().
        "v": node.v.tolist(),
    }


def _node_from_dict(d: dict) -> Node:
    """Reconstruct a Node from a serialised dict.

    WHY re-normalise v:
        JSON float precision (15–17 significant digits) can introduce
        small deviations from unit norm. Re-normalising here ensures the
        invariant holds after round-trip.
    """
    v_raw = np.array(d["v"], dtype=np.float32)
    norm = np.linalg.norm(v_raw)
    if norm > 1e-12:
        v_raw /= norm

    return Node(
        id=d["id"],
        summary=d["summary"],
        ref_uri=d.get("ref_uri"),
        ref_type=d.get("ref_type", "text"),
        importance=Importance(d["importance"]),
        s_base=float(d["s_base"]),
        t=float(d["t"]),
        t_accessed=float(d["t_accessed"]),
        tier=Tier(d["tier"]),
        diffused=bool(d.get("diffused", False)),
        consolidated=bool(d.get("consolidated", False)),
        sources=list(d.get("sources", [])),
        v=v_raw,
    )


def _edge_to_dict(edge: Edge) -> dict:
    return {
        "from_id": edge.from_id,
        "to_id": edge.to_id,
        "cos_sim": float(edge.cos_sim),
        "delta_t": float(edge.delta_t),
        "edge_type": int(edge.edge_type),
    }


def _edge_from_dict(d: dict) -> Edge:
    return Edge(
        from_id=d["from_id"],
        to_id=d["to_id"],
        cos_sim=float(d["cos_sim"]),
        delta_t=float(d["delta_t"]),
        edge_type=EdgeType(d.get("edge_type", 0)),
    )


def save(mem: LiveMem, path: Path | str) -> None:
    """Serialise all LiveMem state to a JSON file.

    Saves all nodes, all edges (from E dict, which is the authoritative
    forward-edge store — no duplicates), config metadata, and last_sleep_end.

    Parameters
    ----------
    mem  : LiveMem — the memory instance to serialise.
    path : Path | str — destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_list = [_node_to_dict(n) for n in mem.graph.V.values()]

    # Collect edges from E (forward index) to avoid duplicates.
    edges_list: list[dict] = []
    for edge_group in mem.graph.E.values():
        for edge in edge_group:
            edges_list.append(_edge_to_dict(edge))

    payload = {
        "version": _FORMAT_VERSION,
        "saved_at": time.time(),
        "last_sleep_end": mem.last_sleep_end,
        "nodes": nodes_list,
        "edges": edges_list,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load(
    path: Path | str,
    cfg: LiveConfig | None = None,
    embedder: BaseEmbedder | None = None,
    mock: bool = False,
) -> LiveMem:
    """Deserialise a LiveMem instance from a JSON file.

    Parameters
    ----------
    path     : Path | str — source file (must exist).
    cfg      : LiveConfig — if None, uses DEFAULT_CONFIG.
    embedder : pre-built embedder (optional).
    mock     : if True and embedder is None, use MockEmbedder.

    Raises
    ------
    FileNotFoundError : if path does not exist.
    ValueError        : if JSON is malformed or missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LiveMem state file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    if "nodes" not in payload or "edges" not in payload:
        raise ValueError(f"Invalid LiveMem state file: missing 'nodes' or 'edges' key")

    if cfg is None:
        cfg = DEFAULT_CONFIG

    # Build empty LiveMem with the provided config.
    mem = LiveMem(cfg=cfg, embedder=embedder, mock=mock)
    mem.last_sleep_end = float(payload.get("last_sleep_end", 0.0))

    # Restore nodes.
    for nd in payload["nodes"]:
        try:
            node = _node_from_dict(nd)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Failed to deserialise node {nd.get('id', '?')}: {exc}") from exc
        # Add to graph (respecting the stored tier).
        mem.graph.add_node(node)
        mem.index.add(node.id, node.v, node.tier)

    # Restore edges.
    for ed in payload["edges"]:
        try:
            edge = _edge_from_dict(ed)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Failed to deserialise edge: {exc}") from exc
        # Only add if both endpoints exist.
        if edge.from_id in mem.graph and edge.to_id in mem.graph:
            mem.graph.add_edge_if_new(edge)

    return mem
