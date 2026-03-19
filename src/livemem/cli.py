"""
cli.py — Rich + Typer CLI for LiveMem.

WHY a CLI:
    Provides a human-friendly interface for testing, demos, and interactive
    exploration of the memory system without writing Python code. State is
    persisted to ~/.livemem/state.json between invocations so the memory
    accumulates across CLI sessions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embedder import make_embedder
from livemem.memory import LiveMem
from livemem.persistence import load, save
from livemem.types import Importance, RefType, Tier

app = typer.Typer(
    name="livemem",
    help="Brain-inspired tiered graph memory: 3 tiers, HNSW ANN, sleep consolidation.",
    add_completion=False,
)
console = Console()

_STATE_PATH = Path.home() / ".livemem" / "state.json"


def _load_or_new(mock: bool) -> LiveMem:
    """Load existing state or create a fresh LiveMem instance."""
    if _STATE_PATH.exists():
        try:
            return load(_STATE_PATH, mock=mock)
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load state ({exc}). Starting fresh.[/]")
    return LiveMem(cfg=DEFAULT_CONFIG, mock=mock)


def _save_state(mem: LiveMem) -> None:
    """Persist state to disk."""
    try:
        save(mem, _STATE_PATH)
    except Exception as exc:
        console.print(f"[red]Error saving state: {exc}[/]")


def _importance_from_str(s: str) -> Importance:
    mapping = {
        "weak": Importance.WEAK,
        "normal": Importance.NORMAL,
        "key": Importance.KEY,
        "capital": Importance.CAPITAL,
    }
    return mapping.get(s.lower(), Importance.NORMAL)


def _tier_name(tier: Tier) -> str:
    colours = {Tier.SHORT: "green", Tier.MEDIUM: "yellow", Tier.LONG: "red"}
    return f"[{colours[tier]}]{tier.name}[/]"


def _importance_name(imp: Importance) -> str:
    colours = {
        Importance.WEAK: "dim",
        Importance.NORMAL: "white",
        Importance.KEY: "cyan",
        Importance.CAPITAL: "bold magenta",
    }
    return f"[{colours[imp]}]{imp.name}[/]"


# ── Commands ───────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    text: str = typer.Argument(..., help="Summary text to ingest."),
    ref_uri: Optional[str] = typer.Option(None, "--ref-uri", help="URI to actual content."),
    ref_type: str = typer.Option("text", "--ref-type", help="Content type: text|image|audio|video|url"),
    importance: str = typer.Option("normal", "--importance", help="weak|normal|key|capital"),
    mock: bool = typer.Option(False, "--mock", help="Use mock embedder (no model download)."),
) -> None:
    """Ingest a new memory unit (awake mode)."""
    mem = _load_or_new(mock)
    imp = _importance_from_str(importance)
    node_id = mem.ingest_awake(
        summary=text,
        ref_uri=ref_uri,
        ref_type=ref_type if ref_type in RefType.ALL else "text",
        importance=imp,
    )
    node = mem.graph.V[node_id]
    _save_state(mem)
    console.print(Panel(
        f"[bold]UUID:[/] {node_id}\n"
        f"[bold]Tier:[/] {_tier_name(node.tier)}\n"
        f"[bold]Importance:[/] {_importance_name(node.importance)}\n"
        f"[bold]Summary:[/] {node.summary[:80]}\n"
        f"[bold]Ref URI:[/] {node.ref_uri or '—'}\n"
        f"[bold]Ref Type:[/] {node.ref_type}",
        title="[green]✓ Memory Ingested[/]",
        expand=False,
    ))


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query text."),
    k: int = typer.Option(10, "--k", help="Number of results to return."),
    mock: bool = typer.Option(False, "--mock", help="Use mock embedder."),
) -> None:
    """Retrieve the top-k most relevant memory nodes."""
    mem = _load_or_new(mock)
    results = mem.retrieve(query, k=k)
    _save_state(mem)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    table = Table(title=f"Results for: '{query}'", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=8)
    table.add_column("Tier", width=8)
    table.add_column("Importance", width=10)
    table.add_column("Summary", min_width=30, max_width=60)
    table.add_column("Ref URI", max_width=40)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            _tier_name(r.tier),
            _importance_name(r.importance),
            r.summary[:60],
            r.ref_uri or "—",
        )

    console.print(table)


@app.command()
def sleep(
    idle: float = typer.Option(0.0, "--idle", help="Simulated idle duration in seconds."),
    mock: bool = typer.Option(False, "--mock", help="Use mock embedder."),
) -> None:
    """Run a manual sleep phase (diffuse + promote + compress)."""
    mem = _load_or_new(mock)
    before = mem.status()

    console.print("[cyan]Running sleep phase...[/]")
    mem.sleep_phase(idle_duration=idle)
    _save_state(mem)

    after = mem.status()
    table = Table(title="Sleep Phase: Before → After")
    table.add_column("Tier")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")

    for tier_name in ("SHORT", "MEDIUM", "LONG"):
        table.add_row(
            tier_name,
            str(before["tier_counts"][tier_name]),
            str(after["tier_counts"][tier_name]),
        )
    table.add_row(
        "TOTAL",
        str(before["total_nodes"]),
        str(after["total_nodes"]),
    )
    console.print(table)


@app.command()
def status(
    mock: bool = typer.Option(False, "--mock", help="Use mock embedder."),
) -> None:
    """Display current memory statistics."""
    mem = _load_or_new(mock)
    s = mem.status()

    table = Table(title="LiveMem Status", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Nodes", str(s["total_nodes"]))
    table.add_row("Total Edges", str(s["total_edges"]))
    table.add_row("SHORT nodes", str(s["tier_counts"]["SHORT"]))
    table.add_row("MEDIUM nodes", str(s["tier_counts"]["MEDIUM"]))
    table.add_row("LONG nodes", str(s["tier_counts"]["LONG"]))
    table.add_row("SHORT index", str(s["index_sizes"]["SHORT"]))
    table.add_row("MEDIUM index", str(s["index_sizes"]["MEDIUM"]))
    table.add_row("LONG index", str(s["index_sizes"]["LONG"]))
    table.add_row("Last sleep", str(s["last_sleep_end"]))

    console.print(table)


@app.command()
def demo(
    real: bool = typer.Option(False, "--real", help="Use real fastembed model (downloads on first run)."),
) -> None:
    """Run a self-contained demo with 25 diverse memory facts."""
    mock = not real
    console.print(Panel(
        "LiveMem Demo — ingesting 25 diverse memory facts",
        title="[bold blue]LiveMem Demo[/]",
    ))

    # Start fresh for demo.
    mem = LiveMem(cfg=DEFAULT_CONFIG, mock=mock)

    facts = [
        # (summary, ref_uri, ref_type, importance)
        ("The Eiffel Tower was built in 1889 by Gustave Eiffel for the World's Fair.",
         "https://en.wikipedia.org/wiki/Eiffel_Tower", "url", "normal"),
        ("Neural networks learn by adjusting weights via backpropagation.",
         "/docs/neural_nets.pdf", "text", "key"),
        ("The HNSW algorithm enables sub-linear ANN search in high dimensions.",
         "/papers/hnsw.pdf", "text", "key"),
        ("Photo of the Milky Way taken from Atacama Desert, Chile.",
         "https://images.nasa.gov/milkyway.jpg", "image", "normal"),
        ("Ebbinghaus forgetting curve: retention decays exponentially without rehearsal.",
         "/docs/memory_science.md", "text", "capital"),
        ("Recording of bird songs from the Amazon rainforest.",
         "/audio/amazon_birds.mp3", "audio", "normal"),
        ("Python 3.11 introduced significant performance improvements via specialising adaptive interpreter.",
         None, "text", "key"),
        ("Docker containers share the host OS kernel, unlike full VMs.",
         "https://docs.docker.com/", "url", "normal"),
        ("Video lecture on transformer architecture by Andrej Karpathy.",
         "https://youtube.com/karpathy_transformers", "video", "capital"),
        ("The cosine similarity of two unit vectors equals their dot product.",
         None, "text", "key"),
        ("Claude Code is Anthropic's official CLI for agentic coding tasks.",
         "https://claude.ai/code", "url", "normal"),
        ("Traefik reverse proxy supports automatic TLS via ACME DNS-01 challenge.",
         "/docs/traefik.md", "text", "normal"),
        ("Portrait of Einstein writing equations on a chalkboard.",
         "/images/einstein_chalkboard.png", "image", "normal"),
        ("The spacing effect: distributed practice outperforms massed practice for long-term retention.",
         "/papers/spacing_effect.pdf", "text", "capital"),
        ("Tailscale creates a WireGuard mesh network across devices without port forwarding.",
         "https://tailscale.com/", "url", "normal"),
        ("NumPy vectorised operations are implemented in C and avoid Python overhead.",
         None, "text", "key"),
        ("Podcast episode on cognitive architectures and working memory capacity.",
         "/audio/cognitive_arch_podcast.mp3", "audio", "normal"),
        ("HNSW index: M=16 gives ~0.99 recall@10 on typical NLP embeddings.",
         None, "text", "key"),
        ("Time-lapse video of the International Space Station crossing the night sky.",
         "https://example.com/iss_timelapse.mp4", "video", "normal"),
        ("Sortedcontainers SortedList gives O(log n) insert and O(1) iteration in Python.",
         None, "text", "normal"),
        ("The hippocampus replays memories during sleep for consolidation into neocortex.",
         "/papers/sleep_consolidation.pdf", "text", "capital"),
        ("FastEmbed uses ONNX Runtime for 3x faster CPU inference vs sentence-transformers.",
         "https://qdrant.github.io/fastembed/", "url", "normal"),
        ("Architecture diagram of KpihX homelab with Proxmox, LXC, and Traefik.",
         "/diagrams/homelab_arch.png", "image", "key"),
        ("Vaultwarden is a self-hosted Bitwarden-compatible password manager.",
         None, "text", "normal"),
        ("The attention mechanism in transformers enables parallel sequence processing.",
         None, "text", "key"),
    ]

    console.print(f"[cyan]Ingesting {len(facts)} facts...[/]")
    node_ids = []
    for summary, ref_uri, ref_type, imp_str in facts:
        nid = mem.ingest_awake(
            summary=summary,
            ref_uri=ref_uri,
            ref_type=ref_type,
            importance=_importance_from_str(imp_str),
        )
        node_ids.append(nid)

    console.print(f"[green]✓ Ingested {len(node_ids)} nodes[/]")
    console.print(mem.status())

    # Run a retrieval.
    console.print("\n[bold]Retrieving: 'memory consolidation during sleep'[/]")
    results = mem.retrieve("memory consolidation during sleep", k=5)
    for i, r in enumerate(results, 1):
        console.print(
            f"  {i}. [{r.score:.3f}] [{r.tier.name}] [{r.importance.name}] "
            f"{r.summary[:70]}"
        )

    # Run a sleep phase.
    console.print("\n[cyan]Running sleep phase (idle=300s to trigger LONG diffusion)...[/]")
    mem.sleep_phase(idle_duration=300.0)

    # Show final stats.
    s = mem.status()
    console.print(Panel(
        f"Total nodes: {s['total_nodes']}\n"
        f"Total edges: {s['total_edges']}\n"
        f"SHORT: {s['tier_counts']['SHORT']}  "
        f"MEDIUM: {s['tier_counts']['MEDIUM']}  "
        f"LONG: {s['tier_counts']['LONG']}",
        title="[green]Demo Complete — Final State[/]",
    ))

    # Save demo state.
    _save_state(mem)
    console.print(f"[dim]State saved to {_STATE_PATH}[/]")


if __name__ == "__main__":
    app()
