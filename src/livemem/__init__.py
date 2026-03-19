"""
livemem — Brain-inspired tiered graph memory system.

Public API
----------
LiveMem         : Main orchestrator (ingest, retrieve, sleep).
LiveConfig      : Immutable configuration dataclass.
DEFAULT_CONFIG  : Singleton default configuration.
Importance      : Enum WEAK=0 … CAPITAL=3.
Tier            : Enum SHORT=0, MEDIUM=1, LONG=2.
Node            : Memory unit (embedding + summary + ref_uri).
Edge            : Directed association between two nodes.
RetrievalResult : Read-only result from LiveMem.retrieve().
SleepDaemon     : Async background daemon that triggers sleep on idle.
save            : Serialise LiveMem state to JSON.
load            : Deserialise LiveMem state from JSON.
"""
from __future__ import annotations

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.daemon import SleepDaemon
from livemem.memory import LiveMem
from livemem.persistence import load, save
from livemem.types import Edge, Importance, Node, RetrievalResult, Tier

__all__ = [
    "LiveMem",
    "LiveConfig",
    "DEFAULT_CONFIG",
    "Importance",
    "Tier",
    "Node",
    "Edge",
    "RetrievalResult",
    "SleepDaemon",
    "save",
    "load",
]

__version__ = "0.1.0"
