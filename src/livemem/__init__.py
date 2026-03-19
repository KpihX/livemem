"""
livemem — Brain-inspired tiered graph memory system.

Public API
----------
LiveMem         : Main orchestrator (ingest, retrieve, sleep).
LiveConfig      : Immutable configuration dataclass.
DEFAULT_CONFIG  : Singleton default configuration.
Tier            : Enum SHORT=0, MEDIUM=1, LONG=2.
Node            : Memory unit (embedding + summary + ref_uri).
              importance: float [0,1] — continuous, no discretization.
              urgency:    float [0,1] — time-pressure, decays from creation.
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
from livemem.types import Edge, Node, RetrievalResult, Tier
from livemem.api import create_app

__all__ = [
    "LiveMem",
    "LiveConfig",
    "DEFAULT_CONFIG",
    "Tier",
    "Node",
    "Edge",
    "RetrievalResult",
    "SleepDaemon",
    "create_app",
    "save",
    "load",
]

__version__ = "0.3.2"
