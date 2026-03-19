"""
daemon.py — Async background daemon that triggers sleep phases on idle.

WHY a daemon (not a cron job):
    A cron job would run at fixed wall-clock times regardless of activity.
    The daemon monitors inactivity (idle time since last ingestion) and only
    triggers sleep when the system is actually idle — matching the biological
    model where sleep consolidation occurs during genuine rest, not on a timer.

WHY asyncio (not threading):
    LiveMem is designed to be embedded in async Python applications
    (e.g., FastAPI, async chat bots). An asyncio.Task runs concurrently
    without needing a separate thread, keeping resource overhead minimal.
    The heavy sleep_phase work is offloaded via asyncio.to_thread so it
    does not block the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import time

from rich.console import Console

from livemem.config import LiveConfig
from livemem.memory import LiveMem

logger = logging.getLogger(__name__)
console = Console()


class SleepDaemon:
    """Background asyncio daemon that triggers LiveMem.sleep_phase on idle.

    Usage
    -----
    daemon = SleepDaemon(mem, cfg)
    await daemon.start()
    # ... application runs ...
    await daemon.stop()

    Call notify_activity() after every external event that counts as
    "user activity" (ingestion, retrieval, etc.) to reset the idle timer.
    """

    def __init__(self, mem: LiveMem, cfg: LiveConfig) -> None:
        self._mem = mem
        self._cfg = cfg
        self._last_activity: float = time.time()
        self._task: asyncio.Task | None = None
        self._running: bool = False

    # ── Activity tracking ──────────────────────────────────────────────────────

    def notify_activity(self) -> None:
        """Reset the idle clock.

        Call this after every user-initiated operation (ingest, retrieve)
        so the daemon knows the system is still active and defers sleep.
        """
        self._last_activity = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop as an asyncio Task.

        WHY create_task:
            The loop runs concurrently with the caller's coroutine without
            blocking. The caller retains a reference via self._task to
            allow clean cancellation via stop().
        """
        if self._running:
            logger.warning("SleepDaemon.start() called while already running")
            return
        self._running = True
        self._task = asyncio.get_event_loop().create_task(
            self._loop(), name="livemem-sleep-daemon"
        )
        logger.info("SleepDaemon started (idle_ttl=%.0fs)", self._cfg.idle_ttl)

    async def stop(self) -> None:
        """Cancel the background task and wait for it to finish.

        WHY await after cancel:
            asyncio.Task.cancel() schedules cancellation but does not
            wait for the coroutine to exit. Awaiting ensures any finally
            blocks in _loop() have run before we return, preventing
            resource leaks.
        """
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("SleepDaemon stopped")

    async def _loop(self) -> None:
        """Main polling loop: check idle time every daemon_check_interval seconds.

        Logic
        -----
        - Sleep for daemon_check_interval seconds.
        - Compute idle = now - last_activity.
        - If idle >= idle_ttl: run sleep_phase in a thread (non-blocking).
        - Reset last_activity after sleep_phase completes so we don't
          immediately re-trigger on the next poll.
        """
        while self._running:
            try:
                await asyncio.sleep(self._cfg.daemon_check_interval)
            except asyncio.CancelledError:
                break

            idle = time.time() - self._last_activity
            if idle >= self._cfg.idle_ttl:
                console.print(
                    f"[bold cyan]💤 Sleep phase triggered after {idle:.0f}s idle[/]"
                )
                logger.info("SleepDaemon: triggering sleep_phase (idle=%.0fs)", idle)
                try:
                    # Run the CPU-bound sleep_phase in a thread pool so the
                    # event loop remains responsive during consolidation.
                    await asyncio.to_thread(self._mem.sleep_phase, idle)
                    # Reset activity to now so we don't immediately re-trigger.
                    self._last_activity = time.time()
                except Exception as exc:
                    logger.error("SleepDaemon: sleep_phase error: %s", exc)

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True if the daemon loop is active."""
        return self._running and self._task is not None and not self._task.done()
