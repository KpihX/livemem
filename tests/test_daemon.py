"""
test_daemon.py — Tests for the SleepDaemon asyncio background task.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from livemem.config import LiveConfig
from livemem.daemon import SleepDaemon
from livemem.memory import LiveMem


@pytest.fixture
def daemon_mem(small_config, mock_embedder) -> LiveMem:
    return LiveMem(cfg=small_config, embedder=mock_embedder)


@pytest.fixture
def fast_config(small_config) -> LiveConfig:
    """Config with very short intervals for daemon tests."""
    return LiveConfig(**{
        **small_config.__dict__,
        "idle_ttl": 0.05,           # 50ms idle threshold.
        "daemon_check_interval": 0.02,  # 20ms polling interval.
    })


# ── Initial state ──────────────────────────────────────────────────────────────

def test_daemon_initial_state(daemon_mem, small_config):
    daemon = SleepDaemon(daemon_mem, small_config)
    assert daemon.is_running is False


# ── notify_activity ────────────────────────────────────────────────────────────

def test_notify_activity_resets_last_activity(daemon_mem, small_config):
    daemon = SleepDaemon(daemon_mem, small_config)
    old_activity = daemon._last_activity
    time.sleep(0.01)
    daemon.notify_activity()
    assert daemon._last_activity >= old_activity


# ── start / stop ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_makes_running(daemon_mem, small_config):
    daemon = SleepDaemon(daemon_mem, small_config)
    await daemon.start()
    assert daemon.is_running is True
    await daemon.stop()


@pytest.mark.asyncio
async def test_stop_cancels_task(daemon_mem, small_config):
    daemon = SleepDaemon(daemon_mem, small_config)
    await daemon.start()
    assert daemon.is_running is True
    await daemon.stop()
    assert daemon.is_running is False


# ── Loop behaviour ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_triggers_sleep_after_idle(daemon_mem, fast_config):
    """The daemon should call sleep_phase when idle >= idle_ttl."""
    daemon = SleepDaemon(daemon_mem, fast_config)

    called = []

    original_sleep_phase = daemon_mem.sleep_phase
    def mock_sleep_phase(idle_duration: float = 0.0) -> None:
        called.append(idle_duration)
        original_sleep_phase(idle_duration)

    daemon_mem.sleep_phase = mock_sleep_phase  # type: ignore[method-assign]

    await daemon.start()
    # Wait long enough for at least one idle trigger.
    await asyncio.sleep(0.15)
    await daemon.stop()

    assert len(called) >= 1, "sleep_phase should have been called at least once"


@pytest.mark.asyncio
async def test_loop_does_not_trigger_when_active(daemon_mem, fast_config):
    """If notify_activity is called frequently, sleep_phase should NOT trigger."""
    daemon = SleepDaemon(daemon_mem, fast_config)

    called = []
    original_sleep_phase = daemon_mem.sleep_phase
    def mock_sleep_phase(idle_duration: float = 0.0) -> None:
        called.append(idle_duration)
        original_sleep_phase(idle_duration)

    daemon_mem.sleep_phase = mock_sleep_phase  # type: ignore[method-assign]

    await daemon.start()
    # Keep notifying activity so idle never reaches idle_ttl.
    end_time = time.time() + 0.12
    while time.time() < end_time:
        daemon.notify_activity()
        await asyncio.sleep(0.01)
    await daemon.stop()

    # sleep_phase should not have been triggered (or at most 1 race condition hit).
    assert len(called) == 0, f"sleep_phase was called unexpectedly: {called}"
