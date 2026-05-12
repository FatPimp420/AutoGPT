"""Tests for TradingScheduler (forge/trading/scheduler.py)."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .scheduler import TradingScheduler
from .swarm import SwarmResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_swarm_result(task_id: str = "test-task") -> SwarmResult:
    return SwarmResult(
        task_id=task_id,
        signals_generated=2,
        trades_executed=1,
        signals_blocked=1,
    )


def _mock_swarm(task_id: str = "test-task") -> MagicMock:
    """Return a TradingSwarm mock whose run_signal_pipeline is an AsyncMock."""
    swarm = MagicMock()
    swarm.run_signal_pipeline = AsyncMock(
        side_effect=lambda task_id, symbols, strategy, **kw: _make_swarm_result(task_id)
    )
    return swarm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_returns_swarm_result():
    """run_once() should return a SwarmResult."""
    swarm = _mock_swarm()
    scheduler = TradingScheduler(db=MagicMock(), swarm=swarm)
    result = await scheduler.run_once()
    assert isinstance(result, SwarmResult)


@pytest.mark.asyncio
async def test_run_once_uses_env_symbols():
    """run_once() should pass the symbols defined in SYMBOLS env var to the swarm."""
    swarm = _mock_swarm()
    scheduler = TradingScheduler(db=MagicMock(), swarm=swarm)

    with patch.dict(os.environ, {"SYMBOLS": "BTC/USDT,ETH/USDT"}):
        await scheduler.run_once()

    call_kwargs = swarm.run_signal_pipeline.call_args
    symbols_passed = call_kwargs.kwargs.get("symbols") or call_kwargs.args[1]
    assert symbols_passed == ["BTC/USDT", "ETH/USDT"]


@pytest.mark.asyncio
async def test_run_once_uses_default_symbols_when_env_missing():
    """run_once() should use SYMBOLS_DEFAULT when SYMBOLS env var is absent."""
    from .swarm import SYMBOLS_DEFAULT

    swarm = _mock_swarm()
    scheduler = TradingScheduler(db=MagicMock(), swarm=swarm)

    env = {k: v for k, v in os.environ.items() if k != "SYMBOLS"}
    with patch.dict(os.environ, env, clear=True):
        await scheduler.run_once()

    call_kwargs = swarm.run_signal_pipeline.call_args
    symbols_passed = call_kwargs.kwargs.get("symbols") or call_kwargs.args[1]
    assert symbols_passed == list(SYMBOLS_DEFAULT)


@pytest.mark.asyncio
async def test_start_creates_background_task():
    """start() should create an asyncio Task that is not yet done."""
    swarm = MagicMock()
    # Make run_signal_pipeline block indefinitely so the task stays alive.
    async def _blocking(*args, **kwargs):
        await asyncio.sleep(3600)

    swarm.run_signal_pipeline = AsyncMock(side_effect=_blocking)
    scheduler = TradingScheduler(db=MagicMock(), swarm=swarm, interval_minutes=60)

    scheduler.start()
    try:
        assert scheduler._task is not None
        assert isinstance(scheduler._task, asyncio.Task)
        assert not scheduler._task.done()
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_stop_cancels_task():
    """After stop(), the background task should be cancelled/done."""
    swarm = MagicMock()

    async def _blocking(*args, **kwargs):
        await asyncio.sleep(3600)

    swarm.run_signal_pipeline = AsyncMock(side_effect=_blocking)
    scheduler = TradingScheduler(db=MagicMock(), swarm=swarm, interval_minutes=60)

    scheduler.start()
    task = scheduler._task
    await scheduler.stop()

    assert task.done()


def test_scheduler_default_interval_from_env():
    """When SIGNAL_INTERVAL_MINUTES=5 is set, interval_minutes should be 5."""
    with patch.dict(os.environ, {"SIGNAL_INTERVAL_MINUTES": "5"}):
        scheduler = TradingScheduler(db=MagicMock())
    assert scheduler.interval_minutes == 5


def test_scheduler_default_interval_fallback():
    """When SIGNAL_INTERVAL_MINUTES is not set, interval_minutes should default to 60."""
    env = {k: v for k, v in os.environ.items() if k != "SIGNAL_INTERVAL_MINUTES"}
    with patch.dict(os.environ, env, clear=True):
        scheduler = TradingScheduler(db=MagicMock())
    assert scheduler.interval_minutes == 60


@pytest.mark.asyncio
async def test_run_once_generates_unique_task_ids():
    """Two successive run_once() calls should produce different task_ids."""
    seen_ids = []

    async def _capture(task_id, symbols, strategy, **kw):
        seen_ids.append(task_id)
        return _make_swarm_result(task_id)

    swarm = MagicMock()
    swarm.run_signal_pipeline = AsyncMock(side_effect=_capture)
    scheduler = TradingScheduler(db=MagicMock(), swarm=swarm)

    await scheduler.run_once()
    await scheduler.run_once()

    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]
