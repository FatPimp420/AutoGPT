"""Background async scheduler for the TradingSwarm signal pipeline.

Usage:
    scheduler = TradingScheduler(db=db, swarm=TradingSwarm(db=db))
    await scheduler.start()   # begins background loop
    ...
    await scheduler.stop()    # cancels background loop cleanly
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

from forge.sdk import ForgeLogger

from .swarm import SYMBOLS_DEFAULT, SwarmResult, TradingSwarm
from .trading_db import TradingDB

LOG = ForgeLogger(__name__)


class TradingScheduler:
    """Runs the TradingSwarm signal pipeline on a configurable interval."""

    def __init__(
        self,
        db: TradingDB,
        swarm: Optional[TradingSwarm] = None,
        interval_minutes: Optional[int] = None,
    ):
        self.db = db
        self.swarm = swarm if swarm is not None else TradingSwarm(db=db)

        if interval_minutes is not None:
            self.interval_minutes = interval_minutes
        else:
            env_val = os.getenv("SIGNAL_INTERVAL_MINUTES")
            self.interval_minutes = int(env_val) if env_val is not None else 60

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_symbols(self) -> list:
        """Return symbol list from SYMBOLS env var, or the hard-coded defaults."""
        raw = os.getenv("SYMBOLS", "")
        if raw.strip():
            return [s.strip() for s in raw.split(",") if s.strip()]
        return list(SYMBOLS_DEFAULT)

    def _get_strategy(self) -> str:
        return os.getenv("DEFAULT_STRATEGY", "rsi")

    def _make_task_id(self) -> str:
        """Generate a unique task_id using the UTC timestamp."""
        return f"scheduler_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run_once(self) -> SwarmResult:
        """Public one-shot call: run the pipeline once and return the result."""
        return await self._run_once()

    async def _run_once(self) -> SwarmResult:
        symbols = self._get_symbols()
        strategy = self._get_strategy()
        task_id = self._make_task_id()

        LOG.info(
            f"[TradingScheduler] Starting pipeline — task_id={task_id} "
            f"symbols={symbols} strategy={strategy}"
        )

        result = await self.swarm.run_signal_pipeline(
            task_id=task_id,
            symbols=symbols,
            strategy=strategy,
        )

        summary = result.summary()
        LOG.info(
            f"[TradingScheduler] Pipeline complete — task_id={task_id} "
            f"signals={summary['signals_generated']} "
            f"trades={summary['paper_trades_executed']} "
            f"blocked={summary['signals_blocked_by_risk']} "
            f"vetoed_by_ml={summary['signals_vetoed_by_ml']}"
        )
        return result

    async def _loop(self) -> None:
        """Background loop: run pipeline, sleep, repeat. Never lets exceptions kill the loop."""
        while self._running:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                # Propagate cancellation so the task can be awaited cleanly.
                raise
            except Exception as exc:
                LOG.error(
                    f"[TradingScheduler] Pipeline error (will retry in "
                    f"{self.interval_minutes}m): {exc}"
                )

            if not self._running:
                break

            try:
                await asyncio.sleep(self.interval_minutes * 60)
            except asyncio.CancelledError:
                raise

    def start(self) -> None:
        """Create a background asyncio task running the signal pipeline loop."""
        if self._running:
            LOG.warning("[TradingScheduler] Already running — ignoring start()")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        LOG.info(
            f"[TradingScheduler] Started — interval={self.interval_minutes}m"
        )

    async def stop(self) -> None:
        """Cancel the background task and wait for it to finish."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None
        LOG.info("[TradingScheduler] Stopped")
