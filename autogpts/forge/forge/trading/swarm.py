"""Ruflo-backed multi-agent trading swarm coordinator.

Pipeline: DataFetcher → SignalGenerator → RiskChecker → PaperExecutor
All symbols run concurrently; each stage awaits the prior stage's output.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from forge.sdk import ForgeLogger

from .actions import check_risk, get_signal, run_backtest

LOG = ForgeLogger(__name__)

SYMBOLS_DEFAULT = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
]


@dataclass
class AgentResult:
    role: str
    symbol: str
    data: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SwarmResult:
    task_id: str
    pipeline_results: List[AgentResult] = field(default_factory=list)
    trades_executed: int = 0
    signals_generated: int = 0
    signals_blocked: int = 0

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "signals_generated": self.signals_generated,
            "signals_blocked_by_risk": self.signals_blocked,
            "paper_trades_executed": self.trades_executed,
            "agents_ran": len(self.pipeline_results),
        }


class TradingSwarm:
    """Coordinates the four trading agent roles across multiple symbols."""

    def __init__(self, db=None):
        self.db = db

    # ------------------------------------------------------------------ #
    # Individual agent roles (coroutines)                                 #
    # ------------------------------------------------------------------ #

    async def _data_fetcher(self, symbol: str, strategy: str, days: int) -> AgentResult:
        """Agent role: fetch/generate OHLCV and compute signals."""
        LOG.info(f"[DataFetcher] {symbol} — fetching {days}d of data")
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: get_signal(symbol, strategy, lookback_days=days)
            )
            return AgentResult(role="data_fetcher", symbol=symbol, data=result)
        except Exception as e:
            LOG.error(f"[DataFetcher] {symbol} error: {e}")
            return AgentResult(role="data_fetcher", symbol=symbol, data={}, error=str(e))

    async def _signal_generator(
        self, fetch_result: AgentResult, strategy: str
    ) -> AgentResult:
        """Agent role: validate and annotate the signal from data_fetcher."""
        symbol = fetch_result.symbol
        if fetch_result.error:
            return AgentResult(
                role="signal_generator", symbol=symbol, data={}, error=fetch_result.error
            )
        signal = fetch_result.data.get("signal")
        LOG.info(f"[SignalGenerator] {symbol} → signal={signal}")
        return AgentResult(
            role="signal_generator",
            symbol=symbol,
            data={"symbol": symbol, "strategy": strategy, "signal": signal},
        )

    async def _risk_checker(
        self,
        signal_result: AgentResult,
        equity: float = 10_000.0,
        price_fallback: float = 1.0,
    ) -> AgentResult:
        """Agent role: apply risk rules and return approved/blocked decision."""
        symbol = signal_result.symbol
        if signal_result.error or signal_result.data.get("signal") is None:
            return AgentResult(
                role="risk_checker",
                symbol=symbol,
                data={"allowed": False, "reason": "no_signal"},
            )
        signal = signal_result.data["signal"]
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: check_risk(signal, equity, price_fallback),
        )
        LOG.info(f"[RiskChecker] {symbol} signal={signal} allowed={result['allowed']}")
        return AgentResult(role="risk_checker", symbol=symbol, data=result)

    async def _paper_executor(self, risk_result: AgentResult) -> AgentResult:
        """Agent role: paper-trade approved signals (log only, no live orders)."""
        symbol = risk_result.symbol
        if not risk_result.data.get("allowed"):
            return AgentResult(
                role="executor",
                symbol=symbol,
                data={"executed": False, "reason": risk_result.data.get("reason", "blocked_by_risk")},
            )
        signal = risk_result.data.get("signal", "buy")
        qty = risk_result.data.get("qty", 0.0)
        price = risk_result.data.get("price", 0.0)
        LOG.info(f"[Executor] PAPER {signal.upper()} {qty} {symbol} @ {price}")
        return AgentResult(
            role="executor",
            symbol=symbol,
            data={"executed": True, "signal": signal, "qty": qty, "price": price},
        )

    # ------------------------------------------------------------------ #
    # Backtest pipeline                                                   #
    # ------------------------------------------------------------------ #

    async def _backtest_worker(
        self, symbol: str, strategy: str, days: int
    ) -> AgentResult:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: run_backtest(symbol, strategy, days)
            )
            return AgentResult(role="backtester", symbol=symbol, data=result)
        except Exception as e:
            return AgentResult(role="backtester", symbol=symbol, data={}, error=str(e))

    # ------------------------------------------------------------------ #
    # Public entry points                                                 #
    # ------------------------------------------------------------------ #

    async def run_signal_pipeline(
        self,
        task_id: str,
        symbols: List[str],
        strategy: str = "rsi",
        equity: float = 10_000.0,
        lookback_days: int = 60,
    ) -> SwarmResult:
        """Run the full 4-stage pipeline for all symbols in parallel."""
        swarm_result = SwarmResult(task_id=task_id)

        async def _process_symbol(symbol: str):
            fetch = await self._data_fetcher(symbol, strategy, lookback_days)
            sig = await self._signal_generator(fetch, strategy)
            risk = await self._risk_checker(sig, equity=equity)
            exec_ = await self._paper_executor(risk)

            swarm_result.pipeline_results.extend([fetch, sig, risk, exec_])

            if sig.data.get("signal"):
                swarm_result.signals_generated += 1
            if risk.data.get("allowed"):
                swarm_result.trades_executed += 1
            else:
                swarm_result.signals_blocked += 1

            if self.db and task_id:
                try:
                    if sig.data.get("signal"):
                        await self.db.log_signal(
                            task_id, symbol, strategy, sig.data["signal"]
                        )
                    if exec_.data.get("executed"):
                        await self.db.log_trade(
                            task_id,
                            symbol,
                            exec_.data["signal"],
                            exec_.data["price"],
                            exec_.data["qty"],
                            fee=0.001 * exec_.data["qty"] * exec_.data["price"],
                            agent_role="executor",
                        )
                except Exception as e:
                    LOG.warning(f"DB log error for {symbol}: {e}")

        await asyncio.gather(*[_process_symbol(sym) for sym in symbols])
        return swarm_result

    async def run_portfolio_backtest(
        self,
        task_id: str,
        symbols: List[str],
        strategy: str = "rsi",
        days: int = 365,
    ) -> SwarmResult:
        """Run parallel backtests for all symbols."""
        swarm_result = SwarmResult(task_id=task_id)
        workers = [self._backtest_worker(sym, strategy, days) for sym in symbols]
        results = await asyncio.gather(*workers)
        swarm_result.pipeline_results.extend(results)
        return swarm_result
