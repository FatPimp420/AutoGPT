"""Ruflo-backed multi-agent trading swarm coordinator.

Pipeline: DataFetcher → SignalGenerator → SentimentAgent → MLVetoAgent → RiskChecker → SmartExecutor
All symbols run concurrently; each stage awaits the prior stage's output.
"""
import asyncio
import os
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


def _default_blender():
    from sentiment.providers.factory import get_provider
    from sentiment.blender import SignalBlender
    return SignalBlender(provider=get_provider())


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
    signals_blocked_by_sentiment: int = 0
    signals_boosted: int = 0
    signals_vetoed_by_ml: int = 0

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "signals_generated": self.signals_generated,
            "signals_blocked_by_sentiment": self.signals_blocked_by_sentiment,
            "signals_boosted_by_sentiment": self.signals_boosted,
            "signals_blocked_by_risk": self.signals_blocked,
            "signals_vetoed_by_ml": self.signals_vetoed_by_ml,
            "paper_trades_executed": self.trades_executed,
            "agents_ran": len(self.pipeline_results),
        }


class TradingSwarm:
    """Coordinates the five trading agent roles across multiple symbols."""

    def __init__(self, db=None, blender=None):
        self.db = db
        self._blender = blender  # None = lazy-init default MockSentimentProvider

    @property
    def blender(self):
        if self._blender is None:
            self._blender = _default_blender()
        return self._blender

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

    async def _sentiment_agent(self, signal_result: AgentResult) -> AgentResult:
        """Agent role: blend strategy signal with news/Reddit sentiment."""
        symbol = signal_result.symbol
        if signal_result.error:
            return AgentResult(
                role="sentiment_agent", symbol=symbol, data={}, error=signal_result.error
            )
        raw_signal = signal_result.data.get("signal")
        loop = asyncio.get_event_loop()
        try:
            blend = await loop.run_in_executor(
                None, lambda: self.blender.blend(symbol, raw_signal)
            )
            result = blend.to_dict()
            LOG.info(
                f"[SentimentAgent] {symbol} signal={raw_signal} "
                f"sentiment={blend.sentiment.label}({blend.sentiment.score:+.2f}) "
                f"→ {blend.action} → final={blend.final_signal}"
            )
            return AgentResult(role="sentiment_agent", symbol=symbol, data=result)
        except Exception as e:
            LOG.error(f"[SentimentAgent] {symbol} error: {e}")
            # Fail open: pass original signal through if sentiment errors
            return AgentResult(
                role="sentiment_agent",
                symbol=symbol,
                data={"final_signal": raw_signal, "blend_action": "error_passthrough"},
            )

    async def _ml_veto_agent(self, sentiment_result: AgentResult) -> AgentResult:
        """Agent role: ML model vetoes low-confidence signals.
        Gracefully skips if no trained model exists at ML_MODEL_PATH."""
        symbol = sentiment_result.symbol
        if sentiment_result.error:
            return AgentResult(
                role="ml_veto_agent", symbol=symbol, data={}, error=sentiment_result.error
            )

        raw_signal = sentiment_result.data.get("final_signal")
        if raw_signal is None:
            LOG.info(f"[MLVeto] {symbol} no signal — skipping ML veto")
            result_data = dict(sentiment_result.data)
            result_data["ml_action"] = "no_signal"
            return AgentResult(role="ml_veto_agent", symbol=symbol, data=result_data)

        # Determine model path from env var or default pattern
        strategy = sentiment_result.data.get("strategy", "rsi")
        default_path = f"./ml_models/{symbol.replace('/', '_')}_{strategy}.pkl"
        model_path = os.getenv("ML_MODEL_PATH", default_path)

        if not os.path.exists(model_path):
            LOG.info(f"[MLVeto] {symbol} no model at {model_path} — passing signal through")
            result_data = dict(sentiment_result.data)
            result_data["ml_action"] = "skipped"
            return AgentResult(role="ml_veto_agent", symbol=symbol, data=result_data)

        # Model exists — load it and run generate_signal on fresh synthetic data
        loop = asyncio.get_event_loop()
        try:
            def _run_ml():
                import pickle
                import sys as _sys
                import os as _os
                # Ensure trading-bot is importable
                from .actions import _ensure_trading_bot_on_path
                _ensure_trading_bot_on_path()

                from backtest.synthetic import generate_ohlcv
                from ml.model import MLSignalModel

                with open(model_path, "rb") as fh:
                    payload = pickle.load(fh)

                # Reconstruct a minimal MLSignalModel from saved payload
                # We need a base_strategy — use the strategy stored in sentiment data
                from .actions import _load_strategy
                base_strategy = _load_strategy(strategy)
                model = MLSignalModel(base_strategy=base_strategy)
                model._clf = payload.get("clf")
                model._scaler = payload.get("scaler")
                model._classes = payload.get("classes", [])
                model.min_confidence = payload.get("min_confidence", 0.6)

                df = generate_ohlcv(symbol=symbol, n_candles=60 * 24)
                return model.generate_signal(df)

            ml_signal = await loop.run_in_executor(None, _run_ml)

            if ml_signal is None:
                ml_action = "vetoed"
                final_signal = None
            else:
                ml_action = "confirmed"
                final_signal = ml_signal

            LOG.info(
                f"[MLVeto] {symbol} raw={raw_signal} → ml_action={ml_action} → final={final_signal}"
            )
            result_data = dict(sentiment_result.data)
            result_data["ml_action"] = ml_action
            result_data["final_signal"] = final_signal
            return AgentResult(role="ml_veto_agent", symbol=symbol, data=result_data)

        except Exception as e:
            LOG.error(f"[MLVeto] {symbol} error loading/running model: {e}")
            # Fail open: pass signal through unchanged if ML errors
            result_data = dict(sentiment_result.data)
            result_data["ml_action"] = "skipped"
            return AgentResult(role="ml_veto_agent", symbol=symbol, data=result_data)

    async def _risk_checker(
        self,
        sentiment_result: AgentResult,
        equity: float = 10_000.0,
        price_fallback: float = 1.0,
    ) -> AgentResult:
        """Agent role: apply risk rules and return approved/blocked decision."""
        symbol = sentiment_result.symbol
        # Accept final_signal from sentiment agent, or fall back to original_signal
        signal = (
            sentiment_result.data.get("final_signal")
            or sentiment_result.data.get("signal")
        )
        if sentiment_result.error or signal is None:
            return AgentResult(
                role="risk_checker",
                symbol=symbol,
                data={"allowed": False, "reason": "no_signal"},
            )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: check_risk(signal, equity, price_fallback),
        )
        result["signal"] = signal  # carry forward for executor
        LOG.info(f"[RiskChecker] {symbol} signal={signal} allowed={result['allowed']}")
        return AgentResult(role="risk_checker", symbol=symbol, data=result)

    async def _smart_executor(
        self, risk_result: AgentResult, task_id: str = ""
    ) -> AgentResult:
        """Agent role: execute approved signals via paper trade or live order.

        Behaviour is controlled by the LIVE_TRADING env var:
        - LIVE_TRADING=true  → real orders via LiveExecutor
        - anything else      → paper trade (log only, no real orders)
        """
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
        live = os.getenv("LIVE_TRADING", "false").lower() == "true"

        if not live:
            # Paper trade: log only, no real orders
            LOG.info(f"[Executor] PAPER {signal.upper()} {qty} {symbol} @ {price}")
            return AgentResult(
                role="executor",
                symbol=symbol,
                data={"executed": True, "signal": signal, "qty": qty, "price": price},
            )

        # Live path: delegate to LiveExecutor
        try:
            from bot.live_executor import LiveExecutor
            executor = LiveExecutor(dry_run=False)
            result = await executor.execute(
                symbol, signal, qty, task_id=task_id, db=self.db
            )
            await executor.close()
            return AgentResult(
                role="executor",
                symbol=symbol,
                data={
                    "executed": result.executed,
                    "signal": signal,
                    "qty": qty,
                    "price": result.price,
                    "order_id": result.order_id,
                    "error": result.error,
                },
            )
        except Exception as e:
            LOG.error(f"[Executor] LiveExecutor error for {symbol}: {e}")
            return AgentResult(
                role="executor",
                symbol=symbol,
                data={"executed": False, "error": str(e)},
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
        """Run the full 6-stage pipeline for all symbols in parallel."""
        swarm_result = SwarmResult(task_id=task_id)

        async def _process_symbol(symbol: str):
            fetch = await self._data_fetcher(symbol, strategy, lookback_days)
            sig = await self._signal_generator(fetch, strategy)
            sent = await self._sentiment_agent(sig)
            ml_veto = await self._ml_veto_agent(sent)
            risk = await self._risk_checker(ml_veto, equity=equity)
            exec_ = await self._smart_executor(risk, task_id=task_id)

            swarm_result.pipeline_results.extend([fetch, sig, sent, ml_veto, risk, exec_])

            if sig.data.get("signal"):
                swarm_result.signals_generated += 1
            if sent.data.get("blend_action") == "contradicted":
                swarm_result.signals_blocked_by_sentiment += 1
            if sent.data.get("blend_action") == "boosted":
                swarm_result.signals_boosted += 1
            if ml_veto.data.get("ml_action") == "vetoed":
                swarm_result.signals_vetoed_by_ml += 1
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
