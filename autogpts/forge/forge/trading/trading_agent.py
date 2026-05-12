"""TradingAgent: a Forge agent that routes tasks to the Ruflo trading swarm.

Supported step commands (JSON input to execute_step):
  {"action": "signal",    "symbols": [...], "strategy": "rsi"}
  {"action": "backtest",  "symbols": [...], "strategy": "rsi", "days": 365}
  {"action": "portfolio", "symbols": [...], "strategy": "rsi", "days": 90}
  {"action": "swarm",     "symbols": [...], "strategy": "rsi", "equity": 10000}
  {"action": "status"}
"""
import json
from typing import Any

from forge.agent import ForgeAgent
from forge.sdk import ForgeLogger, Step, StepRequestBody, Task, TaskRequestBody

from .actions import get_portfolio_summary, get_signal, run_backtest
from .swarm import SYMBOLS_DEFAULT, TradingSwarm

LOG = ForgeLogger(__name__)

_VALID_STRATEGIES = ("ema_cross", "rsi", "macd", "bollinger")


class TradingAgent(ForgeAgent):
    def __init__(self, database, workspace):
        super().__init__(database, workspace)
        self.swarm = TradingSwarm(db=database)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        task = await super().create_task(task_request)
        LOG.info(f"[TradingAgent] Task created: {task.task_id} — {task.input[:60]}")
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )

        raw = (step_request.input or "").strip()
        try:
            cmd = json.loads(raw) if raw.startswith("{") else {"action": raw}
        except json.JSONDecodeError:
            cmd = {"action": raw}

        action = cmd.get("action", "status").lower()
        symbols = cmd.get("symbols", SYMBOLS_DEFAULT)
        strategy = cmd.get("strategy", "rsi")
        days = int(cmd.get("days", 365))
        equity = float(cmd.get("equity", 10_000.0))

        if strategy not in _VALID_STRATEGIES:
            output = json.dumps({"error": f"Unknown strategy '{strategy}'. Choose from {_VALID_STRATEGIES}"})
        elif action == "signal":
            output = await self._handle_signal(task_id, symbols, strategy)
        elif action == "backtest":
            output = await self._handle_backtest(task_id, symbols, strategy, days)
        elif action == "portfolio":
            output = await self._handle_portfolio(symbols, strategy, days)
        elif action == "swarm":
            output = await self._handle_swarm(task_id, symbols, strategy, equity)
        elif action == "status":
            output = json.dumps({"status": "TradingAgent online", "strategies": list(_VALID_STRATEGIES), "default_symbols": SYMBOLS_DEFAULT})
        else:
            output = json.dumps({"error": f"Unknown action '{action}'"})

        await self.db.update_step(task_id, step.step_id, status="completed", output=output)
        step.output = output
        LOG.info(f"[TradingAgent] Step {step.step_id} completed — action={action}")
        return step

    async def _handle_signal(self, task_id: str, symbols: list, strategy: str) -> str:
        swarm_result = await self.swarm.run_signal_pipeline(
            task_id=task_id,
            symbols=symbols,
            strategy=strategy,
        )
        signals = {
            r.symbol: r.data.get("signal")
            for r in swarm_result.pipeline_results
            if r.role == "signal_generator"
        }
        return json.dumps({
            "action": "signal",
            "strategy": strategy,
            "signals": signals,
            "swarm_summary": swarm_result.summary(),
        })

    async def _handle_backtest(
        self, task_id: str, symbols: list, strategy: str, days: int
    ) -> str:
        swarm_result = await self.swarm.run_portfolio_backtest(
            task_id=task_id,
            symbols=symbols,
            strategy=strategy,
            days=days,
        )
        per_symbol = {
            r.symbol: r.data if not r.error else {"error": r.error}
            for r in swarm_result.pipeline_results
        }
        valid = [r.data for r in swarm_result.pipeline_results if not r.error and r.data.get("sharpe") is not None]
        avg_sharpe = sum(r["sharpe"] for r in valid) / len(valid) if valid else 0.0
        avg_pnl = sum(r["total_pnl_pct"] for r in valid) / len(valid) if valid else 0.0
        return json.dumps({
            "action": "backtest",
            "strategy": strategy,
            "days": days,
            "avg_sharpe": round(avg_sharpe, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "per_symbol": per_symbol,
        })

    async def _handle_portfolio(self, symbols: list, strategy: str, days: int) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None, lambda: get_portfolio_summary(symbols, strategy, days)
        )
        return json.dumps({"action": "portfolio", **summary})

    async def _handle_swarm(
        self, task_id: str, symbols: list, strategy: str, equity: float
    ) -> str:
        swarm_result = await self.swarm.run_signal_pipeline(
            task_id=task_id,
            symbols=symbols,
            strategy=strategy,
            equity=equity,
        )
        executions = [
            {
                "symbol": r.symbol,
                "executed": r.data.get("executed"),
                "signal": r.data.get("signal"),
                "qty": r.data.get("qty"),
            }
            for r in swarm_result.pipeline_results
            if r.role == "executor"
        ]
        return json.dumps({
            "action": "swarm",
            "strategy": strategy,
            "equity": equity,
            "executions": executions,
            "summary": swarm_result.summary(),
        })
