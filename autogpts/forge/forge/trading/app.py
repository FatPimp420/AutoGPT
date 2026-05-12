"""Trading agent ASGI entrypoint.

Start with:
  uvicorn forge.trading.app:app --host 0.0.0.0 --port 8000 --reload
"""
import datetime
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from forge.sdk import LocalWorkspace

from .trading_agent import TradingAgent
from .trading_db import TradingDB
from .swarm import TradingSwarm
from .scheduler import TradingScheduler

database_name = os.getenv("DATABASE_STRING", "sqlite:///trading_agent.db")
workspace_path = os.getenv("AGENT_WORKSPACE", "./workspace")

database = TradingDB(database_name, debug_enabled=False)
workspace = LocalWorkspace(workspace_path)

agent = TradingAgent(database=database, workspace=workspace)

# Build a scheduler instance that will be started/stopped via the lifespan.
_scheduler = TradingScheduler(db=database, swarm=TradingSwarm(db=database))


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """FastAPI lifespan: start the background scheduler on startup, stop on shutdown."""
    _scheduler.start()
    try:
        yield
    finally:
        await _scheduler.stop()


# Obtain the base app from the agent SDK, then attach the lifespan.
app = agent.get_agent_app()
app.router.lifespan_context = _lifespan


# ── /trading/dashboard ───────────────────────────────────────────────────────

trading_router = APIRouter()


@trading_router.get("/trading/dashboard")
async def trading_dashboard():
    """Return a live JSON status page for the trading system."""
    recent_trades = await database.get_all_trades(limit=20)
    recent_signals = await database.get_all_signals(limit=20)

    # Trade summary
    total_trades = len(recent_trades)
    symbols_traded = list(dict.fromkeys(t["symbol"] for t in recent_trades))
    trades_with_pnl = [t for t in recent_trades if t.get("pnl_pct") is not None]
    if trades_with_pnl:
        winning = sum(1 for t in trades_with_pnl if t["pnl_pct"] > 0)
        win_rate = round(winning / len(trades_with_pnl), 4)
        total_pnl_pct = round(sum(t["pnl_pct"] for t in trades_with_pnl), 4)
    else:
        win_rate = 0.0
        total_pnl_pct = 0.0

    # Signal summary
    total_signals = len(recent_signals)
    buy_count = sum(1 for s in recent_signals if s.get("signal") == "buy")
    sell_count = sum(1 for s in recent_signals if s.get("signal") == "sell")
    strategies_used = list(dict.fromkeys(s["strategy"] for s in recent_signals))

    return JSONResponse(
        content={
            "status": "online",
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "recent_trades": recent_trades,
            "recent_signals": recent_signals,
            "trade_summary": {
                "total_trades": total_trades,
                "symbols_traded": symbols_traded,
                "win_rate": win_rate,
                "total_pnl_pct": total_pnl_pct,
            },
            "signal_summary": {
                "total_signals": total_signals,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "strategies_used": strategies_used,
            },
        }
    )


app.include_router(trading_router)
