"""Unit tests for the AutoGPT + Ruflo trading integration."""
import asyncio
import json
import os
import sys

import pytest

# Make trading-bot importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_BOT_PATH = os.path.join(_REPO_ROOT, "trading-bot")
if _BOT_PATH not in sys.path:
    sys.path.insert(0, _BOT_PATH)

from forge.sdk import LocalWorkspace
from forge.sdk.model import StepRequestBody, TaskRequestBody

from .trading_db import TradingDB
from .trading_agent import TradingAgent
from .swarm import TradingSwarm
from .actions import run_backtest, get_signal, check_risk


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return TradingDB(f"sqlite:///{tmp_path}/test_trading.db")


@pytest.fixture
def agent(tmp_path):
    db = TradingDB(f"sqlite:///{tmp_path}/agent_test.db")
    ws = LocalWorkspace(str(tmp_path / "workspace"))
    return TradingAgent(database=db, workspace=ws)


# ── TradingDB tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trading_db_creates_tables(tmp_path):
    """trade_log and signal_history tables must be created on init."""
    import sqlite3
    db_path = tmp_path / "check_tables.db"
    TradingDB(f"sqlite:///{db_path}")
    conn = sqlite3.connect(str(db_path))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "trade_log" in tables
    assert "signal_history" in tables


@pytest.mark.asyncio
async def test_log_trade_returns_dict(db):
    result = await db.log_trade(
        task_id="t1", symbol="BTC/USDT", side="buy",
        price=50000.0, qty=0.01, fee=0.5, agent_role="executor"
    )
    assert result["symbol"] == "BTC/USDT"
    assert "trade_id" in result


@pytest.mark.asyncio
async def test_log_signal_returns_dict(db):
    result = await db.log_signal(
        task_id="t1", symbol="ETH/USDT", strategy="rsi", signal="sell"
    )
    assert result["signal"] == "sell"
    assert "signal_id" in result


@pytest.mark.asyncio
async def test_get_trade_history(db):
    await db.log_trade("t2", "SOL/USDT", "buy", 100.0, 1.0, 0.1)
    await db.log_trade("t2", "SOL/USDT", "sell", 110.0, 1.0, 0.11, pnl_pct=0.09)
    history = await db.get_trade_history("t2")
    assert len(history) == 2
    assert history[0]["side"] == "buy"
    assert history[1]["pnl_pct"] == pytest.approx(0.09)


@pytest.mark.asyncio
async def test_get_signal_history(db):
    await db.log_signal("t3", "BNB/USDT", "macd", "buy")
    await db.log_signal("t3", "XRP/USDT", "macd", "sell")
    history = await db.get_signal_history("t3")
    assert len(history) == 2
    symbols = {r["symbol"] for r in history}
    assert symbols == {"BNB/USDT", "XRP/USDT"}


# ── actions.py tests ─────────────────────────────────────────────────────────

def test_run_backtest_returns_summary():
    result = run_backtest("BTC/USDT", strategy_name="rsi", days=30)
    assert result["symbol"] == "BTC/USDT"
    assert result["strategy"] == "rsi"
    assert "total_trades" in result
    assert "sharpe" in result


def test_run_backtest_all_strategies():
    for strat in ("ema_cross", "rsi", "macd", "bollinger"):
        r = run_backtest("ETH/USDT", strategy_name=strat, days=30)
        assert "total_trades" in r, f"{strat} missing total_trades"


def test_get_signal_returns_valid_output():
    result = get_signal("BTC/USDT", strategy_name="rsi", lookback_days=30)
    assert result["symbol"] == "BTC/USDT"
    assert result["signal"] in ("buy", "sell", None)


def test_check_risk_allows_normal_trade():
    result = check_risk("buy", equity=10_000.0, price=50_000.0)
    assert "allowed" in result
    assert "qty" in result
    assert result["qty"] > 0


def test_check_risk_blocks_at_daily_loss_limit():
    result = check_risk("buy", equity=10_000.0, price=100.0, daily_loss=0.99)
    assert result["allowed"] is False


# ── TradingSwarm tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_swarm_signal_pipeline_runs_all_symbols():
    swarm = TradingSwarm()
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    result = await swarm.run_signal_pipeline(
        task_id="swarm-test-1", symbols=symbols, strategy="rsi", lookback_days=30
    )
    # 5 agent roles × 3 symbols = 15 pipeline results
    assert len(result.pipeline_results) == 15
    roles = {r.role for r in result.pipeline_results}
    assert roles == {"data_fetcher", "signal_generator", "sentiment_agent", "risk_checker", "executor"}


@pytest.mark.asyncio
async def test_swarm_backtest_runs_all_symbols():
    swarm = TradingSwarm()
    symbols = ["BTC/USDT", "ETH/USDT"]
    result = await swarm.run_portfolio_backtest(
        task_id="swarm-test-2", symbols=symbols, strategy="ema_cross", days=30
    )
    assert len(result.pipeline_results) == 2
    assert all(not r.error for r in result.pipeline_results)


@pytest.mark.asyncio
async def test_swarm_summary_counts_correct():
    swarm = TradingSwarm()
    result = await swarm.run_signal_pipeline(
        task_id="swarm-test-3",
        symbols=["BTC/USDT"],
        strategy="rsi",
        lookback_days=30,
    )
    summary = result.summary()
    assert summary["task_id"] == "swarm-test-3"
    assert summary["agents_ran"] == 5
    # signals_generated + signals_blocked must equal number of symbols
    assert summary["signals_generated"] + summary["signals_blocked_by_risk"] == 1
    assert "signals_blocked_by_sentiment" in summary
    assert "signals_boosted_by_sentiment" in summary


# ── TradingAgent.execute_step tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_status_action(agent):
    task = await agent.create_task(TaskRequestBody(input="status check"))
    step = await agent.execute_step(task.task_id, StepRequestBody(input='{"action": "status"}'))
    data = json.loads(step.output)
    assert data["status"] == "TradingAgent online"
    assert "strategies" in data


@pytest.mark.asyncio
async def test_agent_signal_action(agent):
    task = await agent.create_task(TaskRequestBody(input="get signals"))
    step = await agent.execute_step(
        task.task_id,
        StepRequestBody(input='{"action": "signal", "symbols": ["BTC/USDT"], "strategy": "rsi"}')
    )
    data = json.loads(step.output)
    assert data["action"] == "signal"
    assert "BTC/USDT" in data["signals"]


@pytest.mark.asyncio
async def test_agent_backtest_action(agent):
    task = await agent.create_task(TaskRequestBody(input="run backtest"))
    step = await agent.execute_step(
        task.task_id,
        StepRequestBody(input='{"action": "backtest", "symbols": ["ETH/USDT"], "strategy": "macd", "days": 30}')
    )
    data = json.loads(step.output)
    assert data["action"] == "backtest"
    assert "ETH/USDT" in data["per_symbol"]
    assert "avg_sharpe" in data


@pytest.mark.asyncio
async def test_agent_unknown_strategy_returns_error(agent):
    task = await agent.create_task(TaskRequestBody(input="bad strategy"))
    step = await agent.execute_step(
        task.task_id,
        StepRequestBody(input='{"action": "signal", "strategy": "nonexistent"}')
    )
    data = json.loads(step.output)
    assert "error" in data


@pytest.mark.asyncio
async def test_agent_unknown_action_returns_error(agent):
    task = await agent.create_task(TaskRequestBody(input="bad action"))
    step = await agent.execute_step(
        task.task_id,
        StepRequestBody(input='{"action": "fly_to_moon"}')
    )
    data = json.loads(step.output)
    assert "error" in data
