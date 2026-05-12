"""Tests for bot/live_executor.py

All tests use mocked ccxt — no real network calls are made.
"""

import os
import sys

# Ensure the project root is on sys.path so `bot.*` imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.live_executor import LiveExecutor, OrderResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ccxt_order(
    price: float = 50_000.0,
    filled: float = 0.001,
    fee_cost: float = 0.05,
    order_id: str = "order-123",
) -> dict:
    """Return a minimal ccxt-style order dict."""
    return {
        "id": order_id,
        "price": price,
        "average": price,
        "filled": filled,
        "amount": filled,
        "fee": {"cost": fee_cost, "currency": "USDT"},
    }


def _make_executor(dry_run_env: str = "false") -> LiveExecutor:
    """Create a LiveExecutor with LIVE_TRADING env var controlled externally."""
    return LiveExecutor(exchange_id="binance", api_key="k", api_secret="s")


# ---------------------------------------------------------------------------
# 1. dry_run does NOT call create_market_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dry_run_does_not_call_exchange(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)

    executor = LiveExecutor()
    mock_exchange = MagicMock()
    mock_exchange.create_market_order = AsyncMock()
    executor._exchange = mock_exchange

    await executor.execute("BTC/USDT", "buy", 0.001)

    mock_exchange.create_market_order.assert_not_called()


# ---------------------------------------------------------------------------
# 2. dry_run returns correct OrderResult shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dry_run_returns_order_result(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)

    executor = LiveExecutor()
    result = await executor.execute("ETH/USDT", "sell", 0.5)

    assert isinstance(result, OrderResult)
    assert result.executed is False
    assert result.dry_run is True
    assert result.error is None
    assert result.symbol == "ETH/USDT"
    assert result.side == "sell"
    assert result.qty == 0.5


# ---------------------------------------------------------------------------
# 3. Live buy calls create_market_order with correct args
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_buy_calls_create_market_order(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    executor = LiveExecutor()
    assert executor.dry_run is False

    mock_exchange = MagicMock()
    mock_exchange.create_market_order = AsyncMock(
        return_value=_make_ccxt_order(price=50_000.0, filled=0.001, order_id="buy-1")
    )
    executor._exchange = mock_exchange

    result = await executor.execute("BTC/USDT", "buy", 0.001)

    mock_exchange.create_market_order.assert_awaited_once_with("BTC/USDT", "buy", 0.001)
    assert result.executed is True
    assert result.dry_run is False
    assert result.side == "buy"
    assert result.order_id == "buy-1"
    assert result.price == 50_000.0


# ---------------------------------------------------------------------------
# 4. Live sell calls create_market_order with correct args
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_sell_calls_create_market_order(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    executor = LiveExecutor()
    mock_exchange = MagicMock()
    mock_exchange.create_market_order = AsyncMock(
        return_value=_make_ccxt_order(price=3_000.0, filled=0.1, order_id="sell-99")
    )
    executor._exchange = mock_exchange

    result = await executor.execute("ETH/USDT", "sell", 0.1)

    mock_exchange.create_market_order.assert_awaited_once_with("ETH/USDT", "sell", 0.1)
    assert result.executed is True
    assert result.side == "sell"
    assert result.order_id == "sell-99"


# ---------------------------------------------------------------------------
# 5. Exchange exception → OrderResult with error, executed=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exchange_error_returns_error_result(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    executor = LiveExecutor()
    mock_exchange = MagicMock()
    mock_exchange.create_market_order = AsyncMock(
        side_effect=RuntimeError("Exchange timeout")
    )
    executor._exchange = mock_exchange

    result = await executor.execute("SOL/USDT", "buy", 1.0)

    assert result.executed is False
    assert result.dry_run is False
    assert result.error is not None
    assert "Exchange timeout" in result.error


# ---------------------------------------------------------------------------
# 6. LIVE_TRADING=true env var sets dry_run=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_trading_env_var_enables_live(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    executor = LiveExecutor(dry_run=True)  # constructor default is True
    # Env var must override the constructor parameter
    assert executor.dry_run is False


# ---------------------------------------------------------------------------
# 7. db.log_trade is awaited on successful live execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_db_log_called_on_live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    executor = LiveExecutor()
    mock_exchange = MagicMock()
    mock_exchange.create_market_order = AsyncMock(
        return_value=_make_ccxt_order(price=2_000.0, filled=0.25, order_id="db-test")
    )
    executor._exchange = mock_exchange

    mock_db = MagicMock()
    mock_db.log_trade = AsyncMock(return_value={"trade_id": "abc"})

    result = await executor.execute("ETH/USDT", "buy", 0.25, task_id="task-42", db=mock_db)

    assert result.executed is True
    mock_db.log_trade.assert_awaited_once()

    # Verify the key arguments forwarded to log_trade
    call_kwargs = mock_db.log_trade.call_args
    args = call_kwargs.args if call_kwargs.args else ()
    kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}

    # log_trade(task_id, symbol, side, price, qty, fee, agent_role=...)
    # Allow positional or keyword invocation
    all_args = list(args) + list(kwargs.values())
    assert "task-42" in all_args or kwargs.get("task_id") == "task-42"
    assert "ETH/USDT" in all_args or kwargs.get("symbol") == "ETH/USDT"
    assert "buy" in all_args or kwargs.get("side") == "buy"


# ---------------------------------------------------------------------------
# Bonus: fee falls back to qty * price * 0.001 when exchange omits it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fee_fallback_when_exchange_omits_fee(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    executor = LiveExecutor()
    mock_exchange = MagicMock()
    order_no_fee = {
        "id": "no-fee-order",
        "price": 1_000.0,
        "average": 1_000.0,
        "filled": 1.0,
        "amount": 1.0,
        "fee": None,  # exchange didn't return a fee
    }
    mock_exchange.create_market_order = AsyncMock(return_value=order_no_fee)
    executor._exchange = mock_exchange

    result = await executor.execute("BNB/USDT", "buy", 1.0)

    assert result.executed is True
    # Expected fallback: 1.0 * 1_000.0 * 0.001 = 1.0
    assert abs(result.fee - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Bonus: close() shuts the exchange and sets _exchange to None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_shuts_exchange(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)

    executor = LiveExecutor()
    mock_exchange = MagicMock()
    mock_exchange.close = AsyncMock()
    executor._exchange = mock_exchange

    await executor.close()

    mock_exchange.close.assert_awaited_once()
    assert executor._exchange is None
