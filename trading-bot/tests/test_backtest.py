"""Tests for BacktestResult.summary() — rich performance metrics."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backtest.runner import BacktestResult, Backtester
from backtest.synthetic import generate_ohlcv
from strategies.rsi import RSIStrategy
from strategies.ema_cross import EMACrossStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "total_trades",
    "win_rate",
    "total_pnl_pct",
    "avg_trade_pnl_pct",
    "max_drawdown",
    "sharpe",
    "calmar_ratio",
    "profit_factor",
    "max_consecutive_losses",
}


def _make_result(trades: list[dict], equity_curve: list[float]) -> BacktestResult:
    r = BacktestResult()
    r.trades = trades
    r.equity_curve = equity_curve
    return r


# ---------------------------------------------------------------------------
# 1. Empty result returns {}
# ---------------------------------------------------------------------------


def test_summary_empty_trades():
    r = BacktestResult()
    assert r.summary() == {}


# ---------------------------------------------------------------------------
# 2. All required keys present
# ---------------------------------------------------------------------------


def test_summary_has_all_required_keys():
    df = generate_ohlcv("BTC/USDT", n_candles=500)
    bt = Backtester(RSIStrategy(), initial_capital=10_000)
    result = bt.run(df)
    s = result.summary()
    if s:  # may be empty if no trades triggered
        assert _REQUIRED_KEYS.issubset(s.keys()), (
            f"Missing keys: {_REQUIRED_KEYS - s.keys()}"
        )


# ---------------------------------------------------------------------------
# 3. max_drawdown is a non-positive float
# ---------------------------------------------------------------------------


def test_max_drawdown_non_positive():
    # Equity starts at 10000, dips to 9000, then recovers
    equity = [10000.0, 10500.0, 10300.0, 9000.0, 9500.0, 10200.0]
    trades = [
        {"pnl": 0.05},
        {"pnl": -0.03},
        {"pnl": -0.10},
        {"pnl": 0.06},
        {"pnl": 0.08},
    ]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["max_drawdown"] <= 0.0


def test_max_drawdown_monotone_up():
    # Strictly increasing equity — drawdown should be 0 (or very close)
    equity = [10000.0 * (1 + 0.01 * i) for i in range(6)]
    trades = [{"pnl": 0.01 * (i + 1)} for i in range(5)]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. calmar_ratio
# ---------------------------------------------------------------------------


def test_calmar_ratio_zero_when_no_drawdown():
    # No drawdown → calmar is 0.0
    equity = [10000.0, 10100.0, 10200.0, 10300.0]
    trades = [{"pnl": 0.01}, {"pnl": 0.01}, {"pnl": 0.01}]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["calmar_ratio"] == pytest.approx(0.0)


def test_calmar_ratio_positive_when_profitable_with_drawdown():
    # Positive return with a real drawdown should give a positive Calmar
    equity = [10000.0, 10500.0, 10200.0, 11000.0]
    trades = [{"pnl": 0.05}, {"pnl": -0.03}, {"pnl": 0.08}]
    r = _make_result(trades, equity)
    s = r.summary()
    # max_drawdown is negative, annualised_return should be positive
    assert s["calmar_ratio"] > 0.0


# ---------------------------------------------------------------------------
# 5. profit_factor
# ---------------------------------------------------------------------------


def test_profit_factor_no_losses():
    # All wins — denominator is 0 → should return 0.0 (not inf / error)
    equity = [10000.0, 10100.0, 10200.0, 10300.0]
    trades = [{"pnl": 0.01}, {"pnl": 0.02}, {"pnl": 0.03}]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["profit_factor"] == pytest.approx(0.0)


def test_profit_factor_known_value():
    # gross_profit = 0.10 + 0.05 = 0.15; gross_loss = abs(-0.05) = 0.05
    # profit_factor = 0.15 / 0.05 = 3.0
    equity = [10000.0, 10100.0, 10050.0, 10150.0]
    trades = [{"pnl": 0.10}, {"pnl": -0.05}, {"pnl": 0.05}]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["profit_factor"] == pytest.approx(3.0, rel=1e-5)


def test_profit_factor_all_losses():
    equity = [10000.0, 9900.0, 9800.0]
    trades = [{"pnl": -0.01}, {"pnl": -0.02}]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["profit_factor"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. max_consecutive_losses
# ---------------------------------------------------------------------------


def test_max_consecutive_losses_none():
    # All wins
    equity = [10000.0, 10100.0, 10200.0, 10300.0]
    trades = [{"pnl": 0.01}, {"pnl": 0.01}, {"pnl": 0.01}]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["max_consecutive_losses"] == 0


def test_max_consecutive_losses_known_streak():
    # Loss, win, loss, loss, loss, win → longest streak = 3
    equity = [10000.0] * 7
    trades = [
        {"pnl": -0.01},  # 1
        {"pnl": 0.02},   # break
        {"pnl": -0.01},  # 1
        {"pnl": -0.01},  # 2
        {"pnl": -0.01},  # 3 ← max
        {"pnl": 0.05},   # break
    ]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["max_consecutive_losses"] == 3


def test_max_consecutive_losses_all_losses():
    equity = [10000.0] * 5
    trades = [{"pnl": -0.01} for _ in range(4)]
    r = _make_result(trades, equity)
    s = r.summary()
    assert s["max_consecutive_losses"] == 4


# ---------------------------------------------------------------------------
# 7. avg_trade_pnl_pct
# ---------------------------------------------------------------------------


def test_avg_trade_pnl_pct():
    equity = [10000.0] * 5
    trades = [{"pnl": 0.02}, {"pnl": -0.04}, {"pnl": 0.06}, {"pnl": 0.08}]
    r = _make_result(trades, equity)
    s = r.summary()
    expected = (0.02 - 0.04 + 0.06 + 0.08) / 4
    assert s["avg_trade_pnl_pct"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 8. Full backtester integration — new keys survive end-to-end
# ---------------------------------------------------------------------------


def test_backtester_full_run_new_keys():
    df = generate_ohlcv("ETH/USDT", n_candles=600)
    bt = Backtester(EMACrossStrategy(), initial_capital=10_000)
    result = bt.run(df)
    s = result.summary()
    if not s:
        pytest.skip("No trades generated — skipping key check")
    missing = _REQUIRED_KEYS - s.keys()
    assert not missing, f"Missing keys from full-run summary: {missing}"
    assert isinstance(s["max_consecutive_losses"], int)
    assert s["max_consecutive_losses"] >= 0
    assert s["profit_factor"] >= 0.0
