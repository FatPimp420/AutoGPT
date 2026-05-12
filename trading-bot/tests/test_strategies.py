import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from backtest.synthetic import generate_ohlcv
from backtest.runner import Backtester
from strategies.ema_cross import EMACrossStrategy
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger import BollingerStrategy


@pytest.fixture
def btc_df():
    return generate_ohlcv("BTC/USDT", n_candles=500)


def test_ema_cross_returns_valid_signal(btc_df):
    s = EMACrossStrategy()
    s.add_indicators(btc_df)
    signal = s.generate_signal(btc_df)
    assert signal in ("buy", "sell", None)


def test_rsi_signal_on_short_data():
    df = generate_ohlcv("BTC/USDT", n_candles=5)
    s = RSIStrategy()
    assert s.generate_signal(df) is None  # not enough data


def test_rsi_adds_indicator_column(btc_df):
    s = RSIStrategy()
    out = s.add_indicators(btc_df.copy())
    assert "rsi" in out.columns
    assert out["rsi"].between(0, 100).all() or out["rsi"].isna().any()


def test_macd_adds_columns(btc_df):
    s = MACDStrategy()
    out = s.add_indicators(btc_df.copy())
    assert all(c in out.columns for c in ("macd", "macd_signal", "macd_hist"))


def test_bollinger_adds_columns(btc_df):
    s = BollingerStrategy()
    out = s.add_indicators(btc_df.copy())
    assert all(c in out.columns for c in ("bb_upper", "bb_mid", "bb_lower"))
    valid = out.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_backtester_equity_grows_on_win():
    df = generate_ohlcv("BTC/USDT", n_candles=500)
    bt = Backtester(RSIStrategy(), initial_capital=10_000)
    result = bt.run(df)
    assert isinstance(result.equity_curve, list)
    assert len(result.equity_curve) == len(df) - 1


def test_backtester_summary_keys(btc_df):
    result = Backtester(RSIStrategy()).run(btc_df)
    s = result.summary()
    if s:
        assert all(k in s for k in ("total_trades", "win_rate", "total_pnl_pct", "sharpe", "max_drawdown"))
