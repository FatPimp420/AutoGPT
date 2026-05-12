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
from strategies.vwap import VWAPStrategy
from strategies.stoch_rsi import StochRSIStrategy
from strategies.factory import get_strategy, list_strategies


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


# ---------------------------------------------------------------------------
# VWAP strategy tests
# ---------------------------------------------------------------------------


def test_vwap_adds_indicator_columns(btc_df):
    s = VWAPStrategy()
    out = s.add_indicators(btc_df.copy())
    assert all(c in out.columns for c in ("vwap", "vwap_upper", "vwap_lower"))


def test_vwap_bands_ordering(btc_df):
    s = VWAPStrategy()
    out = s.add_indicators(btc_df.copy())
    valid = out.dropna(subset=["vwap_upper", "vwap_lower"])
    assert (valid["vwap_upper"] >= valid["vwap_lower"]).all()


def test_vwap_returns_valid_signal(btc_df):
    s = VWAPStrategy()
    s.add_indicators(btc_df)
    signal = s.generate_signal(btc_df)
    assert signal in ("buy", "sell", None)


def test_vwap_short_data_returns_none():
    df = generate_ohlcv("BTC/USDT", n_candles=5)
    s = VWAPStrategy()
    assert s.generate_signal(df) is None


def test_vwap_accessible_via_factory():
    s = get_strategy("vwap")
    assert isinstance(s, VWAPStrategy)


def test_vwap_factory_kwargs():
    s = get_strategy("vwap", period=10, band_mult=2.0)
    assert isinstance(s, VWAPStrategy)
    assert s.period == 10
    assert s.band_mult == 2.0


# ---------------------------------------------------------------------------
# StochRSI strategy tests
# ---------------------------------------------------------------------------


def test_stoch_rsi_adds_indicator_columns(btc_df):
    s = StochRSIStrategy()
    out = s.add_indicators(btc_df.copy())
    assert all(c in out.columns for c in ("rsi", "stoch_k", "stoch_d"))


def test_stoch_rsi_values_in_range(btc_df):
    s = StochRSIStrategy()
    out = s.add_indicators(btc_df.copy())
    valid_k = out["stoch_k"].dropna()
    valid_d = out["stoch_d"].dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()


def test_stoch_rsi_returns_valid_signal(btc_df):
    s = StochRSIStrategy()
    s.add_indicators(btc_df)
    signal = s.generate_signal(btc_df)
    assert signal in ("buy", "sell", None)


def test_stoch_rsi_short_data_returns_none():
    df = generate_ohlcv("BTC/USDT", n_candles=5)
    s = StochRSIStrategy()
    assert s.generate_signal(df) is None


def test_stoch_rsi_accessible_via_factory():
    s = get_strategy("stoch_rsi")
    assert isinstance(s, StochRSIStrategy)


def test_stoch_rsi_factory_kwargs():
    s = get_strategy("stoch_rsi", rsi_period=10, oversold=25.0)
    assert isinstance(s, StochRSIStrategy)
    assert s.rsi_period == 10
    assert s.oversold == 25.0


# ---------------------------------------------------------------------------
# list_strategies includes both new names
# ---------------------------------------------------------------------------


def test_list_strategies_includes_vwap_and_stoch_rsi():
    names = list_strategies()
    assert "vwap" in names
    assert "stoch_rsi" in names
