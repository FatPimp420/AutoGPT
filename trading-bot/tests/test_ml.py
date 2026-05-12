"""Tests for the ML signal layer (ml/ package).

Run from the trading-bot root:
    python -m pytest tests/test_ml.py -v
"""

import os
import sys

# Make sure the trading-bot root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle

import pandas as pd
import pytest

from backtest.synthetic import generate_ohlcv
from ml.features import build_features, label_data
from ml.model import MLSignalModel
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.ema_cross import EMACrossStrategy
from strategies.bollinger import BollingerStrategy

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


@pytest.fixture(scope="module")
def btc_500():
    """500-candle BTC/USDT synthetic OHLCV DataFrame."""
    return generate_ohlcv("BTC/USDT", n_candles=500)


@pytest.fixture(scope="module")
def btc_1000():
    """1000-candle BTC/USDT synthetic OHLCV DataFrame."""
    return generate_ohlcv("BTC/USDT", n_candles=1000)


@pytest.fixture(scope="module")
def btc_500_with_rsi(btc_500):
    """500-candle DataFrame with RSI indicators added."""
    strategy = RSIStrategy()
    return strategy.add_indicators(btc_500.copy())


@pytest.fixture(scope="module")
def btc_1000_with_rsi(btc_1000):
    """1000-candle DataFrame with RSI indicators added."""
    strategy = RSIStrategy()
    return strategy.add_indicators(btc_1000.copy())


# ---------------------------------------------------------------------------
# 1. build_features returns a DataFrame with no OHLCV columns
# ---------------------------------------------------------------------------


def test_build_features_returns_dataframe(btc_500_with_rsi):
    """Output has no OHLCV columns, only derived feature columns."""
    features = build_features(btc_500_with_rsi)

    assert isinstance(features, pd.DataFrame), "build_features must return a DataFrame"
    assert len(features) == len(btc_500_with_rsi), "Feature DF must have same row count"

    # None of the raw OHLCV column names should appear in the feature output
    overlap = set(features.columns) & _OHLCV_COLS
    assert overlap == set(), (
        f"Feature DataFrame must not contain OHLCV columns; found: {overlap}"
    )

    # There must be at least one feature column
    assert len(features.columns) > 0, "Feature DataFrame must not be empty"


# ---------------------------------------------------------------------------
# 2. build_features includes the sentiment_score column
# ---------------------------------------------------------------------------


def test_build_features_includes_sentiment(btc_500_with_rsi):
    """sentiment_score column is present and equals the passed scalar."""
    score = 0.42
    features = build_features(btc_500_with_rsi, sentiment_score=score)

    assert "sentiment_score" in features.columns, (
        "Feature DataFrame must include 'sentiment_score'"
    )
    assert (features["sentiment_score"] == score).all(), (
        "All rows of 'sentiment_score' must equal the passed scalar"
    )


# ---------------------------------------------------------------------------
# 3. label_data returns a Series of the correct length
# ---------------------------------------------------------------------------


def test_label_data_returns_series(btc_500):
    """Return value is a pd.Series of the same length as the input."""
    labels = label_data(btc_500)

    assert isinstance(labels, pd.Series), "label_data must return a pd.Series"
    assert len(labels) == len(btc_500), (
        "Returned Series must have the same length as the input DataFrame"
    )
    # All values must be in {-1, 0, 1}
    unique_vals = set(labels.unique())
    assert unique_vals.issubset({-1, 0, 1}), (
        f"Labels must be in {{-1, 0, 1}}; found {unique_vals}"
    )


# ---------------------------------------------------------------------------
# 4. label_data produces all three classes with 1000 candles
# ---------------------------------------------------------------------------


def test_label_data_has_all_classes(btc_1000):
    """With 1000 candles all three classes (-1, 0, 1) should appear."""
    labels = label_data(btc_1000, forward_periods=6, threshold=0.003)

    unique_vals = set(labels.unique())
    assert -1 in unique_vals, "Label -1 (sell) not present with 1000 candles"
    assert 0 in unique_vals, "Label 0 (hold) not present with 1000 candles"
    assert 1 in unique_vals, "Label 1 (buy) not present with 1000 candles"


# ---------------------------------------------------------------------------
# 5. MLSignalModel.fit + generate_signal produces a valid output
# ---------------------------------------------------------------------------


def test_ml_model_fit_and_predict(btc_500):
    """Fit on 500 candles; generate_signal returns 'buy', 'sell', or None."""
    strategy = RSIStrategy()
    model = MLSignalModel(base_strategy=strategy, min_confidence=0.0)
    model.fit(btc_500)

    # generate_signal should return a valid value (buy, sell, or None)
    enriched = strategy.add_indicators(btc_500.copy())
    signal = model.generate_signal(enriched)

    assert signal in ("buy", "sell", None), (
        f"generate_signal must return 'buy', 'sell', or None; got {signal!r}"
    )

    # predict_proba keys must be exactly {buy, sell, hold}
    proba = model.predict_proba(enriched)
    assert set(proba.keys()) == {"buy", "sell", "hold"}, (
        f"predict_proba must return keys {{buy, sell, hold}}; got {set(proba.keys())}"
    )
    total = sum(proba.values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities must sum to 1.0; got {total}"


# ---------------------------------------------------------------------------
# 6. High min_confidence vetoes all signals
# ---------------------------------------------------------------------------


def test_ml_model_vetoes_low_confidence(btc_500):
    """With min_confidence=0.99 (impossible to meet), every signal is None."""
    strategy = RSIStrategy()
    # Use a threshold so low that we're likely to get buy/sell from the base strategy
    model = MLSignalModel(base_strategy=strategy, min_confidence=0.99)
    model.fit(btc_500)

    # We iterate over many windows; at least one of them should be vetoed
    enriched = strategy.add_indicators(btc_500.copy())
    all_none = True
    for i in range(50, len(enriched)):
        window = enriched.iloc[:i]
        sig = model.generate_signal(window)
        if sig is not None:
            all_none = False
            break

    # With min_confidence=0.99 it is effectively impossible for RandomForest
    # on short windows to report >= 99% confidence; nearly all should be None
    # We assert that the *base strategy* would have produced at least one signal
    # to confirm the veto is happening (not just that there are no raw signals).
    base_signals = []
    for i in range(50, len(enriched)):
        window = enriched.iloc[:i]
        base_sig = strategy.generate_signal(window)
        if base_sig is not None:
            base_signals.append(base_sig)

    if base_signals:
        # We have raw signals to veto; all ML outputs should be None
        assert all_none, (
            "Expected ALL signals to be vetoed when min_confidence=0.99, "
            "but some passed through"
        )
    else:
        # If the strategy itself never fired, the test still passes trivially
        pytest.skip("Base strategy produced no signals on this data window; skipping veto assertion")


# ---------------------------------------------------------------------------
# 7. Save and load produce identical predictions
# ---------------------------------------------------------------------------


def test_ml_model_save_load(btc_500, tmp_path):
    """Save model to disk, load it back, verify predictions match."""
    strategy = RSIStrategy()
    model = MLSignalModel(base_strategy=strategy, min_confidence=0.5)
    model.fit(btc_500)

    save_path = str(tmp_path / "model.pkl")
    model.save(save_path)

    # Load into a new model instance (same strategy / confidence)
    loaded = MLSignalModel(base_strategy=RSIStrategy(), min_confidence=0.5)
    loaded.load(save_path)

    # Both should produce the same probabilities on the same data
    enriched = strategy.add_indicators(btc_500.copy())
    proba_original = model.predict_proba(enriched)
    proba_loaded = loaded.predict_proba(enriched)

    for key in ("buy", "sell", "hold"):
        assert abs(proba_original[key] - proba_loaded[key]) < 1e-9, (
            f"Loaded model predict_proba['{key}'] differs: "
            f"{proba_original[key]} vs {proba_loaded[key]}"
        )

    # Signals must also match
    original_signal = model.generate_signal(enriched)
    loaded_signal = loaded.generate_signal(enriched)
    assert original_signal == loaded_signal, (
        f"Signals differ after save/load: {original_signal!r} vs {loaded_signal!r}"
    )
