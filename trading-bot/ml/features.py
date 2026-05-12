"""Feature engineering for the ML signal layer.

Takes a DataFrame that already has TA indicators computed by a BaseStrategy
and returns a clean feature matrix suitable for sklearn classifiers.
"""

import pandas as pd
import numpy as np


# Columns that belong to raw OHLCV data — never included in feature output
_OHLCV_COLS = {"open", "high", "low", "close", "volume", "timestamp"}


def build_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """Build a feature matrix from a DataFrame with TA indicators already present.

    Args:
        df: DataFrame with at minimum a ``close`` column plus any indicator
            columns added by a BaseStrategy (rsi, macd_hist, ema_fast,
            ema_slow, bb_upper, bb_lower, bb_mid, volume).
        sentiment_score: A scalar sentiment value in [-1, 1] (or any float)
            that is broadcast to all rows as a constant feature.

    Returns:
        DataFrame containing only the derived feature columns (no OHLCV, no
        raw indicators).  All NaN values are filled with 0.
    """
    features: dict[str, pd.Series] = {}

    close = df["close"]

    # --- RSI (normalise 0-100 → 0-1) ---
    if "rsi" in df.columns:
        features["rsi"] = df["rsi"] / 100.0

    # --- MACD histogram ---
    if "macd_hist" in df.columns:
        features["macd_hist"] = df["macd_hist"]

    # --- EMA ratio: ema_fast / ema_slow - 1 ---
    if "ema_fast" in df.columns and "ema_slow" in df.columns:
        features["ema_ratio"] = df["ema_fast"] / (df["ema_slow"] + 1e-9) - 1.0

    # --- Bollinger Band position: where is close within the bands ---
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        band_width = df["bb_upper"] - df["bb_lower"] + 1e-9
        features["bb_pct"] = (close - df["bb_lower"]) / band_width

    # --- Rolling returns ---
    for n in (1, 3, 5):
        features[f"returns_{n}"] = close.pct_change(n)

    # --- Volume z-score (20-period rolling) ---
    if "volume" in df.columns:
        vol = df["volume"]
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        features["volume_zscore"] = (vol - vol_mean) / (vol_std + 1e-9)

    # --- Sentiment (scalar broadcast) ---
    features["sentiment_score"] = pd.Series(
        sentiment_score, index=df.index, dtype=float
    )

    result = pd.DataFrame(features, index=df.index).fillna(0.0)
    return result


def label_data(
    df: pd.DataFrame,
    forward_periods: int = 6,
    threshold: float = 0.01,
) -> pd.Series:
    """Generate supervised learning labels based on forward returns.

    Args:
        df: DataFrame containing at minimum a ``close`` column.
        forward_periods: Number of candles to look ahead when computing the
            future return.
        threshold: Minimum absolute return required to assign a directional
            label.  Returns above +threshold → 1 (profitable buy).
            Returns below -threshold → -1 (profitable sell).
            Otherwise → 0 (no clear edge).

    Returns:
        pd.Series of int labels with values in {-1, 0, 1}, aligned to
        ``df.index`` and of the same length.  The final ``forward_periods``
        rows will have label 0 because future data is unavailable.
    """
    future_close = df["close"].shift(-forward_periods)
    future_return = (future_close - df["close"]) / (df["close"] + 1e-9)

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    # Rows where future data is unavailable default to 0 (already set above)
    return labels
