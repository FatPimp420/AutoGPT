"""Training script for MLSignalModel.

Usage (from the trading-bot root):
    python -m ml.trainer --symbol BTC/USDT --strategy rsi
    python -m ml.trainer --symbol ETH/USDT --strategy macd --days 180
"""

import argparse
import os
import sys

# Ensure the trading-bot root is on the path when run as a script or module
_BOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BOT_ROOT not in sys.path:
    sys.path.insert(0, _BOT_ROOT)

import numpy as np
import pandas as pd

from backtest.synthetic import generate_ohlcv
from ml.features import build_features, label_data
from ml.model import MLSignalModel

# Registry of available strategies
_STRATEGY_MAP: dict[str, type] = {}

def _load_strategies() -> None:
    from strategies.rsi import RSIStrategy
    from strategies.macd import MACDStrategy
    from strategies.ema_cross import EMACrossStrategy
    from strategies.bollinger import BollingerStrategy
    _STRATEGY_MAP.update({
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "ema_cross": EMACrossStrategy,
        "bollinger": BollingerStrategy,
    })


def train(
    symbol: str,
    strategy_name: str,
    days: int = 365,
    forward_periods: int = 6,
    sentiment_score: float = 0.0,
) -> MLSignalModel:
    """Train an MLSignalModel on synthetic OHLCV data and print evaluation stats.

    Args:
        symbol: Asset symbol understood by ``generate_ohlcv``
                (e.g. ``"BTC/USDT"``).
        strategy_name: One of ``rsi``, ``macd``, ``ema_cross``, ``bollinger``.
        days: Approximate number of trading days.  Converted to candle count
              assuming 24 hourly candles per day.
        forward_periods: Look-ahead window used by ``label_data``.
        sentiment_score: Constant sentiment value broadcast to all rows.

    Returns:
        The fitted ``MLSignalModel``.

    Raises:
        ValueError: If ``strategy_name`` is not recognised.
    """
    _load_strategies()
    if strategy_name not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {sorted(_STRATEGY_MAP)}"
        )

    # --- 1. Generate synthetic OHLCV ---
    n_candles = days * 24  # hourly candles
    print(f"\nGenerating {n_candles} synthetic candles for {symbol} …")
    df = generate_ohlcv(symbol, n_candles=n_candles)

    # --- 2. Add indicators via the chosen strategy ---
    strategy_cls = _STRATEGY_MAP[strategy_name]
    strategy = strategy_cls()
    enriched = strategy.add_indicators(df.copy())

    # --- 3. Build features + labels ---
    X = build_features(enriched, sentiment_score=sentiment_score)
    y = label_data(enriched, forward_periods=forward_periods)

    print(f"Feature matrix shape : {X.shape}")
    class_counts = y.value_counts().sort_index()
    print("Label distribution   :")
    for label_val, count in class_counts.items():
        name = {-1: "sell", 0: "hold", 1: "buy"}.get(int(label_val), str(label_val))
        pct = 100 * count / len(y)
        print(f"  {name:>4} ({label_val:+d}) : {count:5d}  ({pct:.1f}%)")

    # --- 4. Train the model ---
    print(f"\nTraining MLSignalModel with strategy='{strategy_name}' …")
    model = MLSignalModel(base_strategy=strategy)
    model.fit(enriched, sentiment_score=sentiment_score, forward_periods=forward_periods)

    # --- 5. Evaluate on the training set (in-sample, as a sanity check) ---
    from sklearn.metrics import accuracy_score

    X_all = build_features(enriched, sentiment_score=sentiment_score)
    if model._scaler is not None and not X_all.empty:
        X_scaled = model._scaler.transform(X_all.values)
        y_pred = model._clf.predict(X_scaled)
        mask = pd.notna(y)
        acc = accuracy_score(y.loc[mask].values, y_pred[mask])
        print(f"\nIn-sample accuracy   : {acc:.4f}")

    # --- 6. Feature importances ---
    if model._clf is not None:
        importances = model._clf.feature_importances_
        feature_names = X.columns.tolist()
        ranked = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        print("\nTop-5 feature importances:")
        for fname, imp in ranked[:5]:
            print(f"  {fname:<20} {imp:.4f}")

    return model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLSignalModel on synthetic OHLCV data."
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Asset symbol (default: BTC/USDT)",
    )
    parser.add_argument(
        "--strategy",
        default="rsi",
        choices=["rsi", "macd", "ema_cross", "bollinger"],
        help="Base strategy to wrap (default: rsi)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of synthetic data to generate (default: 365)",
    )
    parser.add_argument(
        "--forward-periods",
        type=int,
        default=6,
        dest="forward_periods",
        help="Look-ahead candles for label generation (default: 6)",
    )
    parser.add_argument(
        "--sentiment",
        type=float,
        default=0.0,
        help="Constant sentiment score broadcast to all rows (default: 0.0)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    trained_model = train(
        symbol=args.symbol,
        strategy_name=args.strategy,
        days=args.days,
        forward_periods=args.forward_periods,
        sentiment_score=args.sentiment,
    )
    print("\nTraining complete.")
