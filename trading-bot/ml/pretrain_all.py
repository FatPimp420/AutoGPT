"""Batch pre-training script: trains MLSignalModel for every strategy × symbol
combination and saves the resulting .pkl files to ./ml_models/.

Usage (from the trading-bot root):
    python -m ml.pretrain_all
    python -m ml.pretrain_all --days 180 --output-dir /tmp/ml_models
"""

import argparse
import os
import sys
import time

# Ensure the trading-bot root is on the path when run as a script or module
_BOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BOT_ROOT not in sys.path:
    sys.path.insert(0, _BOT_ROOT)

from sklearn.metrics import accuracy_score
import pandas as pd

from backtest.synthetic import generate_ohlcv
from ml.features import build_features, label_data
from ml.model import MLSignalModel
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.ema_cross import EMACrossStrategy
from strategies.bollinger import BollingerStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type] = {
    "rsi": RSIStrategy,
    "ema_cross": EMACrossStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
}

SYMBOLS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_filename(symbol: str, strategy_name: str) -> str:
    """Return the bare filename (no directory) for a model file."""
    safe_symbol = symbol.replace("/", "_")
    return f"{safe_symbol}_{strategy_name}.pkl"


def pretrain_all(
    days: int = 365,
    forward_periods: int = 6,
    sentiment_score: float = 0.0,
    output_dir: str = "./ml_models",
) -> dict[str, float]:
    """Train and save one MLSignalModel per strategy × symbol combination.

    Args:
        days: Approximate number of trading days of synthetic OHLCV to generate
              (converted to hourly candles internally: ``days × 24``).
        forward_periods: Look-ahead window used by ``label_data``.
        sentiment_score: Constant sentiment value broadcast to all training rows.
        output_dir: Directory where ``.pkl`` files are written.  Created if it
                    does not exist.

    Returns:
        Dict mapping ``"SYMBOL_strategy"`` keys to in-sample accuracy floats.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_candles = days * 24  # hourly candles
    results: dict[str, float] = {}

    total = len(STRATEGIES) * len(SYMBOLS)
    done = 0

    for strategy_name, strategy_cls in STRATEGIES.items():
        for symbol in SYMBOLS:
            done += 1
            safe_symbol = symbol.replace("/", "_")
            key = f"{safe_symbol}_{strategy_name}"
            out_path = os.path.join(output_dir, _model_filename(symbol, strategy_name))

            # 1. Generate synthetic OHLCV
            df = generate_ohlcv(symbol, n_candles=n_candles)

            # 2. Add strategy indicators
            strategy = strategy_cls()
            enriched = strategy.add_indicators(df.copy())

            # 3. Train MLSignalModel
            model = MLSignalModel(base_strategy=strategy)
            model.fit(enriched, sentiment_score=sentiment_score, forward_periods=forward_periods)

            # 4. Compute in-sample accuracy for reporting
            X_all = build_features(enriched, sentiment_score=sentiment_score)
            y_all = label_data(enriched, forward_periods=forward_periods)
            mask = pd.notna(y_all)
            acc = 0.0
            if model._scaler is not None and not X_all.empty and mask.any():
                X_scaled = model._scaler.transform(X_all.values)
                y_pred = model._clf.predict(X_scaled)
                acc = accuracy_score(y_all.loc[mask].values, y_pred[mask])

            # 5. Save the model
            model.save(out_path)

            results[key] = acc
            print(
                f"[{done:>2}/{total}] Trained {key:<22} → {out_path}  (acc={acc:.2f})"
            )

    print(f"\nAll {total} models saved to '{output_dir}'.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-train MLSignalModel for all strategies × symbols."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of synthetic OHLCV to generate per model (default: 365)",
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
    parser.add_argument(
        "--output-dir",
        default="./ml_models",
        dest="output_dir",
        help="Directory to write .pkl files to (default: ./ml_models)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.perf_counter()
    pretrain_all(
        days=args.days,
        forward_periods=args.forward_periods,
        sentiment_score=args.sentiment,
        output_dir=args.output_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.1f}s")
