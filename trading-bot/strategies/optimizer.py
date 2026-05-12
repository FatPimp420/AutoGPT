"""Grid-search parameter optimizer for trading strategies.

Finds the best strategy parameters per symbol using the backtest engine on synthetic data,
and saves results to a JSON config file.

Usage:
  python -m strategies.optimizer --symbols BTC/USDT ETH/USDT --strategies rsi ema_cross --days 90
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any

from backtest.runner import Backtester
from backtest.synthetic import generate_ohlcv
from strategies.factory import get_strategy


# ---------------------------------------------------------------------------
# Parameter grids: maps strategy name → list of param dicts (≤27 combos each)
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "rsi": [
        {"period": p, "overbought": ob, "oversold": os_}
        for p in (7, 14, 21)
        for ob in (65, 70, 75)
        for os_ in (25, 30, 35)
    ],  # 3×3×3 = 27 combos
    "ema_cross": [
        {"fast": f, "slow": s}
        for f in (5, 9, 12)
        for s in (21, 50, 100)
    ],  # 3×3 = 9 combos
    "macd": [
        {"fast": f, "slow": s, "signal": sig}
        for f in (8, 12, 16)
        for s in (21, 26, 30)
        for sig in (7, 9)
    ],  # 3×3×2 = 18 combos
    "bollinger": [
        {"period": p, "std_dev": sd}
        for p in (10, 20, 30)
        for sd in (1.5, 2.0, 2.5)
    ],  # 3×3 = 9 combos
    "vwap": [
        {"period": p, "band_mult": bm}
        for p in (10, 20, 30)
        for bm in (1.0, 1.5, 2.0)
    ],  # 3×3 = 9 combos
    "stoch_rsi": [
        {"rsi_period": rp, "stoch_period": sp, "oversold": os_, "overbought": ob}
        for rp in (10, 14)
        for sp in (10, 14)
        for os_ in (15, 20)
        for ob in (80, 85)
    ],  # 2×2×2×2 = 16 combos
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class OptimizeResult:
    strategy: str
    symbol: str
    best_params: dict[str, Any]
    best_sharpe: float
    best_calmar: float
    all_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core optimization functions
# ---------------------------------------------------------------------------
def optimize_strategy(
    symbol: str,
    strategy_name: str,
    days: int = 90,
    metric: str = "sharpe",
) -> OptimizeResult:
    """Run a grid search over PARAM_GRIDS[strategy_name] for the given symbol.

    For each parameter combination, synthetic OHLCV data is generated, the
    strategy is instantiated, and a full backtest is performed.  The
    combination with the highest *metric* value is declared the winner.

    Args:
        symbol:        Asset symbol supported by generate_ohlcv (e.g. "BTC/USDT").
        strategy_name: One of the keys in PARAM_GRIDS / STRATEGY_REGISTRY.
        days:          Number of daily candles (hourly candles = days × 24).
        metric:        Optimisation target: "sharpe" or "calmar".

    Returns:
        OptimizeResult with best_params and aggregated results.

    Raises:
        KeyError: If strategy_name is not in PARAM_GRIDS.
    """
    if strategy_name not in PARAM_GRIDS:
        raise KeyError(
            f"No parameter grid defined for strategy {strategy_name!r}. "
            f"Available: {sorted(PARAM_GRIDS)}"
        )

    # Use hourly candles: days × 24 gives enough signal for warmup periods
    n_candles = days * 24
    df = generate_ohlcv(symbol, n_candles=n_candles)

    grid = PARAM_GRIDS[strategy_name]
    all_results: list[dict[str, Any]] = []

    for params in grid:
        strategy = get_strategy(strategy_name, **params)
        bt = Backtester(strategy)
        result = bt.run(df.copy())
        summary = result.summary()

        if summary:
            sharpe = float(summary.get("sharpe", 0.0))
            calmar = float(summary.get("calmar_ratio", 0.0))
        else:
            # No trades — treat as zero performance
            sharpe = 0.0
            calmar = 0.0

        all_results.append({
            "params": dict(params),
            "sharpe": sharpe,
            "calmar": calmar,
            "total_trades": summary.get("total_trades", 0) if summary else 0,
            "total_pnl_pct": summary.get("total_pnl_pct", 0.0) if summary else 0.0,
        })

    # Select best by requested metric
    metric_key = "sharpe" if metric == "sharpe" else "calmar"
    best_entry = max(all_results, key=lambda r: r[metric_key])

    return OptimizeResult(
        strategy=strategy_name,
        symbol=symbol,
        best_params=dict(best_entry["params"]),
        best_sharpe=float(best_entry["sharpe"]),
        best_calmar=float(best_entry["calmar"]),
        all_results=all_results,
    )


def optimize_portfolio(
    symbols: list[str],
    strategy_names: list[str] | None = None,
    days: int = 90,
    metric: str = "sharpe",
) -> dict[str, dict[str, OptimizeResult]]:
    """Run optimize_strategy for every symbol × strategy combination.

    Args:
        symbols:        List of asset symbols.
        strategy_names: Strategies to optimise; defaults to all keys in PARAM_GRIDS.
        days:           Candle count forwarded to optimize_strategy.
        metric:         Optimisation target forwarded to optimize_strategy.

    Returns:
        Nested dict ``{symbol: {strategy_name: OptimizeResult}}``.
    """
    if strategy_names is None:
        strategy_names = list(PARAM_GRIDS.keys())

    portfolio: dict[str, dict[str, OptimizeResult]] = {}
    for symbol in symbols:
        portfolio[symbol] = {}
        for strat in strategy_names:
            portfolio[symbol][strat] = optimize_strategy(
                symbol, strat, days=days, metric=metric
            )

    return portfolio


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_best_params(
    results: dict[str, dict[str, OptimizeResult]],
    path: str = "./strategy_config.json",
) -> None:
    """Persist the best parameters from an optimize_portfolio result to JSON.

    Output format::

        {
            "BTC/USDT": {
                "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                ...
            },
            ...
        }
    """
    config: dict[str, dict[str, dict[str, Any]]] = {}
    for symbol, strat_map in results.items():
        config[symbol] = {}
        for strat_name, opt_result in strat_map.items():
            config[symbol][strat_name] = opt_result.best_params

    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)


def load_best_params(path: str = "./strategy_config.json") -> dict:
    """Load best parameters from a previously saved JSON config.

    Returns an empty dict if the file is missing (no error raised).
    """
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search strategy parameter optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT"],
        help="Asset symbols to optimise (space-separated)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategy names to optimise; defaults to all grids",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of synthetic candle-days to generate",
    )
    parser.add_argument(
        "--metric",
        choices=["sharpe", "calmar"],
        default="sharpe",
        help="Metric to maximise",
    )
    parser.add_argument(
        "--output",
        default="./strategy_config.json",
        help="Output JSON path",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print(
        f"Optimising {args.symbols} × "
        f"{args.strategies or list(PARAM_GRIDS)} | "
        f"days={args.days}, metric={args.metric}"
    )

    results = optimize_portfolio(
        symbols=args.symbols,
        strategy_names=args.strategies,
        days=args.days,
        metric=args.metric,
    )

    for symbol, strat_map in results.items():
        for strat, opt in strat_map.items():
            print(
                f"  {symbol}/{strat}: best_params={opt.best_params} "
                f"sharpe={opt.best_sharpe:.4f} calmar={opt.best_calmar:.4f}"
            )

    save_best_params(results, path=args.output)
    print(f"\nConfig saved to {args.output}")


if __name__ == "__main__":
    main()
