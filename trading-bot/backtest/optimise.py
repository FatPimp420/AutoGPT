"""
Grid-search parameter optimiser for any strategy.
Finds the parameter combination with the best average Sharpe or PnL across all symbols.
"""
import sys
import itertools
import argparse
import pandas as pd
from loguru import logger
from backtest.runner import Backtester
from backtest.synthetic import generate_ohlcv, ASSET_PARAMS


def grid_search(strategy_cls, param_grid: dict, symbols: list[str], n_candles: int = 1000) -> pd.DataFrame:
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    logger.info(f"Grid search: {len(combos)} combinations x {len(symbols)} symbols")

    rows = []
    data = {sym: generate_ohlcv(sym, n_candles=n_candles) for sym in symbols}

    for combo in combos:
        params = dict(zip(keys, combo))
        sharpes, pnls, wrs = [], [], []
        try:
            strategy = strategy_cls(**params)
        except Exception:
            continue
        for sym, df in data.items():
            result = Backtester(strategy).run(df)
            s = result.summary()
            if s:
                sharpes.append(s["sharpe"])
                pnls.append(s["total_pnl_pct"])
                wrs.append(s["win_rate"])
        if sharpes:
            rows.append({**params, "avg_sharpe": sum(sharpes) / len(sharpes),
                         "avg_pnl": sum(pnls) / len(pnls),
                         "avg_win_rate": sum(wrs) / len(wrs),
                         "symbols_traded": len(sharpes)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="rsi", choices=["rsi", "ema_cross", "macd", "bollinger"])
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--rank-by", default="sharpe", choices=["sharpe", "pnl", "win_rate"],
                        help="Metric to rank results by")
    args = parser.parse_args()

    from bot.symbols import load_symbols
    symbols = load_symbols(args.symbols)

    if args.strategy == "rsi":
        from strategies.rsi import RSIStrategy
        param_grid = {
            "period":    [7, 10, 14, 20],
            "oversold":  [20, 25, 30, 35],
            "overbought":[65, 70, 75, 80],
        }
        strategy_cls = RSIStrategy

    elif args.strategy == "ema_cross":
        from strategies.ema_cross import EMACrossStrategy
        param_grid = {"fast": [5, 7, 9, 12], "slow": [18, 21, 26, 30]}
        strategy_cls = EMACrossStrategy

    elif args.strategy == "macd":
        from strategies.macd import MACDStrategy
        param_grid = {"fast": [8, 12], "slow": [21, 26], "signal": [7, 9]}
        strategy_cls = MACDStrategy

    else:
        from strategies.bollinger import BollingerStrategy
        param_grid = {"period": [15, 20, 25], "std_dev": [1.5, 2.0, 2.5]}
        strategy_cls = BollingerStrategy

    rank_col = {"sharpe": "avg_sharpe", "pnl": "avg_pnl", "win_rate": "avg_win_rate"}[args.rank_by]
    results = grid_search(strategy_cls, param_grid, symbols, n_candles=args.limit)
    results = results.sort_values(rank_col, ascending=False)

    print(f"\n=== Top {args.top} {args.strategy.upper()} — ranked by {args.rank_by.upper()} ===\n")
    top = results.head(args.top)
    param_cols = [c for c in top.columns if c not in ("avg_sharpe", "avg_pnl", "avg_win_rate", "symbols_traded")]
    header = f"{'Rank':>5}  " + "  ".join(f"{c:>12}" for c in param_cols) + \
             f"  {'Sharpe':>8}  {'Avg PnL':>9}  {'Win Rate':>9}"
    print(header)
    print("-" * len(header))
    for i, (_, row) in enumerate(top.iterrows(), 1):
        params_str = "  ".join(f"{row[c]:>12}" for c in param_cols)
        print(f"{i:>5}  {params_str}  {row.avg_sharpe:>8.2f}  {row.avg_pnl:>+9.2%}  {row.avg_win_rate:>9.1%}")

    best = results.iloc[0]
    best_params = {c: best[c] for c in param_cols}
    print(f"\nBest params: {best_params}  →  Sharpe {best.avg_sharpe:.2f}, PnL {best.avg_pnl:+.2%}, Win rate {best.avg_win_rate:.1%}")


if __name__ == "__main__":
    main()
