"""
Strategy comparison: runs all strategies across all symbols and prints a ranked table.
CI gate: exits with code 1 if every strategy falls below --min-sharpe.
"""
import sys
import argparse
import pandas as pd
from backtest.runner import Backtester
from backtest.synthetic import generate_ohlcv, ASSET_PARAMS
from strategies.ema_cross import EMACrossStrategy
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger import BollingerStrategy

STRATEGIES = [
    EMACrossStrategy(),
    RSIStrategy(),
    MACDStrategy(),
    BollingerStrategy(),
]


def run_comparison(symbols: list[str], n_candles: int = 1000) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        df = generate_ohlcv(symbol, n_candles=n_candles)
        for strategy in STRATEGIES:
            result = Backtester(strategy).run(df)
            s = result.summary()
            rows.append({
                "strategy": strategy.name,
                "symbol": symbol,
                "trades": s.get("total_trades", 0),
                "win_rate": s.get("win_rate", 0),
                "total_pnl": s.get("total_pnl_pct", 0),
                "sharpe": s.get("sharpe", 0),
                "max_dd": s.get("max_drawdown", 0),
            })
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame):
    # Per-strategy aggregate (mean across symbols)
    agg = (
        df.groupby("strategy")
        .agg(
            avg_trades=("trades", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_pnl=("total_pnl", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_max_dd=("max_dd", "mean"),
        )
        .sort_values("avg_sharpe", ascending=False)
    )

    print("\n=== Strategy Comparison (avg across all symbols) ===\n")
    print(f"{'Strategy':<14} {'Trades':>8} {'Win Rate':>10} {'Total PnL':>11} {'Sharpe':>8} {'Max DD':>9}")
    print("-" * 65)
    for name, row in agg.iterrows():
        print(
            f"{name:<14} {row.avg_trades:>8.1f} {row.avg_win_rate:>10.1%} "
            f"{row.avg_pnl:>+11.2%} {row.avg_sharpe:>8.2f} {row.avg_max_dd:>9.2%}"
        )

    print("\n=== Per-Symbol Detail ===\n")
    print(f"{'Strategy':<14} {'Symbol':<12} {'Trades':>7} {'Win Rate':>10} {'PnL':>9} {'Sharpe':>8}")
    print("-" * 65)
    for _, row in df.sort_values(["strategy", "symbol"]).iterrows():
        print(
            f"{row.strategy:<14} {row.symbol:<12} {row.trades:>7} "
            f"{row.win_rate:>10.1%} {row.total_pnl:>+9.2%} {row.sharpe:>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=None,
        help="CI gate: fail if best strategy avg Sharpe is below this value",
    )
    args = parser.parse_args()

    from bot.symbols import load_symbols
    symbols = load_symbols(args.symbols)

    df = run_comparison(symbols, n_candles=args.limit)
    print_report(df)

    if args.min_sharpe is not None:
        best_sharpe = df.groupby("strategy")["sharpe"].mean().max()
        if best_sharpe < args.min_sharpe:
            print(f"\n[CI FAIL] Best avg Sharpe {best_sharpe:.2f} < threshold {args.min_sharpe}")
            sys.exit(1)
        else:
            print(f"\n[CI PASS] Best avg Sharpe {best_sharpe:.2f} >= threshold {args.min_sharpe}")


if __name__ == "__main__":
    main()
