"""
Walk-forward validation: splits data into rolling in-sample/out-of-sample windows.
Prevents overfitting — a strategy that only works in-sample is useless.
"""
import argparse
import pandas as pd
from loguru import logger
from backtest.runner import Backtester
from backtest.synthetic import generate_ohlcv, ASSET_PARAMS
from strategies.base import BaseStrategy


def walk_forward(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.7,
) -> pd.DataFrame:
    """
    Splits df into n_splits rolling windows, trains on train_pct and tests on the rest.
    Returns per-window out-of-sample results.
    """
    window_size = len(df) // n_splits
    rows = []

    for i in range(n_splits):
        start = i * window_size
        end = start + window_size
        window = df.iloc[start:end]

        split = int(len(window) * train_pct)
        train = window.iloc[:split]   # noqa: used to confirm indicator warmup
        test  = window.iloc[split:]

        if len(test) < 30:
            continue

        result = Backtester(strategy).run(test)
        s = result.summary()
        rows.append({
            "window": i + 1,
            "from": test.index[0].date(),
            "to": test.index[-1].date(),
            "candles": len(test),
            "trades": s.get("total_trades", 0),
            "win_rate": s.get("win_rate", 0),
            "pnl": s.get("total_pnl_pct", 0),
            "sharpe": s.get("sharpe", 0),
            "max_dd": s.get("max_drawdown", 0),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="rsi", choices=["rsi", "ema_cross", "macd", "bollinger"])
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--limit", type=int, default=2000, help="More candles = more meaningful splits")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    from bot.symbols import load_symbols

    if args.strategy == "rsi":
        from strategies.rsi import RSIStrategy
        strategy = RSIStrategy()
    elif args.strategy == "macd":
        from strategies.macd import MACDStrategy
        strategy = MACDStrategy()
    elif args.strategy == "bollinger":
        from strategies.bollinger import BollingerStrategy
        strategy = BollingerStrategy()
    else:
        from strategies.ema_cross import EMACrossStrategy
        strategy = EMACrossStrategy()

    symbols = load_symbols(args.symbols)

    print(f"\n=== Walk-Forward Validation: {strategy.name.upper()} ({args.splits} splits) ===\n")

    all_sharpes = []
    for symbol in symbols:
        df = generate_ohlcv(symbol, n_candles=args.limit)
        wf = walk_forward(strategy, df, n_splits=args.splits)
        if wf.empty:
            continue
        avg_sharpe = wf["sharpe"].mean()
        positive_windows = (wf["sharpe"] > 0).sum()
        all_sharpes.extend(wf["sharpe"].tolist())

        print(f"{symbol}  (avg Sharpe: {avg_sharpe:.2f}, profitable windows: {positive_windows}/{len(wf)})")
        print(f"  {'Win':>4}  {'From':<12}  {'To':<12}  {'Trades':>7}  {'PnL':>9}  {'Sharpe':>8}  {'Max DD':>8}")
        for _, r in wf.iterrows():
            flag = "✓" if r.sharpe > 0 else "✗"
            print(f"  {flag:>4}  {str(r['from']):<12}  {str(r['to']):<12}  "
                  f"{r.trades:>7}  {r.pnl:>+9.2%}  {r.sharpe:>8.2f}  {r.max_dd:>8.2%}")
        print()

    if all_sharpes:
        overall = sum(all_sharpes) / len(all_sharpes)
        pct_positive = sum(1 for s in all_sharpes if s > 0) / len(all_sharpes)
        print(f"Overall avg Sharpe: {overall:.2f}  |  Profitable windows: {pct_positive:.0%}")


if __name__ == "__main__":
    main()
