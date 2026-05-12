import asyncio
import json
import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger
from backtest.runner import Backtester, BacktestResult
from strategies.base import BaseStrategy


async def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    finally:
        await exchange.close()


async def backtest_symbol(
    symbol: str,
    strategy: BaseStrategy,
    timeframe: str = "1h",
    limit: int = 1000,
) -> tuple[str, BacktestResult]:
    logger.info(f"Fetching {limit} candles for {symbol} ({timeframe})...")
    df = await fetch_ohlcv(symbol, timeframe, limit)
    bt = Backtester(strategy)
    result = bt.run(df)
    return symbol, result


async def run_portfolio_backtest(
    symbols: list[str],
    strategy_cls: type[BaseStrategy],
    timeframe: str = "1h",
    limit: int = 1000,
    **strategy_kwargs,
) -> dict[str, dict]:
    tasks = [
        backtest_symbol(symbol, strategy_cls(**strategy_kwargs), timeframe, limit)
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summary = {}
    for item in results:
        if isinstance(item, Exception):
            logger.error(f"Backtest error: {item}")
            continue
        symbol, result = item
        summary[symbol] = result.summary()

    return summary


def main():
    import argparse
    from bot.symbols import load_symbols
    from strategies.ema_cross import EMACrossStrategy

    parser = argparse.ArgumentParser(description="Multi-symbol backtest")
    parser.add_argument("--symbols", help="Comma-separated symbols (default: top 5)")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000, help="Candles per symbol")
    parser.add_argument("--fast", type=int, default=9)
    parser.add_argument("--slow", type=int, default=21)
    args = parser.parse_args()

    symbols = load_symbols(args.symbols)
    logger.info(f"Running backtest on {len(symbols)} symbols: {symbols}")

    results = asyncio.run(
        run_portfolio_backtest(
            symbols,
            EMACrossStrategy,
            timeframe=args.timeframe,
            limit=args.limit,
            fast=args.fast,
            slow=args.slow,
        )
    )

    print("\n=== Portfolio Backtest Results ===\n")
    for symbol, summary in results.items():
        if not summary:
            print(f"{symbol}: no trades\n")
            continue
        print(f"{symbol}:")
        print(f"  Trades     : {summary['total_trades']}")
        print(f"  Win rate   : {summary['win_rate']:.1%}")
        print(f"  Total PnL  : {summary['total_pnl_pct']:+.2%}")
        print(f"  Avg trade  : {summary['avg_trade_pnl_pct']:+.3%}")
        print(f"  Max DD     : {summary['max_drawdown']:.2%}")
        print(f"  Sharpe     : {summary['sharpe']:.2f}")
        print()


if __name__ == "__main__":
    main()
