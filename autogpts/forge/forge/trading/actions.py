"""Trading actions callable by TradingAgent.execute_step().

Imports from the trading-bot package at TRADING_BOT_PATH (env var) or
auto-detected relative to this repo's root.
"""
import os
import sys
from pathlib import Path


def _ensure_trading_bot_on_path() -> None:
    bot_path = os.getenv("TRADING_BOT_PATH")
    if not bot_path:
        # Locate trading-bot/ relative to the repo root (4 levels up from this file)
        here = Path(__file__).resolve()
        repo_root = here.parents[4]
        bot_path = str(repo_root / "trading-bot")
    if bot_path not in sys.path:
        sys.path.insert(0, bot_path)


_ensure_trading_bot_on_path()

from backtest.runner import Backtester  # noqa: E402
from backtest.synthetic import generate_ohlcv  # noqa: E402
from risk.manager import RiskManager  # noqa: E402


_STRATEGY_MAP = {}


def _load_strategy(name: str):
    if name not in _STRATEGY_MAP:
        if name == "ema_cross":
            from strategies.ema_cross import EMACrossStrategy
            _STRATEGY_MAP[name] = EMACrossStrategy
        elif name == "rsi":
            from strategies.rsi import RSIStrategy
            _STRATEGY_MAP[name] = RSIStrategy
        elif name == "macd":
            from strategies.macd import MACDStrategy
            _STRATEGY_MAP[name] = MACDStrategy
        elif name == "bollinger":
            from strategies.bollinger import BollingerStrategy
            _STRATEGY_MAP[name] = BollingerStrategy
        else:
            raise ValueError(f"Unknown strategy: {name}")
    return _STRATEGY_MAP[name]()


def run_backtest(
    symbol: str,
    strategy_name: str = "rsi",
    days: int = 365,
    initial_capital: float = 10_000.0,
) -> dict:
    """Run a backtest for one symbol using synthetic OHLCV data."""
    candles = days * 24
    df = generate_ohlcv(symbol=symbol, n_candles=candles)
    strategy = _load_strategy(strategy_name)
    bt = Backtester(strategy, initial_capital=initial_capital)
    result = bt.run(df)
    summary = result.summary()
    return {"symbol": symbol, "strategy": strategy_name, "days": days, **summary}


def get_signal(symbol: str, strategy_name: str = "rsi", lookback_days: int = 60) -> dict:
    """Generate the current buy/sell signal for a symbol."""
    df = generate_ohlcv(symbol=symbol, n_candles=lookback_days * 24)
    strategy = _load_strategy(strategy_name)
    df = strategy.add_indicators(df.copy())
    signal = strategy.generate_signal(df)
    return {"symbol": symbol, "strategy": strategy_name, "signal": signal}


def check_risk(
    signal: str,
    equity: float,
    price: float,
    daily_loss: float = 0.0,
) -> dict:
    """Check whether risk rules allow this trade."""
    rm = RiskManager()
    rm._daily_loss = daily_loss
    rm.update_equity(equity)
    qty = rm.position_size(equity, price)
    allowed = rm.allow_trade(signal, qty, price)
    return {"signal": signal, "allowed": allowed, "qty": qty, "price": price}


def get_portfolio_summary(symbols: list, strategy_name: str = "rsi", days: int = 90) -> dict:
    """Run backtest across all symbols and return an aggregated summary."""
    results = []
    for sym in symbols:
        try:
            r = run_backtest(sym, strategy_name, days)
            results.append(r)
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    valid = [r for r in results if "total_pnl_pct" in r]
    avg_pnl = sum(r["total_pnl_pct"] for r in valid) / len(valid) if valid else 0.0
    avg_sharpe = sum(r["sharpe"] for r in valid) / len(valid) if valid else 0.0
    return {
        "strategy": strategy_name,
        "symbols": len(symbols),
        "avg_pnl_pct": round(avg_pnl, 4),
        "avg_sharpe": round(avg_sharpe, 4),
        "per_symbol": results,
    }
