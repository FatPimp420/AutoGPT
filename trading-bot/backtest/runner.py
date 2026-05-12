import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def summary(self) -> dict:
        if not self.trades:
            return {}
        df = self.df
        pnl = df["pnl"]

        # ── Equity-curve drawdown ────────────────────────────────────────────
        equity = pd.Series(self.equity_curve)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown: float = float(drawdown.min())  # negative fraction, e.g. -0.15

        # ── Annualised return (compound) ─────────────────────────────────────
        n_trades = len(df)
        total_pnl_pct: float = float(pnl.sum())
        # Approximate: treat each trade as one day of 252 trading days
        annualised_return: float = total_pnl_pct * (252 / max(n_trades, 1))

        # ── Calmar ratio ─────────────────────────────────────────────────────
        calmar_ratio: float = (
            annualised_return / abs(max_drawdown) if max_drawdown != 0.0 else 0.0
        )

        # ── Profit factor ────────────────────────────────────────────────────
        gross_profit: float = float(pnl[pnl > 0].sum())
        gross_loss: float = float(abs(pnl[pnl < 0].sum()))
        profit_factor: float = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # ── Max consecutive losses ───────────────────────────────────────────
        max_consec_losses: int = 0
        current_streak: int = 0
        for p in pnl:
            if p < 0:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

        return {
            "total_trades": n_trades,
            "win_rate": float((pnl > 0).mean()),
            "total_pnl_pct": total_pnl_pct,
            "avg_trade_pnl_pct": float(pnl.mean()),
            "max_drawdown": max_drawdown,
            "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0,
            "calmar_ratio": calmar_ratio,
            "profit_factor": profit_factor,
            "max_consecutive_losses": max_consec_losses,
        }


class Backtester:
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,    # 0.1% per side — Binance standard taker fee
        slippage_rate: float = 0.0005,  # 0.05% per side — conservative estimate
    ):
        self.strategy = strategy
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def _fill_price(self, price: float, side: str) -> float:
        """Apply slippage: buys fill slightly higher, sells slightly lower."""
        if side == "buy":
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)

    def _apply_fee(self, notional: float) -> float:
        return notional * (1 - self.fee_rate)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        result = BacktestResult()
        # Pre-compute all indicators once — O(n) instead of O(n²)
        df = self.strategy.add_indicators(df.copy())
        position = None

        for i in range(1, len(df)):
            # Pass only the slice needed for signal logic (last 2 rows suffices for
            # crossover strategies; strategies needing more warmup guard internally)
            signal = self.strategy.generate_signal(df.iloc[: i + 1])

            price = df.iloc[i]["close"]
            result.equity_curve.append(self.capital)

            if signal == "buy" and position is None:
                entry = self._fill_price(price, "buy")
                entry_after_fee = self._apply_fee(entry)
                position = {"side": "buy", "entry": entry_after_fee, "i": i}

            elif signal == "sell" and position is not None:
                exit_price = self._fill_price(price, "sell")
                exit_after_fee = self._apply_fee(exit_price)
                pnl_pct = (exit_after_fee - position["entry"]) / position["entry"]
                self.capital *= 1 + pnl_pct
                result.trades.append({
                    "entry": position["entry"],
                    "exit": exit_after_fee,
                    "pnl": pnl_pct,
                })
                position = None

        return result


def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--strategy", default="ema_cross")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")

    if args.strategy == "ema_cross":
        from strategies.ema_cross import EMACrossStrategy
        strategy = EMACrossStrategy()

    bt = Backtester(strategy)
    result = bt.run(df)
    print(json.dumps(result.summary(), indent=2))
