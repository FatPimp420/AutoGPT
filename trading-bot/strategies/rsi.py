import pandas as pd
from strategies.base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """Buy oversold, sell overbought."""

    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        df = self.add_indicators(df.copy())
        if df["rsi"].isna().all():
            return None
        rsi = df["rsi"].iloc[-1]
        if rsi < self.oversold:
            return "buy"
        if rsi > self.overbought:
            return "sell"
        return None

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).ewm(com=self.period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=self.period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, float("inf"))
        df["rsi"] = 100 - (100 / (1 + rs))
        return df
