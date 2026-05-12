import pandas as pd
from strategies.base import BaseStrategy


class EMACrossStrategy(BaseStrategy):
    """Golden/death cross on two EMAs. Replace with your Obsidian strategy logic."""

    name = "ema_cross"

    def __init__(self, fast: int = 9, slow: int = 21):
        self.fast = fast
        self.slow = slow

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        df = self.add_indicators(df.copy())
        if len(df) < self.slow + 1:
            return None

        prev = df.iloc[-2]
        last = df.iloc[-1]

        # Golden cross
        if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
            return "buy"
        # Death cross
        if prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
            return "sell"
        return None

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        return df
