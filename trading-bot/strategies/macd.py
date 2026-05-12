import pandas as pd
from strategies.base import BaseStrategy


class MACDStrategy(BaseStrategy):
    """MACD line crosses above/below signal line."""

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        if "macd_hist" not in df.columns:
            df = self.add_indicators(df.copy())
        if len(df) < 2 or df["macd_hist"].isna().iloc[-2:].any():
            return None
        prev_hist = df["macd_hist"].iloc[-2]
        last_hist = df["macd_hist"].iloc[-1]
        if prev_hist < 0 and last_hist >= 0:
            return "buy"
        if prev_hist > 0 and last_hist <= 0:
            return "sell"
        return None

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df
