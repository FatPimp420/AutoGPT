import pandas as pd
from strategies.base import BaseStrategy


class BollingerStrategy(BaseStrategy):
    """Buy at lower band touch, sell at upper band touch."""

    name = "bollinger"

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        df = self.add_indicators(df.copy())
        if df["bb_lower"].isna().iloc[-1]:
            return None
        close = df["close"].iloc[-1]
        if close <= df["bb_lower"].iloc[-1]:
            return "buy"
        if close >= df["bb_upper"].iloc[-1]:
            return "sell"
        return None

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = df["close"].rolling(self.period).mean()
        std = df["close"].rolling(self.period).std()
        df["bb_mid"] = sma
        df["bb_upper"] = sma + self.std_dev * std
        df["bb_lower"] = sma - self.std_dev * std
        return df
