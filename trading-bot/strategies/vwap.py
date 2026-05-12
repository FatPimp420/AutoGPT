import pandas as pd
from strategies.base import BaseStrategy


class VWAPStrategy(BaseStrategy):
    """VWAP mean reversion: buy when price is below lower band, sell above upper band."""

    name = "vwap"

    def __init__(self, period: int = 20, band_mult: float = 1.5):
        self.period = period
        self.band_mult = band_mult

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Typical price × volume, then rolling cumulative VWAP proxy
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical_price * df["volume"]

        rolling_tp_vol = tp_vol.rolling(self.period).sum()
        rolling_vol = df["volume"].rolling(self.period).sum()
        df["vwap"] = rolling_tp_vol / rolling_vol

        # Rolling std of (close - vwap) to form bands
        deviation = df["close"] - df["vwap"]
        rolling_std = deviation.rolling(self.period).std()
        df["vwap_upper"] = df["vwap"] + self.band_mult * rolling_std
        df["vwap_lower"] = df["vwap"] - self.band_mult * rolling_std
        return df

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        if len(df) < self.period * 2:
            return None
        if "vwap" not in df.columns:
            df = self.add_indicators(df.copy())
        last = df.iloc[-1]
        if pd.isna(last["vwap_lower"]) or pd.isna(last["vwap_upper"]):
            return None
        close = last["close"]
        if close < last["vwap_lower"]:
            return "buy"
        if close > last["vwap_upper"]:
            return "sell"
        return None
