import pandas as pd
from strategies.base import BaseStrategy


class StochRSIStrategy(BaseStrategy):
    """Stochastic RSI: buy when %K crosses above %D below 20, sell when %K crosses below %D above 80."""

    name = "stoch_rsi"

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.oversold = oversold
        self.overbought = overbought

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=self.rsi_period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, float("inf"))
        return 100 - (100 / (1 + rs))

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = self._compute_rsi(df["close"])
        df["rsi"] = rsi

        # Stochastic of RSI
        rsi_min = rsi.rolling(self.stoch_period).min()
        rsi_max = rsi.rolling(self.stoch_period).max()
        rsi_range = rsi_max - rsi_min
        raw_k = (rsi - rsi_min) / rsi_range.replace(0, float("nan")) * 100

        df["stoch_k"] = raw_k.rolling(self.k_smooth).mean()
        df["stoch_d"] = df["stoch_k"].rolling(self.d_smooth).mean()
        return df

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        min_len = self.rsi_period + self.stoch_period + self.k_smooth + self.d_smooth
        if len(df) < min_len:
            return None
        if "stoch_k" not in df.columns:
            df = self.add_indicators(df.copy())

        if len(df) < 2:
            return None

        k_now = df["stoch_k"].iloc[-1]
        k_prev = df["stoch_k"].iloc[-2]
        d_now = df["stoch_d"].iloc[-1]
        d_prev = df["stoch_d"].iloc[-2]

        if any(pd.isna(v) for v in (k_now, k_prev, d_now, d_prev)):
            return None

        # %K crosses above %D (k was below d, now above) while in oversold zone
        if k_prev < d_prev and k_now >= d_now and d_now < self.oversold:
            return "buy"

        # %K crosses below %D (k was above d, now below) while in overbought zone
        if k_prev > d_prev and k_now <= d_now and d_now > self.overbought:
            return "sell"

        return None
