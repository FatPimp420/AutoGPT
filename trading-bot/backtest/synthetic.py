import numpy as np
import pandas as pd

# Approximate annualised volatility and starting prices per asset
ASSET_PARAMS = {
    "BTC/USDT": {"price": 60000, "annual_vol": 0.75},
    "ETH/USDT": {"price": 3200,  "annual_vol": 0.85},
    "SOL/USDT": {"price": 145,   "annual_vol": 1.10},
    "BNB/USDT": {"price": 580,   "annual_vol": 0.70},
    "XRP/USDT": {"price": 0.52,  "annual_vol": 0.90},
}


def generate_ohlcv(symbol: str, n_candles: int = 1000, timeframe_h: int = 1, seed: int = 42) -> pd.DataFrame:
    params = ASSET_PARAMS.get(symbol, {"price": 100, "annual_vol": 0.80})
    rng = np.random.default_rng(seed)

    dt = timeframe_h / 8760  # fraction of a year per candle
    vol_per_candle = params["annual_vol"] * np.sqrt(dt)

    log_returns = rng.normal(0, vol_per_candle, n_candles)
    close = params["price"] * np.exp(np.cumsum(log_returns))

    noise = rng.uniform(0.001, 0.005, n_candles)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = params["price"]
    volume = rng.uniform(1e6, 5e6, n_candles)

    idx = pd.date_range(end="2026-05-12", periods=n_candles, freq=f"{timeframe_h}h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
