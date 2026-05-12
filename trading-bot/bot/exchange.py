import os
import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class Exchange:
    def __init__(self):
        exchange_id = os.getenv("EXCHANGE", "binance")
        exchange_class = getattr(ccxt, exchange_id)
        self._client = exchange_class({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "sandbox": os.getenv("SANDBOX", "true").lower() == "true",
            "enableRateLimit": True,
        })

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def fetch_balance(self) -> dict:
        return await self._client.fetch_balance()

    async def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        logger.info(f"Order: {side} {amount} {symbol}")
        return await self._client.create_market_order(symbol, side, amount)

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._client.fetch_ticker(symbol)

    async def close(self):
        await self._client.close()
