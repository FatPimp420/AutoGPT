import asyncio
from loguru import logger
from bot.exchange import Exchange
from risk.manager import RiskManager
from strategies.base import BaseStrategy


class TradingEngine:
    def __init__(self, strategy: BaseStrategy, symbol: str, timeframe: str):
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = Exchange()
        self.risk = RiskManager()
        self._running = False

    async def run(self):
        self._running = True
        logger.info(f"Engine started: {self.symbol} {self.timeframe} | strategy={self.strategy.name}")
        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self._timeframe_to_seconds(self.timeframe))
        finally:
            await self.exchange.close()

    async def _tick(self):
        df = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe)
        signal = self.strategy.generate_signal(df)

        if signal is None:
            return

        ticker = await self.exchange.fetch_ticker(self.symbol)
        price = ticker["last"]
        balance = await self.exchange.fetch_balance()
        quote_balance = balance["free"].get("USDT", 0)

        size = self.risk.position_size(quote_balance, price)
        if not self.risk.allow_trade(signal, size, price):
            logger.warning("Risk manager blocked trade")
            return

        await self.exchange.create_market_order(self.symbol, signal, size)
        self.risk.record_trade(signal, size, price)

    def stop(self):
        self._running = False

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        units = {"m": 60, "h": 3600, "d": 86400}
        return int(tf[:-1]) * units[tf[-1]]
