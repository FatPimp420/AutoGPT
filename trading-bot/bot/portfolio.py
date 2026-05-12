import asyncio
from loguru import logger
from bot.engine import TradingEngine
from strategies.base import BaseStrategy


class PortfolioEngine:
    """Runs multiple TradingEngine instances concurrently, one per symbol."""

    def __init__(self, strategy_cls: type[BaseStrategy], symbols: list[str], timeframe: str, **strategy_kwargs):
        self.engines = {
            symbol: TradingEngine(strategy_cls(**strategy_kwargs), symbol, timeframe)
            for symbol in symbols
        }
        self._tasks: list[asyncio.Task] = []

    async def run(self):
        logger.info(f"Portfolio engine starting — {len(self.engines)} symbols: {list(self.engines.keys())}")
        self._tasks = [
            asyncio.create_task(engine.run(), name=symbol)
            for symbol, engine in self.engines.items()
        ]
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            self.stop()

    def stop(self):
        for symbol, engine in self.engines.items():
            engine.stop()
            logger.info(f"Stopped engine: {symbol}")
        for task in self._tasks:
            task.cancel()
