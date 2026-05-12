import asyncio
import click
from bot.engine import TradingEngine
from strategies.ema_cross import EMACrossStrategy


@click.command()
@click.option("--symbol", default="BTC/USDT")
@click.option("--timeframe", default="1h")
@click.option("--strategy", default="ema_cross")
def main(symbol, timeframe, strategy):
    strat = EMACrossStrategy()  # TODO: load from --strategy name after vault notes are ingested
    engine = TradingEngine(strat, symbol, timeframe)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
