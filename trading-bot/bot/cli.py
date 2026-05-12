import asyncio
import click
from bot.portfolio import PortfolioEngine
from bot.symbols import load_symbols
from strategies.ema_cross import EMACrossStrategy


@click.command()
@click.option("--symbols", default=None, help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
@click.option("--timeframe", default="1h", show_default=True)
@click.option("--fast", default=9, show_default=True, help="Fast EMA period")
@click.option("--slow", default=21, show_default=True, help="Slow EMA period")
def main(symbols, timeframe, fast, slow):
    symbol_list = load_symbols(symbols)
    click.echo(f"Starting portfolio engine: {symbol_list} @ {timeframe}")
    engine = PortfolioEngine(EMACrossStrategy, symbol_list, timeframe, fast=fast, slow=slow)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
