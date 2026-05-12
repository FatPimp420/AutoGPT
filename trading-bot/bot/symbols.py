DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
]


def load_symbols(symbols_env: str | None = None) -> list[str]:
    """Load symbols from env var SYMBOLS (comma-separated) or fall back to defaults."""
    import os
    raw = symbols_env or os.getenv("SYMBOLS", "")
    if raw.strip():
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    return DEFAULT_SYMBOLS
