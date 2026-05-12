"""Strategy factory — load any strategy by name with optional param overrides.

Usage:
  strategy = get_strategy("rsi")
  strategy = get_strategy("ema_cross", fast=9, slow=21)
  strategy = get_strategy(None)  # reads DEFAULT_STRATEGY env var, fallback "rsi"

Supported names: ema_cross, rsi, macd, bollinger, vwap, stoch_rsi
"""
import os

from strategies.base import BaseStrategy
from strategies.bollinger import BollingerStrategy
from strategies.ema_cross import EMACrossStrategy
from strategies.macd import MACDStrategy
from strategies.rsi import RSIStrategy
from strategies.stoch_rsi import StochRSIStrategy
from strategies.vwap import VWAPStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "ema_cross": EMACrossStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
    "vwap": VWAPStrategy,
    "stoch_rsi": StochRSIStrategy,
}


def get_strategy(name: str | None = None, **kwargs) -> BaseStrategy:
    """Return an instantiated strategy by name.

    Args:
        name: Strategy name (one of ema_cross, rsi, macd, bollinger).
              If None, reads the DEFAULT_STRATEGY environment variable;
              falls back to "rsi" when the variable is unset.
        **kwargs: Optional constructor overrides forwarded to the strategy
                  class (e.g. period=20 for RSIStrategy).

    Returns:
        An instance of the requested BaseStrategy subclass.

    Raises:
        ValueError: When *name* is not present in STRATEGY_REGISTRY.
    """
    if name is None:
        name = os.getenv("DEFAULT_STRATEGY", "rsi")

    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(
            f"Unknown strategy {name!r}. "
            f"Available strategies: {available}"
        )

    return cls(**kwargs)


def list_strategies() -> list[str]:
    """Return a sorted list of all registered strategy names."""
    return sorted(STRATEGY_REGISTRY)


def register_strategy(name: str, cls: type) -> None:
    """Register a strategy class under *name* at runtime.

    Args:
        name: The lookup key used with get_strategy().
        cls: A BaseStrategy subclass (not validated at registration time).
    """
    STRATEGY_REGISTRY[name] = cls
