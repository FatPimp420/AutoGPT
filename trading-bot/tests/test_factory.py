"""Tests for trading-bot strategy factory."""
import sys
from pathlib import Path

import pytest

# Ensure trading-bot root is on sys.path so strategy imports resolve.
_TRADING_BOT_ROOT = str(Path(__file__).resolve().parents[1])
if _TRADING_BOT_ROOT not in sys.path:
    sys.path.insert(0, _TRADING_BOT_ROOT)

from strategies.bollinger import BollingerStrategy
from strategies.ema_cross import EMACrossStrategy
from strategies.factory import (
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    register_strategy,
)
from strategies.macd import MACDStrategy
from strategies.rsi import RSIStrategy


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------


def test_get_strategy_rsi():
    strategy = get_strategy("rsi")
    assert isinstance(strategy, RSIStrategy)


def test_get_strategy_ema_cross():
    strategy = get_strategy("ema_cross")
    assert isinstance(strategy, EMACrossStrategy)


def test_get_strategy_macd():
    strategy = get_strategy("macd")
    assert isinstance(strategy, MACDStrategy)


def test_get_strategy_bollinger():
    strategy = get_strategy("bollinger")
    assert isinstance(strategy, BollingerStrategy)


# ---------------------------------------------------------------------------
# Keyword argument forwarding
# ---------------------------------------------------------------------------


def test_get_strategy_with_kwargs():
    strategy = get_strategy("rsi", period=20)
    assert isinstance(strategy, RSIStrategy)
    assert strategy.period == 20


# ---------------------------------------------------------------------------
# None / env-var behaviour
# ---------------------------------------------------------------------------


def test_get_strategy_none_reads_env(monkeypatch):
    monkeypatch.setenv("DEFAULT_STRATEGY", "macd")
    strategy = get_strategy(None)
    assert isinstance(strategy, MACDStrategy)


def test_get_strategy_default_is_rsi(monkeypatch):
    monkeypatch.delenv("DEFAULT_STRATEGY", raising=False)
    strategy = get_strategy(None)
    assert isinstance(strategy, RSIStrategy)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent_strategy")


def test_get_strategy_unknown_message_lists_available():
    """The error message should hint at valid strategy names."""
    with pytest.raises(ValueError) as exc_info:
        get_strategy("bad_name")
    message = str(exc_info.value)
    # At least one known strategy name should appear in the helpful message.
    assert any(name in message for name in ("rsi", "macd", "ema_cross", "bollinger"))


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_list_strategies_returns_all():
    names = list_strategies()
    assert sorted(names) == names, "list_strategies() should return a sorted list"
    for expected in ("ema_cross", "rsi", "macd", "bollinger"):
        assert expected in names


# ---------------------------------------------------------------------------
# Runtime registration
# ---------------------------------------------------------------------------


def test_register_strategy_custom(monkeypatch):
    """A newly registered strategy should be retrievable via get_strategy."""

    class MockStrategy:
        name = "mock"

        def generate_signal(self, df):
            return None

    # Use monkeypatch to restore the registry after the test.
    original_registry = dict(STRATEGY_REGISTRY)
    try:
        register_strategy("mock", MockStrategy)
        assert "mock" in STRATEGY_REGISTRY
        instance = get_strategy("mock")
        assert isinstance(instance, MockStrategy)
    finally:
        STRATEGY_REGISTRY.clear()
        STRATEGY_REGISTRY.update(original_registry)


# ---------------------------------------------------------------------------
# Integration: actions._load_strategy delegates to factory
# ---------------------------------------------------------------------------


def test_actions_load_strategy_uses_factory():
    """_load_strategy in forge.trading.actions should return the correct type."""
    import importlib.util

    # Locate the actions module file directly to avoid the forge.trading
    # package __init__ which imports sqlalchemy (not available in the
    # trading-bot test environment).
    actions_path = (
        Path(__file__).resolve().parents[2]
        / "autogpts"
        / "forge"
        / "forge"
        / "trading"
        / "actions.py"
    )

    spec = importlib.util.spec_from_file_location("forge_trading_actions", actions_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _load_strategy = module._load_strategy

    assert isinstance(_load_strategy("rsi"), RSIStrategy)
    assert isinstance(_load_strategy("ema_cross"), EMACrossStrategy)
    assert isinstance(_load_strategy("macd"), MACDStrategy)
    assert isinstance(_load_strategy("bollinger"), BollingerStrategy)
