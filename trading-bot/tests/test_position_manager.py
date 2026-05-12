"""Tests for bot/position_manager.py

Covers:
- open / close a position and verify realized PnL
- update_price correctly updates peak and trough
- double-open same symbol raises ValueError
- close non-existent symbol raises KeyError
- get_summary counts open positions correctly
- unrealized PnL for long (buy) positions
- unrealized PnL for short (sell) positions
- max_adverse_excursion tracks the worst price move against the position
- max_favorable_excursion tracks the best price move in favour of the position
- is_open before and after close
- get_open_positions returns all open positions
- hold_duration_seconds is non-negative
- multiple symbols can be open simultaneously
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from bot.position_manager import Position, PositionManager


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pm() -> PositionManager:
    """Return a fresh PositionManager for each test."""
    return PositionManager()


# ---------------------------------------------------------------------------
# 1. Open and close a buy position — verify realized PnL
# ---------------------------------------------------------------------------

def test_open_close_buy_realized_pnl():
    pm = _pm()
    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    result = pm.close_position("BTC/USDT", close_price=55_000.0)

    # (55_000 - 50_000) / 50_000 * 100 = 10 %
    assert abs(result["realized_pnl_pct"] - 10.0) < 1e-9
    assert result["symbol"] == "BTC/USDT"
    assert result["side"] == "buy"
    assert result["entry_price"] == 50_000.0
    assert result["close_price"] == 55_000.0
    assert result["qty"] == 0.01


# ---------------------------------------------------------------------------
# 2. Open and close a sell (short) position — verify realized PnL
# ---------------------------------------------------------------------------

def test_open_close_sell_realized_pnl():
    pm = _pm()
    pm.open_position("ETH/USDT", "sell", price=3_000.0, qty=1.0)
    result = pm.close_position("ETH/USDT", close_price=2_700.0)

    # (3_000 - 2_700) / 3_000 * 100 = 10 %
    assert abs(result["realized_pnl_pct"] - 10.0) < 1e-9
    assert result["side"] == "sell"


# ---------------------------------------------------------------------------
# 3. Negative realized PnL for a losing buy trade
# ---------------------------------------------------------------------------

def test_open_close_buy_losing_trade():
    pm = _pm()
    pm.open_position("SOL/USDT", "buy", price=100.0, qty=10.0)
    result = pm.close_position("SOL/USDT", close_price=90.0)

    # (90 - 100) / 100 * 100 = -10 %
    assert abs(result["realized_pnl_pct"] - (-10.0)) < 1e-9


# ---------------------------------------------------------------------------
# 4. update_price updates peak for long, trough for short
# ---------------------------------------------------------------------------

def test_update_price_updates_peak_and_trough():
    pm = _pm()
    pos = pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.1)

    # Initial state: peak == trough == entry_price
    assert pos.peak_price == 50_000.0
    assert pos.trough_price == 50_000.0

    pm.update_price("BTC/USDT", 55_000.0)  # price goes up → new peak
    assert pos.peak_price == 55_000.0
    assert pos.trough_price == 50_000.0    # trough unchanged

    pm.update_price("BTC/USDT", 48_000.0)  # price goes down → new trough
    assert pos.peak_price == 55_000.0      # peak unchanged
    assert pos.trough_price == 48_000.0

    pm.update_price("BTC/USDT", 51_000.0)  # in between → neither changes
    assert pos.peak_price == 55_000.0
    assert pos.trough_price == 48_000.0


# ---------------------------------------------------------------------------
# 5. Double-open same symbol raises ValueError
# ---------------------------------------------------------------------------

def test_double_open_raises_value_error():
    pm = _pm()
    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)

    with pytest.raises(ValueError, match="BTC/USDT"):
        pm.open_position("BTC/USDT", "buy", price=51_000.0, qty=0.01)


# ---------------------------------------------------------------------------
# 6. Close non-existent symbol raises KeyError
# ---------------------------------------------------------------------------

def test_close_nonexistent_raises_key_error():
    pm = _pm()

    with pytest.raises(KeyError, match="ETH/USDT"):
        pm.close_position("ETH/USDT", close_price=3_000.0)


# ---------------------------------------------------------------------------
# 7. get_summary counts correctly
# ---------------------------------------------------------------------------

def test_get_summary_counts_correctly():
    pm = _pm()
    summary_empty = pm.get_summary()
    assert summary_empty["open_count"] == 0
    assert summary_empty["symbols"] == []

    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    pm.open_position("ETH/USDT", "sell", price=3_000.0, qty=1.0)

    summary = pm.get_summary()
    assert summary["open_count"] == 2
    assert set(summary["symbols"]) == {"BTC/USDT", "ETH/USDT"}

    pm.close_position("BTC/USDT", close_price=50_000.0)

    summary_after = pm.get_summary()
    assert summary_after["open_count"] == 1
    assert summary_after["symbols"] == ["ETH/USDT"]


# ---------------------------------------------------------------------------
# 8. unrealized_pnl_pct is correct for buy positions
# ---------------------------------------------------------------------------

def test_unrealized_pnl_pct_buy():
    pm = _pm()
    pos = pm.open_position("BTC/USDT", "buy", price=40_000.0, qty=0.5)

    # Break even
    assert abs(pos.unrealized_pnl_pct(40_000.0)) < 1e-9

    # +25 %
    assert abs(pos.unrealized_pnl_pct(50_000.0) - 25.0) < 1e-9

    # -25 %
    assert abs(pos.unrealized_pnl_pct(30_000.0) - (-25.0)) < 1e-9


# ---------------------------------------------------------------------------
# 9. unrealized_pnl_pct is correct for sell (short) positions
# ---------------------------------------------------------------------------

def test_unrealized_pnl_pct_sell():
    pm = _pm()
    pos = pm.open_position("ETH/USDT", "sell", price=2_000.0, qty=5.0)

    # Break even
    assert abs(pos.unrealized_pnl_pct(2_000.0)) < 1e-9

    # Price falls → profit for short
    assert abs(pos.unrealized_pnl_pct(1_800.0) - 10.0) < 1e-9

    # Price rises → loss for short
    assert abs(pos.unrealized_pnl_pct(2_200.0) - (-10.0)) < 1e-9


# ---------------------------------------------------------------------------
# 10. max_adverse_excursion tracks worst price move against the position
# ---------------------------------------------------------------------------

def test_max_adverse_excursion_buy():
    pm = _pm()
    pos = pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)

    # No adverse move yet
    assert pos.max_adverse_excursion == 0.0

    pm.update_price("BTC/USDT", 45_000.0)  # −10 %
    assert abs(pos.max_adverse_excursion - (-10.0)) < 1e-9

    pm.update_price("BTC/USDT", 52_000.0)  # favourable move, MAE stays
    assert abs(pos.max_adverse_excursion - (-10.0)) < 1e-9

    pm.update_price("BTC/USDT", 40_000.0)  # new worst: −20 %
    assert abs(pos.max_adverse_excursion - (-20.0)) < 1e-9


def test_max_adverse_excursion_sell():
    pm = _pm()
    pos = pm.open_position("ETH/USDT", "sell", price=3_000.0, qty=1.0)

    # No adverse move yet
    assert pos.max_adverse_excursion == 0.0

    pm.update_price("ETH/USDT", 3_300.0)  # price rose → adverse for short
    # MAE = (entry - peak) / entry * 100 = (3_000 - 3_300) / 3_000 * 100 = -10 %
    assert abs(pos.max_adverse_excursion - (-10.0)) < 1e-9

    pm.update_price("ETH/USDT", 2_700.0)  # favourable move; MAE should not change
    assert abs(pos.max_adverse_excursion - (-10.0)) < 1e-9


# ---------------------------------------------------------------------------
# 11. max_favorable_excursion tracks best price move in favour
# ---------------------------------------------------------------------------

def test_max_favorable_excursion_buy():
    pm = _pm()
    pos = pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)

    assert pos.max_favorable_excursion == 0.0

    pm.update_price("BTC/USDT", 55_000.0)  # +10 %
    assert abs(pos.max_favorable_excursion - 10.0) < 1e-9

    pm.update_price("BTC/USDT", 48_000.0)  # adverse move; MFE should not drop
    assert abs(pos.max_favorable_excursion - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# 12. is_open returns True before close and False after
# ---------------------------------------------------------------------------

def test_is_open_before_and_after_close():
    pm = _pm()
    assert pm.is_open("BTC/USDT") is False

    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    assert pm.is_open("BTC/USDT") is True

    pm.close_position("BTC/USDT", close_price=50_000.0)
    assert pm.is_open("BTC/USDT") is False


# ---------------------------------------------------------------------------
# 13. get_open_positions returns all open positions
# ---------------------------------------------------------------------------

def test_get_open_positions_returns_all():
    pm = _pm()
    assert pm.get_open_positions() == []

    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    pm.open_position("ETH/USDT", "sell", price=3_000.0, qty=1.0)

    positions = pm.get_open_positions()
    assert len(positions) == 2
    symbols = {p.symbol for p in positions}
    assert symbols == {"BTC/USDT", "ETH/USDT"}


# ---------------------------------------------------------------------------
# 14. hold_duration_seconds is non-negative
# ---------------------------------------------------------------------------

def test_hold_duration_seconds_non_negative():
    pm = _pm()
    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    result = pm.close_position("BTC/USDT", close_price=50_000.0)
    assert result["hold_duration_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# 15. Multiple symbols can be open simultaneously without interference
# ---------------------------------------------------------------------------

def test_multiple_symbols_independent():
    pm = _pm()
    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    pm.open_position("ETH/USDT", "buy", price=3_000.0, qty=1.0)
    pm.open_position("SOL/USDT", "sell", price=150.0, qty=10.0)

    # Update prices independently
    pm.update_price("BTC/USDT", 55_000.0)
    pm.update_price("ETH/USDT", 2_500.0)   # adverse for ETH long

    positions = {p.symbol: p for p in pm.get_open_positions()}

    assert positions["BTC/USDT"].peak_price == 55_000.0
    assert positions["ETH/USDT"].trough_price == 2_500.0
    assert positions["SOL/USDT"].peak_price == 150.0  # unchanged

    # Close one; others remain
    pm.close_position("BTC/USDT", close_price=55_000.0)
    assert pm.is_open("BTC/USDT") is False
    assert pm.is_open("ETH/USDT") is True
    assert pm.is_open("SOL/USDT") is True

    summary = pm.get_summary()
    assert summary["open_count"] == 2


# ---------------------------------------------------------------------------
# 16. update_price on unknown symbol does NOT raise
# ---------------------------------------------------------------------------

def test_update_price_unknown_symbol_is_noop():
    pm = _pm()
    # Should not raise even though "BTC/USDT" is not open
    pm.update_price("BTC/USDT", 50_000.0)  # no error


# ---------------------------------------------------------------------------
# 17. Re-open a position after closing it succeeds
# ---------------------------------------------------------------------------

def test_reopen_after_close():
    pm = _pm()
    pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    pm.close_position("BTC/USDT", close_price=50_000.0)

    # Should be allowed to open again
    pos = pm.open_position("BTC/USDT", "buy", price=52_000.0, qty=0.02)
    assert pos.entry_price == 52_000.0
    assert pm.is_open("BTC/USDT") is True
