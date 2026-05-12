import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from risk.manager import RiskManager


def test_position_size_respects_max_pct():
    rm = RiskManager()
    size = rm.position_size(10_000, 60_000)
    assert size == round(10_000 * 0.05 / 60_000, 6)


def test_allow_trade_blocks_on_drawdown():
    rm = RiskManager()
    rm._equity = 8_000
    rm._peak_equity = 10_000  # 20% drawdown, limit is 10%
    assert rm.allow_trade("buy", 0.01, 60_000) is False


def test_allow_trade_permits_normal():
    rm = RiskManager()
    rm._equity = 9_500
    rm._peak_equity = 10_000  # 5% drawdown, under limit
    assert rm.allow_trade("buy", 0.01, 60_000) is True


def test_update_equity_tracks_peak():
    rm = RiskManager()
    rm.update_equity(10_000)
    rm.update_equity(12_000)
    rm.update_equity(11_000)
    assert rm._peak_equity == 12_000
    assert rm._equity == 11_000
