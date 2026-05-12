"""Tests for forge.trading.report — no real database required."""
from __future__ import annotations

import asyncio
import csv
import io
import math
import os

import pytest

from forge.trading.report import (
    compute_symbol_stats,
    export_csv,
    format_table,
    print_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    symbol: str = "BTC/USDT",
    side: str = "sell",
    pnl_pct: float | None = 0.01,
    trade_id: str = "t1",
) -> dict:
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "price": 50000.0,
        "qty": 0.01,
        "fee": 0.5,
        "pnl_pct": pnl_pct,
        "agent_role": "executor",
        "created_at": "2026-05-12T06:41:00",
    }


def _make_signal(
    symbol: str = "BTC/USDT",
    strategy: str = "rsi",
    signal: str = "buy",
    signal_id: str = "s1",
) -> dict:
    return {
        "signal_id": signal_id,
        "symbol": symbol,
        "strategy": strategy,
        "signal": signal,
        "created_at": "2026-05-12T06:41:00",
    }


# ---------------------------------------------------------------------------
# 1. compute_symbol_stats — empty input
# ---------------------------------------------------------------------------


def test_compute_symbol_stats_empty():
    result = compute_symbol_stats([])
    assert result == {}


# ---------------------------------------------------------------------------
# 2. compute_symbol_stats — win_rate calculation
# ---------------------------------------------------------------------------


def test_compute_symbol_stats_win_rate():
    trades = [
        _make_trade(pnl_pct=0.05, trade_id="t1"),  # win
        _make_trade(pnl_pct=0.10, trade_id="t2"),  # win
        _make_trade(pnl_pct=0.02, trade_id="t3"),  # win
        _make_trade(pnl_pct=-0.03, trade_id="t4"),  # loss
    ]
    stats = compute_symbol_stats(trades)
    assert "BTC/USDT" in stats
    assert stats["BTC/USDT"]["win_rate"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 3. compute_symbol_stats — total_pnl sums correctly
# ---------------------------------------------------------------------------


def test_compute_symbol_stats_pnl():
    pnl_values = [0.05, -0.02, 0.03, 0.01]
    trades = [
        _make_trade(pnl_pct=p, trade_id=f"t{i}")
        for i, p in enumerate(pnl_values)
    ]
    stats = compute_symbol_stats(trades)
    expected_total = sum(pnl_values)
    assert stats["BTC/USDT"]["total_pnl"] == pytest.approx(expected_total)


def test_compute_symbol_stats_pnl_skips_none():
    trades = [
        _make_trade(pnl_pct=0.05, trade_id="t1"),
        _make_trade(pnl_pct=None, trade_id="t2"),  # entry trade — no pnl
        _make_trade(pnl_pct=-0.02, trade_id="t3"),
    ]
    stats = compute_symbol_stats(trades)
    assert stats["BTC/USDT"]["total_pnl"] == pytest.approx(0.03)
    assert stats["BTC/USDT"]["trades"] == 3  # all trades counted regardless


# ---------------------------------------------------------------------------
# 4. compute_symbol_stats — multiple symbols separated correctly
# ---------------------------------------------------------------------------


def test_compute_symbol_stats_multiple_symbols():
    trades = [
        _make_trade(symbol="BTC/USDT", pnl_pct=0.05, trade_id="t1"),
        _make_trade(symbol="BTC/USDT", pnl_pct=-0.01, trade_id="t2"),
        _make_trade(symbol="ETH/USDT", pnl_pct=0.03, trade_id="t3"),
        _make_trade(symbol="ETH/USDT", pnl_pct=0.02, trade_id="t4"),
        _make_trade(symbol="ETH/USDT", pnl_pct=-0.04, trade_id="t5"),
    ]
    stats = compute_symbol_stats(trades)

    assert set(stats.keys()) == {"BTC/USDT", "ETH/USDT"}
    assert stats["BTC/USDT"]["trades"] == 2
    assert stats["ETH/USDT"]["trades"] == 3

    # BTC: 1 win out of 2
    assert stats["BTC/USDT"]["win_rate"] == pytest.approx(0.5)
    # ETH: 2 wins out of 3
    assert stats["ETH/USDT"]["win_rate"] == pytest.approx(2 / 3)

    assert stats["BTC/USDT"]["total_pnl"] == pytest.approx(0.04)
    assert stats["ETH/USDT"]["total_pnl"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 5. format_table — output contains all headers
# ---------------------------------------------------------------------------


def test_format_table_correct_columns():
    headers = ["Symbol", "Trades", "Win Rate", "Total PnL", "Max DD", "Sharpe"]
    rows = [
        ["BTC/USDT", "12", "58.3%", "+4.21%", "-2.10%", "0.87"],
        ["ETH/USDT", "10", "50.0%", "-1.33%", "-3.45%", "-0.22"],
    ]
    col_widths = [10, 6, 8, 9, 8, 7]

    table = format_table(headers, rows, col_widths)

    for header in headers:
        assert header in table, f"Header '{header}' missing from table output"

    for row in rows:
        for cell in row:
            assert cell in table, f"Cell value '{cell}' missing from table output"

    # Box drawing characters are present
    assert "┌" in table
    assert "┐" in table
    assert "└" in table
    assert "┘" in table
    assert "│" in table


def test_format_table_row_count():
    headers = ["A", "B"]
    rows = [["x", "1"], ["y", "2"], ["z", "3"]]
    col_widths = [5, 5]
    table = format_table(headers, rows, col_widths)
    lines = table.splitlines()
    # top border + header + mid separator + 3 data rows + bottom border = 7
    assert len(lines) == 7


# ---------------------------------------------------------------------------
# 6. export_csv — creates file with correct headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_csv_creates_file(tmp_path):
    trades = [
        _make_trade(symbol="BTC/USDT", pnl_pct=0.05, trade_id="t1"),
        _make_trade(symbol="ETH/USDT", pnl_pct=-0.02, trade_id="t2"),
    ]
    out = str(tmp_path / "trades.csv")
    await export_csv(trades, out)

    assert os.path.isfile(out)

    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 2
    assert set(rows[0].keys()) >= {
        "trade_id", "symbol", "side", "price", "qty", "fee", "pnl_pct",
        "agent_role", "created_at",
    }
    symbols = {r["symbol"] for r in rows}
    assert symbols == {"BTC/USDT", "ETH/USDT"}


@pytest.mark.asyncio
async def test_export_csv_empty_trades_writes_header(tmp_path):
    out = str(tmp_path / "empty.csv")
    await export_csv([], out)

    assert os.path.isfile(out)
    with open(out, newline="") as fh:
        content = fh.read()
    # At least the header row should be present
    assert "trade_id" in content
    assert "symbol" in content


# ---------------------------------------------------------------------------
# 7. No-trades message
# ---------------------------------------------------------------------------


def test_no_trades_message(capsys):
    print_report(
        stats={},
        signals=[],
        db_path="trading_agent.db",
        total_trades=0,
    )
    captured = capsys.readouterr()
    assert "No trades" in captured.out


def test_no_trades_still_prints_header(capsys):
    print_report(
        stats={},
        signals=[],
        db_path="trading_agent.db",
        total_trades=0,
    )
    captured = capsys.readouterr()
    assert "Performance Report" in captured.out
    assert "trading_agent.db" in captured.out


# ---------------------------------------------------------------------------
# 8. Sharpe ratio edge cases
# ---------------------------------------------------------------------------


def test_sharpe_single_trade():
    trades = [_make_trade(pnl_pct=0.05, trade_id="t1")]
    stats = compute_symbol_stats(trades)
    # Only one data point — std is 0, sharpe must be 0
    assert stats["BTC/USDT"]["sharpe"] == pytest.approx(0.0)


def test_sharpe_identical_pnl():
    # All same PnL → zero variance → sharpe must be 0
    trades = [_make_trade(pnl_pct=0.01, trade_id=f"t{i}") for i in range(5)]
    stats = compute_symbol_stats(trades)
    assert stats["BTC/USDT"]["sharpe"] == pytest.approx(0.0)


def test_sharpe_mixed_pnl():
    pnl = [0.05, -0.02, 0.03, -0.01, 0.04]
    trades = [_make_trade(pnl_pct=p, trade_id=f"t{i}") for i, p in enumerate(pnl)]
    stats = compute_symbol_stats(trades)
    mean = sum(pnl) / len(pnl)
    variance = sum((p - mean) ** 2 for p in pnl) / len(pnl)
    expected = mean / math.sqrt(variance) * math.sqrt(252)
    assert stats["BTC/USDT"]["sharpe"] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 9. Max drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_monotone_gain():
    # Monotonically increasing equity — no drawdown
    trades = [_make_trade(pnl_pct=0.01 * i, trade_id=f"t{i}") for i in range(1, 5)]
    stats = compute_symbol_stats(trades)
    assert stats["BTC/USDT"]["max_drawdown"] == pytest.approx(0.0)


def test_max_drawdown_single_loss():
    trades = [
        _make_trade(pnl_pct=0.10, trade_id="t1"),
        _make_trade(pnl_pct=-0.05, trade_id="t2"),
    ]
    stats = compute_symbol_stats(trades)
    # Cumulative peak = 0.10; then drops to 0.05 → dd = -0.05
    assert stats["BTC/USDT"]["max_drawdown"] == pytest.approx(-0.05)
