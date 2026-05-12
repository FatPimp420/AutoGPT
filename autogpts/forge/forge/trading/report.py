"""
Trade Log Report CLI for the AutoGPT Forge Trading Bot.

Usage:
  poetry run python -m forge.trading.report
  poetry run python -m forge.trading.report --task-id <id>
  poetry run python -m forge.trading.report --csv trades.csv
  poetry run python -m forge.trading.report --last 10
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
from typing import Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


async def load_all_trades(db) -> list[dict]:
    """Return trades across ALL tasks stored in the database."""
    from sqlalchemy import text

    trades: list[dict] = []
    with db.Session() as session:
        task_ids = [
            row[0]
            for row in session.execute(
                text("SELECT DISTINCT task_id FROM trade_log")
            ).fetchall()
        ]

    for task_id in task_ids:
        trades.extend(await db.get_trade_history(task_id))

    return trades


async def load_all_signals(db) -> list[dict]:
    """Return signals across ALL tasks stored in the database."""
    from sqlalchemy import text

    signals: list[dict] = []
    with db.Session() as session:
        task_ids = [
            row[0]
            for row in session.execute(
                text("SELECT DISTINCT task_id FROM signal_history")
            ).fetchall()
        ]

    for task_id in task_ids:
        signals.extend(await db.get_signal_history(task_id))

    return signals


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_symbol_stats(trades: list[dict]) -> dict[str, dict]:
    """Compute per-symbol statistics from a flat list of trade dicts.

    Returns a mapping of symbol -> {trades, win_rate, total_pnl,
    max_drawdown, sharpe}.
    """
    if not trades:
        return {}

    # Group by symbol
    groups: dict[str, list[dict]] = {}
    for t in trades:
        sym = t["symbol"]
        groups.setdefault(sym, []).append(t)

    result: dict[str, dict] = {}
    for sym, sym_trades in groups.items():
        pnl_values = [
            t["pnl_pct"] for t in sym_trades if t.get("pnl_pct") is not None
        ]

        n = len(sym_trades)
        win_rate = (
            sum(1 for p in pnl_values if p > 0) / len(pnl_values)
            if pnl_values
            else 0.0
        )
        total_pnl = sum(pnl_values)

        # Max drawdown from cumulative pnl series
        max_drawdown = _max_drawdown(pnl_values)

        # Sharpe = mean / std * sqrt(252)
        sharpe = _sharpe(pnl_values)

        result[sym] = {
            "trades": n,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
        }

    return result


def _max_drawdown(pnl_values: list[float]) -> float:
    """Compute maximum drawdown from a sequence of PnL percentages."""
    if not pnl_values:
        return 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_values:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = cumulative - peak  # always <= 0
        if dd < max_dd:
            max_dd = dd

    return max_dd


def _sharpe(pnl_values: list[float]) -> float:
    """Sharpe ratio = mean(pnl) / std(pnl) * sqrt(252)."""
    if len(pnl_values) < 2:
        return 0.0

    mean = sum(pnl_values) / len(pnl_values)
    variance = sum((p - mean) ** 2 for p in pnl_values) / len(pnl_values)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0

    return mean / std * math.sqrt(252)


# ---------------------------------------------------------------------------
# Table formatting (stdlib only — no rich/tabulate)
# ---------------------------------------------------------------------------


def format_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
) -> str:
    """Return an ASCII box-drawing table as a string.

    Parameters
    ----------
    headers:    Column header labels.
    rows:       List of rows; each row is a list of pre-formatted strings.
    col_widths: Minimum display width for each column (content is right- or
                left-padded to this width inside the cell).
    """
    # Ensure col_widths accommodates header text
    widths = [max(col_widths[i], len(headers[i])) for i in range(len(headers))]

    def _cell(text: str, width: int, align: str = "left") -> str:
        if align == "right":
            return text.rjust(width)
        return text.ljust(width)

    def _row_line(cells: list[str]) -> str:
        return "│ " + " │ ".join(cells) + " │"

    def _separator(left: str, mid: str, right: str, fill: str) -> str:
        parts = [fill * (w + 2) for w in widths]
        return left + mid.join(parts) + right

    lines: list[str] = []
    lines.append(_separator("┌", "┬", "┐", "─"))
    lines.append(_row_line([_cell(h, widths[i]) for i, h in enumerate(headers)]))
    lines.append(_separator("├", "┼", "┤", "─"))

    for row in rows:
        cells = []
        for i, val in enumerate(row):
            # Numeric-looking values are right-aligned
            stripped = val.strip().lstrip("+-")
            align = "right" if stripped.replace(".", "").replace("%", "").isdigit() else "left"
            cells.append(_cell(val, widths[i], align))
        lines.append(_row_line(cells))

    lines.append(_separator("└", "┴", "┘", "─"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _fmt_pct_neutral(value: float) -> str:
    """Format as percentage without explicit + sign."""
    return f"{value * 100:.1f}%"


def print_report(
    stats: dict[str, dict],
    signals: list[dict],
    db_path: str,
    total_trades: int,
    last_n: Optional[int] = None,
) -> None:
    """Print the full performance report to stdout."""
    task_count = "—"  # computed from DB; callers pass if known

    border = "═" * 55
    print(border)
    print("  Trading Bot — Performance Report")
    print(f"  Database: {os.path.basename(db_path)}   Total trades: {total_trades}")
    print(border)
    print()

    if not stats:
        print("  No trades recorded yet.")
        print()
    else:
        print("Per-Symbol Summary:")

        headers = ["Symbol", "Trades", "Win Rate", "Total PnL", "Max DD", "Sharpe"]
        col_widths = [10, 6, 8, 9, 8, 7]

        rows = []
        for sym, s in sorted(stats.items()):
            rows.append([
                sym,
                str(s["trades"]),
                _fmt_pct_neutral(s["win_rate"]),
                _fmt_pct(s["total_pnl"]),
                _fmt_pct(s["max_drawdown"]),
                f'{s["sharpe"]:.2f}',
            ])

        print(format_table(headers, rows, col_widths))
        print()

    # Recent signals
    limit = last_n if last_n is not None else 20
    recent = sorted(signals, key=lambda s: s.get("created_at", ""), reverse=True)[:limit]
    recent = list(reversed(recent))

    print(f"Recent Signals (last {limit}):")
    if not recent:
        print("  No signals recorded yet.")
    else:
        for sig in recent:
            ts = sig.get("created_at", "")
            if "T" in ts:
                ts = ts.replace("T", " ")[:16]
            sym = sig.get("symbol", "")
            strat = sig.get("strategy", "")
            direction = sig.get("signal", "")
            print(f"  {ts}  {sym:<10}  {strat:<10}  {direction}")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


async def export_csv(trades: list[dict], path: str) -> None:
    """Write trades to a CSV file at *path*."""
    if not trades:
        fieldnames = [
            "trade_id", "symbol", "side", "price", "qty",
            "fee", "pnl_pct", "agent_role", "created_at",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(trades[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a performance report for the AutoGPT trading bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task-id",
        metavar="ID",
        help="Restrict report to a single task ID.",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="Export trades to CSV at the given path.",
    )
    parser.add_argument(
        "--last",
        type=int,
        metavar="N",
        default=20,
        help="Number of recent signals to show (default: 20).",
    )
    parser.add_argument(
        "--db",
        metavar="DB_STRING",
        default=None,
        help="SQLAlchemy database connection string (overrides DATABASE_STRING env var).",
    )
    args = parser.parse_args()

    # Resolve database connection string
    db_string = (
        args.db
        or os.environ.get("DATABASE_STRING")
        or "sqlite:///trading_agent.db"
    )

    # Derive a display path from the connection string
    db_display = db_string
    if db_string.startswith("sqlite:///"):
        db_display = db_string[len("sqlite:///"):]

    from forge.trading.trading_db import TradingDB  # noqa: PLC0415

    db = TradingDB(db_string)

    # Load trades
    if args.task_id:
        trades = await db.get_trade_history(args.task_id)
        signals = await db.get_signal_history(args.task_id)
    else:
        trades = await load_all_trades(db)
        signals = await load_all_signals(db)

    # Optionally trim to last N trades
    if args.last and not args.task_id:
        pass  # last N only applies to the signal display

    # CSV export
    if args.csv:
        await export_csv(trades, args.csv)
        print(f"Exported {len(trades)} trades to {args.csv}")
        return

    # Compute stats and print
    stats = compute_symbol_stats(trades)
    print_report(
        stats=stats,
        signals=signals,
        db_path=db_display,
        total_trades=len(trades),
        last_n=args.last,
    )


if __name__ == "__main__":
    asyncio.run(main())
