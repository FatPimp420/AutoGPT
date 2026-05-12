"""Persistent in-memory PositionManager for tracking open positions across swarm runs.

Designed to be thread-safe: all mutations are guarded by a threading.Lock so the
PositionManager can be shared between threads in a live trading pipeline without
data races.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class Position:
    """Represents a single open trading position.

    Attributes
    ----------
    symbol:
        Market symbol, e.g. ``"BTC/USDT"``.
    side:
        ``"buy"`` (long) or ``"sell"`` (short).
    entry_price:
        The price at which the position was opened.
    qty:
        Base-asset quantity held.
    entry_time:
        UTC datetime when the position was opened.
    peak_price:
        Highest price observed since the position was opened.
    trough_price:
        Lowest price observed since the position was opened.
    """

    symbol: str
    side: str                    # "buy" | "sell"
    entry_price: float
    qty: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    peak_price: float = field(init=False)
    trough_price: float = field(init=False)

    def __post_init__(self) -> None:
        self.peak_price = self.entry_price
        self.trough_price = self.entry_price

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealised P&L as a percentage of entry price.

        For a **long** (buy) position:
            (current_price - entry_price) / entry_price * 100

        For a **short** (sell) position:
            (entry_price - current_price) / entry_price * 100
        """
        if self.entry_price == 0:
            return 0.0
        if self.side == "buy":
            return (current_price - self.entry_price) / self.entry_price * 100.0
        else:  # short / sell
            return (self.entry_price - current_price) / self.entry_price * 100.0

    @property
    def max_favorable_excursion(self) -> float:
        """Maximum Favorable Excursion (MFE) as a percentage of entry price.

        The best price movement *in favour* of the position since open.

        For a long:  (peak_price  - entry_price) / entry_price * 100
        For a short: (entry_price - trough_price) / entry_price * 100
        """
        if self.entry_price == 0:
            return 0.0
        if self.side == "buy":
            return (self.peak_price - self.entry_price) / self.entry_price * 100.0
        else:
            return (self.entry_price - self.trough_price) / self.entry_price * 100.0

    @property
    def max_adverse_excursion(self) -> float:
        """Maximum Adverse Excursion (MAE) as a percentage of entry price.

        The worst price movement *against* the position since open.
        Always returned as a **negative** percentage (a loss).

        For a long:  (trough_price - entry_price) / entry_price * 100
        For a short: (entry_price  - peak_price)  / entry_price * 100
        """
        if self.entry_price == 0:
            return 0.0
        if self.side == "buy":
            return (self.trough_price - self.entry_price) / self.entry_price * 100.0
        else:
            return (self.entry_price - self.peak_price) / self.entry_price * 100.0


class PositionManager:
    """Thread-safe registry of open positions across swarm pipeline runs.

    Usage
    -----
    >>> pm = PositionManager()
    >>> pos = pm.open_position("BTC/USDT", "buy", price=50_000.0, qty=0.01)
    >>> pm.update_price("BTC/USDT", 52_000.0)
    >>> result = pm.close_position("BTC/USDT", close_price=52_000.0)
    >>> result["realized_pnl_pct"]
    4.0
    """

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
    ) -> Position:
        """Open a new position for *symbol*.

        Parameters
        ----------
        symbol:
            Market symbol, e.g. ``"BTC/USDT"``.
        side:
            ``"buy"`` or ``"sell"``.
        price:
            Entry (fill) price.
        qty:
            Quantity of the base asset.

        Returns
        -------
        Position
            The newly created position.

        Raises
        ------
        ValueError
            If a position for *symbol* is already open.
        """
        with self._lock:
            if symbol in self._positions:
                raise ValueError(
                    f"Position for {symbol!r} is already open. "
                    "Close it before opening a new one."
                )
            pos = Position(symbol=symbol, side=side, entry_price=price, qty=qty)
            self._positions[symbol] = pos
            return pos

    def close_position(self, symbol: str, close_price: float) -> dict:
        """Close the open position for *symbol* at *close_price*.

        Parameters
        ----------
        symbol:
            Market symbol to close.
        close_price:
            The exit (fill) price.

        Returns
        -------
        dict
            ``{"symbol", "realized_pnl_pct", "hold_duration_seconds",
               "entry_price", "close_price", "qty", "side"}``

        Raises
        ------
        KeyError
            If no open position exists for *symbol*.
        """
        with self._lock:
            if symbol not in self._positions:
                raise KeyError(f"No open position for {symbol!r}.")
            pos = self._positions.pop(symbol)

        close_time = datetime.now(timezone.utc)
        hold_seconds = (close_time - pos.entry_time).total_seconds()
        realized_pnl_pct = pos.unrealized_pnl_pct(close_price)

        return {
            "symbol": symbol,
            "realized_pnl_pct": realized_pnl_pct,
            "hold_duration_seconds": hold_seconds,
            "entry_price": pos.entry_price,
            "close_price": close_price,
            "qty": pos.qty,
            "side": pos.side,
        }

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update the peak and trough prices for *symbol*.

        Silently ignores symbols that are not currently open so that price
        feeds can broadcast to the manager without per-symbol guards on the
        caller side.

        Parameters
        ----------
        symbol:
            Market symbol to update.
        current_price:
            The latest observed price.
        """
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                return
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            if current_price < pos.trough_price:
                pos.trough_price = current_price

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> List[Position]:
        """Return a snapshot list of all currently open positions."""
        with self._lock:
            return list(self._positions.values())

    def get_summary(self) -> dict:
        """Return a high-level summary of open positions.

        Returns
        -------
        dict
            ``{"open_count": int, "total_unrealized_pnl_pct": float, "symbols": list[str]}``

        Note: ``total_unrealized_pnl_pct`` is the *sum* across all open
        positions evaluated at their current *peak_price* as a proxy for the
        latest known price.  Callers that have a real-time feed should call
        ``update_price`` before ``get_summary`` for accurate figures.
        """
        with self._lock:
            positions = list(self._positions.values())

        total_pnl = sum(
            pos.unrealized_pnl_pct(pos.peak_price if pos.side == "buy" else pos.trough_price)
            for pos in positions
        )
        return {
            "open_count": len(positions),
            "total_unrealized_pnl_pct": total_pnl,
            "symbols": [pos.symbol for pos in positions],
        }

    def is_open(self, symbol: str) -> bool:
        """Return True if there is an open position for *symbol*."""
        with self._lock:
            return symbol in self._positions
