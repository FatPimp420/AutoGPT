"""Live trade executor for the trading bot.

Wraps the Exchange client and adds:
- Dry-run mode (default) controlled by LIVE_TRADING env var
- Structured OrderResult dataclass
- Optional TradingDB logging on successful fills
- Full exception isolation so a failed order never crashes the swarm
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# The Exchange class is the only exchange client in this package; we re-export
# it under the ExchangeClient alias so callers can use either name.
from bot.exchange import Exchange as ExchangeClient  # noqa: F401


@dataclass
class OrderResult:
    """Represents the outcome of a single trade attempt."""

    symbol: str
    side: str           # "buy" | "sell"
    qty: float
    price: float
    fee: float
    order_id: str
    executed: bool
    dry_run: bool
    error: Optional[str] = field(default=None)


class LiveExecutor:
    """Executes market orders via ccxt, with an optional dry-run gate.

    The ``dry_run`` constructor parameter is *always* overridden by the
    ``LIVE_TRADING`` environment variable:

    - ``LIVE_TRADING=true``  → real orders (dry_run=False)
    - anything else          → paper-only mode (dry_run=True)

    This ensures that a misconfigured process never accidentally fires live
    orders; you must explicitly opt-in at the process level.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        dry_run: bool = True,
    ) -> None:
        # Env var takes absolute precedence over the constructor parameter.
        self.dry_run: bool = os.getenv("LIVE_TRADING", "false").lower() != "true"

        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret

        # Lazily create the exchange client so tests can patch it before
        # first use.  We store enough config to build it on demand.
        self._exchange: Optional[ExchangeClient] = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_exchange(self) -> ExchangeClient:
        """Return (or lazily create) the underlying exchange client."""
        if self._exchange is None:
            # Exchange reads EXCHANGE / API_KEY / API_SECRET / SANDBOX from
            # env vars, so set them before construction when explicit values
            # were passed.
            if self._api_key:
                os.environ.setdefault("API_KEY", self._api_key)
            if self._api_secret:
                os.environ.setdefault("API_SECRET", self._api_secret)
            if self._exchange_id:
                os.environ.setdefault("EXCHANGE", self._exchange_id)
            self._exchange = ExchangeClient()
        return self._exchange

    @staticmethod
    def _extract_fill(order: dict, symbol: str, side: str, qty: float) -> tuple[float, float, float, str]:
        """Pull price / fee / order_id out of a ccxt order dict.

        Returns ``(price, fee, order_id, qty_filled)``.
        All fields fall back to safe defaults if the exchange omits them.
        """
        price = float(order.get("price") or order.get("average") or 0.0)
        qty_filled = float(order.get("filled") or order.get("amount") or qty)
        order_id = str(order.get("id") or "")

        # Fee from exchange response; fall back to 0.1 % taker estimate.
        raw_fee = order.get("fee") or {}
        if isinstance(raw_fee, dict):
            fee = float(raw_fee.get("cost") or 0.0)
        else:
            fee = float(raw_fee or 0.0)

        if fee == 0.0:
            fee = qty_filled * price * 0.001

        return price, fee, order_id, qty_filled

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        symbol: str,
        signal: str,
        qty: float,
        task_id: str = "",
        db=None,
    ) -> OrderResult:
        """Execute a market order (or simulate one in dry-run mode).

        Parameters
        ----------
        symbol:
            Market symbol, e.g. ``"BTC/USDT"``.
        signal:
            ``"buy"`` or ``"sell"``.
        qty:
            Base-asset quantity to trade.
        task_id:
            Optional task identifier forwarded to the DB log entry.
        db:
            Optional ``TradingDB`` instance.  When provided and the order
            succeeds, ``db.log_trade()`` is awaited.

        Returns
        -------
        OrderResult
            Always returns a result; never raises.
        """
        side = signal.lower()

        if self.dry_run:
            logger.info(
                f"[LiveExecutor] DRY-RUN {side.upper()} {qty} {symbol} "
                "(LIVE_TRADING not set — no real order placed)"
            )
            result = OrderResult(
                symbol=symbol,
                side=side,
                qty=qty,
                price=0.0,
                fee=0.0,
                order_id="",
                executed=False,
                dry_run=True,
                error=None,
            )
            # Log the paper intent to the DB so dry-run runs leave a full trail.
            if db and task_id:
                try:
                    await db.log_trade(
                        task_id,
                        symbol,
                        side,
                        price=0.0,
                        qty=qty,
                        fee=0.0,
                        agent_role="live_executor_dry",
                    )
                except Exception as db_err:
                    logger.warning(f"[LiveExecutor] DB log error (dry-run): {db_err}")
            return result

        # --- Live path ---------------------------------------------------
        try:
            exchange = self._get_exchange()
            order = await exchange.create_market_order(symbol, side, qty)

            price, fee, order_id, qty_filled = self._extract_fill(
                order, symbol, side, qty
            )

            logger.info(
                f"[LiveExecutor] LIVE FILL {side.upper()} {qty_filled} {symbol} "
                f"@ {price} | fee={fee:.6f} | id={order_id}"
            )

            result = OrderResult(
                symbol=symbol,
                side=side,
                qty=qty_filled,
                price=price,
                fee=fee,
                order_id=order_id,
                executed=True,
                dry_run=False,
                error=None,
            )

            if db and task_id:
                try:
                    await db.log_trade(
                        task_id,
                        symbol,
                        side,
                        price=price,
                        qty=qty_filled,
                        fee=fee,
                        agent_role="live_executor",
                    )
                except Exception as db_err:
                    logger.warning(f"[LiveExecutor] DB log error (live): {db_err}")

            return result

        except Exception as exc:
            logger.error(
                f"[LiveExecutor] Order failed {side.upper()} {qty} {symbol}: {exc}"
            )
            return OrderResult(
                symbol=symbol,
                side=side,
                qty=qty,
                price=0.0,
                fee=0.0,
                order_id="",
                executed=False,
                dry_run=False,
                error=str(exc),
            )

    async def close(self) -> None:
        """Close the underlying exchange connection if one was opened."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
