import os
from loguru import logger


class RiskManager:
    def __init__(self):
        self.max_position_pct = float(os.getenv("MAX_POSITION_PCT", 0.05))
        self.max_daily_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", 0.02))
        self.max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", 0.10))

        self._daily_loss = 0.0
        self._peak_equity = None
        self._equity = None

    def position_size(self, quote_balance: float, price: float) -> float:
        """Kelly-lite: fixed fractional sizing."""
        notional = quote_balance * self.max_position_pct
        return round(notional / price, 6)

    def allow_trade(self, signal: str, size: float, price: float) -> bool:
        if self._daily_loss >= self.max_daily_loss_pct:
            logger.warning(f"Daily loss limit hit ({self._daily_loss:.2%})")
            return False
        if self._equity and self._peak_equity:
            drawdown = (self._peak_equity - self._equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                logger.warning(f"Max drawdown hit ({drawdown:.2%})")
                return False
        return True

    def record_trade(self, signal: str, size: float, price: float):
        pass  # TODO: update equity tracking after fill confirmation

    def update_equity(self, equity: float):
        self._equity = equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity
