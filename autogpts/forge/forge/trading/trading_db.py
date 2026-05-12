import datetime
import uuid
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, String
from sqlalchemy.exc import SQLAlchemyError

from forge.db import ForgeDatabase
from forge.sdk import Base, ForgeLogger

LOG = ForgeLogger(__name__)


class TradeLogModel(Base):
    __tablename__ = "trade_log"

    trade_id = Column(String, primary_key=True, index=True)
    task_id = Column(String, index=True)
    agent_role = Column(String)
    symbol = Column(String)
    side = Column(String)
    price = Column(Float)
    qty = Column(Float)
    fee = Column(Float)
    pnl_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SignalHistoryModel(Base):
    __tablename__ = "signal_history"

    signal_id = Column(String, primary_key=True, index=True)
    task_id = Column(String, index=True)
    symbol = Column(String)
    strategy = Column(String)
    signal = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class TradingDB(ForgeDatabase):
    async def log_trade(
        self,
        task_id: str,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        fee: float,
        agent_role: str = "executor",
        pnl_pct: Optional[float] = None,
    ) -> dict:
        try:
            with self.Session() as session:
                record = TradeLogModel(
                    trade_id=str(uuid.uuid4()),
                    task_id=task_id,
                    agent_role=agent_role,
                    symbol=symbol,
                    side=side,
                    price=price,
                    qty=qty,
                    fee=fee,
                    pnl_pct=pnl_pct,
                )
                session.add(record)
                session.commit()
                return {"trade_id": record.trade_id, "symbol": symbol, "side": side}
        except SQLAlchemyError as e:
            LOG.error(f"Error logging trade: {e}")
            raise

    async def log_signal(
        self, task_id: str, symbol: str, strategy: str, signal: str
    ) -> dict:
        try:
            with self.Session() as session:
                record = SignalHistoryModel(
                    signal_id=str(uuid.uuid4()),
                    task_id=task_id,
                    symbol=symbol,
                    strategy=strategy,
                    signal=signal,
                )
                session.add(record)
                session.commit()
                return {"signal_id": record.signal_id, "symbol": symbol, "signal": signal}
        except SQLAlchemyError as e:
            LOG.error(f"Error logging signal: {e}")
            raise

    async def get_trade_history(self, task_id: str) -> List[dict]:
        with self.Session() as session:
            rows = (
                session.query(TradeLogModel)
                .filter(TradeLogModel.task_id == task_id)
                .order_by(TradeLogModel.created_at)
                .all()
            )
            return [
                {
                    "trade_id": r.trade_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "price": r.price,
                    "qty": r.qty,
                    "fee": r.fee,
                    "pnl_pct": r.pnl_pct,
                    "agent_role": r.agent_role,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def get_signal_history(self, task_id: str) -> List[dict]:
        with self.Session() as session:
            rows = (
                session.query(SignalHistoryModel)
                .filter(SignalHistoryModel.task_id == task_id)
                .order_by(SignalHistoryModel.created_at)
                .all()
            )
            return [
                {
                    "signal_id": r.signal_id,
                    "symbol": r.symbol,
                    "strategy": r.strategy,
                    "signal": r.signal,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def get_all_trades(self, limit: int = 20) -> List[dict]:
        """Return the most recent *limit* trades across all task IDs, newest first."""
        with self.Session() as session:
            rows = (
                session.query(TradeLogModel)
                .order_by(TradeLogModel.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "trade_id": r.trade_id,
                    "task_id": r.task_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "price": r.price,
                    "qty": r.qty,
                    "fee": r.fee,
                    "pnl_pct": r.pnl_pct,
                    "agent_role": r.agent_role,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def get_all_signals(self, limit: int = 20) -> List[dict]:
        """Return the most recent *limit* signals across all task IDs, newest first."""
        with self.Session() as session:
            rows = (
                session.query(SignalHistoryModel)
                .order_by(SignalHistoryModel.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "signal_id": r.signal_id,
                    "task_id": r.task_id,
                    "symbol": r.symbol,
                    "strategy": r.strategy,
                    "signal": r.signal,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]
