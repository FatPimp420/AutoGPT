"""Trading agent ASGI entrypoint.

Start with:
  uvicorn forge.trading.app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from forge.sdk import LocalWorkspace

from .trading_agent import TradingAgent
from .trading_db import TradingDB
from .swarm import TradingSwarm
from .scheduler import TradingScheduler

database_name = os.getenv("DATABASE_STRING", "sqlite:///trading_agent.db")
workspace_path = os.getenv("AGENT_WORKSPACE", "./workspace")

database = TradingDB(database_name, debug_enabled=False)
workspace = LocalWorkspace(workspace_path)

agent = TradingAgent(database=database, workspace=workspace)

# Build a scheduler instance that will be started/stopped via the lifespan.
_scheduler = TradingScheduler(db=database, swarm=TradingSwarm(db=database))


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """FastAPI lifespan: start the background scheduler on startup, stop on shutdown."""
    _scheduler.start()
    try:
        yield
    finally:
        await _scheduler.stop()


# Obtain the base app from the agent SDK, then attach the lifespan.
app = agent.get_agent_app()
app.router.lifespan_context = _lifespan
