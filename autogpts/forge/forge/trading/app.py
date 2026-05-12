"""Trading agent ASGI entrypoint.

Start with:
  uvicorn forge.trading.app:app --host 0.0.0.0 --port 8000 --reload
"""
import os

from forge.sdk import LocalWorkspace

from .trading_agent import TradingAgent
from .trading_db import TradingDB

database_name = os.getenv("DATABASE_STRING", "sqlite:///trading_agent.db")
workspace_path = os.getenv("AGENT_WORKSPACE", "./workspace")

database = TradingDB(database_name, debug_enabled=False)
workspace = LocalWorkspace(workspace_path)

agent = TradingAgent(database=database, workspace=workspace)
app = agent.get_agent_app()
