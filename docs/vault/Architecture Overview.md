# Trading Bot — Architecture Overview

**Repo:** `FatPimp420/AutoGPT` · branch `claude/init-ruvflo-project-7UmHV`
**Last updated:** 2026-05-12

---

## System Map

```
trading-bot/              ← pure Python algo engine
├── strategies/           ← RSI, EMA Cross, MACD, Bollinger, VWAP, StochRSI
├── backtest/             ← Backtester + synthetic OHLCV generator
├── risk/                 ← RiskManager (position sizing, daily loss limit)
├── sentiment/            ← KeywordScorer, SignalBlender, real providers
├── ml/                   ← MLSignalModel (RandomForest), trainer, pre-trainer
└── bot/                  ← LiveExecutor (ccxt), PositionManager, CLI

autogpts/forge/forge/trading/   ← AutoGPT Forge integration
├── trading_db.py         ← SQLite: trade_log + signal_history tables
├── trading_agent.py      ← ForgeAgent subclass, execute_step() dispatcher
├── swarm.py              ← 6-stage multi-agent pipeline
├── scheduler.py          ← Background loop: runs pipeline every N minutes
├── app.py                ← FastAPI ASGI app + lifespan wiring
└── report.py             ← ASCII + CSV trade log CLI
```

---

## 6-Stage Swarm Pipeline

```
DataFetcher → SignalGenerator → SentimentAgent → MLVetoAgent → RiskChecker → SmartExecutor
```

| Stage | Role | Failure mode |
|---|---|---|
| DataFetcher | Generate synthetic OHLCV, compute strategy signal | Skip symbol |
| SignalGenerator | Validate + annotate signal | Pass error through |
| SentimentAgent | Blend signal with news/social sentiment | Fail-open (pass original) |
| MLVetoAgent | Gate signal through RandomForest confidence | Skip if no model file |
| RiskChecker | Apply position sizing + daily loss limit | Block trade |
| SmartExecutor | Paper trade log OR live ccxt order | Log error, no crash |

---

## Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `DEFAULT_STRATEGY` | `rsi` | Strategy loaded by factory |
| `SYMBOLS` | 5 major pairs | Symbols run by scheduler/swarm |
| `SIGNAL_INTERVAL_MINUTES` | `60` | Scheduler run interval |
| `SENTIMENT_PROVIDER` | `mock` | `mock / coingecko / cryptopanic / reddit / multi` |
| `ML_MODEL_PATH` | `./ml_models/{symbol}_{strategy}.pkl` | Path to trained model |
| `LIVE_TRADING` | `false` | `true` = real ccxt orders |
| `CRYPTOPANIC_API_KEY` | — | Required for CryptoPanic provider |
| `REDDIT_CLIENT_ID` | — | Required for Reddit provider |
| `REDDIT_CLIENT_SECRET` | — | Required for Reddit provider |
| `TRADING_BOT_PATH` | auto-detected | Override trading-bot sys.path |
| `DATABASE_STRING` | `sqlite:///agent.db` | SQLAlchemy URL |

---

## Starting the Server

```bash
cd autogpts/forge
export DATABASE_STRING=sqlite:///trading_agent.db
export AGENT_WORKSPACE=./workspace
export TRADING_BOT_PATH=../../trading-bot
poetry run uvicorn forge.trading.app:app --host 0.0.0.0 --port 8000 --reload
# or use the startup script:
bash ../../start_trading_swarm.sh
```

## API Endpoints

| Route | Description |
|---|---|
| `POST /ap/v1/agent/tasks` | Create a task |
| `POST /ap/v1/agent/tasks/{id}/steps` | Execute a step (see actions below) |
| `GET /trading/dashboard` | Live JSON status page |

### execute_step actions

```json
{"action": "status"}
{"action": "signal", "symbols": ["BTC/USDT"], "strategy": "rsi"}
{"action": "backtest", "symbols": ["BTC/USDT"], "strategy": "rsi", "days": 90}
{"action": "portfolio", "symbols": ["BTC/USDT", "ETH/USDT"], "strategy": "rsi"}
{"action": "swarm", "symbols": ["BTC/USDT"], "strategy": "rsi", "equity": 10000}
```

---

## Test Coverage

```
trading-bot/tests/    129 tests
forge/trading/        42 tests
─────────────────────────────
Total                 171 tests  ✅ all passing
```
