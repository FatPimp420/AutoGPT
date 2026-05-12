# Build Log

Chronological record of what was built in each session.

---

## Session 1 — Foundation

- Ruflo `.mcp.json` config (hierarchical-mesh, 50 agents, hybrid memory)
- `.claude-flow/workflows/trading_swarm.yaml`
- `TradingDB` — SQLite trade_log + signal_history tables
- `TradingAgent` — ForgeAgent subclass with execute_step() dispatcher
- `app.py` — FastAPI ASGI entrypoint
- `start_trading_swarm.sh` — startup script
- **18 tests** in test_trading.py

## Session 2 — Sentiment Layer

- `KeywordScorer` — 50-word weighted lexicon, negation handling
- `MockSentimentProvider` — seeded deterministic, no API needed
- `SignalBlender` — confirm/contradict/boost logic
- Real providers: CoinGeckoProvider, CryptoPanicProvider, RedditProvider
- Provider factory (SENTIMENT_PROVIDER env var)
- TradingSwarm expanded from 4 → 5 stages (SentimentAgent added)
- **+40 tests** (test_sentiment.py, test_providers.py)

## Session 3 — ML + Live Executor + Report CLI

- `MLSignalModel` — RandomForest wrapping any BaseStrategy
- `ml/features.py` — feature engineering (RSI, MACD, EMA ratio, BB%, returns, volume zscore, sentiment)
- `ml/trainer.py` — CLI trainer
- `LiveExecutor` — ccxt market orders, LIVE_TRADING env var gate
- `report.py` — ASCII + CSV trade log CLI
- **+16 tests** (test_ml.py, test_live_executor.py, test_report.py)

## Session 4 — 6-Stage Pipeline + Strategy Factory

- TradingSwarm expanded from 5 → 6 stages (MLVetoAgent + SmartExecutor)
- `strategies/factory.py` — get_strategy(), DEFAULT_STRATEGY env var
- test_trading.py updated: 18 pipeline results (6 × 3 symbols), agents_ran=6
- **+12 tests** (test_factory.py)
- CI fix: added `ta` + `scikit-learn` to forge pyproject.toml
- `.github/workflows/trading-ci.yml` — parallel CI for both test suites

## Session 5 — Scheduler + Dashboard + Optimizer + More

- `scheduler.py` — TradingScheduler background loop, wired into app lifespan
- `GET /trading/dashboard` endpoint — live status JSON
- `strategies/optimizer.py` — grid search, save/load JSON config
- `strategies/vwap.py` — VWAP mean-reversion strategy
- `strategies/stoch_rsi.py` — Stochastic RSI strategy
- `bot/position_manager.py` — track open positions, unrealized PnL, MAE/MFE
- `ml/pretrain_all.py` — batch pre-train 20 models (4 strategies × 5 symbols)
- Richer backtest metrics: max_drawdown, calmar_ratio, profit_factor, max_consecutive_losses
- **Total: 171 tests** (129 trading-bot + 42 forge) ✅

---

## Next Steps (when home)

- [ ] Connect Obsidian MCP: set `autoStart: true` in `.mcp.json`
- [ ] Read real strategy notes from vault and replace synthetic params
- [ ] Connect CoinGecko (no key) — `SENTIMENT_PROVIDER=coingecko`
- [ ] Add CryptoPanic key → `SENTIMENT_PROVIDER=multi`
- [ ] Train ML models on real Binance OHLCV data
- [ ] Run paper trading for 1 week, review report CLI
- [ ] Enable `LIVE_TRADING=true` with small position sizing
