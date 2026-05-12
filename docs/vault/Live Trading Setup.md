# Live Trading Setup

Everything runs in **paper mode by default** — no real orders are placed unless `LIVE_TRADING=true` is set.

## Enabling Live Trading

```bash
export LIVE_TRADING=true
export EXCHANGE_ID=binance          # any ccxt-supported exchange
export EXCHANGE_API_KEY=your_key
export EXCHANGE_API_SECRET=your_secret
```

The `SmartExecutor` stage in the swarm checks `LIVE_TRADING` and routes to `LiveExecutor` (real ccxt market orders) when true.

## Paper Mode (default)

When `LIVE_TRADING` is unset or `false`, approved signals are logged as paper trades:
- Full pipeline runs (signal → sentiment → ML veto → risk check)
- Trade is logged to TradingDB with `executed=True`
- No exchange connection required

## Checklist Before Going Live

- [ ] Train ML models on real historical data (not just synthetic)
- [ ] Connect real sentiment providers (CoinGecko + CryptoPanic minimum)
- [ ] Set conservative equity and risk params (start small)
- [ ] Run paper mode for at least 1 week and review report CLI output
- [ ] Set `SIGNAL_INTERVAL_MINUTES` to desired cadence (e.g. `60` for hourly)
- [ ] Confirm `EXCHANGE_ID` supports your trading pairs
- [ ] Set API keys with trade-only permissions (no withdrawal)

## Report CLI

```bash
cd autogpts/forge
poetry run python -m forge.trading.report --last 50
poetry run python -m forge.trading.report --task-id scheduled-20260512T100000 --csv trades.csv
```

## Dashboard

```
GET http://localhost:8000/trading/dashboard
```

Returns live JSON: recent trades, recent signals, win rate, total PnL, symbols traded.

## PositionManager

Tracks open positions in-memory across swarm runs:

```python
from bot.position_manager import PositionManager
pm = PositionManager()
pm.open_position("BTC/USDT", "buy", price=50000, qty=0.01)
pm.update_price("BTC/USDT", current_price=51000)
print(pm.get_summary())
# {"open_count": 1, "total_unrealized_pnl_pct": 0.02, "symbols": ["BTC/USDT"]}
result = pm.close_position("BTC/USDT", close_price=51000)
print(result["realized_pnl_pct"])  # 0.02
```
