# Trading Strategies

All strategies are in `trading-bot/strategies/` and loadable by name via the factory.

## Available Strategies

| Name | File | Signal logic |
|---|---|---|
| `rsi` | `rsi.py` | RSI(14): buy < 30, sell > 70 |
| `ema_cross` | `ema_cross.py` | EMA 9/21 crossover |
| `macd` | `macd.py` | MACD histogram sign change |
| `bollinger` | `bollinger.py` | Price outside Bollinger Bands |
| `vwap` | `vwap.py` | Close vs VWAP ± 1.5σ band mean-reversion |
| `stoch_rsi` | `stoch_rsi.py` | Stoch %K/%D cross in overbought/oversold zones |

## Using the Factory

```python
from strategies.factory import get_strategy, list_strategies

strat = get_strategy("rsi")                    # by name
strat = get_strategy("rsi", period=21)         # with kwargs
strat = get_strategy()                         # reads DEFAULT_STRATEGY env var
strat = get_strategy("rsi", config_path="./strategy_config.json")  # use optimized params

list_strategies()  # ['bollinger', 'ema_cross', 'macd', 'rsi', 'stoch_rsi', 'vwap']
```

## Parameter Optimizer

```bash
cd trading-bot
python -m strategies.optimizer --symbols BTC/USDT ETH/USDT --strategies rsi ema_cross --days 90
```

Runs a grid search over key parameters using synthetic backtest data, then saves best params to `./strategy_config.json`. The factory can load this file automatically.

```python
from strategies.optimizer import optimize_strategy, save_best_params

result = optimize_strategy("BTC/USDT", "rsi", days=90, metric="sharpe")
print(result.best_params)   # e.g. {"period": 21, "overbought": 70, "oversold": 30}
print(result.best_sharpe)
```

## Adding a New Strategy

1. Create `trading-bot/strategies/my_strat.py` subclassing `BaseStrategy`
2. Implement `add_indicators(df) -> df` and `generate_signal(df) -> "buy"|"sell"|None`
3. Register in `factory.py` STRATEGY_REGISTRY dict
4. Update `__init__.py` exports

## ML Veto Layer

Any strategy signal can be gated through a RandomForest model trained on the same data.

```bash
# Train a model for BTC/USDT + RSI strategy
cd trading-bot
python -m ml.trainer --symbol BTC/USDT --strategy rsi

# Pre-train all 4 strategies x 5 symbols at once
python -m ml.pretrain_all
# → saves to ./ml_models/BTC_USDT_rsi.pkl etc.
```

Set `ML_MODEL_PATH=./ml_models/{symbol}_{strategy}.pkl` (or the specific path) and the swarm will automatically use the model to veto low-confidence signals.
