# Sentiment Layer

The sentiment module (`trading-bot/sentiment/`) blends strategy signals with market sentiment from news and social sources.

## Providers

| Provider | Key required | Source |
|---|---|---|
| `mock` | None | Seeded deterministic (same day = same result) |
| `coingecko` | None | CoinGecko community vote% + keyword scoring |
| `cryptopanic` | `CRYPTOPANIC_API_KEY` | Post votes + bullish/bearish labels |
| `reddit` | `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Hot posts from crypto subreddits |
| `multi` | Optional | Weighted average of all available providers |

```bash
export SENTIMENT_PROVIDER=coingecko   # switch provider at runtime
```

## Signal Blending Logic

The `SignalBlender` combines the strategy signal with a sentiment score:

| Condition | Action | Effect |
|---|---|---|
| Confidence < 0.30 | `passed` | Signal passes unchanged |
| Sentiment contradicts signal strongly (> 0.35) | `contradicted` | Signal blocked |
| Sentiment strongly aligns with signal (> 0.40) | `boosted` | Signal confirmed with boost |
| Mild alignment | `confirmed` | Signal passes with note |
| `raw_signal is None` | `neutral` | No signal |

## Keyword Scorer

The `KeywordScorer` uses a weighted lexicon of ~50 bullish/bearish crypto terms with negation handling:

```python
from sentiment.scorer import KeywordScorer
scorer = KeywordScorer()
score = scorer.score_text("Bitcoin breaks all-time high, not bearish at all")
# SentimentScore(score=+0.6, confidence=0.8, ...)
```

## BlendResult

```python
blend = blender.blend("BTC/USDT", "buy")
blend.final_signal    # "buy" | "sell" | None
blend.action          # "confirmed" | "contradicted" | "boosted" | "passed" | "neutral"
blend.sentiment.score # float -1..+1
blend.sentiment.label # "bullish" | "bearish" | "neutral"
blend.to_dict()       # serializable for SwarmResult
```
