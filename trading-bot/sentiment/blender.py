"""Signal blender — merges a strategy signal with a sentiment score.

Blending rules:
  - signal=None         → always None (no technical trigger)
  - low confidence (<0.3) → pass strategy signal through unchanged
  - confirmed: signal agrees with sentiment direction → keep signal
  - contradicted: signal opposes strong sentiment  → block (return None)
  - boosted: signal agrees with strong sentiment   → keep + annotate

The thresholds are configurable so they can be tuned once real data arrives.
"""
from dataclasses import dataclass
from typing import Optional

from .base import SentimentProvider, SentimentScore


@dataclass
class BlendResult:
    original_signal: Optional[str]   # buy / sell / None from strategy
    final_signal: Optional[str]      # buy / sell / None after blending
    sentiment: SentimentScore
    action: str                       # confirmed | contradicted | boosted | neutral | passed

    def to_dict(self) -> dict:
        return {
            "original_signal": self.original_signal,
            "final_signal": self.final_signal,
            "sentiment": self.sentiment.to_dict(),
            "blend_action": self.action,
        }


class SignalBlender:
    """Combine strategy signals with sentiment scores."""

    def __init__(
        self,
        provider: SentimentProvider,
        min_confidence: float = 0.3,    # ignore sentiment below this
        block_threshold: float = 0.35,  # contradict signal if |score| > this
        boost_threshold: float = 0.40,  # annotate as boosted if |score| > this
    ):
        self.provider = provider
        self.min_confidence = min_confidence
        self.block_threshold = block_threshold
        self.boost_threshold = boost_threshold

    def blend(self, symbol: str, strategy_signal: Optional[str]) -> BlendResult:
        sentiment = self.provider.get_score(symbol)

        # No technical signal → nothing to blend
        if strategy_signal is None:
            return BlendResult(
                original_signal=None,
                final_signal=None,
                sentiment=sentiment,
                action="neutral",
            )

        # Low-confidence sentiment → trust the strategy
        if sentiment.confidence < self.min_confidence:
            return BlendResult(
                original_signal=strategy_signal,
                final_signal=strategy_signal,
                sentiment=sentiment,
                action="passed",
            )

        score = sentiment.score
        is_buy = strategy_signal == "buy"
        sentiment_agrees = (is_buy and score > 0) or (not is_buy and score < 0)
        sentiment_strong = abs(score) >= self.block_threshold

        if sentiment_strong and not sentiment_agrees:
            return BlendResult(
                original_signal=strategy_signal,
                final_signal=None,          # blocked
                sentiment=sentiment,
                action="contradicted",
            )

        if abs(score) >= self.boost_threshold and sentiment_agrees:
            return BlendResult(
                original_signal=strategy_signal,
                final_signal=strategy_signal,
                sentiment=sentiment,
                action="boosted",
            )

        return BlendResult(
            original_signal=strategy_signal,
            final_signal=strategy_signal,
            sentiment=sentiment,
            action="confirmed",
        )
