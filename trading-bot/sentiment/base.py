from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Headline:
    text: str
    source: str = "unknown"
    weight: float = 1.0  # source credibility multiplier


@dataclass
class SentimentScore:
    symbol: str
    score: float          # -1.0 (max bearish) … +1.0 (max bullish)
    confidence: float     # 0.0 … 1.0
    headline_count: int = 0
    headlines: List[Headline] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score > 0.3:
            return "bullish"
        if self.score < -0.3:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "headline_count": self.headline_count,
        }


class SentimentProvider(ABC):
    """Fetch and score sentiment for a crypto symbol."""

    @abstractmethod
    def get_headlines(self, symbol: str) -> List[Headline]:
        """Return recent headlines for the symbol."""

    @abstractmethod
    def get_score(self, symbol: str) -> SentimentScore:
        """Return an aggregated SentimentScore for the symbol."""
