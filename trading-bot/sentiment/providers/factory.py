"""Sentiment provider factory.

Reads SENTIMENT_PROVIDER env var (default: mock) and returns the right provider.

  SENTIMENT_PROVIDER=mock          → MockSentimentProvider (no network)
  SENTIMENT_PROVIDER=coingecko     → CoinGeckoProvider (free, no key)
  SENTIMENT_PROVIDER=cryptopanic   → CryptoPanicProvider (needs CRYPTOPANIC_API_KEY)
  SENTIMENT_PROVIDER=reddit        → RedditProvider (needs REDDIT_CLIENT_ID/SECRET)
  SENTIMENT_PROVIDER=multi         → MultiProvider (all available sources averaged)
"""
import os
from typing import Optional

from sentiment.base import SentimentProvider, SentimentScore, Headline


def get_provider(name: Optional[str] = None) -> SentimentProvider:
    """Return a SentimentProvider instance for the given name (or env var)."""
    choice = (name or os.getenv("SENTIMENT_PROVIDER", "mock")).lower().strip()

    if choice == "mock":
        from sentiment.mock import MockSentimentProvider
        return MockSentimentProvider()

    if choice == "coingecko":
        from .coingecko import CoinGeckoProvider
        return CoinGeckoProvider(api_key=os.getenv("COINGECKO_API_KEY"))

    if choice == "cryptopanic":
        from .cryptopanic import CryptoPanicProvider
        return CryptoPanicProvider(api_key=os.getenv("CRYPTOPANIC_API_KEY"))

    if choice == "reddit":
        from .reddit import RedditProvider
        return RedditProvider(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        )

    if choice == "multi":
        return _build_multi_provider()

    raise ValueError(
        f"Unknown SENTIMENT_PROVIDER '{choice}'. "
        "Choose: mock, coingecko, cryptopanic, reddit, multi"
    )


def _build_multi_provider() -> "MultiProvider":
    """Build a MultiProvider from whichever sources have credentials."""
    providers = []

    # CoinGecko always available
    from .coingecko import CoinGeckoProvider
    providers.append(("coingecko", CoinGeckoProvider(), 1.0))

    if os.getenv("CRYPTOPANIC_API_KEY"):
        from .cryptopanic import CryptoPanicProvider
        providers.append(("cryptopanic", CryptoPanicProvider(), 1.2))

    if os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"):
        from .reddit import RedditProvider
        providers.append(("reddit", RedditProvider(), 0.9))

    if not providers:
        from sentiment.mock import MockSentimentProvider
        return MockSentimentProvider()  # type: ignore[return-value]

    return MultiProvider(providers)


class MultiProvider(SentimentProvider):
    """Aggregates scores from multiple providers with configurable weights."""

    def __init__(self, providers: list):
        # providers: list of (name, provider, weight)
        self._providers = providers

    def get_headlines(self, symbol: str) -> list:
        all_headlines = []
        for _name, provider, _weight in self._providers:
            try:
                all_headlines.extend(provider.get_headlines(symbol))
            except Exception:
                pass
        return all_headlines

    def get_score(self, symbol: str) -> SentimentScore:
        scores = []
        for _name, provider, weight in self._providers:
            try:
                s = provider.get_score(symbol)
                if s.confidence > 0:
                    scores.append((s, weight))
            except Exception:
                pass

        if not scores:
            return SentimentScore(symbol=symbol, score=0.0, confidence=0.0,
                                  headline_count=0)

        total_weight = sum(w for _, w in scores)
        blended_score = sum(s.score * w for s, w in scores) / total_weight
        avg_confidence = sum(s.confidence * w for s, w in scores) / total_weight
        total_headlines = sum(s.headline_count for s, _ in scores)

        return SentimentScore(
            symbol=symbol,
            score=round(blended_score, 4),
            confidence=round(avg_confidence, 4),
            headline_count=total_headlines,
        )
